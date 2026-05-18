"""SQLite-backed dashboard backend for Teuton.

Two-tier data sourcing:

- **Outstanding work** comes from the orchestrator's :mod:`teuton_runtime.queue`
  snapshot (``v3/.../runs/{run_id}/queue/{role}.json``). The dashboard reads it
  live on every ``/api/snapshot`` and every ``/api/queue`` -- it's a single
  small object that updates ~every 0.5s in production.
- **Receipts / verdicts / audits / heartbeats / chain** stay in SQLite,
  indexed by background loops, since those prefixes are not bounded by the
  queue model. Completed jobs are derived from receipts JOIN verdicts at
  query time; there is no longer a per-job SQLite table.

Queue-depth history (last 30 min) is kept in a module-level in-memory ring
buffer keyed by ``(netuid, run_id, role)`` -- sampled at the bucket indexer
cadence. It's transient: dashboard restart loses history, which is fine.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from teuton_core import paths
from teuton_core.discovery_ui import INDEX_HTML, _load_logo_bytes
from teuton_core.protocol import AuditResultV3, JobManifestV3, JobReceiptV3, VerificationVerdictV3
from teuton_runtime.discovery import scan_bucket_discovery_records
from teuton_runtime.queue import QueueState, read_queue
from teuton_runtime.storage import ObjectStore


# Default per-hotkey queue cap. Mirrors ``TEUTON_MAX_INFLIGHT_PER_HOTKEY`` on
# the orchestrator -- set the same env on the dashboard host so the "at-cap"
# threshold reflects reality. We default to 8 to match the orchestrator's
# default; a value of 0 means "unknown / unbounded" and disables backpressure
# rendering on the UI.
_DEFAULT_MAX_INFLIGHT = int(os.environ.get("TEUTON_MAX_INFLIGHT_PER_HOTKEY", "8"))

# Queue-depth ring buffer parameters. 30 min @ 5s = 360 samples per stream.
_QUEUE_HISTORY_SECONDS = 30 * 60
_QUEUE_HISTORY_MAX_POINTS = 400

# Per-(netuid, run_id, role) -> deque[QueueHistoryPoint]. Module-level so
# samples survive across requests; reset on dashboard restart.
_QUEUE_HISTORY: dict[tuple[int, str, str], deque] = {}
_QUEUE_HISTORY_LOCK = threading.Lock()


@dataclass(frozen=False)
class DashboardConfig:
    netuid: int
    run_id: str | None = None
    db_path: str = "/var/lib/teuton-dashboard/dashboard.sqlite3"
    host: str = "127.0.0.1"
    port: int = 8765
    refresh_sec: float = 3.0
    heartbeat_ttl_sec: float | None = 30.0
    # Completed-jobs window. Outstanding work is bounded by the orchestrator
    # queue itself, so we no longer need a generous floor here.
    max_jobs: int = 200
    bucket_poll_sec: float = 5.0
    chain_poll_sec: float = 30.0
    network: str = "finney"
    open_browser: bool = False
    # Per-hotkey queue cap mirrored from the orchestrator. Used for "at-cap"
    # detection and the per-miner inflight bar; 0 disables the indicator.
    max_inflight_per_hotkey: int = _DEFAULT_MAX_INFLIGHT

    def __post_init__(self) -> None:
        if self.max_jobs < 50:
            self.max_jobs = 50


@dataclass(frozen=True)
class QueueHistoryPoint:
    ts: int
    depth_total: int
    at_cap_count: int


@dataclass(frozen=True)
class QueueSnapshot:
    """Live view of one ``(run_id, role)`` queue."""

    run_id: str
    role: str
    snapshot_unix: int
    snapshot_id: int
    depth_total: int
    depth_by_hotkey: dict[str, int]
    max_inflight_per_hotkey: int
    at_cap_count: int
    at_cap_hotkeys: list[str]
    oldest_entry_age_sec: float | None
    oldest_job_id: str | None
    history: list[QueueHistoryPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "role": self.role,
            "snapshot_unix": int(self.snapshot_unix),
            "snapshot_id": int(self.snapshot_id),
            "depth_total": int(self.depth_total),
            "depth_by_hotkey": dict(self.depth_by_hotkey),
            "max_inflight_per_hotkey": int(self.max_inflight_per_hotkey),
            "at_cap_count": int(self.at_cap_count),
            "at_cap_hotkeys": list(self.at_cap_hotkeys),
            "oldest_entry_age_sec": self.oldest_entry_age_sec,
            "oldest_job_id": self.oldest_job_id,
            "history": [
                {"ts": p.ts, "depth_total": p.depth_total, "at_cap_count": p.at_cap_count}
                for p in self.history
            ],
        }


class DashboardDB:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self) -> None:
        with self._lock, self.connect() as conn:
            # The legacy ``jobs`` table is intentionally NOT created. Outstanding
            # work comes from the orchestrator queue (live bucket read); completed
            # work is reconstructed from receipts JOIN verdicts. We DROP any
            # leftover ``jobs`` table from older deployments so the schema stays
            # consistent with the new design.
            conn.execute("DROP TABLE IF EXISTS jobs")
            # Migrate existing receipts table to add the ``kind`` column we
            # now project onto completed jobs. ADD COLUMN is idempotent only
            # if we guard on existence.
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(receipts)").fetchall()}
                if cols and "kind" not in cols:
                    conn.execute("ALTER TABLE receipts ADD COLUMN kind TEXT")
            except Exception:
                pass
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    first_seen_unix INTEGER,
                    last_seen_unix INTEGER,
                    receipt_count INTEGER DEFAULT 0,
                    PRIMARY KEY (netuid, run_id)
                );
                CREATE TABLE IF NOT EXISTS workers (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    hotkey TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    host_id TEXT,
                    role TEXT,
                    status TEXT,
                    last_seen_unix INTEGER,
                    capabilities_json TEXT,
                    miner_json TEXT,
                    worker_json TEXT,
                    PRIMARY KEY (netuid, run_id, hotkey, worker_id, role)
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    job_id TEXT,
                    hotkey TEXT,
                    worker_id TEXT,
                    kind TEXT,
                    compute_sec REAL,
                    bytes_read INTEGER,
                    bytes_written INTEGER,
                    finished_unix INTEGER,
                    receipt_json TEXT,
                    PRIMARY KEY (netuid, run_id, receipt_id)
                );
                CREATE TABLE IF NOT EXISTS verdicts (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    job_id TEXT,
                    validator_hotkey TEXT,
                    status TEXT,
                    checked_unix INTEGER,
                    verdict_json TEXT,
                    PRIMARY KEY (netuid, run_id, receipt_id, validator_hotkey)
                );
                CREATE TABLE IF NOT EXISTS audits (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    job_id TEXT,
                    auditor_hotkey TEXT,
                    status TEXT,
                    checked_unix INTEGER,
                    audit_json TEXT,
                    PRIMARY KEY (netuid, run_id, receipt_id, auditor_hotkey)
                );
                CREATE TABLE IF NOT EXISTS chain_hotkeys (
                    netuid INTEGER NOT NULL,
                    hotkey TEXT NOT NULL,
                    uid INTEGER,
                    stake REAL,
                    incentive REAL,
                    emission REAL,
                    validator_permit INTEGER,
                    last_update_block INTEGER,
                    observed_block INTEGER,
                    observed_unix INTEGER,
                    PRIMARY KEY (netuid, hotkey)
                );
                CREATE TABLE IF NOT EXISTS chain_meta (
                    netuid INTEGER PRIMARY KEY,
                    current_block INTEGER,
                    tempo INTEGER,
                    weights_set_rate_limit INTEGER,
                    observed_unix INTEGER,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS indexer_state (
                    name TEXT PRIMARY KEY,
                    cursor_json TEXT,
                    updated_unix INTEGER,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_receipts_run_finished ON receipts(netuid, run_id, finished_unix DESC);
                CREATE INDEX IF NOT EXISTS idx_receipts_run_job ON receipts(netuid, run_id, job_id);
                CREATE INDEX IF NOT EXISTS idx_verdicts_run_job ON verdicts(netuid, run_id, job_id);
                CREATE INDEX IF NOT EXISTS idx_audits_run_job ON audits(netuid, run_id, job_id);
                CREATE INDEX IF NOT EXISTS idx_workers_run ON workers(netuid, run_id);
                """
            )

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        with self._lock, self.connect() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock, self.connect() as conn:
            return list(conn.execute(sql, params))

    def set_state(self, name: str, *, cursor: dict | None = None, error: str | None = None) -> None:
        self.execute(
            """
            INSERT INTO indexer_state(name, cursor_json, updated_unix, error)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                cursor_json=excluded.cursor_json,
                updated_unix=excluded.updated_unix,
                error=excluded.error
            """,
            (name, json.dumps(cursor or {}, sort_keys=True), int(time.time()), error),
        )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def serve_dashboard_backend(*, bucket: ObjectStore, config: DashboardConfig) -> None:
    db = DashboardDB(config.db_path)
    stop = threading.Event()
    threads = [
        threading.Thread(target=_bucket_loop, args=(bucket, db, config, stop), daemon=True),
        threading.Thread(target=_chain_loop, args=(db, config, stop), daemon=True),
    ]
    for thread in threads:
        thread.start()

    handler = _handler(bucket=bucket, db=db, config=config)
    server = ThreadingHTTPServer((config.host, int(config.port)), handler)
    url = f"http://{config.host}:{config.port}/"
    print(
        f"[dashboard-backend] serving {url} netuid={config.netuid} "
        f"run_id={config.run_id or '<auto>'} db={config.db_path}"
    )
    if config.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard-backend] stopped")
    finally:
        stop.set()
        server.server_close()


def _bucket_loop(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, stop: threading.Event) -> None:
    while not stop.is_set():
        t0 = time.time()
        try:
            n = index_bucket_once(bucket=bucket, db=db, config=config)
            db.set_state("bucket", cursor={"indexed": n, "seconds": round(time.time() - t0, 3)}, error=None)
        except Exception:
            db.set_state("bucket", error=traceback.format_exc(limit=6))
        stop.wait(config.bucket_poll_sec)


def _chain_loop(db: DashboardDB, config: DashboardConfig, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            index_chain_once(db=db, config=config)
        except Exception:
            db.execute(
                """
                INSERT INTO chain_meta(netuid, observed_unix, error)
                VALUES (?, ?, ?)
                ON CONFLICT(netuid) DO UPDATE SET observed_unix=excluded.observed_unix, error=excluded.error
                """,
                (config.netuid, int(time.time()), traceback.format_exc(limit=6)),
            )
            db.set_state("chain", error=traceback.format_exc(limit=6))
        stop.wait(config.chain_poll_sec)


# ---------------------------------------------------------------------------
# Indexers
# ---------------------------------------------------------------------------


def index_bucket_once(*, bucket: ObjectStore, db: DashboardDB, config: DashboardConfig) -> int:
    now = int(time.time())
    run_ids = _selected_run_ids(bucket, config)
    indexed = 0
    for run_id in run_ids:
        _upsert_run(db, config.netuid, run_id, now)
        indexed += _index_workers(bucket, db, config, run_id)
        indexed += _index_receipts(bucket, db, config, run_id)
        indexed += _index_verdicts(bucket, db, config, run_id)
        indexed += _index_audits(bucket, db, config, run_id)
        # Sample the queue snapshot into the in-memory ring buffer so the UI
        # can render a 30-min depth timeseries.
        _sample_queue_history(bucket, config, run_id, role="train", now_unix=now)
        _sample_queue_history(bucket, config, run_id, role="audit", now_unix=now)
        _refresh_run_counts(db, config.netuid, run_id)
    return indexed


def _selected_run_ids(bucket: ObjectStore, config: DashboardConfig) -> list[str]:
    if config.run_id:
        return [config.run_id]
    run_ids: set[str] = set()
    for role in ("train", "audit"):
        for record in scan_bucket_discovery_records(bucket, netuid=config.netuid, role=role, heartbeat_ttl_sec=None):
            if record.run_id:
                run_ids.add(record.run_id)
    root = paths.root(config.netuid)
    for prefix in (f"{root}/jobs/", f"{root}/receipts/", f"{root}/audits/", f"{root}/runs/"):
        for uri in bucket.list(bucket.uri_for_key(prefix))[:2000]:
            rid = _run_id_from_uri(uri, prefix)
            if rid:
                run_ids.add(rid)
    return sorted(run_ids, reverse=True)[:10]


def _upsert_run(db: DashboardDB, netuid: int, run_id: str, now: int) -> None:
    db.execute(
        """
        INSERT INTO runs(netuid, run_id, first_seen_unix, last_seen_unix)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(netuid, run_id) DO UPDATE SET last_seen_unix=excluded.last_seen_unix
        """,
        (netuid, run_id, now, now),
    )


def _index_workers(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, run_id: str) -> int:
    rows: list[tuple[Any, ...]] = []
    now = time.time()
    for role in ("train", "audit"):
        for record in scan_bucket_discovery_records(bucket, netuid=config.netuid, run_id=run_id, role=role, heartbeat_ttl_sec=None):
            age = now - record.last_seen_unix
            status = "live" if config.heartbeat_ttl_sec is None or age <= config.heartbeat_ttl_sec else "stale"
            rows.append(
                (
                    config.netuid,
                    run_id,
                    record.worker.hotkey_ss58,
                    record.worker.worker_id or "",
                    record.worker.host_id,
                    record.role,
                    status,
                    int(record.last_seen_unix),
                    json.dumps(record.worker.capabilities, sort_keys=True),
                    json.dumps(record.miner.to_dict(), sort_keys=True),
                    json.dumps(record.worker.to_dict(), sort_keys=True),
                )
            )
    db.executemany(
        """
        INSERT INTO workers(netuid, run_id, hotkey, worker_id, host_id, role, status, last_seen_unix, capabilities_json, miner_json, worker_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, run_id, hotkey, worker_id, role) DO UPDATE SET
            host_id=excluded.host_id,
            status=excluded.status,
            last_seen_unix=excluded.last_seen_unix,
            capabilities_json=excluded.capabilities_json,
            miner_json=excluded.miner_json,
            worker_json=excluded.worker_json
        """,
        rows,
    )
    return len(rows)


def _index_receipts(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, run_id: str) -> int:
    rows: list[tuple[Any, ...]] = []
    for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(config.netuid, run_id)))[: max(config.max_jobs * 4, 500)]:
        if not uri.endswith(".json"):
            continue
        try:
            receipt = JobReceiptV3.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        rows.append(
            (
                config.netuid,
                run_id,
                receipt.receipt_id,
                receipt.job_id,
                receipt.worker.hotkey_ss58,
                receipt.worker.worker_id,
                _receipt_kind(receipt),
                float(receipt.compute_sec or 0.0),
                int(receipt.claimed_bytes_read or 0),
                int(receipt.claimed_bytes_written or 0),
                int(receipt.finished_unix or 0),
                json.dumps(receipt.to_dict(), sort_keys=True),
            )
        )
    db.executemany(
        """
        INSERT INTO receipts(netuid, run_id, receipt_id, job_id, hotkey, worker_id, kind, compute_sec, bytes_read, bytes_written, finished_unix, receipt_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, run_id, receipt_id) DO UPDATE SET
            job_id=excluded.job_id,
            hotkey=excluded.hotkey,
            worker_id=excluded.worker_id,
            kind=excluded.kind,
            compute_sec=excluded.compute_sec,
            bytes_read=excluded.bytes_read,
            bytes_written=excluded.bytes_written,
            finished_unix=excluded.finished_unix,
            receipt_json=excluded.receipt_json
        """,
        rows,
    )
    return len(rows)


def _receipt_kind(receipt: JobReceiptV3) -> str:
    """Extract job ``kind`` from a receipt, falling back to ``""``.

    Receipts don't always carry the manifest kind directly; the executor
    serialises it onto the receipt for v3. Older receipts may be missing it.
    """
    raw = getattr(receipt, "kind", None)
    if raw:
        return str(raw)
    # Some receipt schemas keep the kind inside the worker_report dict.
    report = getattr(receipt, "worker_report", None) or {}
    if isinstance(report, dict):
        return str(report.get("kind") or "")
    return ""


def _index_verdicts(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, run_id: str) -> int:
    rows: list[tuple[Any, ...]] = []
    for uri in bucket.list(bucket.uri_for_key(paths.verdicts_prefix(config.netuid, run_id)))[: max(config.max_jobs * 4, 500)]:
        if not uri.endswith(".json"):
            continue
        try:
            verdict = VerificationVerdictV3.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        rows.append(
            (
                config.netuid,
                run_id,
                verdict.receipt_id,
                verdict.job_id,
                verdict.validator_hotkey,
                verdict.status,
                int(verdict.checked_unix or 0),
                json.dumps(verdict.to_dict(), sort_keys=True),
            )
        )
    db.executemany(
        """
        INSERT INTO verdicts(netuid, run_id, receipt_id, job_id, validator_hotkey, status, checked_unix, verdict_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, run_id, receipt_id, validator_hotkey) DO UPDATE SET
            job_id=excluded.job_id,
            status=excluded.status,
            checked_unix=excluded.checked_unix,
            verdict_json=excluded.verdict_json
        """,
        rows,
    )
    return len(rows)


def _index_audits(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, run_id: str) -> int:
    rows: list[tuple[Any, ...]] = []
    for uri in bucket.list(bucket.uri_for_key(paths.audit_results_prefix(config.netuid, run_id)))[: max(config.max_jobs * 4, 500)]:
        if not uri.endswith(".json"):
            continue
        try:
            audit = AuditResultV3.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        rows.append(
            (
                config.netuid,
                run_id,
                audit.receipt_id,
                audit.job_id,
                audit.auditor_hotkey,
                audit.status,
                int(audit.checked_unix or 0),
                json.dumps(audit.to_dict(), sort_keys=True),
            )
        )
    db.executemany(
        """
        INSERT INTO audits(netuid, run_id, receipt_id, job_id, auditor_hotkey, status, checked_unix, audit_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, run_id, receipt_id, auditor_hotkey) DO UPDATE SET
            job_id=excluded.job_id,
            status=excluded.status,
            checked_unix=excluded.checked_unix,
            audit_json=excluded.audit_json
        """,
        rows,
    )
    return len(rows)


def _refresh_run_counts(db: DashboardDB, netuid: int, run_id: str) -> None:
    db.execute(
        """
        UPDATE runs SET
            receipt_count=(SELECT COUNT(*) FROM receipts WHERE netuid=? AND run_id=?)
        WHERE netuid=? AND run_id=?
        """,
        (netuid, run_id, netuid, run_id),
    )


def index_chain_once(*, db: DashboardDB, config: DashboardConfig) -> None:
    import bittensor as bt

    subtensor = bt.Subtensor(network=config.network)
    current_block = int(subtensor.get_current_block())
    try:
        tempo = int(subtensor.tempo(config.netuid))
    except Exception:
        tempo = 0
    try:
        rate_limit = int(subtensor.query_subtensor("WeightsSetRateLimit", params=[config.netuid]))
    except Exception:
        rate_limit = 0
    metagraph = subtensor.metagraph(config.netuid)
    now = int(time.time())
    rows: list[tuple[Any, ...]] = []
    for uid, hotkey in enumerate(metagraph.hotkeys):
        rows.append(
            (
                config.netuid,
                hotkey,
                int(uid),
                _seq_float(getattr(metagraph, "S", None), uid),
                _seq_float(getattr(metagraph, "I", None), uid),
                _seq_float(getattr(metagraph, "E", None), uid),
                int(bool(_seq_value(getattr(metagraph, "validator_permit", None), uid, False))),
                int(_seq_value(getattr(metagraph, "last_update", None), uid, 0) or 0),
                current_block,
                now,
            )
        )
    db.executemany(
        """
        INSERT INTO chain_hotkeys(netuid, hotkey, uid, stake, incentive, emission, validator_permit, last_update_block, observed_block, observed_unix)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, hotkey) DO UPDATE SET
            uid=excluded.uid,
            stake=excluded.stake,
            incentive=excluded.incentive,
            emission=excluded.emission,
            validator_permit=excluded.validator_permit,
            last_update_block=excluded.last_update_block,
            observed_block=excluded.observed_block,
            observed_unix=excluded.observed_unix
        """,
        rows,
    )
    db.execute(
        """
        INSERT INTO chain_meta(netuid, current_block, tempo, weights_set_rate_limit, observed_unix, error)
        VALUES (?, ?, ?, ?, ?, NULL)
        ON CONFLICT(netuid) DO UPDATE SET
            current_block=excluded.current_block,
            tempo=excluded.tempo,
            weights_set_rate_limit=excluded.weights_set_rate_limit,
            observed_unix=excluded.observed_unix,
            error=NULL
        """,
        (config.netuid, current_block, tempo, rate_limit, now),
    )
    db.set_state("chain", cursor={"current_block": current_block, "hotkeys": len(rows)}, error=None)


# ---------------------------------------------------------------------------
# Queue snapshot + history
# ---------------------------------------------------------------------------


def _sample_queue_history(
    bucket: ObjectStore,
    config: DashboardConfig,
    run_id: str,
    *,
    role: str,
    now_unix: int,
) -> None:
    """Record one queue history point if the queue object exists.

    Best-effort: missing/unreadable queue objects are silently skipped so a
    brand-new run that hasn't published a snapshot yet doesn't fill the log.
    """
    try:
        state = read_queue(bucket, netuid=config.netuid, run_id=run_id, role=role)
    except Exception:
        return
    if state is None:
        return
    snap = _queue_state_to_snapshot(state, run_id=run_id, role=role, config=config, history=[])
    key = (config.netuid, run_id, role)
    with _QUEUE_HISTORY_LOCK:
        buf = _QUEUE_HISTORY.get(key)
        if buf is None:
            buf = deque(maxlen=_QUEUE_HISTORY_MAX_POINTS)
            _QUEUE_HISTORY[key] = buf
        buf.append(QueueHistoryPoint(ts=int(now_unix), depth_total=snap.depth_total, at_cap_count=snap.at_cap_count))


def _queue_history_for(netuid: int, run_id: str, role: str, *, now_unix: int) -> list[QueueHistoryPoint]:
    cutoff = int(now_unix) - _QUEUE_HISTORY_SECONDS
    with _QUEUE_HISTORY_LOCK:
        buf = _QUEUE_HISTORY.get((netuid, run_id, role))
        if not buf:
            return []
        return [p for p in buf if p.ts >= cutoff]


def _queue_state_to_snapshot(
    state: QueueState,
    *,
    run_id: str,
    role: str,
    config: DashboardConfig,
    history: list[QueueHistoryPoint],
) -> QueueSnapshot:
    depth_by_hotkey: dict[str, int] = {}
    now = time.time()
    oldest_age: float | None = None
    oldest_job_id: str | None = None
    for entry in state.outstanding:
        depth_by_hotkey[entry.assigned_hotkey] = depth_by_hotkey.get(entry.assigned_hotkey, 0) + 1
        if entry.created_unix:
            age = max(0.0, now - float(entry.created_unix))
            if oldest_age is None or age > oldest_age:
                oldest_age = age
                oldest_job_id = entry.job_id
    cap = max(0, int(config.max_inflight_per_hotkey))
    at_cap_hotkeys = (
        sorted(hk for hk, n in depth_by_hotkey.items() if cap and n >= cap)
        if cap
        else []
    )
    return QueueSnapshot(
        run_id=run_id,
        role=role,
        snapshot_unix=int(state.snapshot_unix),
        snapshot_id=int(state.snapshot_id),
        depth_total=len(state.outstanding),
        depth_by_hotkey=depth_by_hotkey,
        max_inflight_per_hotkey=cap,
        at_cap_count=len(at_cap_hotkeys),
        at_cap_hotkeys=at_cap_hotkeys,
        oldest_entry_age_sec=oldest_age,
        oldest_job_id=oldest_job_id,
        history=history,
    )


def _queue_snapshot(
    bucket: ObjectStore,
    config: DashboardConfig,
    *,
    run_id: str,
    role: str = "train",
    include_history: bool = True,
) -> QueueSnapshot:
    """Live read of the queue, returning an empty snapshot when missing."""
    state = None
    try:
        state = read_queue(bucket, netuid=config.netuid, run_id=run_id, role=role)
    except Exception:
        state = None
    history = _queue_history_for(config.netuid, run_id, role, now_unix=int(time.time())) if include_history else []
    if state is None:
        return QueueSnapshot(
            run_id=run_id,
            role=role,
            snapshot_unix=0,
            snapshot_id=0,
            depth_total=0,
            depth_by_hotkey={},
            max_inflight_per_hotkey=max(0, int(config.max_inflight_per_hotkey)),
            at_cap_count=0,
            at_cap_hotkeys=[],
            oldest_entry_age_sec=None,
            oldest_job_id=None,
            history=history,
        )
    return _queue_state_to_snapshot(state, run_id=run_id, role=role, config=config, history=history)


def _queue_outstanding_entries(bucket: ObjectStore, config: DashboardConfig, *, run_id: str, role: str = "train") -> list[dict[str, Any]]:
    """Project queue entries into the dashboard's outstanding-job rows."""
    try:
        state = read_queue(bucket, netuid=config.netuid, run_id=run_id, role=role)
    except Exception:
        return []
    if state is None:
        return []
    now = time.time()
    out: list[dict[str, Any]] = []
    for entry in state.outstanding:
        out.append(
            {
                "job_id": entry.job_id,
                "kind": _kind_from_job_id(entry.job_id),
                "assigned_hotkey": entry.assigned_hotkey,
                "assigned_worker": entry.assigned_worker,
                "attempt": entry.attempt,
                "created_unix": entry.created_unix,
                "deadline_unix": entry.deadline_unix,
                "age_sec": max(0.0, now - float(entry.created_unix)) if entry.created_unix else None,
                "deadline_sec": (entry.deadline_unix - now) if entry.deadline_unix else None,
                "manifest_uri": entry.manifest_uri,
                "grant_uri": entry.grant_uri,
                "role": role,
            }
        )
    out.sort(key=lambda r: r.get("created_unix") or 0, reverse=True)
    return out


def _kind_from_job_id(job_id: str) -> str:
    """Cheap kind inference from job_id naming convention.

    Avoids fetching every manifest just to know the kind. Falls back to ``""``
    so the UI just shows a blank kind cell rather than crashing.
    """
    if not job_id:
        return ""
    # Stress / training mode IDs look like ``j-e{epoch}-s{stage}-mb{mb}-fwd``.
    for suffix in ("-fwd", "-bwd", "-outer", "-reduce", "-inner", "-eval"):
        if job_id.endswith(suffix):
            return "pipe_" + suffix.lstrip("-") if suffix in ("-fwd", "-bwd", "-outer") else suffix.lstrip("-")
    if job_id.startswith("audit-"):
        return "audit_replay"
    return ""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _handler(*, bucket: ObjectStore, db: DashboardDB, config: DashboardConfig) -> type[BaseHTTPRequestHandler]:
    logo_bytes = _load_logo_bytes()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send(HTTPStatus.OK, _index_html(config.refresh_sec).encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path in {"/teutonic.png", "/favicon.png", "/favicon.ico"}:
                    self._send(HTTPStatus.OK if logo_bytes else HTTPStatus.NOT_FOUND, logo_bytes, "image/png")
                    return
                if parsed.path == "/healthz":
                    self._send_json(_health(db, config))
                    return
                if parsed.path == "/api/runs":
                    self._send_json(_api_runs(db, config))
                    return
                if parsed.path == "/api/discovery":
                    self._send_json(_api_discovery(db, config, query))
                    return
                if parsed.path == "/api/queue":
                    self._send_json(_api_queue(bucket, db, config, query))
                    return
                if parsed.path == "/api/snapshot":
                    self._send_json(_api_snapshot(bucket, db, config, query))
                    return
                if parsed.path == "/api/job":
                    self._send_json(_api_job(bucket, db, config, query))
                    return
                if parsed.path == "/api/chain/meta":
                    self._send_json(_api_chain_meta(db, config))
                    return
                if parsed.path == "/api/chain/hotkeys":
                    self._send_json(_api_chain_hotkeys(db, config, query))
                    return
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            except Exception:
                body = json.dumps({"error": traceback.format_exc(limit=8)}).encode("utf-8")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, body, "application/json")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, value: dict[str, Any]) -> None:
            self._send(HTTPStatus.OK, json.dumps(value, sort_keys=True).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------


def _health(db: DashboardDB, config: DashboardConfig) -> dict[str, Any]:
    states = {r["name"]: dict(r) for r in db.query("SELECT * FROM indexer_state")}
    chain = _row_dict_one(db.query("SELECT * FROM chain_meta WHERE netuid=?", (config.netuid,)))
    return {"ok": True, "netuid": config.netuid, "run_id": config.run_id, "states": states, "chain": chain}


def _api_runs(db: DashboardDB, config: DashboardConfig) -> dict[str, Any]:
    rows = db.query(
        "SELECT run_id FROM runs WHERE netuid=? ORDER BY last_seen_unix DESC, run_id DESC LIMIT 100",
        (config.netuid,),
    )
    return {"runs": [r["run_id"] for r in rows], "default_run_id": config.run_id or ""}


def _api_discovery(db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query)
    role = _first(query, "role") or "all"
    params: list[Any] = [config.netuid]
    where = "netuid=?"
    if run_id is not None:
        where += " AND run_id=?"
        params.append(run_id)
    if role in {"train", "audit"}:
        where += " AND role=?"
        params.append(role)
    rows = db.query(f"SELECT * FROM workers WHERE {where} ORDER BY role, host_id, worker_id", tuple(params))
    now = time.time()
    return {
        "meta": {
            "bucket": os.environ.get("S3_BUCKET", ""),
            "netuid": config.netuid,
            "run_id": run_id or "all",
            "role": role,
            "heartbeat_ttl_sec": config.heartbeat_ttl_sec,
            "generated_unix": int(now),
            "source": "sqlite",
        },
        "records": [_worker_record_from_row(r, now=now) for r in rows],
    }


def _api_queue(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query) or ""
    role = _first(query, "role") or "train"
    if role not in ("train", "audit"):
        role = "train"
    if not run_id:
        return {"queue": None, "meta": {"netuid": config.netuid, "run_id": "", "role": role}}
    snap = _queue_snapshot(bucket, config, run_id=run_id, role=role)
    return {
        "queue": snap.to_dict(),
        "meta": {"netuid": config.netuid, "run_id": run_id, "role": role, "generated_unix": int(time.time())},
    }


def _api_snapshot(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query)
    now = time.time()
    queue_snap = _queue_snapshot(bucket, config, run_id=run_id, role="train") if run_id else None
    audit_queue = _queue_snapshot(bucket, config, run_id=run_id, role="audit", include_history=False) if run_id else None
    machines = _machines_from_sql(db, config, run_id, queue_snap=queue_snap, now=now)
    outstanding = _queue_outstanding_entries(bucket, config, run_id=run_id, role="train") if run_id else []
    audit_outstanding = _queue_outstanding_entries(bucket, config, run_id=run_id, role="audit") if run_id else []
    completed = _completed_jobs_from_sql(db, config, run_id, limit=config.max_jobs)
    return {
        "meta": {
            "bucket": os.environ.get("S3_BUCKET", ""),
            "netuid": config.netuid,
            "run_id": run_id or "all",
            "generated_unix": int(now),
            "max_jobs": config.max_jobs,
            "max_inflight_per_hotkey": config.max_inflight_per_hotkey,
            "heartbeat_ttl_sec": config.heartbeat_ttl_sec,
            "source": "sqlite",
            "health": _health(db, config),
        },
        "run": {"run_id": run_id or "all"},
        "queue": queue_snap.to_dict() if queue_snap else None,
        "audit_queue": audit_queue.to_dict() if audit_queue else None,
        "machines": machines,
        "jobs": {
            "outstanding": outstanding,
            "completed": completed,
            "audit_outstanding": audit_outstanding,
        },
    }


def _api_job(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query)
    job_id = _first(query, "job_id") or ""
    if not job_id:
        return {"meta": {"netuid": config.netuid, "run_id": run_id or "all"}, "job": None, "manifest": None}

    # Try to locate the run if not explicit: pick the most recent receipt for this job_id.
    resolved_run = run_id
    if resolved_run is None:
        rows = db.query(
            "SELECT run_id FROM receipts WHERE netuid=? AND job_id=? ORDER BY finished_unix DESC LIMIT 1",
            (config.netuid, job_id),
        )
        resolved_run = rows[0]["run_id"] if rows else None

    manifest: dict[str, Any] | None = None
    if resolved_run:
        for manifest_key_fn in (paths.job_manifest_key, paths.audit_job_manifest_key):
            uri = bucket.uri_for_key(manifest_key_fn(config.netuid, resolved_run, job_id))
            try:
                manifest = bucket.get_json(uri)
                break
            except Exception:
                manifest = None

    completion = None
    if resolved_run:
        rows = db.query(
            """
            SELECT r.*, (SELECT verdict_json FROM verdicts v WHERE v.netuid=r.netuid AND v.run_id=r.run_id AND v.job_id=r.job_id ORDER BY checked_unix DESC LIMIT 1) verdict_json
            FROM receipts r
            WHERE r.netuid=? AND r.run_id=? AND r.job_id=?
            ORDER BY r.finished_unix DESC LIMIT 1
            """,
            (config.netuid, resolved_run, job_id),
        )
        if rows:
            completion = _completed_row(rows[0])

    return {
        "meta": {"netuid": config.netuid, "run_id": resolved_run or "all", "source": "sqlite"},
        "job": completion,
        "manifest": manifest,
    }


def _api_chain_meta(db: DashboardDB, config: DashboardConfig) -> dict[str, Any]:
    return {"chain": _row_dict_one(db.query("SELECT * FROM chain_meta WHERE netuid=?", (config.netuid,)))}


def _api_chain_hotkeys(db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query)
    if run_id is None:
        rows = db.query(
            """
            SELECT DISTINCT w.hotkey, c.* FROM workers w
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=?
            ORDER BY c.uid IS NULL, c.uid, w.hotkey
            """,
            (config.netuid,),
        )
    else:
        rows = db.query(
            """
            SELECT DISTINCT w.hotkey, c.* FROM workers w
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=? AND w.run_id=?
            ORDER BY c.uid IS NULL, c.uid, w.hotkey
            """,
            (config.netuid, run_id),
        )
    return {"run_id": run_id or "all", "hotkeys": [_chain_dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# SQL projectors
# ---------------------------------------------------------------------------


def _selected_run(db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> str | None:
    explicit = _first(query, "run_id")
    if explicit in {"", "all", "*", "network"}:
        return None
    if explicit:
        return explicit
    return config.run_id or None


def _machines_from_sql(
    db: DashboardDB,
    config: DashboardConfig,
    run_id: str | None,
    *,
    queue_snap: QueueSnapshot | None,
    now: float,
) -> list[dict[str, Any]]:
    if run_id is None:
        rows = db.query(
            """
            SELECT w.*, c.uid, c.stake, c.incentive, c.emission, c.validator_permit, c.last_update_block, c.observed_block
            FROM workers w
            JOIN (
                SELECT netuid, hotkey, worker_id, role, MAX(last_seen_unix) AS max_seen
                FROM workers
                WHERE netuid=?
                GROUP BY netuid, hotkey, worker_id, role
            ) latest
              ON latest.netuid=w.netuid
             AND latest.hotkey=w.hotkey
             AND latest.worker_id=w.worker_id
             AND latest.role=w.role
             AND latest.max_seen=w.last_seen_unix
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=?
            ORDER BY w.host_id, w.worker_id
            """,
            (config.netuid, config.netuid),
        )
    else:
        rows = db.query(
            """
            SELECT w.*, c.uid, c.stake, c.incentive, c.emission, c.validator_permit, c.last_update_block, c.observed_block
            FROM workers w
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=? AND w.run_id=?
            ORDER BY w.host_id, w.worker_id
            """,
            (config.netuid, run_id),
        )
    receipt_counts = _receipts_by_hotkey(db, config, run_id)
    inflight = (queue_snap.depth_by_hotkey if queue_snap else {}) or {}
    cap = queue_snap.max_inflight_per_hotkey if queue_snap else config.max_inflight_per_hotkey
    by_host: dict[str, dict[str, Any]] = {}
    for r in rows:
        host_id = r["host_id"] or "(unknown)"
        machine = by_host.setdefault(
            host_id,
            {"host_id": host_id, "roles": [], "hotkeys": [], "workers": [], "last_seen_unix": 0, "age_sec": 0},
        )
        if r["role"] not in machine["roles"]:
            machine["roles"].append(r["role"])
        if r["hotkey"] not in machine["hotkeys"]:
            machine["hotkeys"].append(r["hotkey"])
        machine["last_seen_unix"] = max(machine["last_seen_unix"] or 0, r["last_seen_unix"] or 0)
        machine["age_sec"] = max(0.0, now - (machine["last_seen_unix"] or 0)) if machine["last_seen_unix"] else None
        worker = json.loads(r["worker_json"] or "{}")
        miner = json.loads(r["miner_json"] or "{}")
        chain = _chain_dict(r)
        if chain:
            miner["chain"] = chain
        depth = int(inflight.get(r["hotkey"], 0))
        at_cap = bool(cap and depth >= cap)
        machine["workers"].append(
            {
                "role": r["role"],
                "status": r["status"] or "seen",
                "miner": miner,
                "worker": worker,
                "chain": chain,
                "last_seen_unix": r["last_seen_unix"],
                "age_sec": max(0.0, now - (r["last_seen_unix"] or 0)) if r["last_seen_unix"] else None,
                "n_receipts": int(receipt_counts.get(r["hotkey"], 0)),
                "queue_depth": depth,
                "queue_cap": int(cap or 0),
                "at_cap": at_cap,
                "sources": ["heartbeat", "sqlite"],
            }
        )
    return sorted(by_host.values(), key=lambda m: m["host_id"])


def _receipts_by_hotkey(db: DashboardDB, config: DashboardConfig, run_id: str | None) -> dict[str, int]:
    where = "netuid=?" + (" AND run_id=?" if run_id is not None else "")
    params = (config.netuid, run_id) if run_id is not None else (config.netuid,)
    out: dict[str, int] = {}
    for r in db.query(f"SELECT hotkey, COUNT(*) n FROM receipts WHERE {where} GROUP BY hotkey", params):
        out[r["hotkey"]] = int(r["n"])
    return out


def _completed_jobs_from_sql(db: DashboardDB, config: DashboardConfig, run_id: str | None, *, limit: int) -> list[dict[str, Any]]:
    where = "r.netuid=?" + (" AND r.run_id=?" if run_id is not None else "")
    params: tuple[Any, ...] = (config.netuid, run_id, limit) if run_id is not None else (config.netuid, limit)
    rows = db.query(
        f"""
        SELECT r.*,
               (SELECT verdict_json FROM verdicts v WHERE v.netuid=r.netuid AND v.run_id=r.run_id AND v.job_id=r.job_id ORDER BY checked_unix DESC LIMIT 1) verdict_json,
               (SELECT audit_json   FROM audits   a WHERE a.netuid=r.netuid AND a.run_id=r.run_id AND a.job_id=r.job_id ORDER BY checked_unix DESC LIMIT 1) audit_json
        FROM receipts r
        WHERE {where}
        ORDER BY r.finished_unix DESC, r.receipt_id DESC
        LIMIT ?
        """,
        params,
    )
    return [_completed_row(r) for r in rows]


def _completed_row(r: sqlite3.Row) -> dict[str, Any]:
    receipt = json.loads(r["receipt_json"] or "{}")
    verdict = json.loads(r["verdict_json"]) if r["verdict_json"] else None
    audit = json.loads(r["audit_json"]) if r["audit_json"] else None
    if verdict and verdict.get("status") == "fail":
        status = "failed"
    elif verdict and verdict.get("status") == "pass":
        status = "verified"
    else:
        status = "completed"
    finished = int(r["finished_unix"] or 0)
    started = int((receipt.get("started_unix") or 0)) if isinstance(receipt, dict) else 0
    return {
        "job_id": r["job_id"],
        "kind": r["kind"] or "",
        "status": status,
        "assigned_hotkey": r["hotkey"],
        "assigned_worker": r["worker_id"],
        "finished_unix": finished,
        "started_unix": started or None,
        "duration_sec": max(0.0, (finished - started)) if (started and finished) else None,
        "checked_unix": int(verdict.get("checked_unix") or 0) if isinstance(verdict, dict) else None,
        "compute_sec": float(r["compute_sec"] or 0.0),
        "bytes_read": int(r["bytes_read"] or 0),
        "bytes_written": int(r["bytes_written"] or 0),
        "receipt_id": r["receipt_id"],
        "verdict": verdict,
        "audit": audit,
    }


# ---------------------------------------------------------------------------
# Small helpers (unchanged from prior implementation)
# ---------------------------------------------------------------------------


def _run_id_from_uri(uri: str, prefix: str) -> str | None:
    marker = prefix.rstrip("/") + "/"
    if marker not in uri:
        return None
    rest = uri.split(marker, 1)[1]
    return rest.split("/", 1)[0] if rest else None


def _worker_record_from_row(row: sqlite3.Row, *, now: float) -> dict[str, Any]:
    miner = json.loads(row["miner_json"] or "{}")
    worker = json.loads(row["worker_json"] or "{}")
    return {
        "miner": miner,
        "worker": worker,
        "run_id": row["run_id"],
        "role": row["role"],
        "last_seen_unix": row["last_seen_unix"],
        "age_sec": max(0.0, now - (row["last_seen_unix"] or 0)) if row["last_seen_unix"] else None,
    }


def _chain_dict(row: sqlite3.Row) -> dict[str, Any] | None:
    if "uid" not in row.keys() or row["uid"] is None:
        return None
    current = row["observed_block"]
    last_update = row["last_update_block"]
    return {
        "uid": row["uid"],
        "stake": row["stake"],
        "incentive": row["incentive"],
        "emission": row["emission"],
        "validator_permit": bool(row["validator_permit"]),
        "last_update_block": last_update,
        "observed_block": current,
        "blocks_since_last_update": (current - last_update) if current is not None and last_update is not None else None,
        "observed_unix": row["observed_unix"] if "observed_unix" in row.keys() else None,
    }


def _row_dict_one(rows: list[sqlite3.Row]) -> dict[str, Any] | None:
    return dict(rows[0]) if rows else None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(value.item())
        except Exception:
            return 0.0


def _seq_value(seq: Any, idx: int, default: Any = None) -> Any:
    if seq is None:
        return default
    try:
        if len(seq) <= idx:
            return default
        return seq[idx]
    except Exception:
        try:
            return seq[idx]
        except Exception:
            return default


def _seq_float(seq: Any, idx: int) -> float:
    return _safe_float(_seq_value(seq, idx, 0.0))


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    return values[0] if values else None


def _index_html(refresh_sec: float) -> str:
    return INDEX_HTML.replace("__REFRESH_MS__", str(max(500, int(refresh_sec * 1000))))
