"""SQLite-backed dashboard backend for Teuton.

The legacy discovery UI built each ``/api/snapshot`` response by synchronously
walking S3. That is fine for tiny local smoke runs, but live buckets can contain
enough objects that a browser times out before the response is ready. This
module moves the expensive work into background indexer threads:

* bucket indexer: S3 heartbeats, manifests, receipts, verdicts, audit results
* chain indexer: Bittensor metagraph/current-block hotkey state

Request handlers only read SQLite and serialize pre-indexed rows.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from teuton_core import paths
from teuton_core.discovery_ui import INDEX_HTML, _load_logo_bytes
from teuton_core.job_index import list_job_ids
from teuton_core.protocol import AuditResultV3, JobManifestV3, JobReceiptV3, VerificationVerdictV3
from teuton_runtime.discovery import scan_bucket_discovery_records
from teuton_runtime.storage import ObjectStore


@dataclass(frozen=True)
class DashboardConfig:
    netuid: int
    run_id: str | None = None
    db_path: str = "/var/lib/teuton-dashboard/dashboard.sqlite3"
    host: str = "127.0.0.1"
    port: int = 8765
    refresh_sec: float = 3.0
    heartbeat_ttl_sec: float | None = 30.0
    max_jobs: int = 500
    max_artifacts: int = 300
    bucket_poll_sec: float = 5.0
    chain_poll_sec: float = 30.0
    network: str = "finney"
    open_browser: bool = False


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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    first_seen_unix INTEGER,
                    last_seen_unix INTEGER,
                    job_count INTEGER DEFAULT 0,
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
                CREATE TABLE IF NOT EXISTS jobs (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    kind TEXT,
                    status TEXT,
                    assigned_hotkey TEXT,
                    assigned_worker TEXT,
                    created_unix INTEGER,
                    deadline_unix INTEGER,
                    role TEXT,
                    manifest_json TEXT,
                    PRIMARY KEY (netuid, run_id, job_id)
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    netuid INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    job_id TEXT,
                    hotkey TEXT,
                    worker_id TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_jobs_run_created ON jobs(netuid, run_id, created_unix DESC);
                CREATE INDEX IF NOT EXISTS idx_receipts_run_job ON receipts(netuid, run_id, job_id);
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


def serve_dashboard_backend(*, bucket: ObjectStore, config: DashboardConfig) -> None:
    db = DashboardDB(config.db_path)
    stop = threading.Event()
    threads = [
        threading.Thread(target=_bucket_loop, args=(bucket, db, config, stop), daemon=True),
        threading.Thread(target=_chain_loop, args=(db, config, stop), daemon=True),
    ]
    for thread in threads:
        thread.start()

    handler = _handler(db=db, config=config)
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


def index_bucket_once(*, bucket: ObjectStore, db: DashboardDB, config: DashboardConfig) -> int:
    now = int(time.time())
    run_ids = _selected_run_ids(bucket, config)
    indexed = 0
    for run_id in run_ids:
        _upsert_run(db, config.netuid, run_id, now)
        indexed += _index_workers(bucket, db, config, run_id)
        indexed += _index_jobs(bucket, db, config, run_id)
        indexed += _index_receipts(bucket, db, config, run_id)
        indexed += _index_verdicts(bucket, db, config, run_id)
        indexed += _index_audits(bucket, db, config, run_id)
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


def _index_jobs(bucket: ObjectStore, db: DashboardDB, config: DashboardConfig, run_id: str) -> int:
    rows: list[tuple[Any, ...]] = []
    seen: set[str] = set()
    for role, index_key, prefix, manifest_key in (
        ("train", paths.job_index_key(config.netuid, run_id), paths.jobs_prefix(config.netuid, run_id), paths.job_manifest_key),
        ("audit", paths.audit_job_index_key(config.netuid, run_id), paths.audit_jobs_prefix(config.netuid, run_id), paths.audit_job_manifest_key),
    ):
        for job_id in list_job_ids(
            bucket,
            index_key=index_key,
            jobs_prefix_key=prefix,
            manifest_list_max_uris=2000,
        )[: config.max_jobs]:
            if job_id in seen:
                continue
            manifest = _load_manifest(bucket, bucket.uri_for_key(manifest_key(config.netuid, run_id, job_id)))
            if manifest is None:
                continue
            seen.add(manifest.job_id)
            rows.append(
                (
                    config.netuid,
                    run_id,
                    manifest.job_id,
                    manifest.kind,
                    "created",
                    manifest.assigned_hotkey,
                    manifest.assigned_worker,
                    int(manifest.created_unix or 0),
                    int(manifest.deadline_unix or 0),
                    role,
                    json.dumps(manifest.to_dict(), sort_keys=True),
                )
            )
    db.executemany(
        """
        INSERT INTO jobs(netuid, run_id, job_id, kind, status, assigned_hotkey, assigned_worker, created_unix, deadline_unix, role, manifest_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, run_id, job_id) DO UPDATE SET
            kind=excluded.kind,
            assigned_hotkey=excluded.assigned_hotkey,
            assigned_worker=excluded.assigned_worker,
            created_unix=excluded.created_unix,
            deadline_unix=excluded.deadline_unix,
            role=excluded.role,
            manifest_json=excluded.manifest_json
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
                float(receipt.compute_sec or 0.0),
                int(receipt.claimed_bytes_read or 0),
                int(receipt.claimed_bytes_written or 0),
                int(receipt.finished_unix or 0),
                json.dumps(receipt.to_dict(), sort_keys=True),
            )
        )
    db.executemany(
        """
        INSERT INTO receipts(netuid, run_id, receipt_id, job_id, hotkey, worker_id, compute_sec, bytes_read, bytes_written, finished_unix, receipt_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(netuid, run_id, receipt_id) DO UPDATE SET
            job_id=excluded.job_id,
            hotkey=excluded.hotkey,
            worker_id=excluded.worker_id,
            compute_sec=excluded.compute_sec,
            bytes_read=excluded.bytes_read,
            bytes_written=excluded.bytes_written,
            finished_unix=excluded.finished_unix,
            receipt_json=excluded.receipt_json
        """,
        rows,
    )
    return len(rows)


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
            job_count=(SELECT COUNT(*) FROM jobs WHERE netuid=? AND run_id=?),
            receipt_count=(SELECT COUNT(*) FROM receipts WHERE netuid=? AND run_id=?)
        WHERE netuid=? AND run_id=?
        """,
        (netuid, run_id, netuid, run_id, netuid, run_id),
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


def _handler(*, db: DashboardDB, config: DashboardConfig) -> type[BaseHTTPRequestHandler]:
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
                if parsed.path == "/api/snapshot":
                    self._send_json(_api_snapshot(db, config, query))
                    return
                if parsed.path == "/api/job":
                    self._send_json(_api_job(db, config, query))
                    return
                if parsed.path == "/api/artifact":
                    self._send_json({"uri": _first(query, "uri") or "", "exists": None, "source": "sqlite_backend"})
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


def _health(db: DashboardDB, config: DashboardConfig) -> dict[str, Any]:
    states = {r["name"]: dict(r) for r in db.query("SELECT * FROM indexer_state")}
    chain = _row_dict_one(db.query("SELECT * FROM chain_meta WHERE netuid=?", (config.netuid,)))
    return {"ok": True, "netuid": config.netuid, "run_id": config.run_id, "states": states, "chain": chain}


def _api_runs(db: DashboardDB, config: DashboardConfig) -> dict[str, Any]:
    rows = db.query("SELECT run_id FROM runs WHERE netuid=? ORDER BY last_seen_unix DESC, run_id DESC LIMIT 100", (config.netuid,))
    runs = [r["run_id"] for r in rows]
    # Empty default means "full-network view". A specific run_id can still be
    # selected by passing ?run_id=... or starting the backend with --run-id.
    return {"runs": runs, "default_run_id": config.run_id or ""}


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


def _api_snapshot(db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query)
    now = time.time()
    machines = _machines_from_sql(db, config, run_id, now=now)
    jobs = _jobs_from_sql(db, config, run_id)
    artifacts: list[dict[str, Any]] = []
    summary = _summary(machines=machines, jobs=jobs, artifacts=artifacts)
    return {
        "meta": {
            "bucket": os.environ.get("S3_BUCKET", ""),
            "netuid": config.netuid,
            "run_id": run_id or "all",
            "generated_unix": int(now),
            "max_jobs": config.max_jobs,
            "max_artifacts": config.max_artifacts,
            "heartbeat_ttl_sec": config.heartbeat_ttl_sec,
            "source": "sqlite",
            "health": _health(db, config),
        },
        "run": {"run_id": run_id or "all"},
        "machines": machines,
        "jobs": jobs,
        "artifacts": artifacts,
        "edges": [],
        "summary": summary,
    }


def _api_job(db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _selected_run(db, config, query)
    job_id = _first(query, "job_id") or ""
    if run_id is None:
        rows = db.query(
            "SELECT manifest_json, run_id FROM jobs WHERE netuid=? AND job_id=? ORDER BY created_unix DESC LIMIT 1",
            (config.netuid, job_id),
        )
        row_run = rows[0]["run_id"] if rows else ""
    else:
        rows = db.query("SELECT manifest_json, run_id FROM jobs WHERE netuid=? AND run_id=? AND job_id=?", (config.netuid, run_id, job_id))
        row_run = run_id
    manifest = json.loads(rows[0]["manifest_json"]) if rows else None
    job = next((j for j in _jobs_from_sql(db, config, row_run if row_run else None, limit=config.max_jobs) if j["job_id"] == job_id), None)
    return {"meta": {"netuid": config.netuid, "run_id": row_run or "all", "source": "sqlite"}, "job": job, "manifest": manifest, "artifacts": []}


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


def _selected_run(db: DashboardDB, config: DashboardConfig, query: dict[str, list[str]]) -> str | None:
    explicit = _first(query, "run_id")
    if explicit in {"", "all", "*", "network"}:
        return None
    if explicit:
        return explicit
    return config.run_id or None


def _machines_from_sql(db: DashboardDB, config: DashboardConfig, run_id: str | None, *, now: float) -> list[dict[str, Any]]:
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
    by_host: dict[str, dict[str, Any]] = {}
    counts = _worker_counts(db, config, run_id)
    for r in rows:
        host_id = r["host_id"] or "(unknown)"
        machine = by_host.setdefault(host_id, {"host_id": host_id, "roles": [], "hotkeys": [], "workers": [], "last_seen_unix": 0, "age_sec": 0})
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
        key = (r["hotkey"], r["worker_id"])
        c = counts.get(key, {"jobs": 0, "receipts": 0})
        machine["workers"].append(
            {
                "role": r["role"],
                "status": r["status"] or "seen",
                "miner": miner,
                "worker": worker,
                "chain": chain,
                "last_seen_unix": r["last_seen_unix"],
                "age_sec": max(0.0, now - (r["last_seen_unix"] or 0)) if r["last_seen_unix"] else None,
                "n_jobs": c["jobs"],
                "n_receipts": c["receipts"],
                "sources": ["heartbeat", "sqlite"],
            }
        )
    return sorted(by_host.values(), key=lambda m: m["host_id"])


def _worker_counts(db: DashboardDB, config: DashboardConfig, run_id: str | None) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = {}
    job_where = "netuid=?" + (" AND run_id=?" if run_id is not None else "")
    params = (config.netuid, run_id) if run_id is not None else (config.netuid,)
    for r in db.query(f"SELECT assigned_hotkey hotkey, COALESCE(assigned_worker,'') worker_id, COUNT(*) n FROM jobs WHERE {job_where} GROUP BY 1,2", params):
        out.setdefault((r["hotkey"], r["worker_id"]), {"jobs": 0, "receipts": 0})["jobs"] = int(r["n"])
    for r in db.query(f"SELECT hotkey, COALESCE(worker_id,'') worker_id, COUNT(*) n FROM receipts WHERE {job_where} GROUP BY 1,2", params):
        out.setdefault((r["hotkey"], r["worker_id"]), {"jobs": 0, "receipts": 0})["receipts"] = int(r["n"])
    return out


def _jobs_from_sql(db: DashboardDB, config: DashboardConfig, run_id: str | None, *, limit: int | None = None) -> list[dict[str, Any]]:
    limit = limit or config.max_jobs
    where = "j.netuid=?" + (" AND j.run_id=?" if run_id is not None else "")
    params = (config.netuid, run_id, limit) if run_id is not None else (config.netuid, limit)
    rows = db.query(
        f"""
        SELECT j.*, r.receipt_json, r.compute_sec, r.bytes_read, r.bytes_written, r.finished_unix,
               (SELECT verdict_json FROM verdicts v WHERE v.netuid=j.netuid AND v.run_id=j.run_id AND v.job_id=j.job_id ORDER BY checked_unix DESC LIMIT 1) verdict_json,
               (SELECT audit_json FROM audits a WHERE a.netuid=j.netuid AND a.run_id=j.run_id AND a.job_id=j.job_id ORDER BY checked_unix DESC LIMIT 1) audit_json
        FROM jobs j
        LEFT JOIN receipts r ON r.netuid=j.netuid AND r.run_id=j.run_id AND r.job_id=j.job_id
        WHERE {where}
        ORDER BY j.created_unix DESC
        LIMIT ?
        """,
        params,
    )
    now = time.time()
    out: list[dict[str, Any]] = []
    for r in rows:
        manifest = json.loads(r["manifest_json"] or "{}")
        receipt = json.loads(r["receipt_json"] or "null")
        verdict = json.loads(r["verdict_json"]) if r["verdict_json"] else None
        audit = json.loads(r["audit_json"]) if r["audit_json"] else None
        status = _job_status_from_sql(r, receipt, verdict, now=now)
        started = receipt.get("started_unix") if isinstance(receipt, dict) else None
        finished = receipt.get("finished_unix") if isinstance(receipt, dict) else None
        duration = (finished - started) if started and finished else max(0.0, now - (r["created_unix"] or 0))
        out.append(
            {
                "job_id": r["job_id"],
                "role": r["role"],
                "kind": r["kind"],
                "status": status,
                "run_id": run_id,
                "step_id": manifest.get("step_id", 0),
                "created_unix": r["created_unix"] or 0,
                "deadline_unix": r["deadline_unix"] or 0,
                "assigned_hotkey": r["assigned_hotkey"],
                "assigned_worker": r["assigned_worker"],
                "attempt": manifest.get("attempt", 0),
                "critical": ((manifest.get("verification_policy") or {}).get("critical") or False),
                "input_count": len(manifest.get("inputs") or []),
                "output_count": len(manifest.get("outputs") or []),
                "started_unix": started,
                "finished_unix": finished,
                "duration_sec": duration,
                "bytes_read": r["bytes_read"] or 0,
                "bytes_written": r["bytes_written"] or 0,
                "compute_sec": r["compute_sec"] or 0.0,
                "score_points": float(r["compute_sec"] or 0.0),
                "receipt_id": receipt.get("receipt_id") if isinstance(receipt, dict) else None,
                "verdicts": [verdict] if verdict else [],
                "audit_results": [audit] if audit else [],
                "params": manifest.get("params") or {},
            }
        )
    return out


def _job_status_from_sql(row: sqlite3.Row, receipt: dict | None, verdict: dict | None, *, now: float) -> str:
    if verdict and verdict.get("status") == "fail":
        return "failed"
    if verdict and verdict.get("status") == "pass":
        return "verified"
    if receipt:
        return "completed"
    if row["deadline_unix"] and now > row["deadline_unix"]:
        return "stale"
    return row["status"] or "created"


def _summary(*, machines: list[dict[str, Any]], jobs: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_worker: dict[str, int] = {}
    audits = {"pending": 0, "pass": 0, "fail": 0}
    for job in jobs:
        by_status[job["status"]] = by_status.get(job["status"], 0) + 1
        by_kind[job["kind"]] = by_kind.get(job["kind"], 0) + 1
        worker = job.get("assigned_worker") or job.get("assigned_hotkey") or "unknown"
        by_worker[worker] = by_worker.get(worker, 0) + 1
        for audit in job.get("audit_results", []):
            if audit and audit.get("status") in ("pass", "fail"):
                audits[audit["status"]] += 1
            else:
                audits["pending"] += 1
    return {
        "machines": len(machines),
        "workers": sum(len(m["workers"]) for m in machines),
        "jobs": len(jobs),
        "in_flight_jobs": sum(1 for j in jobs if j["status"] in {"created", "running", "outputs_written"}),
        "completed_jobs": sum(1 for j in jobs if j["status"] in {"completed", "verified"}),
        "failed_or_stale_jobs": sum(1 for j in jobs if j["status"] in {"failed", "stale"}),
        "bytes_read": sum(int(j.get("bytes_read") or 0) for j in jobs),
        "bytes_written": sum(int(j.get("bytes_written") or 0) for j in jobs),
        "artifacts": len(artifacts),
        "present_artifacts": sum(1 for a in artifacts if a.get("exists")),
        "missing_artifacts": sum(1 for a in artifacts if not a.get("exists")),
        "audits": audits,
        "by_status": by_status,
        "by_kind": by_kind,
        "by_worker": by_worker,
    }


def _load_manifest(bucket: ObjectStore, uri: str) -> JobManifestV3 | None:
    try:
        return JobManifestV3.from_dict(bucket.get_json(uri))
    except Exception:
        return None


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

