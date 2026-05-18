"""Bucket -> SQLite indexer.

Periodically scans the bucket and upserts ``workers``, ``receipts``,
``verdicts``, ``audits`` (one pass per discovered run). Pure I/O wrapping
of synchronous boto3 calls; we use ``asyncio.to_thread`` so the event loop
keeps serving HTTP / SSE while the scan runs.

The legacy ``_index_jobs`` step is gone -- outstanding work comes from the
queue (handled by :mod:`queue_sampler`); completed work is reconstructed
from receipts JOIN verdicts at query time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Any

from teuton_core import paths
from teuton_core.protocol import AuditResultV3, JobReceiptV3, VerificationVerdictV3
from teuton_runtime.discovery import scan_bucket_discovery_records
from teuton_runtime.storage import ObjectStore

from ..db import DashboardDB
from ..settings import Settings


LOG = logging.getLogger(__name__)


async def run_bucket_indexer_loop(
    *,
    bucket: ObjectStore,
    db: DashboardDB,
    settings: Settings,
    stop_event: asyncio.Event,
) -> None:
    """Loop until ``stop_event`` is set, running one indexer pass per cycle."""
    while not stop_event.is_set():
        t0 = time.time()
        try:
            indexed = await index_bucket_once(bucket=bucket, db=db, settings=settings)
            await db.set_state(
                "bucket",
                cursor={"indexed": indexed, "seconds": round(time.time() - t0, 3)},
                error=None,
            )
        except Exception:
            err = traceback.format_exc(limit=6)
            LOG.warning("bucket indexer error: %s", err)
            try:
                await db.set_state("bucket", error=err)
            except Exception:
                pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.bucket_poll_sec)
        except asyncio.TimeoutError:
            pass


async def index_bucket_once(
    *,
    bucket: ObjectStore,
    db: DashboardDB,
    settings: Settings,
) -> int:
    """Run one indexer pass (workers + receipts + verdicts + audits) per run.

    Returns the total number of upserted rows across all runs/tables.
    """
    run_ids = await asyncio.to_thread(_selected_run_ids, bucket, settings)
    now = int(time.time())
    total = 0
    for run_id in run_ids:
        await _upsert_run(db, settings.netuid, run_id, now)
        total += await _index_workers(bucket, db, settings, run_id)
        total += await _index_receipts(bucket, db, settings, run_id)
        total += await _index_verdicts(bucket, db, settings, run_id)
        total += await _index_audits(bucket, db, settings, run_id)
        await _refresh_run_counts(db, settings.netuid, run_id)
    return total


# ---------------------------------------------------------------------------
# Per-table workers (each runs the synchronous bucket scan in a worker thread)
# ---------------------------------------------------------------------------


def _selected_run_ids(bucket: ObjectStore, settings: Settings) -> list[str]:
    if settings.run_id:
        return [settings.run_id]
    run_ids: set[str] = set()
    for role in ("train", "audit"):
        for record in scan_bucket_discovery_records(
            bucket, netuid=settings.netuid, role=role, heartbeat_ttl_sec=None
        ):
            if record.run_id:
                run_ids.add(record.run_id)
    root = paths.root(settings.netuid)
    for prefix in (f"{root}/jobs/", f"{root}/receipts/", f"{root}/audits/", f"{root}/runs/"):
        for uri in bucket.list(bucket.uri_for_key(prefix))[:2000]:
            marker = prefix.rstrip("/") + "/"
            if marker not in uri:
                continue
            rid = uri.split(marker, 1)[1].split("/", 1)[0]
            if rid:
                run_ids.add(rid)
    return sorted(run_ids, reverse=True)[:10]


async def _upsert_run(db: DashboardDB, netuid: int, run_id: str, now: int) -> None:
    await db.execute(
        """
        INSERT INTO runs(netuid, run_id, first_seen_unix, last_seen_unix)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(netuid, run_id) DO UPDATE SET last_seen_unix=excluded.last_seen_unix
        """,
        (netuid, run_id, now, now),
    )


async def _index_workers(
    bucket: ObjectStore, db: DashboardDB, settings: Settings, run_id: str
) -> int:
    def scan() -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        now = time.time()
        for role in ("train", "audit"):
            for record in scan_bucket_discovery_records(
                bucket,
                netuid=settings.netuid,
                run_id=run_id,
                role=role,
                heartbeat_ttl_sec=None,
            ):
                age = now - record.last_seen_unix
                status = (
                    "live"
                    if settings.heartbeat_ttl_sec is None or age <= settings.heartbeat_ttl_sec
                    else "stale"
                )
                rows.append(
                    (
                        settings.netuid,
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
        return rows

    rows = await asyncio.to_thread(scan)
    return await db.executemany(
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


async def _index_receipts(
    bucket: ObjectStore, db: DashboardDB, settings: Settings, run_id: str
) -> int:
    def scan() -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        cap = max(settings.max_completed_jobs * 4, 500)
        for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(settings.netuid, run_id)))[:cap]:
            if not uri.endswith(".json"):
                continue
            try:
                receipt = JobReceiptV3.from_dict(bucket.get_json(uri))
            except Exception:
                continue
            rows.append(
                (
                    settings.netuid,
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
        return rows

    rows = await asyncio.to_thread(scan)
    return await db.executemany(
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


def _receipt_kind(receipt: JobReceiptV3) -> str:
    raw = getattr(receipt, "kind", None)
    if raw:
        return str(raw)
    report = getattr(receipt, "worker_report", None) or {}
    if isinstance(report, dict):
        return str(report.get("kind") or "")
    return ""


async def _index_verdicts(
    bucket: ObjectStore, db: DashboardDB, settings: Settings, run_id: str
) -> int:
    def scan() -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        cap = max(settings.max_completed_jobs * 4, 500)
        for uri in bucket.list(bucket.uri_for_key(paths.verdicts_prefix(settings.netuid, run_id)))[:cap]:
            if not uri.endswith(".json"):
                continue
            try:
                verdict = VerificationVerdictV3.from_dict(bucket.get_json(uri))
            except Exception:
                continue
            rows.append(
                (
                    settings.netuid,
                    run_id,
                    verdict.receipt_id,
                    verdict.job_id,
                    verdict.validator_hotkey,
                    verdict.status,
                    int(verdict.checked_unix or 0),
                    json.dumps(verdict.to_dict(), sort_keys=True),
                )
            )
        return rows

    rows = await asyncio.to_thread(scan)
    return await db.executemany(
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


async def _index_audits(
    bucket: ObjectStore, db: DashboardDB, settings: Settings, run_id: str
) -> int:
    def scan() -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        cap = max(settings.max_completed_jobs * 4, 500)
        for uri in bucket.list(bucket.uri_for_key(paths.audit_results_prefix(settings.netuid, run_id)))[:cap]:
            if not uri.endswith(".json"):
                continue
            try:
                audit = AuditResultV3.from_dict(bucket.get_json(uri))
            except Exception:
                continue
            rows.append(
                (
                    settings.netuid,
                    run_id,
                    audit.receipt_id,
                    audit.job_id,
                    audit.auditor_hotkey,
                    audit.status,
                    int(audit.checked_unix or 0),
                    json.dumps(audit.to_dict(), sort_keys=True),
                )
            )
        return rows

    rows = await asyncio.to_thread(scan)
    return await db.executemany(
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


async def _refresh_run_counts(db: DashboardDB, netuid: int, run_id: str) -> None:
    await db.execute(
        """
        UPDATE runs SET
            receipt_count=(SELECT COUNT(*) FROM receipts WHERE netuid=? AND run_id=?)
        WHERE netuid=? AND run_id=?
        """,
        (netuid, run_id, netuid, run_id),
    )
