"""/api/snapshot: aggregated view used by the main dashboard page."""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query

from teuton_runtime.queue import QueueState, read_queue
from teuton_runtime.storage import ObjectStore

from ..db import DashboardDB
from ..indexers.queue_sampler import project_state
from ..models import (
    ChainMeta,
    ChainSummary,
    CompletedJobRow,
    HealthResponse,
    IndexerState,
    JobsSplit,
    Machine,
    OutstandingJobRow,
    QueueSnapshot,
    SnapshotMeta,
    SnapshotResponse,
    WorkerRow,
)
from ..queue_bus import QueueBus
from ..settings import Settings
from .deps import get_bucket, get_bus, get_db, get_settings


router = APIRouter()


@router.get("/api/snapshot", response_model=SnapshotResponse)
async def snapshot(
    db: DashboardDB = Depends(get_db),
    bucket: ObjectStore = Depends(get_bucket),
    bus: QueueBus = Depends(get_bus),
    settings: Settings = Depends(get_settings),
    run_id: Optional[str] = Query(default=None),
) -> SnapshotResponse:
    resolved = _selected_run(settings, run_id)
    now = time.time()
    # Live queue reads (cache-first via the bus) for both roles. Run them in
    # threads only on cache miss; the common steady-state hit is in-memory.
    queue_snap = (
        await _queue_for(bus, bucket, settings, resolved, "train") if resolved else None
    )
    audit_queue = (
        await _queue_for(bus, bucket, settings, resolved, "audit") if resolved else None
    )
    machines = await _machines(db, settings, resolved, queue_snap=queue_snap, now=now)
    outstanding = _outstanding_rows(queue_snap, role="train", now=now)
    audit_outstanding = _outstanding_rows(audit_queue, role="audit", now=now)
    completed = await _completed(db, settings, resolved, limit=settings.max_completed_jobs)
    health = await _health(db, settings)

    return SnapshotResponse(
        meta=SnapshotMeta(
            bucket=os.environ.get("S3_BUCKET", ""),
            netuid=settings.netuid,
            run_id=resolved or "all",
            generated_unix=int(now),
            max_jobs=settings.max_completed_jobs,
            max_inflight_per_hotkey=settings.max_inflight_per_hotkey,
            heartbeat_ttl_sec=settings.heartbeat_ttl_sec,
            source="sqlite",
            health=health,
        ),
        run={"run_id": resolved or "all"},
        queue=queue_snap,
        audit_queue=audit_queue,
        machines=machines,
        jobs=JobsSplit(
            outstanding=outstanding,
            completed=completed,
            audit_outstanding=audit_outstanding,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selected_run(settings: Settings, run_id: Optional[str]) -> Optional[str]:
    if run_id in {None, "", "all", "*", "network"}:
        return settings.run_id or None
    return run_id


async def _queue_for(
    bus: QueueBus,
    bucket: ObjectStore,
    settings: Settings,
    run_id: str,
    role: str,
) -> Optional[QueueSnapshot]:
    cached = bus.latest(run_id, role)
    if cached is not None:
        return cached
    state: Optional[QueueState] = await asyncio.to_thread(
        _safe_read_queue, bucket, settings.netuid, run_id, role
    )
    return project_state(
        state,
        run_id=run_id,
        role=role,
        cap=settings.max_inflight_per_hotkey,
        bus=bus,
        now_unix=int(time.time()),
    )


def _safe_read_queue(bucket: ObjectStore, netuid: int, run_id: str, role: str) -> Optional[QueueState]:
    try:
        return read_queue(bucket, netuid=netuid, run_id=run_id, role=role)
    except Exception:
        return None


def _outstanding_rows(
    snap: Optional[QueueSnapshot], *, role: str, now: float
) -> list[OutstandingJobRow]:
    """Project the queue snapshot's entries into the dashboard's row schema."""
    if snap is None:
        return []
    rows: list[OutstandingJobRow] = []
    for entry in snap.outstanding:
        rows.append(
            OutstandingJobRow(
                job_id=entry.job_id,
                kind=_kind_from_job_id(entry.job_id),
                assigned_hotkey=entry.assigned_hotkey,
                assigned_worker=entry.assigned_worker,
                attempt=entry.attempt,
                created_unix=entry.created_unix,
                deadline_unix=entry.deadline_unix,
                age_sec=max(0.0, now - float(entry.created_unix)) if entry.created_unix else None,
                deadline_sec=(entry.deadline_unix - now) if entry.deadline_unix else None,
                manifest_uri=entry.manifest_uri,
                grant_uri=entry.grant_uri,
                role=role,
            )
        )
    rows.sort(key=lambda r: r.created_unix or 0, reverse=True)
    return rows


def _kind_from_job_id(job_id: str) -> str:
    if not job_id:
        return ""
    for suffix in ("-fwd", "-bwd", "-outer", "-reduce", "-inner", "-eval"):
        if job_id.endswith(suffix):
            base = suffix.lstrip("-")
            return "pipe_" + base if suffix in ("-fwd", "-bwd", "-outer") else base
    if job_id.startswith("audit-"):
        return "audit_replay"
    return ""


async def _completed(
    db: DashboardDB, settings: Settings, run_id: Optional[str], *, limit: int
) -> list[CompletedJobRow]:
    where = "r.netuid=?"
    params: tuple[Any, ...]
    if run_id is not None:
        where += " AND r.run_id=?"
        params = (settings.netuid, run_id, limit)
    else:
        params = (settings.netuid, limit)
    rows = await db.query(
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
    out: list[CompletedJobRow] = []
    for r in rows:
        receipt = json.loads(r["receipt_json"] or "{}") if r["receipt_json"] else {}
        verdict = json.loads(r["verdict_json"]) if r["verdict_json"] else None
        audit = json.loads(r["audit_json"]) if r["audit_json"] else None
        if verdict and verdict.get("status") == "fail":
            status = "failed"
        elif verdict and verdict.get("status") == "pass":
            status = "verified"
        else:
            status = "completed"
        finished = int(r["finished_unix"] or 0)
        started = int(receipt.get("started_unix") or 0) if isinstance(receipt, dict) else 0
        out.append(
            CompletedJobRow(
                job_id=r["job_id"],
                kind=r["kind"] or "",
                status=status,
                assigned_hotkey=r["hotkey"],
                assigned_worker=r["worker_id"],
                finished_unix=finished,
                started_unix=started or None,
                duration_sec=(finished - started) if (started and finished) else None,
                checked_unix=int(verdict.get("checked_unix") or 0) if isinstance(verdict, dict) else None,
                compute_sec=float(r["compute_sec"] or 0.0),
                bytes_read=int(r["bytes_read"] or 0),
                bytes_written=int(r["bytes_written"] or 0),
                receipt_id=r["receipt_id"],
                verdict=verdict,
                audit=audit,
            )
        )
    return out


async def _machines(
    db: DashboardDB,
    settings: Settings,
    run_id: Optional[str],
    *,
    queue_snap: Optional[QueueSnapshot],
    now: float,
) -> list[Machine]:
    if run_id is None:
        rows = await db.query(
            """
            SELECT w.*, c.uid, c.stake, c.incentive, c.emission, c.validator_permit, c.last_update_block, c.observed_block
            FROM workers w
            JOIN (
                SELECT netuid, hotkey, worker_id, role, MAX(last_seen_unix) AS max_seen
                FROM workers WHERE netuid=?
                GROUP BY netuid, hotkey, worker_id, role
            ) latest
              ON latest.netuid=w.netuid AND latest.hotkey=w.hotkey AND latest.worker_id=w.worker_id
             AND latest.role=w.role AND latest.max_seen=w.last_seen_unix
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=?
            ORDER BY w.host_id, w.worker_id
            """,
            (settings.netuid, settings.netuid),
        )
    else:
        rows = await db.query(
            """
            SELECT w.*, c.uid, c.stake, c.incentive, c.emission, c.validator_permit, c.last_update_block, c.observed_block
            FROM workers w
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=? AND w.run_id=?
            ORDER BY w.host_id, w.worker_id
            """,
            (settings.netuid, run_id),
        )

    receipt_counts = await _receipts_by_hotkey(db, settings, run_id)
    inflight = (queue_snap.depth_by_hotkey if queue_snap else {}) or {}
    cap = queue_snap.max_inflight_per_hotkey if queue_snap else settings.max_inflight_per_hotkey

    by_host: dict[str, Machine] = {}
    for r in rows:
        host_id = r["host_id"] or "(unknown)"
        m = by_host.get(host_id)
        if m is None:
            m = Machine(host_id=host_id)
            by_host[host_id] = m
        if r["role"] not in m.roles:
            m.roles.append(r["role"])
        if r["hotkey"] not in m.hotkeys:
            m.hotkeys.append(r["hotkey"])
        m.last_seen_unix = max(m.last_seen_unix or 0, r["last_seen_unix"] or 0)
        m.age_sec = max(0.0, now - (m.last_seen_unix or 0)) if m.last_seen_unix else None

        worker = json.loads(r["worker_json"] or "{}")
        miner = json.loads(r["miner_json"] or "{}")
        chain = _chain_summary(r)
        if chain is not None:
            miner["chain"] = chain.model_dump()
        depth = int(inflight.get(r["hotkey"], 0))
        at_cap = bool(cap and depth >= cap)
        m.workers.append(
            WorkerRow(
                role=r["role"],
                status=r["status"] or "seen",
                miner=miner,
                worker=worker,
                chain=chain,
                last_seen_unix=r["last_seen_unix"],
                age_sec=max(0.0, now - (r["last_seen_unix"] or 0)) if r["last_seen_unix"] else None,
                n_receipts=int(receipt_counts.get(r["hotkey"], 0)),
                queue_depth=depth,
                queue_cap=int(cap or 0),
                at_cap=at_cap,
                sources=["heartbeat", "sqlite"],
            )
        )
    return sorted(by_host.values(), key=lambda m: m.host_id)


def _chain_summary(row) -> Optional[ChainSummary]:
    if "uid" not in row.keys() or row["uid"] is None:
        return None
    current = row["observed_block"]
    last_update = row["last_update_block"]
    return ChainSummary(
        uid=row["uid"],
        stake=row["stake"],
        incentive=row["incentive"],
        emission=row["emission"],
        validator_permit=bool(row["validator_permit"]) if row["validator_permit"] is not None else None,
        last_update_block=last_update,
        observed_block=current,
        blocks_since_last_update=(current - last_update) if current is not None and last_update is not None else None,
    )


async def _receipts_by_hotkey(
    db: DashboardDB, settings: Settings, run_id: Optional[str]
) -> dict[str, int]:
    where = "netuid=?"
    params: tuple[Any, ...]
    if run_id is not None:
        where += " AND run_id=?"
        params = (settings.netuid, run_id)
    else:
        params = (settings.netuid,)
    rows = await db.query(
        f"SELECT hotkey, COUNT(*) n FROM receipts WHERE {where} GROUP BY hotkey",
        params,
    )
    return {r["hotkey"]: int(r["n"]) for r in rows}


async def _health(db: DashboardDB, settings: Settings) -> HealthResponse:
    state_rows = await db.query("SELECT * FROM indexer_state")
    chain_rows = await db.query("SELECT * FROM chain_meta WHERE netuid=?", (settings.netuid,))
    states = {r["name"]: IndexerState(**dict(r)) for r in state_rows}
    chain = ChainMeta(**dict(chain_rows[0])) if chain_rows else None
    return HealthResponse(
        ok=True,
        netuid=settings.netuid,
        run_id=settings.run_id or None,
        states=states,
        chain=chain,
    )
