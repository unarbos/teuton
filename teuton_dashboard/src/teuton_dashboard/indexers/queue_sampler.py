"""Queue snapshot poller.

Polls the orchestrator's ``queue.json`` files on a short cadence (default 0.5s)
and:

1. Publishes the decoded snapshot to :class:`QueueBus` whenever the
   ``snapshot_id`` advances (this is what the SSE stream listens to).
2. Records a (ts, depth_total, at_cap_count) point in the bus's history ring
   buffer so the per-snapshot ``history`` field is always populated.

The list of active ``(run_id, role)`` pairs is rediscovered every cycle from
the dashboard's ``runs`` table so newly-created runs are picked up without a
process restart.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Iterable

from teuton_runtime.queue import QueueState, read_queue
from teuton_runtime.storage import ObjectStore

from ..db import DashboardDB
from ..models import QueueEntry, QueueHistoryPoint, QueueSnapshot
from ..queue_bus import QueueBus
from ..settings import Settings


LOG = logging.getLogger(__name__)


_ROLES = ("train", "audit")


async def run_queue_sampler_loop(
    *,
    bucket: ObjectStore,
    db: DashboardDB,
    bus: QueueBus,
    settings: Settings,
    stop_event: asyncio.Event,
) -> None:
    seen_snapshot_id: dict[tuple[str, str], int] = {}
    while not stop_event.is_set():
        try:
            run_ids = await _active_run_ids(db, settings)
            for run_id in run_ids:
                for role in _ROLES:
                    await _sample_one(
                        bucket=bucket,
                        bus=bus,
                        settings=settings,
                        run_id=run_id,
                        role=role,
                        seen=seen_snapshot_id,
                    )
        except Exception:
            LOG.warning("queue sampler error: %s", traceback.format_exc(limit=6))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.queue_sample_sec)
        except asyncio.TimeoutError:
            pass


async def _active_run_ids(db: DashboardDB, settings: Settings) -> list[str]:
    if settings.run_id:
        return [settings.run_id]
    rows = await db.query(
        "SELECT run_id FROM runs WHERE netuid=? ORDER BY last_seen_unix DESC LIMIT 10",
        (settings.netuid,),
    )
    return [r["run_id"] for r in rows]


async def _sample_one(
    *,
    bucket: ObjectStore,
    bus: QueueBus,
    settings: Settings,
    run_id: str,
    role: str,
    seen: dict[tuple[str, str], int],
) -> None:
    state = await asyncio.to_thread(
        _safe_read_queue, bucket, settings.netuid, run_id, role
    )
    if state is None:
        return
    now = int(time.time())
    snap = _state_to_snapshot(
        state=state, run_id=run_id, role=role, cap=settings.max_inflight_per_hotkey, bus=bus, now_unix=now
    )
    # Record one history point per sample (regardless of whether the snapshot
    # changed) so the sparkline shows a continuous line even when the queue
    # is idle.
    bus.record_history_point(
        run_id,
        role,
        QueueHistoryPoint(ts=now, depth_total=snap.depth_total, at_cap_count=snap.at_cap_count),
    )
    key = (run_id, role)
    if seen.get(key) != snap.snapshot_id:
        seen[key] = snap.snapshot_id
        await bus.publish(snap)


def _safe_read_queue(bucket: ObjectStore, netuid: int, run_id: str, role: str) -> QueueState | None:
    try:
        return read_queue(bucket, netuid=netuid, run_id=run_id, role=role)
    except Exception as exc:
        LOG.debug("queue read failed for %s/%s: %r", run_id, role, exc)
        return None


def _state_to_snapshot(
    *,
    state: QueueState,
    run_id: str,
    role: str,
    cap: int,
    bus: QueueBus,
    now_unix: int,
) -> QueueSnapshot:
    depth_by_hotkey: dict[str, int] = {}
    oldest_age: float | None = None
    oldest_job_id: str | None = None
    entries: list[QueueEntry] = []
    for raw in state.outstanding:
        depth_by_hotkey[raw.assigned_hotkey] = depth_by_hotkey.get(raw.assigned_hotkey, 0) + 1
        if raw.created_unix:
            age = max(0.0, now_unix - float(raw.created_unix))
            if oldest_age is None or age > oldest_age:
                oldest_age = age
                oldest_job_id = raw.job_id
        entries.append(
            QueueEntry(
                job_id=raw.job_id,
                assigned_hotkey=raw.assigned_hotkey,
                assigned_worker=raw.assigned_worker,
                manifest_uri=raw.manifest_uri,
                grant_uri=raw.grant_uri,
                deadline_unix=int(raw.deadline_unix or 0),
                attempt=int(raw.attempt or 0),
                created_unix=int(raw.created_unix or 0),
            )
        )
    cap_eff = max(0, int(cap))
    at_cap_hotkeys = (
        sorted(hk for hk, n in depth_by_hotkey.items() if cap_eff and n >= cap_eff)
        if cap_eff
        else []
    )
    history = bus.history_for(run_id, role, now_unix=now_unix)
    return QueueSnapshot(
        run_id=run_id,
        role=role,
        snapshot_unix=int(state.snapshot_unix),
        snapshot_id=int(state.snapshot_id),
        depth_total=len(entries),
        depth_by_hotkey=depth_by_hotkey,
        max_inflight_per_hotkey=cap_eff,
        at_cap_count=len(at_cap_hotkeys),
        at_cap_hotkeys=at_cap_hotkeys,
        oldest_entry_age_sec=oldest_age,
        oldest_job_id=oldest_job_id,
        outstanding=entries,
        history=history,
    )


# ----------------------------------------------------------------------
# Helper reused by HTTP handlers for on-demand reads (cache miss).
# ----------------------------------------------------------------------


def empty_snapshot(run_id: str, role: str, *, cap: int, history: Iterable[QueueHistoryPoint]) -> QueueSnapshot:
    return QueueSnapshot(
        run_id=run_id,
        role=role,
        snapshot_unix=0,
        snapshot_id=0,
        depth_total=0,
        depth_by_hotkey={},
        max_inflight_per_hotkey=max(0, int(cap)),
        at_cap_count=0,
        at_cap_hotkeys=[],
        oldest_entry_age_sec=None,
        oldest_job_id=None,
        history=list(history),
    )


def project_state(
    state: QueueState | None,
    *,
    run_id: str,
    role: str,
    cap: int,
    bus: QueueBus,
    now_unix: int,
) -> QueueSnapshot:
    if state is None:
        return empty_snapshot(
            run_id=run_id,
            role=role,
            cap=cap,
            history=bus.history_for(run_id, role, now_unix=now_unix),
        )
    return _state_to_snapshot(
        state=state, run_id=run_id, role=role, cap=cap, bus=bus, now_unix=now_unix
    )
