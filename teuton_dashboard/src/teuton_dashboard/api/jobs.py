"""/api/jobs: thin convenience wrapper over /api/snapshot's jobs split.

Lets the frontend hit only what it needs (just outstanding, or just completed)
when the full snapshot would be wasteful. The underlying data comes from the
same sources as ``/api/snapshot``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from teuton_runtime.queue import read_queue
from teuton_runtime.storage import ObjectStore

from ..db import DashboardDB
from ..indexers.queue_sampler import project_state
from ..models import CompletedJobRow, JobsSplit, OutstandingJobRow
from ..queue_bus import QueueBus
from ..settings import Settings
from .deps import get_bucket, get_bus, get_db, get_settings
from .snapshot import _completed, _kind_from_job_id, _outstanding_rows, _selected_run


router = APIRouter()


@router.get("/api/jobs", response_model=JobsSplit)
async def jobs(
    db: DashboardDB = Depends(get_db),
    bucket: ObjectStore = Depends(get_bucket),
    bus: QueueBus = Depends(get_bus),
    settings: Settings = Depends(get_settings),
    run_id: Optional[str] = Query(default=None),
    kind: Literal["both", "outstanding", "completed"] = Query(default="both"),
    role: Literal["train", "audit"] = Query(default="train"),
) -> JobsSplit:
    resolved = _selected_run(settings, run_id)
    now = time.time()
    out_rows: list[OutstandingJobRow] = []
    audit_rows: list[OutstandingJobRow] = []
    completed: list[CompletedJobRow] = []

    if kind in ("both", "outstanding") and resolved:
        snap = bus.latest(resolved, role)
        if snap is None:
            state = await asyncio.to_thread(
                _read_queue_safe, bucket, settings.netuid, resolved, role
            )
            snap = project_state(
                state,
                run_id=resolved,
                role=role,
                cap=settings.max_inflight_per_hotkey,
                bus=bus,
                now_unix=int(now),
            )
        rows = _outstanding_rows(snap, role=role, now=now)
        if role == "audit":
            audit_rows = rows
        else:
            out_rows = rows

    if kind in ("both", "completed"):
        completed = await _completed(db, settings, resolved, limit=settings.max_completed_jobs)

    return JobsSplit(
        outstanding=out_rows,
        completed=completed,
        audit_outstanding=audit_rows,
    )


def _read_queue_safe(bucket: ObjectStore, netuid: int, run_id: str, role: str):
    try:
        return read_queue(bucket, netuid=netuid, run_id=run_id, role=role)
    except Exception:
        return None
