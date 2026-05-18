"""/api/discovery: raw heartbeat records joined from the workers table."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query

from ..db import DashboardDB
from ..models import DiscoveryRecord, DiscoveryResponse
from ..settings import Settings
from .deps import get_db, get_settings


router = APIRouter()


@router.get("/api/discovery", response_model=DiscoveryResponse)
async def discovery(
    db: DashboardDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
    run_id: Optional[str] = Query(default=None),
    role: str = Query(default="all"),
) -> DiscoveryResponse:
    where = "netuid=?"
    params: list[Any] = [settings.netuid]
    resolved = _resolved_run(settings, run_id)
    if resolved is not None:
        where += " AND run_id=?"
        params.append(resolved)
    if role in {"train", "audit"}:
        where += " AND role=?"
        params.append(role)
    rows = await db.query(
        f"SELECT * FROM workers WHERE {where} ORDER BY role, host_id, worker_id",
        tuple(params),
    )
    now = time.time()
    records = [
        DiscoveryRecord(
            miner=json.loads(r["miner_json"] or "{}"),
            worker=json.loads(r["worker_json"] or "{}"),
            run_id=r["run_id"],
            role=r["role"],
            last_seen_unix=r["last_seen_unix"],
            age_sec=max(0.0, now - (r["last_seen_unix"] or 0)) if r["last_seen_unix"] else None,
        )
        for r in rows
    ]
    return DiscoveryResponse(
        meta={
            "bucket": os.environ.get("S3_BUCKET", ""),
            "netuid": settings.netuid,
            "run_id": resolved or "all",
            "role": role,
            "heartbeat_ttl_sec": settings.heartbeat_ttl_sec,
            "generated_unix": int(now),
            "source": "sqlite",
        },
        records=records,
    )


def _resolved_run(settings: Settings, run_id: Optional[str]) -> Optional[str]:
    if run_id in {None, "", "all", "*", "network"}:
        return settings.run_id or None
    return run_id
