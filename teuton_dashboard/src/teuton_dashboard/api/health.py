"""/healthz and /api/runs."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..db import DashboardDB
from ..models import ChainMeta, HealthResponse, IndexerState, RunsResponse
from ..settings import Settings
from .deps import get_db, get_settings


router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz(
    db: DashboardDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
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


@router.get("/api/runs", response_model=RunsResponse)
async def runs(
    db: DashboardDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RunsResponse:
    rows = await db.query(
        "SELECT run_id FROM runs WHERE netuid=? ORDER BY last_seen_unix DESC, run_id DESC LIMIT 100",
        (settings.netuid,),
    )
    return RunsResponse(runs=[r["run_id"] for r in rows], default_run_id=settings.run_id or "")
