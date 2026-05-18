"""/api/chain/{meta,hotkeys}."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..db import DashboardDB
from ..models import ChainHotkey, ChainHotkeysResponse, ChainMeta, ChainMetaResponse
from ..settings import Settings
from .deps import get_db, get_settings


router = APIRouter()


@router.get("/api/chain/meta", response_model=ChainMetaResponse)
async def chain_meta(
    db: DashboardDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ChainMetaResponse:
    rows = await db.query("SELECT * FROM chain_meta WHERE netuid=?", (settings.netuid,))
    return ChainMetaResponse(chain=ChainMeta(**dict(rows[0])) if rows else None)


@router.get("/api/chain/hotkeys", response_model=ChainHotkeysResponse)
async def chain_hotkeys(
    db: DashboardDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
    run_id: Optional[str] = Query(default=None),
) -> ChainHotkeysResponse:
    resolved = run_id if run_id and run_id not in {"all", "*", "network"} else settings.run_id or None
    if resolved is None:
        rows = await db.query(
            """
            SELECT DISTINCT w.hotkey, c.uid, c.stake, c.incentive, c.emission, c.validator_permit,
                            c.last_update_block, c.observed_block, c.observed_unix
            FROM workers w
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=?
            ORDER BY c.uid IS NULL, c.uid, w.hotkey
            """,
            (settings.netuid,),
        )
    else:
        rows = await db.query(
            """
            SELECT DISTINCT w.hotkey, c.uid, c.stake, c.incentive, c.emission, c.validator_permit,
                            c.last_update_block, c.observed_block, c.observed_unix
            FROM workers w
            LEFT JOIN chain_hotkeys c ON c.netuid=w.netuid AND c.hotkey=w.hotkey
            WHERE w.netuid=? AND w.run_id=?
            ORDER BY c.uid IS NULL, c.uid, w.hotkey
            """,
            (settings.netuid, resolved),
        )
    hotkeys = [_to_hotkey(r) for r in rows]
    return ChainHotkeysResponse(run_id=resolved or "all", hotkeys=hotkeys)


def _to_hotkey(row) -> ChainHotkey:
    current = row["observed_block"]
    last_update = row["last_update_block"]
    return ChainHotkey(
        uid=row["uid"],
        hotkey=row["hotkey"],
        stake=row["stake"],
        incentive=row["incentive"],
        emission=row["emission"],
        validator_permit=bool(row["validator_permit"]) if row["validator_permit"] is not None else None,
        last_update_block=last_update,
        observed_block=current,
        blocks_since_last_update=(current - last_update) if current is not None and last_update is not None else None,
        observed_unix=row["observed_unix"],
    )
