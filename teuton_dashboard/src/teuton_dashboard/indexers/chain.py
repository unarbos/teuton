"""Bittensor metagraph -> SQLite indexer.

bittensor's RPC client is blocking; we run each pass in a worker thread so the
event loop keeps serving the dashboard. Indexer is optional: if bittensor
isn't installed (e.g. local dev / tests) we mark the chain state as disabled
and the loop never runs.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any

from ..db import DashboardDB
from ..settings import Settings


LOG = logging.getLogger(__name__)


async def run_chain_indexer_loop(
    *,
    db: DashboardDB,
    settings: Settings,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(_index_chain_once, db, settings)
        except Exception:
            err = traceback.format_exc(limit=6)
            LOG.warning("chain indexer error: %s", err)
            try:
                await db.execute(
                    """
                    INSERT INTO chain_meta(netuid, observed_unix, error)
                    VALUES (?, ?, ?)
                    ON CONFLICT(netuid) DO UPDATE SET observed_unix=excluded.observed_unix, error=excluded.error
                    """,
                    (settings.netuid, int(time.time()), err),
                )
                await db.set_state("chain", error=err)
            except Exception:
                pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.chain_poll_sec)
        except asyncio.TimeoutError:
            pass


def _index_chain_once(db: DashboardDB, settings: Settings) -> None:
    """Synchronous chain scan; runs in a thread executor."""
    import bittensor as bt  # local import keeps the dep optional

    subtensor = bt.Subtensor(network=settings.bt_network)
    current_block = int(subtensor.get_current_block())
    try:
        tempo = int(subtensor.tempo(settings.netuid))
    except Exception:
        tempo = 0
    try:
        rate_limit = int(subtensor.query_subtensor("WeightsSetRateLimit", params=[settings.netuid]))
    except Exception:
        rate_limit = 0
    metagraph = subtensor.metagraph(settings.netuid)
    now = int(time.time())
    rows: list[tuple[Any, ...]] = []
    for uid, hotkey in enumerate(metagraph.hotkeys):
        rows.append(
            (
                settings.netuid,
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

    # ``db`` is async-only but called from a worker thread; build a synchronous
    # transaction via the raw sqlite3 stdlib module instead of dragging the
    # event loop in here. We connect with the same path + PRAGMAs DashboardDB
    # uses so behaviour matches.
    import sqlite3

    conn = sqlite3.connect(db.path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executemany(
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
        conn.execute(
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
            (settings.netuid, current_block, tempo, rate_limit, now),
        )
        conn.execute(
            """
            INSERT INTO indexer_state(name, cursor_json, updated_unix, error)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET cursor_json=excluded.cursor_json, updated_unix=excluded.updated_unix, error=excluded.error
            """,
            (
                "chain",
                __import__("json").dumps({"current_block": current_block, "hotkeys": len(rows)}, sort_keys=True),
                now,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


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
