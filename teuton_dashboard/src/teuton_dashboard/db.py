"""Async SQLite layer for the dashboard.

The schema preserves the legacy ``teuton_core.dashboard_backend`` shape minus
the dropped ``jobs`` table (outstanding work comes from the orchestrator queue
in :mod:`teuton_runtime.queue`; completed work is reconstructed from receipts
JOIN verdicts). Each call opens a short-lived aiosqlite connection with WAL
mode + NORMAL sync so readers (HTTP handlers) can overlap with writers
(indexer tasks) without lock contention.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import aiosqlite


LOG = logging.getLogger(__name__)


_SCHEMA = """
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


class DashboardDB:
    """Async SQLite wrapper. Opens a short-lived connection per call."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with self._open() as conn:
            await conn.execute("DROP TABLE IF EXISTS jobs")
            try:
                cur = await conn.execute("PRAGMA table_info(receipts)")
                cols = {row[1] for row in await cur.fetchall()}
                if cols and "kind" not in cols:
                    await conn.execute("ALTER TABLE receipts ADD COLUMN kind TEXT")
            except Exception as exc:
                LOG.debug("receipts kind probe failed: %r", exc)
            await conn.executescript(_SCHEMA)
            await conn.commit()

    @asynccontextmanager
    async def _open(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await aiosqlite.connect(self.path, timeout=30.0)
        try:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            await conn.close()

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        async with self._open() as conn:
            await conn.execute(sql, params)
            await conn.commit()

    async def executemany(self, sql: str, rows: Iterable[tuple[Any, ...]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        async with self._open() as conn:
            await conn.executemany(sql, rows)
            await conn.commit()
        return len(rows)

    async def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        async with self._open() as conn:
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute(sql, params)
            return list(await cur.fetchall())

    async def set_state(
        self,
        name: str,
        *,
        cursor: dict | None = None,
        error: str | None = None,
    ) -> None:
        await self.execute(
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
