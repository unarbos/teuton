"""Async indexer cycle tests against the LocalBucket fixture."""
from __future__ import annotations

import time

import pytest

from teuton_dashboard.indexers.bucket import index_bucket_once
from teuton_dashboard.indexers.queue_sampler import _safe_read_queue, project_state
from teuton_dashboard.queue_bus import QueueBus


async def test_bucket_indexer_populates_workers_and_receipts_table(app):
    db = app.state.db
    indexed = await index_bucket_once(
        bucket=app.state.bucket, db=db, settings=app.state.settings
    )
    assert indexed >= 1
    workers = await db.query("SELECT * FROM workers WHERE netuid=?", (0,))
    assert any(r["hotkey"] == "hk-a" for r in workers)


def test_queue_sampler_projection_includes_entries(app):
    state = _safe_read_queue(app.state.bucket, 0, "test-run", "train")
    assert state is not None
    snap = project_state(
        state, run_id="test-run", role="train", cap=4, bus=app.state.bus, now_unix=int(time.time())
    )
    assert snap.depth_total == 1
    assert snap.depth_by_hotkey == {"hk-a": 1}
    assert len(snap.outstanding) == 1
    assert snap.outstanding[0].job_id == "j-a"
