"""/healthz + /api/runs surface tests."""
from __future__ import annotations

import pytest


async def test_healthz(client):
    res = await client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["netuid"] == 0
    assert "states" in body


async def test_runs_returns_active_run(client, app):
    # Force one bucket-indexer pass so the runs table is populated.
    from teuton_dashboard.indexers.bucket import index_bucket_once

    n = await index_bucket_once(
        bucket=app.state.bucket, db=app.state.db, settings=app.state.settings
    )
    assert n >= 1
    res = await client.get("/api/runs")
    assert res.status_code == 200
    body = res.json()
    assert "test-run" in body["runs"]
    assert body["default_run_id"] == "test-run"
