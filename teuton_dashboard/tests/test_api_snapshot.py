"""/api/snapshot and /api/queue + /api/jobs surface tests."""
from __future__ import annotations

import pytest

from teuton_dashboard.indexers.bucket import index_bucket_once


async def test_snapshot_returns_queue_and_machines(client, app):
    await index_bucket_once(
        bucket=app.state.bucket, db=app.state.db, settings=app.state.settings
    )
    res = await client.get("/api/snapshot")
    assert res.status_code == 200
    body = res.json()
    assert sorted(body.keys()) == ["audit_queue", "jobs", "machines", "meta", "queue", "run"]
    assert body["meta"]["run_id"] == "test-run"
    assert body["meta"]["max_inflight_per_hotkey"] == 4
    # queue may be empty until /api/queue is hit (bus is lazy-populated by
    # the snapshot handler itself on cache miss).
    assert body["queue"]["depth_total"] == 1
    assert body["queue"]["depth_by_hotkey"] == {"hk-a": 1}
    assert len(body["queue"]["outstanding"]) == 1
    assert body["jobs"]["outstanding"][0]["job_id"] == "j-a"
    assert body["jobs"]["completed"] == []
    # Machines has the inflight column populated.
    machines = body["machines"]
    assert len(machines) == 1
    worker = machines[0]["workers"][0]
    assert worker["queue_depth"] == 1
    assert worker["queue_cap"] == 4
    assert worker["at_cap"] is False


async def test_queue_endpoint(client, app):
    res = await client.get("/api/queue?role=train")
    assert res.status_code == 200
    body = res.json()
    assert body["queue"]["depth_total"] == 1
    assert body["meta"]["role"] == "train"


async def test_jobs_endpoint_outstanding_only(client, app):
    res = await client.get("/api/jobs?kind=outstanding")
    assert res.status_code == 200
    body = res.json()
    assert len(body["outstanding"]) == 1
    assert body["completed"] == []


async def test_snapshot_for_unknown_run_returns_empty_queue(client):
    res = await client.get("/api/snapshot?run_id=nope")
    assert res.status_code == 200
    body = res.json()
    assert body["queue"] is not None
    assert body["queue"]["depth_total"] == 0
    assert body["jobs"]["outstanding"] == []
