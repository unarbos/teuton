"""Shared fixtures for the dashboard test suite."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from teuton_core import paths
from teuton_core.protocol import MinerIdentity, WorkerIdentity
from teuton_dashboard.app import create_app
from teuton_dashboard.bucket_factory import build_bucket
from teuton_dashboard.settings import Settings
from teuton_runtime.discovery import BucketDiscoveryBackend
from teuton_runtime.queue import OrchestratorQueue, QueueEntry
from teuton_runtime.storage import LocalBucket


@pytest.fixture
def bucket(tmp_path) -> LocalBucket:
    return LocalBucket(root=str(tmp_path / "bucket"), bucket="test")


@pytest.fixture
def settings(tmp_path, bucket) -> Settings:
    db_path = str(tmp_path / "dash.sqlite3")
    return Settings(
        S3_BUCKET="",  # forces LocalBucket
        TEUTON_LOCAL_BUCKET_ROOT=str(tmp_path / "bucket"),
        TEUTON_LOCAL_BUCKET_NAME="test",
        TEUTON_NETUID=0,
        TEUTON_RUN_ID="test-run",
        TEUTON_DASHBOARD_DB_PATH=db_path,
        TEUTON_DASHBOARD_BUCKET_POLL_SEC=0.1,
        TEUTON_DASHBOARD_CHAIN_POLL_SEC=600.0,
        TEUTON_DASHBOARD_QUEUE_SAMPLE_SEC=0.05,
        TEUTON_DASHBOARD_ENABLE_CHAIN=False,
        TEUTON_MAX_INFLIGHT_PER_HOTKEY=4,
        TEUTON_DASHBOARD_HOST="127.0.0.1",
        TEUTON_DASHBOARD_PORT=8767,
        TEUTON_DASHBOARD_STATIC_DIR="missing",
        TEUTON_DASHBOARD_SSE_KEEPALIVE_SEC=0.5,
    )


@pytest.fixture
def seeded_bucket(bucket, settings) -> LocalBucket:
    """Bucket with one heartbeat + one outstanding queue entry."""
    netuid, run_id = settings.netuid, settings.run_id
    disc = BucketDiscoveryBackend(bucket=bucket, netuid=netuid, run_id=run_id)
    disc.advertise_worker(
        miner=MinerIdentity(netuid=netuid, hotkey_ss58="hk-a", capabilities={}),
        worker=WorkerIdentity(
            hotkey_ss58="hk-a", worker_id="gpu0", host_id="host-a",
            gpu_index=0, session_nonce="n", software_hash="dev",
            device_group=[0], worker_group_id=None, capabilities={},
        ),
    )
    q = OrchestratorQueue(bucket=bucket, netuid=netuid, run_id=run_id, role="train")
    q.add(QueueEntry(
        job_id="j-a", assigned_hotkey="hk-a", assigned_worker="gpu0",
        manifest_uri="s3://test/m-a", grant_uri=None,
        deadline_unix=int(time.time()) + 600, attempt=0, created_unix=int(time.time()),
    ))
    q.flush()
    q.stop()
    return bucket


@pytest.fixture
async def app(settings, seeded_bucket):
    app = create_app(settings)
    # Drive the lifespan manually so background tasks start + clean up.
    async with _LifespanContext(app):
        yield app


@pytest.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class _LifespanContext:
    """Trigger FastAPI's lifespan context manager outside an HTTP server."""

    def __init__(self, app):
        self.app = app
        self._cm = None

    async def __aenter__(self):
        # FastAPI stores the lifespan context on the router; invoke it directly.
        self._cm = self.app.router.lifespan_context(self.app)
        await self._cm.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc, tb)
