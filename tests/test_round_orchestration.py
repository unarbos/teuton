from __future__ import annotations

import pytest

from teuton_core import paths
from teuton_miner.worker import MinerWorker, WorkerConfig
from teuton_orchestrator.run_manager import RunConfig, RunManager


def test_round_flow_writes_step_index_and_outputs(local_bucket, run_id, start_miners) -> None:
    start_miners(run_id=run_id, count=2)
    manager = RunManager(
        bucket=local_bucket,
        config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
    )

    manager.run_loop(timeout_sec=30)

    step_index_uri = local_bucket.uri_for_key(paths.job_step_index_key(0, run_id, 0))
    jobs = local_bucket.get_json(step_index_uri)
    assert "step0-eval" in jobs
    assert len(jobs) == 10
    assert local_bucket.exists(local_bucket.uri_for_key(paths.weights_key(0, run_id, 1, 0)))
    assert local_bucket.exists(local_bucket.uri_for_key(paths.weights_key(0, run_id, 1, 1)))


def test_orchestrator_ignores_stale_run_heartbeats(local_bucket, run_id) -> None:
    old_worker = MinerWorker(
        bucket=local_bucket,
        config=WorkerConfig(
            netuid=0,
            run_id="old-run",
            hotkey_ss58="miner-old",
            worker_id="miner-old-gpu0",
        ),
    )
    try:
        manager = RunManager(
            bucket=local_bucket,
            config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
        )
        manager.bootstrap()
        assert manager.discover_workers() == []
        with pytest.raises(TimeoutError, match="no miners"):
            manager.run_loop(timeout_sec=0.05, poll_interval=0.01)
    finally:
        old_worker.stop()


def test_timeout_marks_stale_and_releases_quota(local_bucket, run_id) -> None:
    worker = MinerWorker(
        bucket=local_bucket,
        config=WorkerConfig(
            netuid=0,
            run_id=run_id,
            hotkey_ss58="miner",
            worker_id="miner-gpu0",
        ),
    )
    try:
        manager = RunManager(
            bucket=local_bucket,
            config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
        )
        manager.bootstrap()
        manager.discover_workers()
        manifest = manager.emit_forward(0)

        with pytest.raises(TimeoutError):
            manager.wait_outputs(manifest, timeout_sec=0.01)

        stale_uri = local_bucket.uri_for_key(f"{paths.jobs_prefix(0, run_id)}{manifest.job_id}/stale.json")
        assert local_bucket.exists(stale_uri)
        assert manager.quota.accounts["miner"].inflight_cu == 0.0
    finally:
        worker.stop()
