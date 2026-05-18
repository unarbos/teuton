"""Unit tests for the orchestrator-owned outstanding-work queue."""
from __future__ import annotations

import threading
import time

import pytest

from teuton_core import paths
from teuton_runtime.queue import (
    OrchestratorQueue,
    QueueEntry,
    QueueState,
    queue_uri,
    read_queue,
)
from teuton_runtime.storage import PreconditionFailed


def _entry(job_id: str, *, hotkey: str = "hk1", worker: str | None = "w0", deadline: int = 0) -> QueueEntry:
    return QueueEntry(
        job_id=job_id,
        assigned_hotkey=hotkey,
        assigned_worker=worker,
        manifest_uri=f"s3://bkt/manifests/{job_id}.json",
        grant_uri=f"s3://bkt/grants/{job_id}.json",
        deadline_unix=deadline,
        attempt=0,
        created_unix=int(time.time()),
    )


def test_add_remove_depth(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a", hotkey="hk1"))
    queue.add(_entry("b", hotkey="hk1"))
    queue.add(_entry("c", hotkey="hk2"))
    assert queue.depth() == 3
    assert queue.depth("hk1") == 2
    assert queue.depth("hk2") == 1
    assert queue.depth("hk-missing") == 0
    assert queue.remove("a") is True
    assert queue.remove("a") is False
    assert queue.depth("hk1") == 1


def test_flush_publishes_snapshot(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a"))
    queue.add(_entry("b"))
    assert queue.flush() is True

    state = read_queue(local_bucket, netuid=0, run_id=run_id)
    assert state is not None
    assert state.role == "train"
    assert {e.job_id for e in state.outstanding} == {"a", "b"}
    assert state.snapshot_id == 1


def test_flush_skips_when_clean(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a"))
    queue.flush()
    assert queue.flush() is False


def test_idempotent_add_replaces(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a", deadline=100))
    queue.add(_entry("a", deadline=200))
    queue.flush()
    state = read_queue(local_bucket, netuid=0, run_id=run_id)
    assert state is not None
    assert len(state.outstanding) == 1
    assert state.outstanding[0].deadline_unix == 200


def test_replay_increments_attempt(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a", deadline=100))
    queue.replay("a", attempt=1, deadline_unix=999)
    queue.flush()
    state = read_queue(local_bucket, netuid=0, run_id=run_id)
    assert state is not None
    entry = state.outstanding[0]
    assert entry.attempt == 1
    assert entry.deadline_unix == 999


def test_replay_noop_when_missing(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.replay("nonexistent", attempt=1, deadline_unix=999)
    assert queue.depth() == 0


def test_prune_expired(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    now = time.time()
    queue.add(_entry("future", deadline=int(now + 1000)))
    queue.add(_entry("past", deadline=int(now - 1000)))
    queue.add(_entry("nodeadline", deadline=0))
    dropped = queue.prune_expired(now=now)
    assert {e.job_id for e in dropped} == {"past"}
    assert queue.depth() == 2


def test_reconcile_from_bucket(local_bucket, run_id) -> None:
    """A restarted orchestrator picks up the previous queue."""
    first = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    first.add(_entry("a"))
    first.add(_entry("b"))
    first.flush()
    first.stop(final_flush=False)

    second = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    second.reconcile_from_bucket()
    assert second.depth() == 2
    assert "a" in second
    assert "b" in second


def test_reconcile_drops_completed_receipts(local_bucket, run_id) -> None:
    first = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    first.add(_entry("a"))
    first.add(_entry("b"))
    first.flush()
    first.stop(final_flush=False)

    second = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    second.reconcile_from_bucket(recent_receipt_job_ids={"a"})
    assert second.depth() == 1
    assert "b" in second


def test_read_queue_if_none_match_returns_none_when_unchanged(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a"))
    queue.flush()

    state1 = read_queue(local_bucket, netuid=0, run_id=run_id)
    assert state1 is not None
    state2 = read_queue(local_bucket, netuid=0, run_id=run_id, if_none_match=state1.etag)
    assert state2 is None


def test_read_queue_if_none_match_returns_state_when_changed(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    queue.add(_entry("a"))
    queue.flush()
    state1 = read_queue(local_bucket, netuid=0, run_id=run_id)
    assert state1 is not None

    # Wait past the LocalBucket etag granularity (mtime_ns + size) so the
    # next put yields a distinct etag.
    time.sleep(0.01)
    queue.add(_entry("b"))
    queue.flush()
    state2 = read_queue(local_bucket, netuid=0, run_id=run_id, if_none_match=state1.etag)
    assert state2 is not None
    assert {e.job_id for e in state2.outstanding} == {"a", "b"}


def test_filter_for_worker(local_bucket, run_id) -> None:
    state = QueueState(
        role="train",
        snapshot_unix=0,
        snapshot_id=0,
        outstanding=[
            _entry("a", hotkey="alice", worker="w0"),
            _entry("b", hotkey="alice", worker="w1"),
            _entry("c", hotkey="bob", worker="w0"),
            _entry("d", hotkey="alice", worker=None),
        ],
    )
    out = state.filter_for_worker(hotkey_ss58="alice", worker_id="w0")
    assert {e.job_id for e in out} == {"a", "d"}


def test_conditional_write_detects_concurrent_writer(local_bucket, run_id) -> None:
    """Two queues sharing the same (run_id, role) detect each other via etag."""
    a = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    a.add(_entry("a"))
    a.flush()

    b = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id)
    b.reconcile_from_bucket()
    b.add(_entry("b"))

    # ``a`` writes again behind ``b``'s back; ``b`` should detect the etag
    # mismatch on its next flush, reconcile, and successfully publish a
    # merged snapshot.
    time.sleep(0.01)
    a.add(_entry("a2"))
    a.flush()

    time.sleep(0.01)
    b.flush()
    state = read_queue(local_bucket, netuid=0, run_id=run_id)
    assert state is not None
    job_ids = {e.job_id for e in state.outstanding}
    # b reconciled before retrying so a's writes are preserved alongside b's.
    assert {"a", "a2", "b"}.issubset(job_ids)


def test_audit_role_separate_file(local_bucket, run_id) -> None:
    """Train and audit queues are independent files."""
    train = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id, role="train")
    audit = OrchestratorQueue(bucket=local_bucket, netuid=0, run_id=run_id, role="audit")
    train.add(_entry("t1"))
    audit.add(_entry("a1"))
    train.flush()
    audit.flush()

    train_state = read_queue(local_bucket, netuid=0, run_id=run_id, role="train")
    audit_state = read_queue(local_bucket, netuid=0, run_id=run_id, role="audit")
    assert train_state is not None and audit_state is not None
    assert {e.job_id for e in train_state.outstanding} == {"t1"}
    assert {e.job_id for e in audit_state.outstanding} == {"a1"}


def test_queue_uri_path_shape(local_bucket, run_id) -> None:
    """Queue files live under publicly-readable prefixes.

    ``role="train"`` -> ``v3/netuid=N/jobs/{run_id}/queue.json``
    ``role="audit"`` -> ``v3/netuid=N/audits/{run_id}/jobs/queue.json``

    Both prefixes are covered by the bucket's ``PublicReadMinerCoordination``
    statement, so unauth'd miners and auditors can fetch the queue without
    AWS creds.
    """
    train_uri = queue_uri(local_bucket, netuid=0, run_id=run_id, role="train")
    assert train_uri.endswith(f"jobs/{run_id}/queue.json")
    assert paths.queue_key(0, run_id, "train") in train_uri

    audit_uri = queue_uri(local_bucket, netuid=0, run_id=run_id, role="audit")
    assert audit_uri.endswith(f"audits/{run_id}/jobs/queue.json")


def test_invalid_role_raises() -> None:
    with pytest.raises(ValueError):
        paths.queue_key(0, "any-run", "stranger")


def test_background_flush_thread_publishes(local_bucket, run_id) -> None:
    queue = OrchestratorQueue(
        bucket=local_bucket,
        netuid=0,
        run_id=run_id,
        flush_interval_sec=0.05,
    )
    queue.start_background_flush()
    try:
        queue.add(_entry("bg"))
        for _ in range(50):
            time.sleep(0.05)
            state = read_queue(local_bucket, netuid=0, run_id=run_id)
            if state is not None and any(e.job_id == "bg" for e in state.outstanding):
                break
        else:
            pytest.fail("background flush thread did not publish within 2.5s")
    finally:
        queue.stop()
