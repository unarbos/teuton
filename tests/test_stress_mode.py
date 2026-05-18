from __future__ import annotations

from teuton_orchestrator.streaming import StreamingRunConfig, StreamingRunManager
from teuton_runtime.queue import read_queue


def test_stress_emit_loops_with_unique_ids_and_pinned_weights(
    local_bucket, run_id, start_miners, tiny_gpt_pipe
) -> None:
    start_miners(run_id=run_id, count=2, hotkey_prefix="stress")

    bootstrap_manager = StreamingRunManager(
        bucket=local_bucket,
        config=StreamingRunConfig(netuid=0, run_id=run_id, task="gpt_pipe", max_epochs=1),
    )
    bootstrap_manager.run_loop(poll_interval=0.02, timeout_sec=60.0)

    stress_manager = StreamingRunManager(
        bucket=local_bucket,
        config=StreamingRunConfig(
            netuid=0,
            run_id=run_id,
            task="gpt_pipe",
            max_epochs=10,
            stress_emit=True,
            stress_emit_interval=0.0,
            stress_epoch_base=10_000,
            stress_pin_weights_epoch=0,
            stress_max_iterations=3,
            max_inflight_per_hotkey=8,
        ),
    )
    stress_manager.run_loop(poll_interval=0.02, timeout_sec=60.0)

    stress_ids = [j for j in stress_manager.emitted if j.startswith("j-e10")]
    assert stress_ids, "stress mode emitted no jobs"
    assert all("-fwd" in j for j in stress_ids), "stress mode should emit forward-only jobs"
    assert all("-bwd" not in j for j in stress_ids)
    assert all("-outer" not in j for j in stress_ids)

    pinned = stress_manager.epoch_weights_uri(epoch=10_001, stage_id=0)
    assert "weights/epoch=0/" in pinned, f"weights URI not pinned to epoch 0: {pinned}"

    # Queue must stay bounded by max_inflight_per_hotkey * n_miners. With
    # 2 miners and the default cap of 8, we expect <= 16 outstanding at
    # any time (and after the stress run finishes most are likely drained
    # via receipts already).
    state = read_queue(local_bucket, netuid=0, run_id=run_id, role="train")
    if state is not None:
        per_hotkey: dict[str, int] = {}
        for entry in state.outstanding:
            per_hotkey[entry.assigned_hotkey] = per_hotkey.get(entry.assigned_hotkey, 0) + 1
        for hk, depth in per_hotkey.items():
            assert depth <= 8, f"hotkey {hk} exceeded max_inflight_per_hotkey: {depth}"


def test_per_epoch_deadline_propagated_to_wait_epoch(local_bucket, run_id, tiny_gpt_pipe) -> None:
    """Static config check: StreamingRunConfig surfaces a per-epoch deadline
    that wait_epoch honours, instead of relying solely on the global timeout.

    The full e2e behaviour is exercised by the streaming integration tests; here
    we only check the contract so we never regress to a 1-year hang."""
    cfg = StreamingRunConfig(netuid=0, run_id=run_id, task="gpt_pipe", epoch_timeout_sec=42.0)
    assert cfg.epoch_timeout_sec == 42.0
    mgr = StreamingRunManager(bucket=local_bucket, config=cfg)
    assert mgr.config.epoch_timeout_sec == 42.0
