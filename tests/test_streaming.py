from __future__ import annotations

from teuton_orchestrator.streaming import StreamingRunConfig, StreamingRunManager
from teuton_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


def test_tiny_gpt_pipe_streaming_bridge(local_bucket, run_id, start_miners, tiny_gpt_pipe) -> None:
    start_miners(run_id=run_id, count=2, hotkey_prefix="stream")
    manager = StreamingRunManager(
        bucket=local_bucket,
        config=StreamingRunConfig(netuid=0, run_id=run_id, task="gpt_pipe", max_epochs=1),
    )

    manager.run_loop(poll_interval=0.02, timeout_sec=60)

    validator = ValidatorNeuron(
        bucket=local_bucket,
        config=ValidatorNeuronConfig(netuid=0, run_id=run_id, validator_hotkey="validator", sample_rate=1.0),
    )
    first = validator.run_once(max_receipts=100, publish_weights=True)
    second = validator.run_once(max_receipts=100, publish_weights=True)

    total_checked = first["checked"] + second["checked"]
    scores = second["scores"] or first["scores"]
    assert total_checked > 0
    assert scores
    assert all(score["fail_cu"] == 0.0 for score in scores.values())
