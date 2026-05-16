from __future__ import annotations

from teuton_core import paths
from teuton_orchestrator.run_manager import RunConfig, RunManager
from teuton_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


def test_validator_scores_honest_miners(local_bucket, run_id, start_miners) -> None:
    start_miners(run_id=run_id, count=2)
    manager = RunManager(
        bucket=local_bucket,
        config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
    )
    manager.run_loop(timeout_sec=30)

    validator = ValidatorNeuron(
        bucket=local_bucket,
        config=ValidatorNeuronConfig(netuid=0, run_id=run_id, validator_hotkey="validator", sample_rate=1.0),
    )
    result = validator.run_once(max_receipts=100, publish_weights=True)

    assert result["checked"] == 10
    assert result["scores"]
    assert all(score["fail_cu"] == 0.0 for score in result["scores"].values())
    assert all(score["trust_multiplier"] == 1.0 for score in result["scores"].values())


def test_validator_zeroes_corrupt_miner(local_bucket, run_id, start_miners) -> None:
    start_miners(run_id=run_id, count=2, fault_index=0, fault_mode="partial_corrupt")
    manager = RunManager(
        bucket=local_bucket,
        config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
    )
    manager.run_loop(timeout_sec=30)

    validator = ValidatorNeuron(
        bucket=local_bucket,
        config=ValidatorNeuronConfig(netuid=0, run_id=run_id, validator_hotkey="validator", sample_rate=1.0),
    )
    result = validator.run_once(max_receipts=100, publish_weights=True)

    assert result["scores"]["miner0"]["fail_cu"] > 0.0
    assert result["scores"]["miner0"]["score"] == 0.0
    assert result["scores"]["miner0"]["trust_multiplier"] == 0.0
    assert result["weight_update"]["weights"][0] == 0.0


def test_bad_miner_signature_is_a_failed_verdict(local_bucket, run_id, start_miners) -> None:
    start_miners(run_id=run_id, count=1)
    manager = RunManager(
        bucket=local_bucket,
        config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
    )
    manager.run_loop(timeout_sec=30)

    for receipt_uri in local_bucket.list(local_bucket.uri_for_key(paths.receipts_prefix(0, run_id))):
        if not receipt_uri.endswith(".json"):
            continue
        receipt = local_bucket.get_json(receipt_uri)
        receipt["miner_signature"] = "bad-signature"
        local_bucket.put_json(receipt_uri, receipt)

    validator = ValidatorNeuron(
        bucket=local_bucket,
        config=ValidatorNeuronConfig(netuid=0, run_id=run_id, validator_hotkey="validator", sample_rate=1.0),
    )
    validator.run_once(max_receipts=100, publish_weights=False)

    verdicts = [
        local_bucket.get_json(uri)
        for uri in local_bucket.list(local_bucket.uri_for_key(paths.verdicts_prefix(0, run_id)))
        if uri.endswith(".json")
    ]
    assert verdicts
    assert all(verdict["status"] == "fail" for verdict in verdicts)
    assert all(verdict["reason"] == "bad miner signature" for verdict in verdicts)
