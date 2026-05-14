from __future__ import annotations

import json
import os
import time

import pytest
import torch

from locus_core.ir import GraphBuilder
from locus_core.protocol import ArtifactCryptoPolicy, ArtifactRef, CryptoMode, GraphRef, JobManifestV3, VerificationPolicy, WorkerIdentity
from locus_core.signatures import HmacSigner
from locus_runtime import tensor_io
from locus_runtime.crypto import BittensorDrandTimelockProvider, MockDrandTimelockProvider, TimelockPending, decode_envelope, encode_envelope
from locus_runtime.executor import JobExecutor
from locus_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


def _graph():
    gb = GraphBuilder()
    x = gb.input("x", [2], "float32")
    gb.output("y", gb.mul(x, gb.const(2.0)))
    return gb.build()


def _worker() -> WorkerIdentity:
    return WorkerIdentity("miner", "worker", "host", None, "nonce")


def _manifest(local_bucket, policy: ArtifactCryptoPolicy | None) -> JobManifestV3:
    graph = _graph()
    sha = graph.graph_id()
    graph_uri = local_bucket.uri_for_key(f"graphs/{sha}.json")
    local_bucket.put(graph_uri, graph.to_canonical_json())
    x_uri = local_bucket.uri_for_key("inputs/x.bin")
    local_bucket.put(x_uri, tensor_io.encode_tensor(torch.tensor([1.0, 3.0])))
    return JobManifestV3(
        job_id="job",
        run_id="crypto-run",
        step_id=0,
        kind="unit",
        graph_ref=GraphRef(sha, graph_uri),
        params={},
        inputs=[ArtifactRef("x", x_uri)],
        outputs=[ArtifactRef("y", local_bucket.uri_for_key("outputs/y.bin"), crypto=policy)],
        assigned_hotkey="miner",
        assigned_worker="worker",
        attempt=0,
        deadline_unix=10,
        created_unix=1,
        verification_policy=VerificationPolicy(),
    ).sign("owner-dev-secret")


def test_signed_artifact_envelope_round_trip() -> None:
    policy = ArtifactCryptoPolicy(mode=CryptoMode.SIGNED.value, required_signer="miner")
    signer = HmacSigner("secret", identity="miner")

    blob = encode_envelope(b"payload", policy, signer=signer)
    plaintext = decode_envelope(blob, policy, verifier=HmacSigner("secret"))

    assert plaintext == b"payload"


def test_tampered_signed_artifact_fails() -> None:
    policy = ArtifactCryptoPolicy(mode=CryptoMode.SIGNED.value, required_signer="miner")
    signer = HmacSigner("secret", identity="miner")
    env = json.loads(encode_envelope(b"payload", policy, signer=signer).decode("utf-8"))
    env["payload_b64"] = "dGFtcGVyZWQ="

    with pytest.raises(ValueError):
        decode_envelope(json.dumps(env).encode("utf-8"), policy, verifier=HmacSigner("secret"))


def test_plaintext_policy_preserves_bytes() -> None:
    assert encode_envelope(b"raw", None) == b"raw"
    assert decode_envelope(b"raw", None) == b"raw"


def test_executor_writes_signed_output(local_bucket) -> None:
    policy = ArtifactCryptoPolicy(mode=CryptoMode.SIGNED.value, required_signer="miner")
    manifest = _manifest(local_bucket, policy)
    executor = JobExecutor(bucket=local_bucket)

    receipt = executor.execute(manifest, worker=_worker(), miner_secret="miner-dev-secret")
    stored = local_bucket.get(manifest.outputs[0].uri)
    decoded = decode_envelope(stored, policy, verifier=HmacSigner("miner-dev-secret"))

    assert torch.equal(tensor_io.decode_tensor(decoded), torch.tensor([2.0, 6.0]))
    assert receipt.output_digests[0].crypto_mode == CryptoMode.SIGNED.value
    assert receipt.output_digests[0].signature


def test_validator_rejects_unsigned_output_when_signed_required(local_bucket) -> None:
    policy = ArtifactCryptoPolicy(mode=CryptoMode.SIGNED.value, required_signer="miner")
    manifest = _manifest(local_bucket, policy)
    executor = JobExecutor(bucket=local_bucket)
    receipt = executor.execute(manifest, worker=_worker(), miner_secret="miner-dev-secret")
    local_bucket.put_json(local_bucket.uri_for_key("v3/netuid=0/jobs/crypto-run/job/manifest.json"), manifest.to_dict())
    local_bucket.put_json(local_bucket.uri_for_key("v3/netuid=0/receipts/crypto-run/hotkey=miner/job/attempt=0.json"), receipt.to_dict())
    # Overwrite the signed envelope with raw tensor bytes.
    local_bucket.put(manifest.outputs[0].uri, tensor_io.encode_tensor(torch.tensor([2.0, 6.0])))

    validator = ValidatorNeuron(bucket=local_bucket, config=ValidatorNeuronConfig(netuid=0, run_id="crypto-run", validator_hotkey="validator"))
    result = validator.run_once(max_receipts=1, publish_weights=False)

    verdict_uri = local_bucket.list(local_bucket.uri_for_key("v3/netuid=0/verdicts/crypto-run/"))[0]
    verdict = local_bucket.get_json(verdict_uri)
    assert result["checked"] == 1
    assert verdict["status"] == "fail"


def test_mock_timelock_blocks_until_reveal() -> None:
    policy = ArtifactCryptoPolicy(mode=CryptoMode.DRAND_TIMELOCK.value, drand_round=10)
    locked = MockDrandTimelockProvider(revealed_round=0)
    revealed = MockDrandTimelockProvider(revealed_round=10)
    signer = HmacSigner("secret", identity="miner")

    blob = encode_envelope(b"secret", policy, signer=signer, timelock_provider=locked)
    with pytest.raises(TimelockPending):
        decode_envelope(blob, policy, verifier=HmacSigner("secret"), timelock_provider=locked)
    assert decode_envelope(blob, policy, verifier=HmacSigner("secret"), timelock_provider=revealed) == b"secret"


@pytest.mark.drand
def test_real_drand_tlock_round_trip() -> None:
    if os.environ.get("LOCUS_TEST_DRAND") != "1":
        pytest.skip("set LOCUS_TEST_DRAND=1 to run real drand tlock test")

    provider = BittensorDrandTimelockProvider()
    target_round = provider.latest_round() + 1
    policy = ArtifactCryptoPolicy(
        mode=CryptoMode.DRAND_TIMELOCK.value,
        drand_round=target_round,
    )
    signer = HmacSigner("secret", identity="miner")
    blob = encode_envelope(b"real drand secret", policy, signer=signer, timelock_provider=provider)

    deadline = time.time() + 20
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            assert decode_envelope(blob, policy, verifier=HmacSigner("secret"), timelock_provider=provider) == b"real drand secret"
            return
        except TimelockPending as e:
            last_error = e
            time.sleep(1)
    raise AssertionError(f"drand round {target_round} did not reveal before timeout: {last_error}")
