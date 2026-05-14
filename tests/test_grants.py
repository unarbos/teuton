from __future__ import annotations

import os
import threading
import time

import pytest
import torch

from locus_core import paths
from locus_core.ir import GraphBuilder
from locus_core.protocol import ArtifactRef, AssignmentGrantV3, GraphRef, JobManifestV3, MetagraphMinerIdentity, PresignedUrlGrant, VerificationPolicy, WorkerIdentity
from locus_core.wallet_crypto import BittensorWalletCrypto, DevAssignmentCrypto
from locus_miner.worker import MinerWorker, WorkerConfig
from locus_orchestrator.run_manager import RunConfig, RunManager
from locus_runtime import tensor_io
from locus_runtime.executor import JobExecutor
from locus_runtime.grants import LocalGrantBroker, S3PresignedUrlBroker
from locus_runtime.storage import S3Bucket
from locus_runtime.transport import PresignedArtifactTransport
from locus_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


def test_metagraph_identity_round_trip() -> None:
    identity = MetagraphMinerIdentity(netuid=1, uid=7, hotkey_ss58="hotkey", coldkey_ss58="cold", stake=1.5)
    restored = MetagraphMinerIdentity.from_dict(identity.to_dict())
    assert restored.hotkey_ss58 == "hotkey"
    assert restored.uid == 7


def test_dev_assignment_grant_round_trip() -> None:
    grant = AssignmentGrantV3(
        job_id="job",
        run_id="run",
        assigned_hotkey="miner",
        output_puts=[PresignedUrlGrant("PUT", "s3://bucket/out", "s3://bucket/out", int(time.time()) + 60)],
    )
    crypto = DevAssignmentCrypto("secret")

    encrypted = crypto.encrypt_for_hotkey(grant, recipient_hotkey="miner", recipient_uid=3)
    restored = crypto.decrypt(encrypted, expected_hotkey="miner")

    assert encrypted.recipient_uid == 3
    assert restored.job_id == "job"
    with pytest.raises(ValueError):
        crypto.decrypt(encrypted, expected_hotkey="other")


def test_bittensor_wallet_crypto_reports_no_encryption() -> None:
    class DummyKeypair:
        ss58_address = "hotkey"

        def sign(self, payload):
            return b"sig"

        def verify(self, payload, signature):
            return signature == b"sig"

    adapter = BittensorWalletCrypto(DummyKeypair())
    assert adapter.verify(b"payload", adapter.sign(b"payload"))
    with pytest.raises(NotImplementedError):
        adapter.encrypt_for_hotkey(AssignmentGrantV3("job", "run", "hotkey"), recipient_hotkey="hotkey")


def test_presigned_transport_rejects_mismatched_or_expired_grant(local_bucket) -> None:
    uri = local_bucket.uri_for_key("x.bin")
    transport = PresignedArtifactTransport(local_bucket)

    with pytest.raises(ValueError, match="canonical URI mismatch"):
        transport.put(uri, b"x", PresignedUrlGrant("PUT", local_bucket.uri_for_key("y.bin"), uri, int(time.time()) + 60))
    with pytest.raises(ValueError, match="expired"):
        transport.put(uri, b"x", PresignedUrlGrant("PUT", uri, uri, int(time.time()) - 1))


def test_presigned_executor_does_not_need_direct_artifact_access(local_bucket) -> None:
    gb = GraphBuilder()
    x = gb.input("x", [2], "float32")
    gb.output("y", gb.mul(x, gb.const(2.0)))
    graph = gb.build()
    graph_uri = local_bucket.uri_for_key(f"graphs/{graph.graph_id()}.json")
    input_uri = local_bucket.uri_for_key("weights/x.bin")
    output_uri = local_bucket.uri_for_key("outputs/y.bin")
    local_bucket.put(graph_uri, graph.to_canonical_json())
    local_bucket.put(input_uri, tensor_io.encode_tensor(torch.tensor([2.0, 5.0])))

    manifest = JobManifestV3(
        job_id="job",
        run_id="run",
        step_id=0,
        kind="unit",
        graph_ref=GraphRef(sha256=graph.graph_id(), uri=graph_uri),
        params={},
        inputs=[ArtifactRef(name="x", uri=input_uri)],
        outputs=[ArtifactRef(name="y", uri=output_uri)],
        assigned_hotkey="miner",
        assigned_worker="worker",
        attempt=0,
        deadline_unix=int(time.time()) + 60,
        created_unix=int(time.time()),
        verification_policy=VerificationPolicy(),
    ).sign("owner")
    broker = LocalGrantBroker()
    grants = {
        input_uri: broker.get_grant(input_uri, expires_in=60),
        output_uri: broker.put_grant(output_uri, expires_in=60),
    }

    class MetadataOnlyBucket:
        bucket = local_bucket.bucket

        def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
            return local_bucket.uri_for_key(key, bucket=bucket)

        def get(self, uri: str) -> bytes:
            if uri == graph_uri:
                return local_bucket.get(uri)
            raise AssertionError(f"direct artifact GET was attempted for {uri}")

        def put(self, uri: str, data: bytes) -> None:
            raise AssertionError(f"direct artifact PUT was attempted for {uri}")

        def exists(self, uri: str) -> bool:
            return local_bucket.exists(uri)

        def list(self, prefix_uri: str) -> list[str]:
            return local_bucket.list(prefix_uri)

        def get_json(self, uri: str) -> dict:
            return local_bucket.get_json(uri)

        def put_json(self, uri: str, value: dict | list) -> None:
            local_bucket.put_json(uri, value)

    executor = JobExecutor(
        bucket=MetadataOnlyBucket(),
        transport=PresignedArtifactTransport(local_bucket),
    )
    worker = WorkerIdentity(
        hotkey_ss58="miner",
        worker_id="worker",
        host_id="host",
        gpu_index=None,
        session_nonce="nonce",
    )
    executor.execute(manifest, worker=worker, miner_secret="miner-secret", grants=grants)

    assert torch.equal(tensor_io.decode_tensor(local_bucket.get(output_uri)), torch.tensor([4.0, 10.0]))


def test_local_grant_mode_round_flow(local_bucket, run_id) -> None:
    worker = MinerWorker(
        bucket=local_bucket,
        config=WorkerConfig(
            netuid=0,
            run_id=run_id,
            hotkey_ss58="miner",
            worker_id="miner-gpu0",
            grant_mode="local",
        ),
    )
    thread = threading.Thread(target=worker.loop, daemon=True)
    thread.start()
    try:
        manager = RunManager(
            bucket=local_bucket,
            config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1, grant_mode="local"),
        )
        manager.run_loop(timeout_sec=30, poll_interval=0.02)
        validator = ValidatorNeuron(
            bucket=local_bucket,
            config=ValidatorNeuronConfig(netuid=0, run_id=run_id, validator_hotkey="validator", sample_rate=1.0),
        )
        result = validator.run_once(max_receipts=100, publish_weights=True)
        assert result["checked"] > 0
        assert local_bucket.list(local_bucket.uri_for_key(f"{paths.root(0)}/assignments/{run_id}/"))
    finally:
        worker.stop()
        thread.join(timeout=2.0)


@pytest.mark.s3
def test_s3_presigned_broker_smoke() -> None:
    if os.environ.get("LOCUS_TEST_S3") != "1":
        pytest.skip("set LOCUS_TEST_S3=1 to run S3 presigned broker test")
    required = ["S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    if any(not os.environ.get(k) for k in required):
        pytest.skip("missing S3 env")
    bucket = S3Bucket(
        bucket=os.environ["S3_BUCKET"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
    )
    uri = bucket.uri_for_key(f"v3/test-presigned/{int(time.time())}.txt")
    broker = S3PresignedUrlBroker(bucket)
    grant = broker.put_grant(uri, expires_in=60)
    transport = PresignedArtifactTransport()
    transport.put(uri, b"ok", grant)
    assert bucket.get(uri) == b"ok"
    bucket.delete(uri)
