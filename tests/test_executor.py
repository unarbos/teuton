from __future__ import annotations

import json

import torch

from teuton_core.ir import GraphBuilder
from teuton_core.protocol import ArtifactRef, GraphRef, JobManifestV3, VerificationPolicy, WorkerIdentity
from teuton_runtime import tensor_io
from teuton_runtime.executor import JobExecutor


def _double_graph(json_output: bool = False):
    gb = GraphBuilder()
    x = gb.input("x", [2], "float32")
    y = gb.mul(x, gb.const(2.0))
    if json_output:
        gb.output("metrics", y)
    else:
        gb.output("y", y)
    return gb.build()


def _manifest(bucket, graph, *, run_id: str = "run", json_output: bool = False) -> JobManifestV3:
    sha = graph.graph_id()
    graph_uri = bucket.uri_for_key(f"graphs/{sha}.json")
    bucket.put(graph_uri, graph.to_canonical_json())
    x_uri = bucket.uri_for_key("weights/x.bin")
    bucket.put(x_uri, tensor_io.encode_tensor(torch.tensor([1.0, 3.0])))
    out_name = "metrics" if json_output else "y"
    out_suffix = ".json" if json_output else ".bin"
    return JobManifestV3(
        job_id="job",
        run_id=run_id,
        step_id=0,
        kind="unit",
        graph_ref=GraphRef(sha256=sha, uri=graph_uri),
        params={},
        inputs=[ArtifactRef(name="x", uri=x_uri)],
        outputs=[ArtifactRef(name=out_name, uri=bucket.uri_for_key(f"outputs/{out_name}{out_suffix}"))],
        assigned_hotkey="miner",
        assigned_worker="worker",
        attempt=0,
        deadline_unix=10,
        created_unix=1,
        verification_policy=VerificationPolicy(),
    ).sign("owner")


def _worker() -> WorkerIdentity:
    return WorkerIdentity(
        hotkey_ss58="miner",
        worker_id="worker",
        host_id="host",
        gpu_index=None,
        session_nonce="nonce",
    )


def test_executor_writes_tensor_output_and_receipt(local_bucket) -> None:
    manifest = _manifest(local_bucket, _double_graph())
    executor = JobExecutor(bucket=local_bucket)

    receipt = executor.execute(manifest, worker=_worker(), miner_secret="miner-secret")

    output = tensor_io.decode_tensor(local_bucket.get(manifest.outputs[0].uri))
    assert torch.equal(output, torch.tensor([2.0, 6.0]))
    assert receipt.output_digests[0].sha256 == JobExecutor.digest_artifact(executor, manifest.outputs[0]).sha256
    assert receipt.claimed_bytes_read > 0
    assert receipt.claimed_bytes_written > 0


def test_executor_encodes_json_tensor_outputs(local_bucket) -> None:
    manifest = _manifest(local_bucket, _double_graph(json_output=True), json_output=True)
    executor = JobExecutor(bucket=local_bucket)

    executor.execute(manifest, worker=_worker(), miner_secret="miner-secret")

    body = json.loads(local_bucket.get(manifest.outputs[0].uri).decode("utf-8"))
    assert body["value"] == [2.0, 6.0]


def test_executor_caches_weight_inputs(local_bucket) -> None:
    manifest = _manifest(local_bucket, _double_graph())
    executor = JobExecutor(bucket=local_bucket)

    executor.execute(manifest, worker=_worker(), miner_secret="miner-secret")

    assert manifest.inputs[0].uri in executor._input_cache


def test_fault_mode_corrupts_output(local_bucket) -> None:
    manifest = _manifest(local_bucket, _double_graph())
    executor = JobExecutor(bucket=local_bucket)

    executor.execute(
        manifest,
        worker=_worker(),
        miner_secret="miner-secret",
        fault_mode="partial_corrupt",
        fault_rate=1.0,
    )

    output = tensor_io.decode_tensor(local_bucket.get(manifest.outputs[0].uri))
    assert not torch.equal(output, torch.tensor([2.0, 6.0]))
