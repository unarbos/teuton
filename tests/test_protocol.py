from __future__ import annotations

from teuton_core.protocol import (
    ArtifactDigest,
    ArtifactRef,
    GraphRef,
    JobManifestV3,
    JobReceiptV3,
    MinerIdentity,
    VerificationPolicy,
    VerificationVerdictV3,
    WorkerIdentity,
)
from teuton_core.signatures import verify_dict


def test_manifest_hash_and_signature_round_trip() -> None:
    manifest = JobManifestV3(
        job_id="job-1",
        run_id="run-1",
        step_id=0,
        kind="unit",
        graph_ref=GraphRef(sha256="abc", uri="s3://bucket/graph.json"),
        params={"x": 1},
        inputs=[ArtifactRef(name="x", uri="s3://bucket/x.bin")],
        outputs=[ArtifactRef(name="y", uri="s3://bucket/y.bin")],
        assigned_hotkey="miner",
        assigned_worker="worker",
        attempt=0,
        deadline_unix=10,
        created_unix=1,
        verification_policy=VerificationPolicy(sample_seed=7),
    ).sign("owner")

    as_dict = manifest.to_dict()
    restored = JobManifestV3.from_dict(as_dict)

    assert restored.manifest_hash() == manifest.manifest_hash()
    assert as_dict["manifest_hash"] == manifest.manifest_hash()
    assert verify_dict(restored.unsigned_dict(), "owner", restored.owner_signature or "")

    tampered = restored.unsigned_dict()
    tampered["params"] = {"x": 2}
    assert not verify_dict(tampered, "owner", restored.owner_signature or "")


def test_receipt_and_verdict_signatures_round_trip() -> None:
    worker = WorkerIdentity(
        hotkey_ss58="miner",
        worker_id="worker",
        host_id="host",
        gpu_index=0,
        session_nonce="nonce",
    )
    digest = ArtifactDigest(name="x", uri="s3://bucket/x.bin", sha256="abc", size_bytes=3)
    receipt = JobReceiptV3(
        receipt_id="receipt",
        manifest_hash="manifest",
        job_id="job",
        run_id="run",
        step_id=0,
        kind="unit",
        worker=worker,
        input_digests=[digest],
        output_digests=[digest],
        started_unix=1.0,
        finished_unix=2.0,
        compute_sec=1.0,
        claimed_bytes_read=3,
        claimed_bytes_written=3,
    ).sign("miner-secret")

    restored = JobReceiptV3.from_dict(receipt.to_dict())
    assert verify_dict(restored.unsigned_dict(), "miner-secret", restored.miner_signature or "")

    verdict = VerificationVerdictV3(
        verdict_id="verdict",
        receipt_id=restored.receipt_id,
        manifest_hash=restored.manifest_hash,
        job_id=restored.job_id,
        run_id=restored.run_id,
        miner_hotkey=restored.worker.hotkey_ss58,
        validator_hotkey="validator",
        status="pass",
        reason="ok",
        estimated_cu=1.0,
        replay_compute_sec=1.0,
        checked_unix=3.0,
    ).sign("validator-secret")

    restored_verdict = VerificationVerdictV3.from_dict(verdict.to_dict())
    assert verify_dict(
        restored_verdict.unsigned_dict(),
        "validator-secret",
        restored_verdict.validator_signature or "",
    )


def test_identity_and_artifacts_round_trip() -> None:
    miner = MinerIdentity(
        netuid=1,
        hotkey_ss58="hotkey",
        uid=42,
        endpoint="https://example.com",
        commitment_hash="hash",
        capabilities={"gpu": "RTX3090"},
    )
    assert MinerIdentity.from_dict(miner.to_dict()).capabilities["gpu"] == "RTX3090"

    ref = ArtifactRef(name="weights", uri="s3://bucket/w.bin", sha256="abc", size_bytes=123)
    assert ArtifactRef.from_dict(ref.to_dict()).size_bytes == 123
