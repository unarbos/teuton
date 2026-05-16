from __future__ import annotations

from bench.presigned_ssh_worker import build_worker, write_heartbeat
from teuton_core import paths
from teuton_core.protocol import EncryptedAssignmentGrantV3, JobManifestV3, JobReceiptV3
from teuton_core.wallet_crypto import DevAssignmentCrypto
from teuton_orchestrator.run_manager import RunConfig, RunManager
from teuton_validator.audit import AuditReplayConfig, AuditReplayRunner
from teuton_validator.audit_jobs import AuditJobConfig, AuditJobManager
from teuton_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


def _run_training(local_bucket, run_id: str, start_miners) -> None:
    start_miners(run_id=run_id, count=2)
    manager = RunManager(
        bucket=local_bucket,
        config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
    )
    manager.run_loop(timeout_sec=30)


def test_audit_jobs_are_assigned_across_auditors(local_bucket, run_id, start_miners) -> None:
    _run_training(local_bucket, run_id, start_miners)
    for i in range(2):
        worker = build_worker(f"auditor{i}", f"auditor{i}-worker")
        write_heartbeat(local_bucket, netuid=0, run_id=run_id, worker=worker, role="audit")

    manager = AuditJobManager(
        bucket=local_bucket,
        config=AuditJobConfig(netuid=0, run_id=run_id, validator_hotkey="validator", grant_mode="local"),
    )
    emitted = manager.run_once(max_jobs=4)

    assert emitted == 4
    job_ids = local_bucket.get_json(local_bucket.uri_for_key(paths.audit_job_index_key(0, run_id)))
    manifests = [
        JobManifestV3.from_dict(local_bucket.get_json(local_bucket.uri_for_key(paths.audit_job_manifest_key(0, run_id, job_id))))
        for job_id in job_ids
    ]
    assert {manifest.kind for manifest in manifests} == {"audit_replay"}
    assert {manifest.assigned_hotkey for manifest in manifests} == {"auditor0", "auditor1"}


def test_validator_consumes_auditor_result(local_bucket, run_id, start_miners) -> None:
    _run_training(local_bucket, run_id, start_miners)
    auditor = build_worker("auditor0", "auditor0-worker")
    write_heartbeat(local_bucket, netuid=0, run_id=run_id, worker=auditor, role="audit")
    manager = AuditJobManager(
        bucket=local_bucket,
        config=AuditJobConfig(netuid=0, run_id=run_id, validator_hotkey="validator", grant_mode="local"),
    )
    assert manager.run_once(max_jobs=1) == 1

    job_id = local_bucket.get_json(local_bucket.uri_for_key(paths.audit_job_index_key(0, run_id)))[0]
    audit_manifest = JobManifestV3.from_dict(
        local_bucket.get_json(local_bucket.uri_for_key(paths.audit_job_manifest_key(0, run_id, job_id)))
    )
    encrypted = EncryptedAssignmentGrantV3.from_dict(
        local_bucket.get_json(local_bucket.uri_for_key(paths.audit_assignment_key(0, run_id, job_id, "auditor0")))
    )
    grant = DevAssignmentCrypto().decrypt(encrypted, expected_hotkey="auditor0")
    target = JobManifestV3.from_dict(audit_manifest.params["target_manifest"])
    receipt = JobReceiptV3.from_dict(audit_manifest.params["receipt"])
    audit = AuditReplayRunner(
        bucket=local_bucket,
        config=AuditReplayConfig(owner_secret="owner-dev-secret", miner_secret="miner-dev-secret"),
    ).run(
        receipt_uri=audit_manifest.params["receipt_uri"],
        manifest=target,
        receipt=receipt,
        auditor_hotkey="auditor0",
    ).sign("auditor0")
    local_bucket.put_json(grant.output_puts[0].canonical_uri, audit.to_dict())

    validator = ValidatorNeuron(
        bucket=local_bucket,
        config=ValidatorNeuronConfig(
            netuid=0,
            run_id=run_id,
            validator_hotkey="validator",
            sample_rate=1.0,
            audit_mode="consume",
        ),
    )
    result = validator.run_once(max_receipts=100, publish_weights=True)

    assert result["checked"] == 1
    verdicts = [
        local_bucket.get_json(uri)
        for uri in local_bucket.list(local_bucket.uri_for_key(paths.verdicts_prefix(0, run_id)))
        if uri.endswith(".json")
    ]
    assert len(verdicts) == 1
    assert verdicts[0]["status"] == "pass"
