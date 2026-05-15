"""Audit-job emission and local execution helpers."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from locus_core import paths
from locus_core.metagraph import BtcliMetagraphHotkeyResolver, MetagraphHotkeyResolver
from locus_core.protocol import (
    ArtifactRef,
    AssignmentGrantV3,
    EncryptedAssignmentGrantV3,
    GraphRef,
    JobManifestV3,
    JobReceiptV3,
    VerificationPolicy,
    WorkerIdentity,
)
from locus_core.wallet_crypto import AssignmentEncryptor, DevAssignmentCrypto, Ed25519SealedBoxAssignmentCrypto
from locus_orchestrator.scheduler import QuotaBook
from locus_runtime.discovery import build_discovery_backend
from locus_runtime.grants import broker_for_mode
from locus_runtime.storage import ObjectStore
from .verifier import ValidatorConfig, ReplayVerifier


@dataclass
class AuditJobConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    owner_secret: str = "owner-dev-secret"
    assignment_secret: str = "locus-dev-assignment"
    grant_mode: str = "direct"
    grant_ttl_sec: int = 600
    sample_rate: float = 1.0
    assignment_crypto: str = "dev"
    network: str = "finney"
    discovery_backend: str = "bucket"
    discovery_heartbeat_ttl_sec: float | None = 30.0


class AuditJobManager:
    def __init__(self, *, bucket: ObjectStore, config: AuditJobConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.quota = QuotaBook()
        self.discovery = build_discovery_backend(
            config.discovery_backend,
            bucket=bucket,
            netuid=config.netuid,
            run_id=config.run_id,
            role="audit",
            heartbeat_ttl_sec=config.discovery_heartbeat_ttl_sec,
        )
        self.grant_broker = broker_for_mode(config.grant_mode, bucket)
        self.assignment_crypto: AssignmentEncryptor = (
            Ed25519SealedBoxAssignmentCrypto()
            if config.assignment_crypto == "ed25519"
            else DevAssignmentCrypto(config.assignment_secret)
        )
        self.hotkey_resolver: MetagraphHotkeyResolver | None = (
            BtcliMetagraphHotkeyResolver(netuid=config.netuid, network=config.network)
            if config.assignment_crypto == "ed25519"
            else None
        )

    def run_once(self, *, max_jobs: int | None = None) -> int:
        emitted = 0
        self.discover_auditors()
        verifier = ReplayVerifier(
            bucket=self.bucket,
            config=ValidatorConfig(
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                validator_hotkey=self.config.validator_hotkey,
                owner_secret=self.config.owner_secret,
                sample_rate=self.config.sample_rate,
            ),
        )
        for receipt_uri, receipt in verifier.sample_receipts():
            if max_jobs is not None and emitted >= max_jobs:
                break
            if verifier.has_verdict(receipt) or verifier.find_audit_result(receipt) is not None:
                continue
            job_id = self.audit_job_id(receipt)
            if self.bucket.exists(self.bucket.uri_for_key(paths.audit_job_manifest_key(self.config.netuid, self.config.run_id, job_id))):
                continue
            self.emit_audit_job(receipt_uri, receipt, verifier.find_manifest(receipt), job_id)
            emitted += 1
        return emitted

    def discover_auditors(self) -> list[WorkerIdentity]:
        records = self.discovery.discover_workers()
        workers = [record.worker for record in records]
        self.quota.update_workers([record.miner for record in records], workers)
        return workers

    def emit_audit_job(self, receipt_uri: str, receipt: JobReceiptV3, target: JobManifestV3, job_id: str) -> JobManifestV3:
        worker = self.quota.pick_worker()
        output = ArtifactRef(
            name="audit_result",
            uri=self.bucket.uri_for_key(paths.audit_result_key(self.config.netuid, self.config.run_id, worker.hotkey_ss58, receipt.receipt_id)),
        )
        inputs = self.unique_refs([*target.inputs, *target.outputs])
        manifest = JobManifestV3(
            job_id=job_id,
            run_id=self.config.run_id,
            step_id=target.step_id,
            kind="audit_replay",
            graph_ref=GraphRef(sha256=target.graph_ref.sha256, uri=target.graph_ref.uri),
            params={
                "receipt_uri": receipt_uri,
                "receipt": receipt.to_dict(),
                "target_manifest": target.to_dict(),
            },
            inputs=inputs,
            outputs=[output],
            assigned_hotkey=worker.hotkey_ss58,
            assigned_worker=worker.worker_id,
            attempt=0,
            deadline_unix=int(time.time()) + int(self.config.grant_ttl_sec),
            created_unix=int(time.time()),
            verification_policy=VerificationPolicy(critical=False),
        ).sign(self.config.owner_secret)
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.audit_job_manifest_key(self.config.netuid, self.config.run_id, job_id)),
            manifest.to_dict(),
        )
        self.emit_assignment_grant(manifest)
        self.append_index(job_id)
        return manifest

    def emit_assignment_grant(self, manifest: JobManifestV3) -> None:
        if self.grant_broker is None:
            return
        now = int(time.time())
        grant = AssignmentGrantV3(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            assigned_hotkey=manifest.assigned_hotkey,
            input_gets=[self.grant_broker.get_grant(ref.uri, expires_in=self.config.grant_ttl_sec) for ref in manifest.inputs],
            output_puts=[self.grant_broker.put_grant(ref.uri, expires_in=self.config.grant_ttl_sec) for ref in manifest.outputs],
            receipt_put=None,
            created_unix=now,
            expires_unix=now + int(self.config.grant_ttl_sec),
        )
        if self.hotkey_resolver is not None:
            hotkey_info = self.hotkey_resolver.resolve(manifest.assigned_hotkey)
            encrypted = self.assignment_crypto.encrypt_for_hotkey(
                grant,
                recipient_hotkey=manifest.assigned_hotkey,
                recipient_uid=hotkey_info.uid,
                metagraph_block=hotkey_info.block,
                metagraph_hash=hotkey_info.metagraph_hash,
                recipient_public_key=hotkey_info.public_key,
            )
        else:
            encrypted = self.assignment_crypto.encrypt_for_hotkey(grant, recipient_hotkey=manifest.assigned_hotkey)
        self.bucket.put_json(
            self.bucket.uri_for_key(
                paths.audit_assignment_key(self.config.netuid, manifest.run_id, manifest.job_id, manifest.assigned_hotkey)
            ),
            encrypted.to_dict(),
        )

    def append_index(self, job_id: str) -> None:
        uri = self.bucket.uri_for_key(paths.audit_job_index_key(self.config.netuid, self.config.run_id))
        try:
            jobs = self.bucket.get_json(uri) if self.bucket.exists(uri) else []
        except Exception:
            jobs = []
        if job_id not in jobs:
            jobs.append(job_id)
            self.bucket.put_json(uri, jobs)

    @staticmethod
    def unique_refs(refs: list[ArtifactRef]) -> list[ArtifactRef]:
        out: list[ArtifactRef] = []
        seen: set[str] = set()
        for ref in refs:
            if ref.uri in seen:
                continue
            seen.add(ref.uri)
            out.append(ref)
        return out

    @staticmethod
    def audit_job_id(receipt: JobReceiptV3) -> str:
        suffix = hashlib.sha256(receipt.receipt_id.encode("utf-8")).hexdigest()[:16]
        return f"audit-{receipt.job_id}-{suffix}"
