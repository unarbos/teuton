"""Hotkey-bound worker process for Locus v3."""
from __future__ import annotations

import os
import json
import threading
import time
import uuid
from dataclasses import dataclass

from locus_core import paths
from locus_core.protocol import AssignmentGrantV3, EncryptedAssignmentGrantV3, JobManifestV3, MinerIdentity, WorkerIdentity
from locus_core.wallet_crypto import DevAssignmentCrypto
from locus_runtime.executor import JobExecutor
from locus_runtime.storage import ObjectStore
from locus_runtime.transport import DirectArtifactTransport, PresignedArtifactTransport
from .capabilities import detect_capabilities, gpu_index, probe_torch_device


@dataclass
class WorkerConfig:
    netuid: int
    run_id: str
    hotkey_ss58: str
    worker_id: str
    device: str = "cpu"
    miner_secret: str = "miner-dev-secret"
    poll_interval: float = 0.1
    heartbeat_interval: float = 1.0
    fault_mode: str = ""
    fault_rate: float = 1.0
    encryption_secret: str = "locus-dev-encryption"
    grant_mode: str = "direct"
    assignment_secret: str = "locus-dev-assignment"
    max_idle_iters: int | None = None


class MinerWorker:
    def __init__(self, *, bucket: ObjectStore, config: WorkerConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.stop_event = threading.Event()
        transport = DirectArtifactTransport(bucket) if config.grant_mode == "direct" else PresignedArtifactTransport(bucket)
        self.executor = JobExecutor(bucket=bucket, device=config.device, encryption_secret=config.encryption_secret, transport=transport)
        self.transport = transport
        self.assignment_crypto = DevAssignmentCrypto(config.assignment_secret)
        self.capabilities = detect_capabilities(bucket, run_id=config.run_id, worker_id=config.worker_id, device=config.device)
        self.identity = WorkerIdentity(
            hotkey_ss58=config.hotkey_ss58,
            worker_id=config.worker_id,
            host_id=self.capabilities.get("hostname", "unknown"),
            gpu_index=gpu_index(config.device),
            session_nonce=str(uuid.uuid4()),
            software_hash=os.environ.get("LOCUS_SOFTWARE_HASH", "dev"),
        )
        self.last_heartbeat = 0.0
        self.idle_iters = 0
        self._gpu_probe()
        self.heartbeat(force=True)

    def _gpu_probe(self) -> None:
        probe_torch_device(self.config.device)

    def stop(self) -> None:
        self.stop_event.set()

    def loop(self) -> None:
        while not self.stop_event.is_set():
            self.tick()
            time.sleep(self.config.poll_interval)

    def tick(self) -> bool:
        self.heartbeat()
        index_uri = self.bucket.uri_for_key(paths.job_index_key(self.config.netuid, self.config.run_id))
        job_ids = self._job_ids(index_uri)
        if not job_ids:
            self._idle()
            return False
        for job_id in job_ids:
            manifest_uri = self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, self.config.run_id, job_id))
            if not self.bucket.exists(manifest_uri):
                continue
            manifest = JobManifestV3.from_dict(self.bucket.get_json(manifest_uri))
            if not self._eligible(manifest):
                continue
            if all(self.bucket.exists(ref.uri) for ref in manifest.outputs):
                continue
            if not all(self.bucket.exists(ref.uri) for ref in manifest.inputs):
                continue
            grant = self.load_assignment_grant(manifest) if self.config.grant_mode != "direct" else None
            grants = self.grants_by_uri(grant) if grant is not None else None
            receipt = self.executor.execute(
                manifest,
                worker=self.identity,
                miner_secret=self.config.miner_secret,
                fault_mode=self.config.fault_mode,
                fault_rate=self.config.fault_rate,
                grants=grants,
            )
            receipt_uri = self.bucket.uri_for_key(
                paths.receipt_key(
                    self.config.netuid,
                    self.config.run_id,
                    self.config.hotkey_ss58,
                    manifest.job_id,
                    manifest.attempt,
                )
            )
            receipt_body = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
            if grant is not None and grant.receipt_put is not None:
                self.transport.put(receipt_uri, receipt_body, grant.receipt_put)
            else:
                self.bucket.put_json(receipt_uri, receipt.to_dict())
            self.idle_iters = 0
            return True
        self._idle()
        return False

    def _job_ids(self, fallback_index_uri: str) -> list[str]:
        out: list[str] = []
        jobs_prefix = self.bucket.uri_for_key(paths.jobs_prefix(self.config.netuid, self.config.run_id))
        for uri in self.bucket.list(jobs_prefix):
            if not uri.endswith("/index.json"):
                continue
            try:
                for job_id in self.bucket.get_json(uri):
                    if job_id not in out:
                        out.append(job_id)
            except Exception:
                continue
        if out:
            return out
        if self.bucket.exists(fallback_index_uri):
            try:
                return list(self.bucket.get_json(fallback_index_uri))
            except Exception:
                return []
        return []

    def load_assignment_grant(self, manifest: JobManifestV3) -> AssignmentGrantV3:
        uri = self.bucket.uri_for_key(
            paths.assignment_key(
                self.config.netuid,
                self.config.run_id,
                manifest.job_id,
                self.config.hotkey_ss58,
            )
        )
        if not self.bucket.exists(uri):
            raise FileNotFoundError(f"missing assignment grant for {manifest.job_id}")
        encrypted = EncryptedAssignmentGrantV3.from_dict(self.bucket.get_json(uri))
        grant = self.assignment_crypto.decrypt(encrypted, expected_hotkey=self.config.hotkey_ss58)
        self.validate_assignment_grant(manifest, grant)
        return grant

    @staticmethod
    def grants_by_uri(grant: AssignmentGrantV3) -> dict[str, object]:
        out: dict[str, object] = {}
        for item in [*grant.input_gets, *grant.output_puts]:
            out[item.canonical_uri] = item
        if grant.receipt_put is not None:
            out[grant.receipt_put.canonical_uri] = grant.receipt_put
        return out

    def validate_assignment_grant(self, manifest: JobManifestV3, grant: AssignmentGrantV3) -> None:
        now = int(time.time())
        if grant.job_id != manifest.job_id or grant.run_id != manifest.run_id:
            raise ValueError("assignment grant job mismatch")
        if grant.assigned_hotkey != self.config.hotkey_ss58:
            raise ValueError("assignment grant hotkey mismatch")
        if grant.expires_unix < now:
            raise ValueError("assignment grant expired")
        put_uris = {g.canonical_uri for g in grant.output_puts}
        expected_outputs = {ref.uri for ref in manifest.outputs}
        if put_uris != expected_outputs:
            raise ValueError("assignment grant output URI mismatch")
        if grant.receipt_put is None:
            raise ValueError("assignment grant missing receipt PUT")

    def _eligible(self, manifest: JobManifestV3) -> bool:
        if manifest.assigned_hotkey != self.config.hotkey_ss58:
            return False
        return manifest.assigned_worker in (None, self.config.worker_id)

    def _idle(self) -> None:
        self.idle_iters += 1
        if self.config.max_idle_iters is not None and self.idle_iters >= self.config.max_idle_iters:
            self.stop()

    def heartbeat(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_heartbeat < self.config.heartbeat_interval:
            return
        self.last_heartbeat = now
        info = MinerIdentity(
            netuid=self.config.netuid,
            hotkey_ss58=self.config.hotkey_ss58,
            capabilities=dict(self.capabilities),
        )
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.worker_heartbeat_key(self.config.netuid, self.config.hotkey_ss58, self.config.worker_id)),
            {
                "miner": info.to_dict(),
                "worker": self.identity.to_dict(),
                "run_id": self.config.run_id,
                "last_seen_unix": int(now),
            },
        )
