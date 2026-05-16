"""Shared driver for running one audit-replay job from a JobManifestV3.

This is the common "I have a signed audit_replay manifest assigned to me;
I want to produce a signed AuditResultV3" sequence used by both:

- the legacy dedicated auditor box (``bench.auditor_worker``); and
- the audit-eligible branch of the regular miner worker
  (``teuton_miner.worker.MinerWorker``).

Keeping it here keeps the actual replay choreography (target manifest
signature check, AuditReplayRunner setup, result signing) in exactly one
place so both code paths can never drift.
"""
from __future__ import annotations

from typing import Any

from teuton_core.protocol import AuditResultV3, JobManifestV3, JobReceiptV3
from teuton_core.signatures import verify_dict
from teuton_runtime.storage import ObjectStore
from teuton_runtime.transport import ArtifactTransport
from .audit import AuditReplayConfig, AuditReplayRunner


def run_audit_replay(
    *,
    bucket: ObjectStore,
    manifest: JobManifestV3,
    worker_hotkey: str,
    owner_secret: str,
    miner_secret: str,
    device: str,
    grants: dict[str, Any] | None,
    transport: ArtifactTransport | None = None,
) -> AuditResultV3:
    """Run one audit_replay manifest end-to-end and return the signed result.

    The caller is responsible for:
      - Loading and decrypting the assignment grant (if any) and passing the
        per-uri grants map plus a matching transport.
      - Uploading the returned ``AuditResultV3`` to the bucket.

    Raises ``ValueError`` if the carried target manifest fails owner-signature
    verification (and ``owner_secret`` isn't ``"skip"``).
    """
    target = JobManifestV3.from_dict(manifest.params["target_manifest"])
    if owner_secret != "skip":
        if not target.owner_signature or not verify_dict(
            target.unsigned_dict(), owner_secret, target.owner_signature
        ):
            raise ValueError(
                f"audit target manifest has bad owner signature: {target.job_id}"
            )
    receipt = JobReceiptV3.from_dict(manifest.params["receipt"])

    runner = AuditReplayRunner(
        bucket=bucket,
        config=AuditReplayConfig(
            owner_secret=owner_secret,
            miner_secret=miner_secret,
            device=device,
        ),
        transport=transport,
        grants=grants or {},
    )
    audit = runner.run(
        receipt_uri=manifest.params["receipt_uri"],
        manifest=target,
        receipt=receipt,
        auditor_hotkey=worker_hotkey,
    )
    return audit.sign(worker_hotkey)
