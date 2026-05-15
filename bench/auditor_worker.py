"""In-process auditor worker for Locus.

This is the dockerized counterpart of `bench.presigned_ssh_worker dispatch
--role audit`, but without the SSH split. The container has bucket creds, so
the same process can:

  1. Heartbeat as an auditor under role=audit.
  2. Scan the audit job index for jobs assigned to this auditor hotkey.
  3. Decrypt the assignment grant (ed25519 by default) using the on-disk
     hotkey file mounted at /root/.bittensor/wallets/.
  4. Run `AuditReplayRunner` locally against the target receipt.
  5. Sign the resulting `AuditResultV3` with the auditor hotkey and PUT it to
     the bucket at `paths.audit_result_key`.

The validator in audit_mode=consume picks these results up and turns them into
signed verdicts (see locus_validator/verifier.py:consume_audit_results).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from bench.presigned_ssh_worker import (
    build_worker,
    decrypt_grant,
    eligible_jobs,
    grants_by_uri,
    write_heartbeat,
)
from locus_core import paths
from locus_core.cli import build_bucket
from locus_core.protocol import AuditResultV3, JobManifestV3, JobReceiptV3
from locus_core.signatures import verify_dict
from locus_runtime.storage import ObjectStore
from locus_runtime.transport import DirectArtifactTransport
from locus_validator.audit import AuditReplayConfig, AuditReplayRunner


def _process_audit_job(
    *,
    args: argparse.Namespace,
    bucket: ObjectStore,
    manifest: JobManifestV3,
    worker_hotkey: str,
) -> str:
    target = JobManifestV3.from_dict(manifest.params["target_manifest"])
    if args.owner_secret != "skip":
        if not target.owner_signature or not verify_dict(
            target.unsigned_dict(), args.owner_secret, target.owner_signature
        ):
            raise ValueError(f"audit target manifest has bad owner signature: {target.job_id}")
    grant = decrypt_grant(
        bucket,
        netuid=args.netuid,
        run_id=args.run_id,
        manifest=manifest,
        assignment_secret=args.assignment_secret,
        assignment_crypto=args.assignment_crypto,
        wallet_path=args.wallet_path,
        wallet_name=args.wallet_name,
        hotkey_name=args.hotkey_name,
        role="audit",
    )
    receipt = JobReceiptV3.from_dict(manifest.params["receipt"])

    transport = DirectArtifactTransport(bucket)
    runner = AuditReplayRunner(
        bucket=bucket,
        config=AuditReplayConfig(
            owner_secret=args.owner_secret,
            miner_secret=args.miner_secret,
            device=args.device,
        ),
        transport=transport,
        grants=grants_by_uri(grant),
    )
    result = runner.run(
        receipt_uri=manifest.params["receipt_uri"],
        manifest=target,
        receipt=receipt,
        auditor_hotkey=worker_hotkey,
    ).sign(worker_hotkey)
    body = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    key = paths.audit_result_key(args.netuid, args.run_id, worker_hotkey, receipt.receipt_id)
    bucket.put(bucket.uri_for_key(key), body)
    return result.status


def _index_done(bucket: ObjectStore, args: argparse.Namespace, manifest: JobManifestV3, worker_hotkey: str) -> bool:
    receipt = JobReceiptV3.from_dict(manifest.params["receipt"])
    key = paths.audit_result_key(args.netuid, args.run_id, worker_hotkey, receipt.receipt_id)
    return bucket.exists(bucket.uri_for_key(key))


def cmd_dispatch(args: argparse.Namespace) -> int:
    args.run_id = (args.run_id or "").strip()
    if not args.run_id:
        raise SystemExit(
            "error: --run-id is empty. Provide --run-id, set LOCUS_RUN_ID/RUN_ID "
            "in the environment, or rebuild the image with --build-arg LOCUS_RUN_ID=..."
        )
    bucket = build_bucket(args)
    device_group = args.device_group.split(",") if args.device_group else None
    worker = build_worker(
        args.hotkey,
        args.worker_id,
        host_id=os.environ.get("HOSTNAME", "auditor"),
        gpu_index=args.gpu_index,
        device_group=device_group,
    )
    deadline = time.time() + args.timeout_sec
    print(
        f"[auditor] start hotkey={args.hotkey} worker={args.worker_id} "
        f"device={args.device} run_id={args.run_id} netuid={args.netuid}"
    )
    while time.time() < deadline:
        try:
            write_heartbeat(
                bucket,
                netuid=args.netuid,
                run_id=args.run_id,
                worker=worker,
                role="audit",
            )
            jobs = eligible_jobs(
                bucket, netuid=args.netuid, run_id=args.run_id, worker=worker, role="audit"
            )
            for manifest in jobs:
                if _index_done(bucket, args, manifest, worker.hotkey_ss58):
                    continue
                t0 = time.time()
                try:
                    status = _process_audit_job(
                        args=args, bucket=bucket, manifest=manifest, worker_hotkey=worker.hotkey_ss58
                    )
                    print(
                        f"[auditor] job={manifest.job_id} status={status} "
                        f"elapsed={time.time() - t0:.2f}s"
                    )
                except Exception as e:
                    print(f"[auditor] job={manifest.job_id} ERROR: {e!r}")
        except Exception as e:
            print(f"[auditor] loop error: {e!r}")
        time.sleep(args.poll_interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument(
        "--run-id",
        default=os.environ.get("LOCUS_RUN_ID", ""),
        help="Defaults to $LOCUS_RUN_ID (set by the entrypoint from compose override or image-baked value).",
    )
    p.add_argument("--hotkey", required=True, help="auditor SS58")
    p.add_argument("--worker-id", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-group", default="")
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument("--assignment-secret", default=os.environ.get("LOCUS_ASSIGNMENT_SECRET", "locus-dev-assignment"))
    p.add_argument("--assignment-crypto", choices=["dev", "ed25519"], default=os.environ.get("LOCUS_ASSIGNMENT_CRYPTO", "ed25519"))
    p.add_argument("--miner-secret", default=os.environ.get("LOCUS_MINER_SECRET", "hotkey"))
    p.add_argument("--owner-secret", default=os.environ.get("LOCUS_OWNER_SECRET", "owner-dev-secret"))
    p.add_argument("--wallet-path", default=os.environ.get("BT_WALLET_PATH", "/root/.bittensor/wallets"))
    p.add_argument("--wallet-name", default=os.environ.get("BT_WALLET_NAME", "locus_mining"))
    p.add_argument("--hotkey-name", default=os.environ.get("BT_HOTKEY_NAME", ""))
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--timeout-sec", type=float, default=86400.0)
    p.add_argument("--local-root", default="/tmp/locus-v3")
    p.add_argument("--bucket", default="locus-v3")
    p.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", ""))
    p.add_argument("--s3-region", default=os.environ.get("S3_REGION", "us-east-1"))
    p.add_argument("--s3-endpoint-url", default=os.environ.get("S3_ENDPOINT_URL", ""))
    p.add_argument("--aws-access-key-id", default=os.environ.get("AWS_ACCESS_KEY_ID", ""))
    p.add_argument("--aws-secret-access-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
