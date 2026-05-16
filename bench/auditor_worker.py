"""In-process auditor worker for Teuton (legacy dedicated-auditor stack).

NOTE: As of the trusted-anchor peer-audit rework, dedicated off-chain auditors
are deprecated. The preferred deployment is to flag specific on-chain miner
hotkeys as audit-eligible (see ``TEUTON_AUDIT_ELIGIBLE_HOTKEYS``); those
miners will pick up ``audit_replay`` jobs from inside the regular miner loop
via ``teuton_miner.worker.MinerWorker._tick_audit_jobs``.

This file is retained for one release as a debug/fallback tool and as a
working example of the audit-replay choreography. It now delegates the
actual signature check + replay + signing to
``teuton_validator.audit_dispatch.run_audit_replay`` so the on-chain and
off-chain auditor paths can't drift.

What this process does end-to-end:

  1. Heartbeat as an auditor under role=audit.
  2. Scan the audit job index for jobs assigned to this auditor hotkey.
  3. Decrypt the assignment grant (ed25519 by default) using the on-disk
     hotkey file mounted at /root/.bittensor/wallets/.
  4. Call ``run_audit_replay`` to verify the target manifest, re-execute
     the graph against ``receipt.input_digests``, compare outputs, and
     produce a signed ``AuditResultV3``.
  5. PUT the result to the bucket at ``paths.audit_result_key``.

The validator in audit_mode=consume picks these results up and turns them into
signed verdicts (see ``teuton_validator/verifier.py:consume_audit_results``).
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
from teuton_core import paths
from teuton_core.cli import build_bucket
from teuton_core.protocol import JobManifestV3, JobReceiptV3
from teuton_runtime.storage import ObjectStore
from teuton_core.telemetry import TelemetryWriter
from teuton_runtime.transport import DirectArtifactTransport
from teuton_validator.audit_dispatch import run_audit_replay


def _device_info(device: str) -> dict:
    info: dict = {"device": device}
    try:
        import torch
        info["torch"] = getattr(torch, "__version__", "")
        if torch.cuda.is_available():
            idx = 0
            if device.startswith("cuda:"):
                try:
                    idx = int(device.split(":", 1)[1])
                except ValueError:
                    idx = 0
            info["name"] = torch.cuda.get_device_name(idx)
            free, total = torch.cuda.mem_get_info(idx)
            info["free_vram_mb"] = int(free / 1024 / 1024)
            info["total_vram_mb"] = int(total / 1024 / 1024)
    except Exception as e:
        info["error"] = repr(e)
    return info


def _process_audit_job(
    *,
    args: argparse.Namespace,
    bucket: ObjectStore,
    manifest: JobManifestV3,
    worker_hotkey: str,
) -> tuple[str, dict]:
    """Run one audit job. Returns (status, phase_timings_ms)."""
    timings_ms: dict = {}

    t_decrypt = time.time()
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
    timings_ms["decrypt_grant"] = round((time.time() - t_decrypt) * 1000, 2)
    receipt = JobReceiptV3.from_dict(manifest.params["receipt"])

    t_replay = time.time()
    transport = DirectArtifactTransport(bucket)
    result = run_audit_replay(
        bucket=bucket,
        manifest=manifest,
        worker_hotkey=worker_hotkey,
        owner_secret=args.owner_secret,
        miner_secret=args.miner_secret,
        device=args.device,
        grants=grants_by_uri(grant),
        transport=transport,
    )
    timings_ms["replay"] = round((time.time() - t_replay) * 1000, 2)

    t_up = time.time()
    body = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = paths.audit_result_key(args.netuid, args.run_id, worker_hotkey, receipt.receipt_id)
    bucket.put(bucket.uri_for_key(key), body)
    timings_ms["upload"] = round((time.time() - t_up) * 1000, 2)
    timings_ms["total"] = round(sum(timings_ms.values()), 2)
    return result.status, timings_ms


def _index_done(bucket: ObjectStore, args: argparse.Namespace, manifest: JobManifestV3, worker_hotkey: str) -> bool:
    receipt = JobReceiptV3.from_dict(manifest.params["receipt"])
    key = paths.audit_result_key(args.netuid, args.run_id, worker_hotkey, receipt.receipt_id)
    return bucket.exists(bucket.uri_for_key(key))


def cmd_dispatch(args: argparse.Namespace) -> int:
    args.run_id = (args.run_id or "").strip()
    if not args.run_id:
        raise SystemExit(
            "error: --run-id is empty. Provide --run-id, set TEUTON_RUN_ID/RUN_ID "
            "in the environment, or rebuild the image with --build-arg TEUTON_RUN_ID=..."
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

    telemetry = TelemetryWriter(
        bucket=bucket,
        netuid=args.netuid,
        run_id=args.run_id,
        component=f"auditor/{args.worker_id}",
    )
    dev_info = _device_info(args.device)
    rollup: dict = {
        "counts": {"pass": 0, "fail": 0, "inconclusive": 0, "error": 0},
        "timings_ms_sum": {},
        "n_jobs": 0,
        "window_started_unix": int(time.time()),
    }
    last_rollup_emit = time.time()
    rollup_interval = 30.0

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
                    status, timings_ms = _process_audit_job(
                        args=args, bucket=bucket, manifest=manifest, worker_hotkey=worker.hotkey_ss58
                    )
                    print(
                        f"[auditor] job={manifest.job_id} status={status} "
                        f"elapsed={time.time() - t0:.2f}s timings_ms={timings_ms}"
                    )
                    rollup["n_jobs"] += 1
                    rollup["counts"][status] = rollup["counts"].get(status, 0) + 1
                    for k, v in timings_ms.items():
                        rollup["timings_ms_sum"][k] = rollup["timings_ms_sum"].get(k, 0.0) + float(v)
                except Exception as e:
                    rollup["n_jobs"] += 1
                    rollup["counts"]["error"] = rollup["counts"].get("error", 0) + 1
                    print(f"[auditor] job={manifest.job_id} ERROR: {e!r}")
        except Exception as e:
            print(f"[auditor] loop error: {e!r}")

        now = time.time()
        if now - last_rollup_emit >= rollup_interval:
            denom = max(1, rollup["n_jobs"])
            avg_ms = {k: round(float(v) / denom, 2) for k, v in rollup["timings_ms_sum"].items()}
            try:
                telemetry.audits(
                    {
                        "auditor_hotkey": args.hotkey,
                        "worker_id": args.worker_id,
                        "device_info": dev_info,
                        "window_started_unix": rollup["window_started_unix"],
                        "window_seconds": int(now - last_rollup_emit),
                        "n_jobs": rollup["n_jobs"],
                        "counts": rollup["counts"],
                        "avg_timings_ms": avg_ms,
                    }
                )
            except Exception:
                pass
            last_rollup_emit = now
            rollup["counts"] = {"pass": 0, "fail": 0, "inconclusive": 0, "error": 0}
            rollup["timings_ms_sum"] = {}
            rollup["n_jobs"] = 0
            rollup["window_started_unix"] = int(now)

        time.sleep(args.poll_interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Defaults to $TEUTON_RUN_ID (set by the entrypoint from compose override or image-baked value).",
    )
    p.add_argument("--hotkey", required=True, help="auditor SS58")
    p.add_argument("--worker-id", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-group", default="")
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument("--assignment-secret", default=os.environ.get("TEUTON_ASSIGNMENT_SECRET", "teuton-dev-assignment"))
    p.add_argument("--assignment-crypto", choices=["dev", "ed25519"], default=os.environ.get("TEUTON_ASSIGNMENT_CRYPTO", "ed25519"))
    p.add_argument("--miner-secret", default=os.environ.get("TEUTON_MINER_SECRET", "hotkey"))
    p.add_argument("--owner-secret", default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"))
    p.add_argument("--wallet-path", default=os.environ.get("BT_WALLET_PATH", "/root/.bittensor/wallets"))
    p.add_argument("--wallet-name", default=os.environ.get("BT_WALLET_NAME", "teuton_mining"))
    p.add_argument("--hotkey-name", default=os.environ.get("BT_HOTKEY_NAME", ""))
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--timeout-sec", type=float, default=86400.0)
    p.add_argument("--local-root", default="/tmp/teuton-v3")
    p.add_argument("--bucket", default="teuton-v3")
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
