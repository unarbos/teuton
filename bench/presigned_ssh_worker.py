"""Run Teuton jobs on an SSH worker using only presigned grants.

The dispatcher runs on a trusted machine with bucket access. The remote worker
receives only public job metadata, graph JSON, its hotkey identity, and
presigned URLs for the specific artifacts it may read/write.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teuton_core import paths
from teuton_core.cli import build_bucket
from teuton_core.protocol import AssignmentGrantV3, EncryptedAssignmentGrantV3, JobManifestV3, JobReceiptV3, MinerIdentity, WorkerIdentity
from teuton_core.signatures import verify_dict
from teuton_core.wallet_crypto import DevAssignmentCrypto, Ed25519SealedBoxAssignmentCrypto
from teuton_runtime.discovery import build_discovery_backend
from teuton_runtime.distributed_executor import DistributedJobExecutor
from teuton_runtime.executor import JobExecutor
from teuton_runtime.queue import read_queue
from teuton_runtime.storage import ObjectStore
from teuton_runtime.transport import PresignedArtifactTransport
from teuton_validator.audit import AuditReplayConfig, AuditReplayRunner


@dataclass
class GraphBundleBucket:
    graph_uri: str
    graph_body: bytes
    bucket: str = "presigned-worker"

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return f"s3://{bucket or self.bucket}/{key}"

    def get(self, uri: str) -> bytes:
        if uri != self.graph_uri:
            raise FileNotFoundError(uri)
        return self.graph_body

    def put(self, uri: str, data: bytes) -> None:
        raise RuntimeError("presigned worker bucket is read-only")

    def exists(self, uri: str) -> bool:
        return uri == self.graph_uri

    def delete(self, uri: str) -> None:
        return None

    def list(self, prefix_uri: str) -> list[str]:
        return []

    def get_json(self, uri: str) -> dict:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: dict | list) -> None:
        raise RuntimeError("presigned worker bucket is read-only")


def build_worker(
    hotkey: str,
    worker_id: str,
    *,
    host_id: str = "ssh-presigned",
    gpu_index: int | None = None,
    device_group: list[str] | None = None,
) -> WorkerIdentity:
    gpu_indices: list[int] = []
    for device in device_group or []:
        if device.startswith("cuda:"):
            try:
                gpu_indices.append(int(device.split(":", 1)[1]))
            except ValueError:
                pass
    return WorkerIdentity(
        hotkey_ss58=hotkey,
        worker_id=worker_id,
        host_id=host_id,
        gpu_index=gpu_index,
        session_nonce=str(uuid.uuid4()),
        software_hash="ssh-presigned",
        device_group=gpu_indices,
        worker_group_id=worker_id if len(gpu_indices) > 1 else None,
        capabilities={
            "device_group": list(device_group or []),
            "gpu_indices": gpu_indices,
            "world_size": max(1, len(device_group or [])),
            "placement": "single_host",
            "multi_gpu": len(device_group or []) > 1,
        },
    )


def grants_by_uri(grant: AssignmentGrantV3) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in [*grant.input_gets, *grant.output_puts]:
        out[item.canonical_uri] = item
    if grant.receipt_put is not None:
        out[grant.receipt_put.canonical_uri] = grant.receipt_put
    return out


def write_heartbeat(bucket: ObjectStore, *, netuid: int, run_id: str, worker: WorkerIdentity, role: str = "train") -> None:
    info = MinerIdentity(netuid=netuid, hotkey_ss58=worker.hotkey_ss58, capabilities={"transport": "ssh-presigned", "role": role})
    discovery = build_discovery_backend("bucket", bucket=bucket, netuid=netuid, run_id=run_id, role=role)
    discovery.advertise_worker(miner=info, worker=worker)


def decrypt_grant(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    manifest: JobManifestV3,
    assignment_secret: str,
    assignment_crypto: str = "dev",
    wallet_path: str = "~/.bittensor/wallets",
    wallet_name: str = "",
    hotkey_name: str = "",
    role: str = "train",
) -> AssignmentGrantV3:
    key = (
        paths.audit_assignment_key(netuid, run_id, manifest.job_id, manifest.assigned_hotkey)
        if role == "audit"
        else paths.assignment_key(netuid, run_id, manifest.job_id, manifest.assigned_hotkey)
    )
    uri = bucket.uri_for_key(key)
    encrypted = EncryptedAssignmentGrantV3.from_dict(bucket.get_json(uri))
    if assignment_crypto == "ed25519":
        if not wallet_name or not hotkey_name:
            raise ValueError("ed25519 assignment crypto requires wallet_name and hotkey_name")
        keyfile = Path(wallet_path).expanduser() / wallet_name / "hotkeys" / hotkey_name
        return Ed25519SealedBoxAssignmentCrypto.from_keyfile(keyfile).decrypt(
            encrypted,
            expected_hotkey=manifest.assigned_hotkey,
        )
    return DevAssignmentCrypto(assignment_secret).decrypt(encrypted, expected_hotkey=manifest.assigned_hotkey)


def eligible_jobs(bucket: ObjectStore, *, netuid: int, run_id: str, worker: WorkerIdentity, role: str = "train") -> list[JobManifestV3]:
    state = read_queue(bucket, netuid=netuid, run_id=run_id, role=role)
    if state is None:
        return []
    out: list[JobManifestV3] = []
    for entry in state.outstanding:
        if entry.assigned_hotkey != worker.hotkey_ss58:
            continue
        if entry.assigned_worker not in (None, "", worker.worker_id):
            continue
        try:
            manifest = JobManifestV3.from_dict(bucket.get_json(entry.manifest_uri))
        except Exception:
            continue
        if manifest.resource_requirements.min_gpus > max(1, len(worker.device_group)):
            continue
        if all(bucket.exists(ref.uri) for ref in manifest.outputs):
            continue
        if not all(bucket.exists(ref.uri) for ref in manifest.inputs):
            continue
        out.append(manifest)
    return out


def dispatch_job(args: argparse.Namespace, bucket: ObjectStore, manifest: JobManifestV3, worker: WorkerIdentity) -> None:
    if args.role == "audit":
        target = JobManifestV3.from_dict(manifest.params["target_manifest"])
        if not target.owner_signature or not verify_dict(target.unsigned_dict(), args.owner_secret, target.owner_signature):
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
        role=args.role,
    )
    bundle = {
        "role": args.role,
        "manifest": manifest.to_dict(),
        "graph": json.loads(bucket.get(manifest.graph_ref.uri).decode("utf-8")),
        "graph_uri": manifest.graph_ref.uri,
        "grant": grant.to_dict(),
        "worker": worker.to_dict(),
        "device": args.device,
        "device_group": args.device_group.split(",") if args.device_group else [args.device],
        "miner_secret": args.miner_secret,
    }
    target = f"{args.user}@{args.host}"
    remote = "cd /root/teuton && source .venv/bin/activate && python -m bench.presigned_ssh_worker execute"
    cmd = ["ssh", "-p", str(args.port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", target, remote]
    subprocess.run(cmd, input=json.dumps(bundle).encode("utf-8"), check=True)


def cmd_dispatch(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    device_group = args.device_group.split(",") if args.device_group else None
    worker = build_worker(args.hotkey, args.worker_id, gpu_index=args.gpu_index, device_group=device_group)
    deadline = time.time() + args.timeout_sec
    while time.time() < deadline:
        write_heartbeat(bucket, netuid=args.netuid, run_id=args.run_id, worker=worker, role=args.role)
        for manifest in eligible_jobs(bucket, netuid=args.netuid, run_id=args.run_id, worker=worker, role=args.role):
            dispatch_job(args, bucket, manifest, worker)
        time.sleep(args.poll_interval)
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    bundle = json.loads(args.input.read())
    role = bundle.get("role", "train")
    manifest = JobManifestV3.from_dict(bundle["manifest"])
    grant = AssignmentGrantV3.from_dict(bundle["grant"])
    worker = WorkerIdentity.from_dict(bundle["worker"])
    graph_body = json.dumps(bundle["graph"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    bucket = GraphBundleBucket(graph_uri=bundle["graph_uri"], graph_body=graph_body)
    transport = PresignedArtifactTransport()
    if role == "audit":
        target_manifest = JobManifestV3.from_dict(manifest.params["target_manifest"])
        receipt = JobReceiptV3.from_dict(manifest.params["receipt"])
        result = AuditReplayRunner(
            bucket=bucket,
            config=AuditReplayConfig(
                owner_secret="skip",
                miner_secret=bundle.get("miner_secret", "hotkey"),
                device=bundle["device"],
            ),
            transport=transport,
            grants=grants_by_uri(grant),
        ).run(
            receipt_uri=manifest.params["receipt_uri"],
            manifest=target_manifest,
            receipt=receipt,
            auditor_hotkey=worker.hotkey_ss58,
        ).sign(worker.hotkey_ss58)
        body = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        if not manifest.outputs:
            raise ValueError("audit job missing output")
        transport.put(manifest.outputs[0].uri, body, grants_by_uri(grant).get(manifest.outputs[0].uri))
        return 0
    device_group = list(bundle.get("device_group") or [bundle["device"]])
    executor = (
        DistributedJobExecutor(bucket=bucket, devices=device_group, transport=transport)
        if len(device_group) > 1
        else JobExecutor(bucket=bucket, device=bundle["device"], transport=transport)
    )
    receipt = executor.execute(
        manifest,
        worker=worker,
        miner_secret=worker.hotkey_ss58,
        grants=grants_by_uri(grant),
    )
    if grant.receipt_put is None:
        raise ValueError("assignment grant missing receipt PUT")
    body = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    transport.put(grant.receipt_put.canonical_uri, body, grant.receipt_put)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    dispatch = sub.add_parser("dispatch")
    dispatch.add_argument("--netuid", type=int, required=True)
    dispatch.add_argument("--run-id", required=True)
    dispatch.add_argument("--hotkey", required=True)
    dispatch.add_argument("--worker-id", required=True)
    dispatch.add_argument("--role", choices=["train", "audit"], default="train")
    dispatch.add_argument("--device", default="cpu")
    dispatch.add_argument("--device-group", default="")
    dispatch.add_argument("--gpu-index", type=int, default=None)
    dispatch.add_argument("--host", required=True)
    dispatch.add_argument("--port", type=int, required=True)
    dispatch.add_argument("--user", default="root")
    dispatch.add_argument("--assignment-secret", required=True)
    dispatch.add_argument("--miner-secret", default="hotkey")
    dispatch.add_argument("--assignment-crypto", choices=["dev", "ed25519"], default="dev")
    dispatch.add_argument("--wallet-path", default="~/.bittensor/wallets")
    dispatch.add_argument("--wallet-name", default="")
    dispatch.add_argument("--hotkey-name", default="")
    dispatch.add_argument("--owner-secret", default="owner-dev-secret")
    dispatch.add_argument("--poll-interval", type=float, default=0.2)
    dispatch.add_argument("--timeout-sec", type=float, default=900.0)
    dispatch.add_argument("--local-root", default="/tmp/teuton-v3")
    dispatch.add_argument("--bucket", default="teuton-v3")
    dispatch.add_argument("--s3-bucket", default="")
    dispatch.add_argument("--s3-region", default="us-east-1")
    dispatch.add_argument("--s3-endpoint-url", default="")
    dispatch.add_argument("--aws-access-key-id", default="")
    dispatch.add_argument("--aws-secret-access-key", default="")
    dispatch.set_defaults(fn=cmd_dispatch)

    execute = sub.add_parser("execute")
    execute.add_argument("input", nargs="?", type=argparse.FileType("r"), default="-")
    execute.set_defaults(fn=cmd_execute)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
