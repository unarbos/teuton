"""Push a wave of synthetic jobs through the visualizer demo run.

Reads `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `S3_REGION`
from the environment (or `.env`).

Run:
    python -m bench.viz_demo_feed
    python -m bench.viz_demo_feed --run-id viz-demo-1778865107 --batch 10

Produces, per wave, jobs that span every state the visualizer renders (created,
running, outputs_written, completed, verified, failed, stale) plus refreshed
heartbeats for the three demo workers so the UI shows them as live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import uuid
from dataclasses import dataclass

import boto3

from teuton_core import paths
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


NETUID = 0
GRAPH_SHA = "demograph"
GRAPH_KEY_SUFFIX = "graphs/demograph.json"


@dataclass
class DemoWorker:
    hotkey: str
    worker_id: str
    host_id: str
    gpu_index: int
    gpu_name: str
    role: str = "train"


WORKERS: list[DemoWorker] = [
    DemoWorker("miner-alpha", "demo-A-gpu0", "demo-host-A", 0, "A100"),
    DemoWorker("miner-beta", "demo-A-gpu1", "demo-host-A", 1, "A100"),
    DemoWorker("miner-gamma", "demo-B-gpu0", "demo-host-B", 0, "H100"),
]

VALIDATOR_HOTKEY = "validator-demo"


STATUS_RECIPE: list[str] = [
    "verified",
    "completed",
    "outputs_written",
    "running",
    "created",
    "stale",
    "failed",
    "verified",
    "completed",
    "running",
]


def make_bucket():
    return boto3.client(
        "s3",
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def put_json(s3, bucket: str, key: str, body: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
        ContentType="application/json",
    )


def put_bytes(s3, bucket: str, key: str, body: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/octet-stream")


def ensure_run_skeleton(s3, bucket: str, run_id: str) -> None:
    graph_key = f"{paths.run_root(NETUID, run_id)}/{GRAPH_KEY_SUFFIX}"
    try:
        s3.head_object(Bucket=bucket, Key=graph_key)
    except Exception:
        put_json(s3, bucket, graph_key, {"nodes": []})
    weights_key = paths.weights_key(NETUID, run_id, 0, 0)
    try:
        s3.head_object(Bucket=bucket, Key=weights_key)
    except Exception:
        put_bytes(s3, bucket, weights_key, b"\x00" * 1024)


def refresh_heartbeats(s3, bucket: str, run_id: str, now: int, session: str) -> None:
    for worker in WORKERS:
        capabilities = {
            "gpu_name": worker.gpu_name,
            "role": worker.role,
            "transport": "direct",
            "world_size": 1,
        }
        miner = MinerIdentity(netuid=NETUID, hotkey_ss58=worker.hotkey, capabilities=capabilities)
        wid = WorkerIdentity(
            hotkey_ss58=worker.hotkey,
            worker_id=worker.worker_id,
            host_id=worker.host_id,
            gpu_index=worker.gpu_index,
            session_nonce=session,
            software_hash="demo",
            device_group=[worker.gpu_index],
            capabilities=capabilities,
        )
        body = {
            "miner": miner.to_dict(),
            "worker": wid.to_dict(),
            "run_id": run_id,
            "role": worker.role,
            "last_seen_unix": now,
        }
        put_json(s3, bucket, paths.worker_heartbeat_key(NETUID, worker.hotkey, worker.worker_id), body)


def append_index(s3, bucket: str, run_id: str, new_ids: list[str]) -> list[str]:
    """Maintain the demo run's queue.json with the supplied job ids.

    The visualizer demo runs a synthetic feed without an orchestrator, so it
    has to maintain its own queue snapshot. We write a minimal compatible
    ``queue/train.json`` body containing the entries the demo just emitted.
    """
    key = paths.queue_key(NETUID, run_id, "train")
    body = {
        "version": 1,
        "role": "train",
        "snapshot_unix": int(time.time()),
        "snapshot_id": int(time.time()),
        "outstanding": [
            {
                "job_id": jid,
                "assigned_hotkey": "demo-hotkey",
                "assigned_worker": None,
                "manifest_uri": s3_uri(bucket, paths.job_manifest_key(NETUID, run_id, jid)),
                "grant_uri": None,
                "deadline_unix": int(time.time()) + 600,
                "attempt": 0,
                "created_unix": int(time.time()),
            }
            for jid in new_ids
        ],
    }
    put_json(s3, bucket, key, body)
    return new_ids


def fake_sha(salt: str) -> str:
    return hashlib.sha256(salt.encode()).hexdigest()


def build_manifest(
    *,
    bucket: str,
    run_id: str,
    job_id: str,
    step_id: int,
    worker: DemoWorker,
    now: int,
    deadline_offset: int,
    critical: bool = False,
) -> JobManifestV3:
    weights_uri = s3_uri(bucket, paths.weights_key(NETUID, run_id, 0, 0))
    out_uri = s3_uri(
        bucket,
        paths.artifact_key(NETUID, run_id, job_id, worker.hotkey, worker.worker_id, 0, "out"),
    )
    return JobManifestV3(
        job_id=job_id,
        run_id=run_id,
        step_id=step_id,
        kind="demo_step",
        graph_ref=GraphRef(sha256=GRAPH_SHA, uri=s3_uri(bucket, f"{paths.run_root(NETUID, run_id)}/{GRAPH_KEY_SUFFIX}")),
        params={"demo": True, "wave": True, "i": step_id},
        inputs=[ArtifactRef(name="weights", uri=weights_uri, sha256=fake_sha(f"weights-{run_id}"), size_bytes=1024)],
        outputs=[ArtifactRef(name="out", uri=out_uri)],
        assigned_hotkey=worker.hotkey,
        assigned_worker=worker.worker_id,
        attempt=0,
        deadline_unix=now + deadline_offset,
        created_unix=now,
        verification_policy=VerificationPolicy(critical=critical),
    )


def write_manifest(s3, bucket: str, manifest: JobManifestV3) -> None:
    key = paths.job_manifest_key(NETUID, manifest.run_id, manifest.job_id)
    put_json(s3, bucket, key, manifest.to_dict())


def write_output(s3, bucket: str, manifest: JobManifestV3, payload: bytes) -> None:
    output = manifest.outputs[0]
    key = output.uri.split(f"s3://{bucket}/", 1)[1]
    put_bytes(s3, bucket, key, payload)


def write_receipt(
    s3,
    bucket: str,
    *,
    manifest: JobManifestV3,
    worker: DemoWorker,
    started: float,
    finished: float,
    bytes_read: int,
    bytes_written: int,
    payload_sha: str,
) -> JobReceiptV3:
    receipt = JobReceiptV3(
        receipt_id=f"{manifest.run_id}:{manifest.job_id}:{worker.hotkey}:{worker.worker_id}:{manifest.attempt}",
        manifest_hash=manifest.manifest_hash(),
        job_id=manifest.job_id,
        run_id=manifest.run_id,
        step_id=manifest.step_id,
        kind=manifest.kind,
        worker=WorkerIdentity(
            hotkey_ss58=worker.hotkey,
            worker_id=worker.worker_id,
            host_id=worker.host_id,
            gpu_index=worker.gpu_index,
            session_nonce="demo-session",
            software_hash="demo",
            device_group=[worker.gpu_index],
            capabilities={
                "gpu_name": worker.gpu_name,
                "role": worker.role,
                "transport": "direct",
                "world_size": 1,
            },
        ),
        input_digests=[
            ArtifactDigest(
                name="weights",
                uri=manifest.inputs[0].uri,
                sha256=manifest.inputs[0].sha256 or fake_sha("weights"),
                size_bytes=manifest.inputs[0].size_bytes or 1024,
            )
        ],
        output_digests=[
            ArtifactDigest(
                name="out",
                uri=manifest.outputs[0].uri,
                sha256=payload_sha,
                size_bytes=bytes_written,
            )
        ],
        started_unix=float(started),
        finished_unix=float(finished),
        compute_sec=float(max(0.05, finished - started)),
        claimed_bytes_read=bytes_read,
        claimed_bytes_written=bytes_written,
    )
    key = paths.receipt_key(NETUID, manifest.run_id, worker.hotkey, manifest.job_id, manifest.attempt)
    put_json(s3, bucket, key, receipt.to_dict())
    return receipt


def write_verdict(
    s3,
    bucket: str,
    *,
    manifest: JobManifestV3,
    receipt: JobReceiptV3,
    worker: DemoWorker,
    status: str,
    reason: str,
    now: float,
) -> None:
    verdict = VerificationVerdictV3(
        verdict_id=f"{VALIDATOR_HOTKEY}:{receipt.receipt_id}",
        receipt_id=receipt.receipt_id,
        manifest_hash=manifest.manifest_hash(),
        job_id=manifest.job_id,
        run_id=manifest.run_id,
        miner_hotkey=worker.hotkey,
        validator_hotkey=VALIDATOR_HOTKEY,
        status=status,
        reason=reason,
        estimated_cu=float(random.uniform(8.0, 20.0)),
        replay_compute_sec=float(receipt.compute_sec * random.uniform(0.8, 1.1)),
        checked_unix=float(now),
        comparison={"sample_count": 1024, "max_abs_err": 0.0 if status == "pass" else 0.42},
    )
    key = paths.verdict_key(NETUID, manifest.run_id, VALIDATOR_HOTKEY, receipt.receipt_id)
    put_json(s3, bucket, key, verdict.to_dict())


def feed_wave(s3, bucket: str, run_id: str, count: int) -> list[dict]:
    now = int(time.time())
    session = uuid.uuid4().hex
    ensure_run_skeleton(s3, bucket, run_id)
    refresh_heartbeats(s3, bucket, run_id, now, session)

    wave_tag = now
    rows: list[dict] = []
    new_ids: list[str] = []

    for k in range(count):
        status = STATUS_RECIPE[k % len(STATUS_RECIPE)]
        worker = WORKERS[k % len(WORKERS)]
        step_id = 100 + k
        job_id = f"wave-{wave_tag}-step{k:02d}-{status}"
        if status == "stale":
            created = now - 1200
            deadline_offset = -600
        else:
            created = now - random.randint(2, 30)
            deadline_offset = 600
        manifest = build_manifest(
            bucket=bucket,
            run_id=run_id,
            job_id=job_id,
            step_id=step_id,
            worker=worker,
            now=created,
            deadline_offset=deadline_offset,
            critical=status == "verified" and k % 2 == 0,
        )
        write_manifest(s3, bucket, manifest)
        new_ids.append(job_id)

        if status in {"outputs_written", "completed", "verified", "failed"}:
            payload = (f"demo-output-{job_id}".encode() * 64)[: 1024 + (k * 53) % 4096]
            write_output(s3, bucket, manifest, payload)
            payload_sha = hashlib.sha256(payload).hexdigest()
            bytes_written = len(payload)
            if status in {"completed", "verified", "failed"}:
                started = created + 1
                finished = created + random.randint(2, 9)
                receipt = write_receipt(
                    s3,
                    bucket,
                    manifest=manifest,
                    worker=worker,
                    started=started,
                    finished=finished,
                    bytes_read=1024,
                    bytes_written=bytes_written,
                    payload_sha=payload_sha,
                )
                if status == "verified":
                    write_verdict(
                        s3,
                        bucket,
                        manifest=manifest,
                        receipt=receipt,
                        worker=worker,
                        status="pass",
                        reason="demo replay matched",
                        now=finished + 1,
                    )
                elif status == "failed":
                    write_verdict(
                        s3,
                        bucket,
                        manifest=manifest,
                        receipt=receipt,
                        worker=worker,
                        status="fail",
                        reason="demo replay mismatch",
                        now=finished + 1,
                    )

        rows.append({"job_id": job_id, "status": status, "worker": worker.worker_id})

    append_index(s3, bucket, run_id, new_ids)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=os.environ.get("VIZ_DEMO_RUN_ID", "viz-demo-1778865107"))
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    parser.add_argument("--batch", type=int, default=10, help="Number of jobs in this wave.")
    parser.add_argument("--waves", type=int, default=1, help="Number of waves to push (sleeps 3s between).")
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("S3_BUCKET env or --bucket required")

    s3 = make_bucket()
    print(f"feeding viz demo  bucket={args.bucket}  run={args.run_id}  batch={args.batch}  waves={args.waves}")
    total = 0
    for w in range(args.waves):
        rows = feed_wave(s3, args.bucket, args.run_id, args.batch)
        total += len(rows)
        print(f"[wave {w + 1}/{args.waves}] pushed {len(rows)} jobs")
        for row in rows:
            print(f"  + {row['job_id']:60s} status={row['status']:16s} worker={row['worker']}")
        if w + 1 < args.waves:
            time.sleep(3)
    print(f"done. total jobs added: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
