"""Standalone heartbeat advertiser for the visualizer demo.

Run on a Lium pod (or any GPU box). Publishes one Teuton v3 heartbeat per GPU
to the configured S3 bucket, refreshing every `INTERVAL` seconds.

Required env vars:
    S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    TEUTON_RUN_ID                run id to advertise on (e.g. viz-demo-...)
    TEUTON_HOTKEY_BASE           hotkey prefix (e.g. lium-noble-hawk-a3)

Optional:
    S3_REGION (default us-east-1), S3_ENDPOINT_URL, TEUTON_NETUID (default 0),
    TEUTON_HEARTBEAT_INTERVAL_SEC (default 5), TEUTON_HOST_LABEL (default hostname),
    TEUTON_GPU_TYPE (overrides nvidia-smi name), TEUTON_GPU_COUNT (overrides smi count)
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid

import boto3


def gpu_rows() -> list[tuple[int, str, int]]:
    forced_type = os.environ.get("TEUTON_GPU_TYPE")
    forced_count = int(os.environ.get("TEUTON_GPU_COUNT") or 0)
    if forced_type and forced_count:
        return [(i, forced_type, 24576) for i in range(forced_count)]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return [(0, forced_type or "unknown-gpu", 0)]
    rows: list[tuple[int, str, int]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            name = parts[1].replace("NVIDIA ", "").replace("GeForce ", "").strip()
            rows.append((int(parts[0]), name, int(float(parts[2]))))
    return rows or [(0, forced_type or "unknown-gpu", 0)]


def main() -> None:
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("S3_REGION") or "us-east-1"
    endpoint = os.environ.get("S3_ENDPOINT_URL") or None
    netuid = int(os.environ.get("TEUTON_NETUID", "0"))
    run_id = os.environ["TEUTON_RUN_ID"]
    base = os.environ["TEUTON_HOTKEY_BASE"]
    interval = int(os.environ.get("TEUTON_HEARTBEAT_INTERVAL_SEC", "5"))
    host = os.environ.get("TEUTON_HOST_LABEL") or socket.gethostname()
    s3 = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    nonce = uuid.uuid4().hex
    rows = gpu_rows()
    print(f"[viz-demo-heartbeat] run={run_id} host={host} gpus={[r[1] for r in rows]} interval={interval}s")
    while True:
        now = int(time.time())
        for idx, name, vram in rows:
            hotkey = f"{base}-gpu{idx}"
            worker_id = f"{base}-gpu{idx}"
            capabilities = {
                "device": f"cuda:{idx}",
                "worker_id": worker_id,
                "gpu_available": True,
                "n_gpus": len(rows),
                "device_group": [f"cuda:{idx}"],
                "gpu_indices": [idx],
                "world_size": 1,
                "placement": "single_host",
                "multi_gpu": False,
                "gpu_class": name,
                "gpu_name": name,
                "vram_mb": vram,
                "hostname": host,
                "transport": "lium-pod-heartbeat",
                "role": "train",
            }
            worker = {
                "hotkey_ss58": hotkey,
                "worker_id": worker_id,
                "host_id": host,
                "gpu_index": idx,
                "session_nonce": nonce,
                "software_hash": "viz-demo-heartbeat",
                "device_group": [idx],
                "capabilities": capabilities,
            }
            miner = {
                "netuid": netuid,
                "hotkey_ss58": hotkey,
                "uid": None,
                "endpoint": None,
                "commitment_hash": None,
                "capabilities": dict(capabilities),
            }
            body = {
                "miner": miner,
                "worker": worker,
                "run_id": run_id,
                "role": "train",
                "last_seen_unix": now,
            }
            key = f"v3/netuid={netuid}/miners/{hotkey}/workers/{worker_id}/heartbeat.json"
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
                ContentType="application/json",
            )
        time.sleep(interval)


if __name__ == "__main__":
    main()
