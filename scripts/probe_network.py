"""Print a one-line snapshot of churn on the live run, intended to be run in a
loop while validating the network. Usage:

  watch -n 30 'python scripts/probe_network.py'

or for a single shot:

  python scripts/probe_network.py
"""
from __future__ import annotations

import os
import sys
import time

import boto3


def count(p, bucket: str, prefix: str, now: float):
    total = 0
    last_1m = 0
    last_5m = 0
    latest = 0.0
    for page in p.paginate(Bucket=bucket, Prefix=prefix):
        for obj in (page.get("Contents") or []):
            total += 1
            ts = obj["LastModified"].timestamp()
            age = now - ts
            if age < 60:
                last_1m += 1
            if age < 300:
                last_5m += 1
            if ts > latest:
                latest = ts
    age = now - latest if latest else float("inf")
    return total, last_1m, last_5m, age


def main() -> int:
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("S3_REGION", "us-east-1")
    rid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    rid = rid_arg or open("/tmp/teuton_sn3_run_id").read().strip()
    s3 = boto3.client("s3", region_name=region)
    p = s3.get_paginator("list_objects_v2")
    now = time.time()

    parts = []
    for label, prefix in (
        ("manifests", f"v3/netuid=3/jobs/{rid}/"),
        ("assignments", f"v3/netuid=3/assignments/{rid}/"),
        ("receipts", f"v3/netuid=3/receipts/{rid}/"),
        ("verdicts", f"v3/netuid=3/verdicts/{rid}/"),
    ):
        total, m1, m5, age = count(p, bucket, prefix, now)
        parts.append(f"{label}={total} ({m1}/1m, {m5}/5m, latest={age:.0f}s)")

    fresh_miners = 0
    for page in p.paginate(Bucket=bucket, Prefix="v3/netuid=3/miners/"):
        for obj in (page.get("Contents") or []):
            if not obj["Key"].endswith("/heartbeat.json"):
                continue
            if now - obj["LastModified"].timestamp() < 60:
                fresh_miners += 1
    parts.append(f"miners_fresh_60s={fresh_miners}")
    print(time.strftime("[%H:%M:%S] ") + " | ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
