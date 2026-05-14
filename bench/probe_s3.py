"""Per-host S3 probe.

Loads /root/.env, probes the bucket with a small write+read+delete round-trip,
and prints elapsed times. Used during pool setup to confirm each rented box
can actually reach the shared bucket and that credentials work.
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path


def main() -> int:
    from dotenv import load_dotenv

    for p in [Path("/root/.env"), Path("/root/Locus/.env"), Path("/root/Locus/.env")]:
        if p.exists():
            load_dotenv(p, override=True)
            break

    from locus_legacy_v2.storage import S3Bucket

    bucket_name = os.environ["S3_BUCKET"]
    region = os.environ.get("S3_REGION", "us-east-1")
    bucket = S3Bucket(
        bucket=bucket_name,
        region=region,
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    host = socket.gethostname()
    payload = f"probe from {host} at {int(time.time())}".encode()
    key = f"_probe/{host}_{int(time.time())}.txt"
    uri = bucket.uri_for_key(key)

    t0 = time.time()
    bucket.put(uri, payload)
    t_put = time.time() - t0

    t0 = time.time()
    body = bucket.get(uri)
    t_get = time.time() - t0

    t0 = time.time()
    bucket.delete(uri)
    t_del = time.time() - t0

    assert body == payload, "round-trip mismatch"
    print(
        f"OK host={host} bucket={bucket_name} region={region} "
        f"put={t_put*1000:.0f}ms get={t_get*1000:.0f}ms del={t_del*1000:.0f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
