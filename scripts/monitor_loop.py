"""Continuous dev-box monitor for the Teuton fleet.

Every `--interval` seconds, sample the bucket + Bittensor metagraph and:
  1. Write a summary JSON to v3/netuid=N/telemetry/<run_id>/monitor/<unix>.json
  2. Append a one-line summary to `--log` (default /tmp/teuton_monitor.log).

Run as:
    nohup python -m scripts.monitor_loop --netuid 3 > /tmp/teuton_monitor_stdout.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from teuton_core.telemetry import TelemetryWriter
from teuton_runtime.storage import S3Bucket


def make_bucket() -> S3Bucket:
    return S3Bucket(
        bucket=os.environ["S3_BUCKET"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
    )


def count_prefix(s3, bucket: str, prefix: str) -> int:
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for _ in page.get("Contents", []) or []:
            n += 1
    return n


def newest_age(s3, bucket: str, prefix: str) -> int | None:
    now = time.time()
    newest = 0.0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []) or []:
            ts = o["LastModified"].timestamp()
            if ts > newest:
                newest = ts
    if newest == 0.0:
        return None
    return int(now - newest)


def heartbeat_counts(s3, bucket: str, netuid: int, run_id: str) -> tuple[int, int]:
    now = time.time()
    miners = 0
    auditors = 0
    for role, prefix in (("m", f"v3/netuid={netuid}/miners/"), ("a", f"v3/netuid={netuid}/auditors/")):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []) or []:
                if not o["Key"].endswith("/heartbeat.json"):
                    continue
                if now - o["LastModified"].timestamp() > 120:
                    continue
                try:
                    body = json.loads(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read())
                    if body.get("run_id") == run_id:
                        if role == "m":
                            miners += 1
                        else:
                            auditors += 1
                except Exception:
                    pass
    return miners, auditors


def chain_state(netuid: int) -> dict:
    try:
        import bittensor as bt  # type: ignore

        st = bt.Subtensor(network="finney")
        block = int(st.get_current_block())
        mg = st.metagraph(netuid, lite=False)
        val_uid = 204
        last_upd = int(mg.last_update[val_uid])
        import numpy as np

        W = np.array(mg.W[val_uid])
        ours = {31, 36, 37, 80, 91, 101, 158, 216, 240, 249}
        return {
            "block": block,
            "validator_uid": val_uid,
            "validator_last_update": last_upd,
            "blocks_since_update": block - last_upd,
            "tempo_blocks": 360,
            "weights_non_zero_total": int((W > 0).sum()),
            "weights_non_zero_ours": sum(1 for u in ours if float(W[u]) > 0),
        }
    except Exception as e:
        return {"error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    ap.add_argument("--run-id", default=None, help="defaults to /tmp/teuton_sn3_run_id contents")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--log", default="/tmp/teuton_monitor.log")
    ap.add_argument("--chain-every", type=int, default=5, help="poll chain only every N iters")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("monitor")

    run_id = args.run_id or Path("/tmp/teuton_sn3_run_id").read_text().strip()
    if not run_id:
        print("RUN_ID unset", file=sys.stderr)
        return 2

    bucket = make_bucket()
    s3 = bucket._client  # type: ignore[attr-defined]
    bkt = bucket.bucket  # type: ignore[attr-defined]
    tw = TelemetryWriter(bucket=bucket, netuid=args.netuid, run_id=run_id, component="monitor")
    log.info("monitor netuid=%s run_id=%s interval=%ss", args.netuid, run_id, args.interval)

    chain_cache: dict | None = None
    chain_counter = 0
    log_fh = open(args.log, "a")

    while True:
        t0 = time.time()
        try:
            counts = {
                "miners": count_prefix(s3, bkt, f"v3/netuid={args.netuid}/miners/"),
                "auditors": count_prefix(s3, bkt, f"v3/netuid={args.netuid}/auditors/"),
                "job_manifests": count_prefix(s3, bkt, f"v3/netuid={args.netuid}/jobs/{run_id}/"),
                "receipts": count_prefix(s3, bkt, f"v3/netuid={args.netuid}/receipts/{run_id}/"),
                "verdicts": count_prefix(s3, bkt, f"v3/netuid={args.netuid}/verdicts/{run_id}/"),
                "audit_results": count_prefix(
                    s3, bkt, f"v3/netuid={args.netuid}/audits/{run_id}/results/"
                ),
                "streaming_outputs": count_prefix(s3, bkt, f"runs/{run_id}/streaming/"),
                "weight_blobs": count_prefix(s3, bkt, f"runs/{run_id}/weights/"),
            }
            recent_miner, recent_auditor = heartbeat_counts(s3, bkt, args.netuid, run_id)
            ages = {
                "last_manifest_age_sec": newest_age(s3, bkt, f"v3/netuid={args.netuid}/jobs/{run_id}/"),
                "last_receipt_age_sec": newest_age(s3, bkt, f"v3/netuid={args.netuid}/receipts/{run_id}/"),
                "last_verdict_age_sec": newest_age(s3, bkt, f"v3/netuid={args.netuid}/verdicts/{run_id}/"),
                "last_audit_result_age_sec": newest_age(
                    s3, bkt, f"v3/netuid={args.netuid}/audits/{run_id}/results/"
                ),
            }
            if chain_cache is None or (chain_counter % args.chain_every == 0):
                chain_cache = chain_state(args.netuid)
            chain_counter += 1
            payload = {
                "counts": counts,
                "ages": ages,
                "recent_heartbeats": {"miners_120s": recent_miner, "auditors_120s": recent_auditor},
                "chain": chain_cache,
                "sample_wall_seconds": round(time.time() - t0, 3),
            }
            tw.monitor(payload)

            line = (
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"miners_120s={recent_miner} auditors_120s={recent_auditor} "
                f"manifests={counts['job_manifests']} receipts={counts['receipts']} "
                f"verdicts={counts['verdicts']} audits={counts['audit_results']} "
                f"last_receipt_age={ages['last_receipt_age_sec']} "
                f"chain_delta={(chain_cache or {}).get('blocks_since_update')}"
            )
            log.info(line)
            log_fh.write(line + "\n")
            log_fh.flush()
        except Exception as e:
            log.exception("monitor iteration failed: %r", e)

        elapsed = time.time() - t0
        time.sleep(max(0.0, args.interval - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
