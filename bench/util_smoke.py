"""5-minute utilization smoke driver.

Purpose: tight feedback loop for utilization-first iteration. Run this on the
master to bootstrap a fresh run, wait until N_ROUNDS rounds complete, read
the per-round telemetry, and print a single utilization number plus the
breakdown.

Usage on master node:
    python -m bench.util_smoke --task pluralis_gpt_10M --n-rounds 3 \
        --label "phase2 baseline"

If --launch-pool is set, the script also calls bench/pool.sh start to spawn
workers (assumes pool.sh is on PATH and bootstrap was done elsewhere).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _load_env() -> None:
    from dotenv import load_dotenv
    for p in [Path("/root/.env"), Path("/root/Locus/.env"),
              Path(__file__).resolve().parent.parent.parent / ".env"]:
        if p.exists():
            load_dotenv(p, override=True)
            return


def _build_bucket():
    from locus_legacy_v2.storage import S3Bucket
    return S3Bucket(
        bucket=os.environ["S3_BUCKET"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def main() -> int:
    _load_env()

    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--n-rounds", type=int, default=3,
                   help="How many completed rounds to wait for (excluding warmup round 0)")
    p.add_argument("--include-warmup", action="store_true",
                   help="Include round 0 in stats (otherwise excluded as warmup)")
    p.add_argument("--timeout-sec", type=float, default=600.0)
    p.add_argument("--label", type=str, default="",
                   help="Free-form label printed in the report for diff comparisons")
    args = p.parse_args()

    from locus import paths
    bucket = _build_bucket()

    state_uri = bucket.uri_for_key(paths.state_key(args.run_id))
    deadline = time.time() + args.timeout_sec
    target_round = args.n_rounds + (0 if args.include_warmup else 1)

    print(f"[util-smoke] watching {bucket.bucket}/runs/{args.run_id}/  "
          f"(want {args.n_rounds} non-warmup rounds, label={args.label!r})")
    last_round = -1
    while time.time() < deadline:
        try:
            st = bucket.get_json(state_uri)
        except Exception:
            time.sleep(2.0)
            continue
        cur = int(st.get("current_round", 0))
        if cur != last_round:
            print(f"[util-smoke] state.current_round = {cur}/{target_round}")
            last_round = cur
        if cur >= target_round:
            break
        time.sleep(2.0)
    if last_round < target_round:
        print(f"[util-smoke] TIMEOUT after {args.timeout_sec}s "
              f"(only reached round {last_round}/{target_round})")
        return 2

    rows = []
    skip = 0 if args.include_warmup else 1
    for r in range(skip, target_round):
        uri = bucket.uri_for_key(f"runs/{args.run_id}/telemetry/round={r}.json")
        if not bucket.exists(uri):
            continue
        try:
            rows.append(bucket.get_json(uri))
        except Exception:
            continue
    if not rows:
        print("[util-smoke] no telemetry rows found")
        return 3

    # Aggregate across the included rounds.
    total_wall = sum(float(r.get("wallclock_sec") or 0) for r in rows)
    total_busy = sum(float(r.get("pool_busy_sec") or 0) for r in rows)
    total_compute = sum(float(r.get("pool_compute_sec") or 0) for r in rows)
    total_io = sum(float(r.get("pool_io_sec") or 0) for r in rows)
    total_bytes = sum(int(r.get("total_bytes") or 0) for r in rows)
    total_jobs = sum(int(r.get("n_jobs") or 0) for r in rows)
    pool_size = max((int(r.get("pool_size") or 0) for r in rows), default=0)
    util = (total_busy / (total_wall * pool_size)) if (total_wall and pool_size) else 0.0
    compute_share = total_compute / max(total_busy, 1e-9)
    io_share = total_io / max(total_busy, 1e-9)

    losses = []
    for r_idx in range(skip, target_round):
        muri = bucket.uri_for_key(paths.metrics_key(args.run_id, r_idx))
        if bucket.exists(muri):
            try:
                losses.append(float(bucket.get_json(muri)["value"][0]))
            except Exception:
                pass

    print()
    print(f"=== util-smoke report  label={args.label!r}  rounds={skip}..{target_round-1} ===")
    print(f"  pool_size            : {pool_size}")
    print(f"  total wallclock      : {total_wall:.1f}s")
    print(f"  total pool_busy      : {total_busy:.1f}s")
    print(f"  total pool_compute   : {total_compute:.1f}s")
    print(f"  total pool_io_input  : {total_io:.1f}s")
    print(f"  total bytes          : {total_bytes/1e6:.2f} MB")
    print(f"  total n_jobs         : {total_jobs}")
    print()
    print(f"  UTILIZATION          : {util*100:.1f}%   (busy / (wall * pool))")
    print(f"  compute share of busy: {compute_share*100:.1f}%")
    print(f"  io share of busy     : {io_share*100:.1f}%")
    print(f"  bytes/sec            : {total_bytes/max(total_wall, 1e-9)/1e6:.2f} MB/s")
    print(f"  jobs/sec             : {total_jobs/max(total_wall, 1e-9):.2f}")
    if losses:
        print(f"  loss[first]/[last]   : {losses[0]:.4f} / {losses[-1]:.4f}  "
              f"(d={losses[-1]-losses[0]:+.4f})")

    out = {
        "label": args.label,
        "run_id": args.run_id,
        "rounds": list(range(skip, target_round)),
        "pool_size": pool_size,
        "wallclock_sec": total_wall,
        "pool_busy_sec": total_busy,
        "pool_compute_sec": total_compute,
        "pool_io_sec": total_io,
        "total_bytes": total_bytes,
        "n_jobs": total_jobs,
        "utilization": util,
        "compute_share_of_busy": compute_share,
        "losses": losses,
    }
    print()
    print("=== machine-readable ===")
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
