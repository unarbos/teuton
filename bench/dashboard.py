"""Live telemetry dashboard for a distributed run.

Polls the bucket and prints, every K seconds, a per-round table of:

  round | wallclock | total_bytes | active_workers | bytes_per_worker
        | val_loss

plus an inventory line of which workers are live (heartbeats < stale).

Usage:
    python -m bench.dashboard --run-id RUN [--max-rounds N] [--refresh 5.0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _load_env() -> None:
    from dotenv import load_dotenv
    for p in [Path("/root/.env"), Path("/root/Teuton/.env"),
              Path(__file__).resolve().parent.parent.parent / ".env"]:
        if p.exists():
            load_dotenv(p, override=True)
            return


def _build_bucket():
    from teuton_legacy_v2.storage import S3Bucket
    return S3Bucket(
        bucket=os.environ["S3_BUCKET"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _format_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n/1024/1024:.2f}MB"
    return f"{n/1024/1024/1024:.2f}GB"


def main() -> int:
    _load_env()

    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-rounds", type=int, default=100)
    p.add_argument("--refresh", type=float, default=5.0)
    p.add_argument("--once", action="store_true",
                   help="Print the table once and exit (instead of polling).")
    args = p.parse_args()

    from teuton import paths
    from teuton_legacy_v2.types import WorkerInfo
    from teuton_legacy_v2.config import WORKER_STALE_SEC

    bucket = _build_bucket()

    while True:
        rows: list[dict] = []
        for r in range(args.max_rounds):
            tele_uri = bucket.uri_for_key(
                f"runs/{args.run_id}/telemetry/round={r}.json"
            )
            metrics_uri = bucket.uri_for_key(paths.metrics_key(args.run_id, r))
            row: dict = {"round": r}
            if bucket.exists(tele_uri):
                try:
                    row.update(bucket.get_json(tele_uri))
                except Exception:
                    pass
            if bucket.exists(metrics_uri):
                try:
                    row["val_loss"] = float(bucket.get_json(metrics_uri)["value"][0])
                except Exception:
                    pass
            if "total_bytes" in row or "val_loss" in row:
                rows.append(row)

        os.system("clear" if sys.platform != "win32" else "cls")
        print(f"[dashboard] run={args.run_id}  bucket={bucket.bucket}  refresh={args.refresh}s")
        print()
        print(f"{'round':>5} {'wall':>6} {'busy':>7} {'util%':>7} {'cmp%':>6} "
              f"{'jobs':>5} {'wk':>4} {'bytes':>9} {'val_loss':>9}")
        print("-" * 76)
        if not rows:
            print("  (no telemetry yet)")
        for row in rows[-30:]:
            wc = row.get("wallclock_sec")
            wc_s = f"{wc:.1f}s" if wc is not None else "  -  "
            tb = row.get("total_bytes", 0)
            tb_s = _format_bytes(tb) if tb else "    -"
            nw = row.get("active_workers", 0)
            ps = row.get("pool_size") or nw
            busy = row.get("pool_busy_sec")
            busy_s = f"{busy:.1f}s" if busy is not None else "  -  "
            util = row.get("utilization")
            util_s = f"{util*100:.1f}" if util is not None else "  -"
            cmp_share = row.get("compute_share_of_busy")
            cmp_s = f"{cmp_share*100:.0f}" if cmp_share is not None else " -"
            n_jobs = row.get("n_jobs", 0)
            vl = row.get("val_loss")
            vl_s = f"{vl:.4f}" if vl is not None else "      -"
            print(f"{row['round']:>5} {wc_s:>6} {busy_s:>7} {util_s:>7} {cmp_s:>6} "
                  f"{n_jobs:>5} {nw}/{ps:<2} {tb_s:>9} {vl_s:>9}")

        if rows:
            last = rows[-1]
            by_kind = last.get("by_kind") or {}
            print()
            print(f"--- last round (r={last['round']}) by kind ---")
            for k, slot in sorted(by_kind.items()):
                print(f"  {k:<14}  n={slot['n']:>3}  {_format_bytes(slot['bytes']):>9}")

            # Per-worker heatmap: one row per worker, ASCII bar of busy time
            # within the round's wallclock.
            per_worker = last.get("per_worker_jobs") or {}
            wall = float(last.get("wallclock_sec") or 1.0)
            t0 = None
            for jobs in per_worker.values():
                for j in jobs:
                    s = j.get("start")
                    if s is not None:
                        t0 = s if t0 is None else min(t0, s)
            if per_worker and t0 is not None:
                width = 50
                print()
                print(f"--- last round (r={last['round']}) per-worker activity ({wall:.1f}s wallclock) ---")
                # Sort workers by total busy desc
                def _wk_busy(jobs):
                    return sum((j.get("end", 0) - j.get("start", 0)) for j in jobs)
                items = sorted(per_worker.items(), key=lambda kv: (-_wk_busy(kv[1]), kv[0]))
                for wid, jobs in items:
                    bar = [" "] * width
                    for j in jobs:
                        s = max(0.0, (j.get("start", t0) - t0) / wall) * width
                        e = max(s + 0.5, (j.get("end", t0) - t0) / wall * width)
                        kind = (j.get("kind") or "?")[0].upper()
                        for px in range(int(s), min(width, int(e) + 1)):
                            bar[px] = kind
                    busy = _wk_busy(jobs)
                    print(f"  {wid:<10} |{''.join(bar)}| {busy:.1f}s")
                print(f"  legend: F=forward I=inner_step R=reduce O=outer_step E=eval")

        # Live workers: who has heartbeated recently?
        try:
            workers_prefix = bucket.uri_for_key(paths.workers_prefix(args.run_id))
            cutoff = time.time() - WORKER_STALE_SEC
            live: list[WorkerInfo] = []
            for u in bucket.list(workers_prefix):
                if not u.endswith(".json"):
                    continue
                try:
                    wi = WorkerInfo.from_dict(bucket.get_json(u))
                    if wi.last_seen_unix >= cutoff:
                        live.append(wi)
                except Exception:
                    continue
            print()
            print(f"--- live workers ({len(live)}) ---")
            for wi in sorted(live, key=lambda w: w.worker_id):
                cap = wi.capabilities or {}
                gpu = cap.get("gpu_class", "?")
                rtt = cap.get("rtt_to_bucket_ms", "?")
                pin = cap.get("resident_ub", "?")
                print(f"  {wi.worker_id:<10}  gpu={gpu:<10}  rtt={rtt}ms  pin_ub={pin}")
        except Exception as e:
            print(f"[live workers] error: {e}")

        if args.once:
            return 0
        try:
            time.sleep(args.refresh)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
