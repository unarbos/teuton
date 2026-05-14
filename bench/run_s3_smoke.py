"""Smoke test: run the Stage-0 `mlp` task against the real S3 bucket.

Usage:
    cd Locus && source .venv/bin/activate
    python -m bench.run_s3_smoke

Reads bucket name + AWS creds from `../.env`. Runs the orchestrator + 4
workers in-process (threads) for 5 rounds against the bucket, prints the
loss trajectory, then wipes the run's prefix.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _load_env() -> None:
    """Load the .env file living at the project root (one level above Locus)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv not installed; relying on inherited env")
        return
    # Locus/.env or ../.env
    here = Path(__file__).resolve().parent.parent           # Locus/
    candidates = [here / ".env", here.parent / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            return
    print(f"WARNING: no .env found in {candidates}", file=sys.stderr)


def main() -> int:
    _load_env()

    from locus_legacy_v2.storage import S3Bucket
    from locus_legacy_v2.main import run_in_process
    from locus import paths

    bucket_name = os.environ.get("S3_BUCKET")
    region = os.environ.get("S3_REGION", "us-east-1")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("S3_ENDPOINT") or None

    if not bucket_name:
        print("ERROR: S3_BUCKET not set in environment", file=sys.stderr)
        return 1
    if not access_key or not secret_key:
        print("ERROR: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set", file=sys.stderr)
        return 1

    bucket = S3Bucket(
        bucket=bucket_name,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        endpoint_url=endpoint if (endpoint and "amazonaws.com" not in endpoint) else None,
    )

    # Sanity check: round-trip a tiny probe blob to confirm credentials.
    probe_uri = bucket.uri_for_key(f"_smoke_probe/probe_{int(time.time())}.txt")
    print(f"[probe] writing -> {probe_uri}")
    bucket.put(probe_uri, b"locus-smoke-test")
    body = bucket.get(probe_uri)
    assert body == b"locus-smoke-test", "probe round-trip failed"
    bucket.delete(probe_uri)
    print("[probe] credentials verified")

    run_id = f"smoke-{int(time.time())}"
    n_workers = 4
    max_rounds = 5
    print(f"\n[run] s3://{bucket_name}/runs/{run_id}/  region={region}")
    print(f"[run] task=mlp workers={n_workers} rounds={max_rounds}")

    t0 = time.time()
    final_state = run_in_process(
        bucket=bucket,
        run_id=run_id,
        task_name="mlp",
        n_workers=n_workers,
        max_rounds=max_rounds,
        timeout_sec=600.0,
        drain_sec=20.0,
        poll_interval=1.0,
        heartbeat_interval=2.0,
        state_poll_interval=0.5,
    )
    elapsed = time.time() - t0
    print(
        f"\n[run] final round: {final_state.current_round}/{max_rounds}  "
        f"elapsed: {elapsed:.1f}s ({elapsed / max(final_state.current_round, 1):.1f}s/round)"
    )

    # Loss trajectory
    print("\n[metrics] val_loss trajectory:")
    losses = []
    for r in range(max_rounds):
        uri = bucket.uri_for_key(paths.metrics_key(run_id, r))
        if bucket.exists(uri):
            v = bucket.get_json(uri)["value"][0]
            losses.append(float(v))
            print(f"  round {r}: {v:.4f}")
        else:
            print(f"  round {r}: MISSING")

    # Final-round weights
    print("\n[weights] final-round byte sizes:")
    from locus_legacy_v2.tasks.mlp import N_UB
    for ub in range(N_UB):
        uri = bucket.uri_for_key(paths.weights_key(run_id, max_rounds, ub))
        h = bucket.head(uri)
        if h is not None:
            print(f"  weights/round={max_rounds}/UB-{ub}.bin  {h['size_bytes']} bytes")
        else:
            print(f"  weights/round={max_rounds}/UB-{ub}.bin  MISSING")

    # Cleanup
    print(f"\n[cleanup] wiping runs/{run_id}/...")
    deleted = bucket.wipe_run(run_id)
    print(f"[cleanup] deleted {deleted} objects")
    remaining = bucket.list(bucket.uri_for_key(f"runs/{run_id}/"))
    print(f"[cleanup] remaining under runs/{run_id}/: {len(remaining)}")

    # Exit code
    ok = (
        final_state.current_round == max_rounds
        and len(losses) == max_rounds
        and losses[-1] < losses[0]
        and len(remaining) == 0
    )
    print(f"\n[result] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
