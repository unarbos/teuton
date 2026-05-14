"""Distributed-pool driver — same orchestrator/worker classes as locus.main,
but the bucket is `S3Bucket` constructed from /root/.env.

Usage on master node:
    python -m bench.legacy_dist bootstrap --task mlp --run-id RUN --max-rounds 5
    python -m bench.legacy_dist orchestrator --task mlp --run-id RUN

Usage on each worker box (one process per GPU):
    python -m bench.legacy_dist worker --run-id RUN --worker-id <unique>

Optionally tail metrics:
    python -m bench.legacy_dist tail --run-id RUN --max-rounds 5
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path


def _load_env() -> None:
    from dotenv import load_dotenv
    for p in [Path("/root/.env"), Path("/root/Locus/.env"), Path("/root/Locus/.env"),
              Path(__file__).resolve().parent.parent.parent / ".env"]:
        if p.exists():
            load_dotenv(p, override=True)
            return
    print("WARNING: no .env found", file=sys.stderr)


def _build_bucket():
    from locus_legacy_v2.storage import S3Bucket
    return S3Bucket(
        bucket=os.environ["S3_BUCKET"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _load_task(name: str):
    from locus_legacy_v2.main import _KNOWN_TASKS
    if name not in _KNOWN_TASKS:
        raise SystemExit(f"unknown task: {name!r}; known: {sorted(_KNOWN_TASKS)}")
    return importlib.import_module(f"locus.tasks.{name}")


def cmd_bootstrap(args: argparse.Namespace) -> int:
    bucket = _build_bucket()
    task = _load_task(args.task)
    task.bootstrap(bucket=bucket, run_id=args.run_id, max_rounds=args.max_rounds)
    print(f"[bootstrap] bucket={bucket.bucket} run_id={args.run_id} task={args.task} rounds={args.max_rounds}")
    return 0


def cmd_orchestrator(args: argparse.Namespace) -> int:
    from locus_legacy_v2.orchestrator import Orchestrator
    bucket = _build_bucket()
    task = _load_task(args.task)
    graphs, params = task.build_orchestrator_inputs(bucket=bucket, run_id=args.run_id)
    o = Orchestrator(
        bucket=bucket, run_id=args.run_id, graphs=graphs, params=params,
        poll_interval=args.poll_interval,
        startup_grace_sec=args.startup_grace_sec,
    )
    print(f"[orch] starting bucket={bucket.bucket} run_id={args.run_id} "
          f"max_rounds={params.max_rounds} grace={args.startup_grace_sec}s", flush=True)
    o.loop()
    return 0


def cmd_streaming_orchestrator(args: argparse.Namespace) -> int:
    """Run the streaming-pipeline orchestrator instead of the sync-round one."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    from locus_legacy_v2.streaming import StreamingOrchestrator
    bucket = _build_bucket()
    task = _load_task(args.task)
    if not hasattr(task, "build_streaming_inputs"):
        raise SystemExit(f"task {args.task!r} has no build_streaming_inputs() — "
                         f"not a streaming task")
    stages, params = task.build_streaming_inputs(bucket=bucket, run_id=args.run_id)
    o = StreamingOrchestrator(
        bucket=bucket, run_id=args.run_id, stages=stages, params=params,
        poll_interval=args.poll_interval,
        startup_grace_sec=args.startup_grace_sec,
        resume_from_epoch=args.resume_from_epoch,
    )
    print(f"[stream-orch] starting run_id={args.run_id} stages={params.n_stages} "
          f"mb={params.n_microbatches} epochs={params.max_epochs}", flush=True)
    o.loop()
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from locus_legacy_v2.worker import Worker
    bucket = _build_bucket()
    caps: dict = {}
    if args.resident_ub:
        ubs = [int(x) for x in args.resident_ub.split(",") if x.strip() != ""]
        caps["resident_ub"] = ubs[0] if len(ubs) == 1 else ubs
    if args.pipe_stage:
        stages = [int(x) for x in args.pipe_stage.split(",") if x.strip() != ""]
        caps["pipe_stage"] = stages[0] if len(stages) == 1 else stages
    if args.gpu_class:
        caps["gpu_class"] = args.gpu_class
    w = Worker(
        bucket=bucket,
        run_id=args.run_id,
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
        max_idle_iters=args.max_idle_iters,
        capabilities=caps or None,
        device=args.device,
        fault_mode=args.fault_mode,
        fault_rate=args.fault_rate,
    )
    print(
        f"[worker {args.worker_id}] starting run_id={args.run_id} "
        f"caps={w.capabilities} device={args.device}",
        flush=True,
    )
    w.loop()
    return 0


def cmd_validator(args: argparse.Namespace) -> int:
    from locus_legacy_v2.validator import ReplayValidator
    bucket = _build_bucket()
    v = ReplayValidator(
        bucket=bucket,
        run_id=args.run_id,
        validator_id=args.validator_id,
        device=args.device,
        sample_rate=args.sample_rate,
        poll_interval=args.poll_interval,
        max_sample_elements=args.max_sample_elements,
    )
    print(
        f"[validator {args.validator_id}] starting run_id={args.run_id} "
        f"sample_rate={args.sample_rate} device={args.device}",
        flush=True,
    )
    n = v.loop(max_jobs=args.max_jobs, timeout_sec=args.timeout_sec)
    print(f"[validator {args.validator_id}] checked={n}", flush=True)
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from locus_legacy_v2.validator import summarize_ledger
    bucket = _build_bucket()
    summary = summarize_ledger(bucket, run_id=args.run_id)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    """Poll state.json and metrics, print as they become available."""
    from locus import paths
    bucket = _build_bucket()
    state_uri = bucket.uri_for_key(paths.state_key(args.run_id))
    seen_metrics = set()
    seen_round = -1
    deadline = time.time() + args.timeout_sec
    print(f"[tail] watching {bucket.bucket}/runs/{args.run_id}/", flush=True)
    while time.time() < deadline:
        try:
            st = bucket.get_json(state_uri)
        except Exception:
            time.sleep(1.0)
            continue
        cur = int(st.get("current_round", 0))
        if cur != seen_round:
            print(f"[tail] state.current_round = {cur}/{st.get('max_rounds')}", flush=True)
            seen_round = cur
        for r in range(args.max_rounds):
            if r in seen_metrics:
                continue
            mu = bucket.uri_for_key(paths.metrics_key(args.run_id, r))
            if bucket.exists(mu):
                try:
                    v = bucket.get_json(mu)["value"][0]
                    print(f"[tail] round {r}: val_loss = {float(v):.6f}", flush=True)
                except Exception as e:
                    print(f"[tail] round {r}: <error reading metrics: {e}>", flush=True)
                seen_metrics.add(r)
        if len(seen_metrics) >= args.max_rounds:
            print(f"[tail] all {args.max_rounds} rounds reported; done", flush=True)
            return 0
        time.sleep(args.poll_interval)
    print(f"[tail] timeout after {args.timeout_sec}s; metrics seen: {sorted(seen_metrics)}", flush=True)
    return 2


def cmd_wipe(args: argparse.Namespace) -> int:
    bucket = _build_bucket()
    n = bucket.wipe_run(args.run_id)
    print(f"[wipe] deleted {n} objects under runs/{args.run_id}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()

    p = argparse.ArgumentParser(prog="dist")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap")
    b.add_argument("--task", required=True)
    b.add_argument("--run-id", required=True)
    b.add_argument("--max-rounds", type=int, default=5)
    b.set_defaults(fn=cmd_bootstrap)

    o = sub.add_parser("orchestrator")
    o.add_argument("--task", required=True)
    o.add_argument("--run-id", required=True)
    o.add_argument("--poll-interval", type=float, default=0.2)
    o.add_argument("--startup-grace-sec", type=float, default=60.0,
                   help="Seconds to wait after orch startup before emitting round 0, "
                        "so all workers can heartbeat and pin assignments lock in correctly.")
    o.set_defaults(fn=cmd_orchestrator)

    so = sub.add_parser("streaming-orchestrator")
    so.add_argument("--task", required=True)
    so.add_argument("--run-id", required=True)
    so.add_argument("--poll-interval", type=float, default=0.2)
    so.add_argument("--startup-grace-sec", type=float, default=60.0)
    so.add_argument("--resume-from-epoch", type=int, default=None,
                    help="If set, skip bootstrap and start emission from this epoch. "
                         "Requires weights/epoch=E/stage_K_W.bin to already exist.")
    so.set_defaults(fn=cmd_streaming_orchestrator)

    w = sub.add_parser("worker")
    w.add_argument("--run-id", required=True)
    w.add_argument("--worker-id", required=True)
    w.add_argument("--poll-interval", type=float, default=0.5)
    w.add_argument("--heartbeat-interval", type=float, default=2.0)
    w.add_argument("--max-idle-iters", type=int, default=None)
    w.add_argument("--resident-ub", type=str, default="",
                   help="UB id this worker is pinned to (comma-separated for multiple)")
    w.add_argument("--pipe-stage", type=str, default="",
                   help="Pipeline stage id for streaming mode (comma-separated for multiple)")
    w.add_argument("--gpu-class", type=str, default="",
                   help="Override GPU class string (default: auto-detected)")
    w.add_argument("--device", type=str, default="cpu",
                   help="Torch device for compute: 'cpu' or 'cuda' or 'cuda:0' etc.")
    w.add_argument("--fault-mode", type=str, default="",
                   choices=["", "none", "wrong_output", "skip_compute", "stale_output", "partial_corrupt", "timing_lie"],
                   help="Experiment-only adversarial mode for validator tests")
    w.add_argument("--fault-rate", type=float, default=None,
                   help="Probability a worker applies its adversarial fault mode")
    w.set_defaults(fn=cmd_worker)

    v = sub.add_parser("validator")
    v.add_argument("--run-id", default="all",
                   help="Run id to validate, or 'all' to scan all run receipts")
    v.add_argument("--validator-id", required=True)
    v.add_argument("--device", type=str, default="cpu")
    v.add_argument("--sample-rate", type=float, default=1.0)
    v.add_argument("--poll-interval", type=float, default=2.0)
    v.add_argument("--max-jobs", type=int, default=None)
    v.add_argument("--timeout-sec", type=float, default=None)
    v.add_argument("--max-sample-elements", type=int, default=4096)
    v.set_defaults(fn=cmd_validator)

    lg = sub.add_parser("ledger")
    lg.add_argument("--run-id", default="all")
    lg.set_defaults(fn=cmd_ledger)

    t = sub.add_parser("tail")
    t.add_argument("--run-id", required=True)
    t.add_argument("--max-rounds", type=int, default=5)
    t.add_argument("--poll-interval", type=float, default=2.0)
    t.add_argument("--timeout-sec", type=float, default=600.0)
    t.set_defaults(fn=cmd_tail)

    wp = sub.add_parser("wipe")
    wp.add_argument("--run-id", required=True)
    wp.set_defaults(fn=cmd_wipe)

    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
