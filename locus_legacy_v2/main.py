"""CLI + in-process launcher.

Subcommands:
    bootstrap      — initialize a run (state.json, config.json, initial weights)
    orchestrator   — run the orchestrator loop
    worker         — run a worker loop

The bootstrap command accepts a `--task` selector that picks the graph
builder + initial-weights helper from `locus/tasks/<task>.py`.

`run_in_process` is a programmatic launcher used by tests; it spawns the
orchestrator and N workers in threads in this process and waits for max_rounds
to complete (plus a grace drain for trailing eval jobs).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import threading
import time
from typing import Any, Callable

from . import config, paths
from .orchestrator import Orchestrator, OrchestratorParams
from .schedule import TaskGraphs
from .storage import LocalBucket, open_bucket
from .types import RunState
from .worker import Worker


# --------------------------------------------------------------------------- #
# Task registry
# --------------------------------------------------------------------------- #


_KNOWN_TASKS = {
    "mlp",
    "relu_mlp",
    "locoprop_mlp",
    "adam_mlp",
    "sign_loco_mlp",
    "data_parallel_mlp",
    "tiny_gpt",
    "pluralis_subspace",
    "pluralis_lossless_wire",
    "pluralis_asymmetric",
    "pluralis_int8",
    "pluralis_demo",
    "pluralis_tied",
    "pluralis_grouped",
    "pluralis_full",
    "pluralis_grassmann",
    "pluralis_gpt_10M",
    "pluralis_gpt_10M_v3",
    "pipe_demo",
    "pipe_train",
    "gpt_pipe",
}


def _load_task(name: str):
    if name not in _KNOWN_TASKS:
        raise ValueError(f"unknown task: {name!r}; known: {sorted(_KNOWN_TASKS)}")
    return importlib.import_module(f".tasks.{name}", package="locus_legacy_v2")


# --------------------------------------------------------------------------- #
# CLI subcommands
# --------------------------------------------------------------------------- #


def cmd_bootstrap(args: argparse.Namespace) -> int:
    bucket = open_bucket(args.local_root, args.bucket)
    bucket.ensure_bucket()
    task = _load_task(args.task)
    task.bootstrap(
        bucket=bucket,
        run_id=args.run_id,
        max_rounds=args.max_rounds,
    )
    print(f"bootstrap complete: bucket={args.bucket} run_id={args.run_id}")
    return 0


def cmd_orchestrator(args: argparse.Namespace) -> int:
    bucket = open_bucket(args.local_root, args.bucket)
    task = _load_task(args.task)
    graphs, params = task.build_orchestrator_inputs(bucket=bucket, run_id=args.run_id)
    o = Orchestrator(bucket=bucket, run_id=args.run_id, graphs=graphs, params=params)
    o.loop()
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    bucket = open_bucket(args.local_root, args.bucket)
    w = Worker(
        bucket=bucket,
        run_id=args.run_id,
        worker_id=args.worker_id,
        capabilities={"resident_ub": args.resident_ub} if args.resident_ub is not None else None,
        max_idle_iters=args.max_idle_iters,
        device=args.device,
        fault_mode=args.fault_mode,
        fault_rate=args.fault_rate,
    )
    w.loop()
    return 0


def cmd_validator(args: argparse.Namespace) -> int:
    from .validator import ReplayValidator
    bucket = open_bucket(args.local_root, args.bucket)
    v = ReplayValidator(
        bucket=bucket,
        run_id=args.run_id,
        validator_id=args.validator_id,
        device=args.device,
        sample_rate=args.sample_rate,
        poll_interval=args.poll_interval,
        max_sample_elements=args.max_sample_elements,
    )
    n = v.loop(max_jobs=args.max_jobs, timeout_sec=args.timeout_sec)
    print(f"validator {args.validator_id}: checked {n} receipts")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from .validator import summarize_ledger
    bucket = open_bucket(args.local_root, args.bucket)
    summary = summarize_ledger(bucket, run_id=args.run_id)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# In-process launcher (for tests)
# --------------------------------------------------------------------------- #


def run_in_process(
    *,
    bucket: Any,
    run_id: str,
    task_name: str,
    n_workers: int,
    max_rounds: int,
    timeout_sec: float = 60.0,
    drain_sec: float = 10.0,
    poll_interval: float = 0.05,
    heartbeat_interval: float = 0.1,
    state_poll_interval: float | None = None,
) -> RunState:
    """Bootstrap (if needed), run orchestrator + N workers in threads, wait
    until `max_rounds` are complete or `timeout_sec` elapses, drain pending
    eval jobs, then stop and return the final `RunState`.

    `poll_interval` / `heartbeat_interval` are forwarded to Orchestrator and
    each Worker. Defaults are tuned for fast local-fs runs; bump them for
    high-latency backends (e.g. S3) to ~1.0 / ~2.0 to avoid hammering the
    bucket with HEAD/LIST calls.

    `state_poll_interval` is how often the launcher itself polls state.json
    to detect round advance / drain progress. Defaults to `poll_interval`.
    """
    task = _load_task(task_name)
    state_uri = bucket.uri_for_key(paths.state_key(run_id))
    if not bucket.exists(state_uri):
        task.bootstrap(bucket=bucket, run_id=run_id, max_rounds=max_rounds)

    graphs, params = task.build_orchestrator_inputs(bucket=bucket, run_id=run_id)
    params.max_rounds = max_rounds
    sleep_for = state_poll_interval if state_poll_interval is not None else poll_interval

    orch = Orchestrator(
        bucket=bucket, run_id=run_id, graphs=graphs, params=params,
        poll_interval=poll_interval,
    )
    workers = [
        Worker(
            bucket=bucket,
            run_id=run_id,
            worker_id=f"W{i}",
            poll_interval=poll_interval,
            heartbeat_interval=heartbeat_interval,
        )
        for i in range(n_workers)
    ]
    threads: list[threading.Thread] = []
    threads.append(threading.Thread(target=orch.loop, name="orchestrator", daemon=True))
    for w in workers:
        threads.append(threading.Thread(target=w.loop, name=f"worker-{w.worker_id}", daemon=True))
    for t in threads:
        t.start()

    deadline = time.time() + timeout_sec
    final_state = RunState(run_id=run_id)
    while time.time() < deadline:
        try:
            final_state = RunState.from_dict(bucket.get_json(state_uri))
        except Exception:
            time.sleep(sleep_for)
            continue
        if final_state.current_round >= max_rounds:
            break
        time.sleep(sleep_for)

    drain_deadline = time.time() + drain_sec
    while time.time() < drain_deadline:
        all_done = True
        for r in range(max_rounds):
            metrics_uri = bucket.uri_for_key(paths.metrics_key(run_id, r))
            if not bucket.exists(metrics_uri):
                all_done = False
                break
        if all_done:
            break
        time.sleep(sleep_for)

    orch.stop()
    for w in workers:
        w.stop()
    for t in threads:
        t.join(timeout=5.0)

    return final_state


# --------------------------------------------------------------------------- #
# argparse plumbing
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locus")
    p.add_argument("--local-root", default=config.DEFAULT_LOCAL_ROOT)
    p.add_argument("--bucket", default=config.DEFAULT_BUCKET)
    p.add_argument("--run-id", default=config.DEFAULT_RUN_ID)
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap")
    b.add_argument("--task", required=True)
    b.add_argument("--max-rounds", type=int, default=5)

    o = sub.add_parser("orchestrator")
    o.add_argument("--task", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--worker-id", required=True)
    w.add_argument("--resident-ub", type=int, default=None)
    w.add_argument("--max-idle-iters", type=int, default=None)
    w.add_argument("--device", type=str, default="cpu")
    w.add_argument("--fault-mode", type=str, default="")
    w.add_argument("--fault-rate", type=float, default=None)

    v = sub.add_parser("validator")
    v.add_argument("--validator-id", required=True)
    v.add_argument("--device", type=str, default="cpu")
    v.add_argument("--sample-rate", type=float, default=1.0)
    v.add_argument("--poll-interval", type=float, default=1.0)
    v.add_argument("--max-jobs", type=int, default=None)
    v.add_argument("--timeout-sec", type=float, default=30.0)
    v.add_argument("--max-sample-elements", type=int, default=4096)

    l = sub.add_parser("ledger")
    l.set_defaults(fn=cmd_ledger)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    fn: Callable[[argparse.Namespace], int] = {
        "bootstrap": cmd_bootstrap,
        "orchestrator": cmd_orchestrator,
        "worker": cmd_worker,
        "validator": cmd_validator,
        "ledger": cmd_ledger,
    }[args.cmd]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
