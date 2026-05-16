"""Orchestrator poll loop.

Per tick the orchestrator:

  1. Refreshes the worker registry by listing manifest/workers/.
  2. Builds the pin table.
  3. If `jobs/round=R/index.json` is absent, builds the initial DAG and
     writes it.
  4. For each UB, lists landed deltas; emits reduce when quorum is met.
  5. Advances `state.json` once next-round weights all exist.
  6. Tail-logs `metrics/round=R.json`.

The orchestrator is the only writer of `state.json`, the jobs index, and the
reduce-emitted markers. Workers never write any of those.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import config, paths
from .ir import Graph
from .schedule import (
    TaskGraphs,
    build_eval_job,
    build_initial_round_jobs,
    build_outer_job,
    build_reduce_job,
    jid_eval,
    jid_outer,
    jid_reduce,
    round_completion_uris,
)
from .storage import LocalBucket, parse_uri
from .types import JobManifest, RunState, WorkerInfo


log = logging.getLogger(__name__)


_GPU_RANK = {
    "h200": 0, "h100": 1, "a100": 2, "rtx4090": 3, "4090": 3,
    "rtx3090": 4, "3090": 4, "rtxa6000": 5, "cpu": 9,
}


def _worker_cost(worker_id: str, infos: "list[WorkerInfo]") -> float:
    """Lower = better. Combines GPU class + S3 RTT.

    Used to order pin lists so the strongest worker for each UB is picked
    first (which matters because `_round_robin` uses index 0 for the inner
    replica, and the highest-indexed pinned worker tends to be the outer step
    target via `(ub + 7919) % len(pinned)`).
    """
    info = next((w for w in infos if w.worker_id == worker_id), None)
    if info is None:
        return 1e9
    cap = info.capabilities or {}
    gpu = str(cap.get("gpu_class", "cpu")).lower().replace(" ", "")
    rank = _GPU_RANK.get(gpu, 6)
    rtt = float(cap.get("rtt_to_bucket_ms", 1000.0))
    return rank * 10000.0 + rtt


@dataclass
class OrchestratorParams:
    n_unique_blocks: int
    inner_replicas_per_ub: int
    common_params: dict[str, Any]
    inner_params: dict[str, Any]
    outer_params: dict[str, Any]
    eval_params: dict[str, Any]
    reduce_params: dict[str, Any]
    max_rounds: int
    m_target: int | None = None
    m_min: int | None = None
    t_max_sec: float | None = None
    # Step 2: number of concurrent microbatches per round. Each microbatch
    # is an independent forward pass with a different data seed; their
    # gradients average in reduce. Multiplies inner_step job count by M.
    n_microbatches: int = 1
    # Optional task-driven extras passed to schedule.build_initial_round_jobs.
    forward_extra_outputs: list[str] = field(default_factory=list)
    inner_extra_outputs: list[str] = field(default_factory=list)
    outer_extra_state: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        *,
        bucket: LocalBucket,
        run_id: str,
        graphs: TaskGraphs,
        params: OrchestratorParams,
        poll_interval: float | None = None,
        startup_grace_sec: float = 0.5,
    ) -> None:
        self.bucket = bucket
        self.run_id = run_id
        self.graphs = graphs
        self.params = params
        self.poll_interval = (
            poll_interval if poll_interval is not None else config.env_poll_interval()
        )
        # Wait at least this long after orchestrator startup before emitting
        # round 0, to give workers from heterogeneous boxes a chance to come
        # online and heartbeat. After this grace period, emit even if some
        # UBs lack a pinned worker (they'll get unpinned-worker fallback).
        self.startup_grace_sec = startup_grace_sec
        self._stop = threading.Event()
        self._round_started_at: dict[int, float] = {}
        self._started_at = time.time()

    def stop(self) -> None:
        self._stop.set()

    def loop(self) -> None:
        log.info("orchestrator starting (run=%s)", self.run_id)
        state = self._load_or_init_state()
        if state.max_rounds is None:
            state.max_rounds = self.params.max_rounds
            self._save_state(state)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("orchestrator tick failure")
            state = self._load_state()
            if state.max_rounds is not None and state.current_round >= state.max_rounds:
                log.info("reached max_rounds=%s; orchestrator exiting", state.max_rounds)
                break
            time.sleep(self.poll_interval)
        log.info("orchestrator stopped")

    def _load_or_init_state(self) -> RunState:
        uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        if self.bucket.exists(uri):
            return RunState.from_dict(self.bucket.get_json(uri))
        state = RunState(run_id=self.run_id, current_round=0)
        self._save_state(state)
        return state

    def _load_state(self) -> RunState:
        uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        return RunState.from_dict(self.bucket.get_json(uri))

    def _save_state(self, state: RunState) -> None:
        uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        self.bucket.put_json(uri, state.to_dict())

    def _list_workers(self) -> list[WorkerInfo]:
        prefix = self.bucket.uri_for_key(paths.workers_prefix(self.run_id))
        out: list[WorkerInfo] = []
        cutoff = time.time() - config.WORKER_STALE_SEC
        for u in self.bucket.list(prefix):
            if not u.endswith(".json"):
                continue
            try:
                d = self.bucket.get_json(u)
                wi = WorkerInfo.from_dict(d)
            except Exception:
                continue
            if wi.last_seen_unix < cutoff:
                continue
            out.append(wi)
        out.sort(key=lambda w: w.worker_id)
        return out

    def _build_pin_table(self, workers_info: list[WorkerInfo]) -> dict[int, list[str]]:
        """Build UB → [worker_id] pin table.

        Honours `WorkerInfo.capabilities["resident_ub"]` (int or list[int]).
        Critically: workers pinned to UB X are NEVER placed in UB Y's pin
        list, even as a fallback — the original implementation's fallback
        (which put pinned workers from other UBs into empty UBs as a "graceful
        degrade") caused work for UB Y to land on the wrong specialist and
        starve UB X. Instead, an empty UB falls back ONLY to unpinned workers;
        if there are no unpinned workers either, the UB is left empty and the
        emit-side caller is expected to delay emission until coverage exists.

        Within a UB the pin order favours capability hints (gpu_class) and
        then lower S3 RTT, so the first replica slot lands on the strongest
        worker.
        """
        nu = self.params.n_unique_blocks
        pin: dict[int, list[str]] = {ub: [] for ub in range(nu)}
        unpinned: list[WorkerInfo] = []
        for wi in workers_info:
            wanted = wi.capabilities.get("resident_ub") if wi.capabilities else None
            if wanted is None:
                unpinned.append(wi)
                continue
            wants = wanted if isinstance(wanted, (list, tuple)) else [wanted]
            attached = False
            for ub in wants:
                ub = int(ub)
                if 0 <= ub < nu:
                    pin[ub].append(wi.worker_id)
                    attached = True
            if not attached:
                unpinned.append(wi)
        for ub in range(nu):
            if not pin[ub] and unpinned:
                pin[ub] = [unpinned[(ub + j) % len(unpinned)].worker_id for j in range(len(unpinned))]
                continue
            pin[ub].sort(key=lambda wid: _worker_cost(wid, workers_info))
        return pin

    def _coverage_ok(self, pin: dict[int, list[str]]) -> bool:
        """True if every UB has enough workers to fill `inner_replicas_per_ub`
        distinct slots (so `_round_robin` doesn't wrap around 2 workers and
        stack 5 replicas onto them)."""
        need = max(1, int(self.params.inner_replicas_per_ub))
        return all(len(pin.get(ub, [])) >= need for ub in range(self.params.n_unique_blocks))

    def _pick_critical_worker(self, workers_info: list[WorkerInfo]) -> str:
        """Pick the worker that should run forward / outer / eval (fast GPU,
        low RTT, ideally the H200 box). Falls back to first sorted worker."""
        if not workers_info:
            raise ValueError("no workers")
        scored = sorted(
            workers_info,
            key=lambda w: _worker_cost(w.worker_id, [w]),
        )
        return scored[0].worker_id

    def _tick(self) -> None:
        state = self._load_state()
        if state.max_rounds is not None and state.current_round >= int(state.max_rounds):
            return
        round_id = int(state.current_round)
        workers_info = self._list_workers()
        worker_ids = [w.worker_id for w in workers_info]
        if not worker_ids:
            return

        pin = self._build_pin_table(workers_info)

        idx_uri = self.bucket.uri_for_key(paths.jobs_index_key(self.run_id, round_id))
        if not self.bucket.exists(idx_uri):
            # For the FIRST round of the run, always wait `startup_grace_sec`
            # so all workers come online before pin assignments lock in.
            # Without this, fast-RTT workers heartbeat first, the orchestrator
            # emits round 0 with a sparse pin, and slow-link workers sit idle
            # for the entire run because all jobs landed on the few that
            # were online at emit time. Subsequent rounds don't need the wait.
            if round_id == 0:
                elapsed = time.time() - self._started_at
                if elapsed < self.startup_grace_sec:
                    coverage = {ub: len(pin[ub]) for ub in range(self.params.n_unique_blocks)}
                    log.info(
                        "orch waiting startup_grace (pin=%s elapsed=%.1fs/%.1fs)",
                        coverage, elapsed, self.startup_grace_sec,
                    )
                    return
            self._emit_initial_round(round_id, workers_info, pin)
            self._round_started_at[round_id] = time.time()

        self._maybe_emit_reduces(round_id, worker_ids, pin)
        self._maybe_emit_eval(round_id, workers_info)

        if self._round_complete(round_id):
            state = self._load_state()
            if round_id == state.current_round:
                state.current_round = round_id + 1
                state.completed_rounds = list(state.completed_rounds) + [round_id]
                self._save_state(state)
                self._aggregate_round_telemetry(round_id)
                log.info("orchestrator advanced to round %s", state.current_round)

                # Step 1: speculative forward emission. As soon as round R+1
                # weights all exist (which is the moment we advance state),
                # emit round R+1 forward in the SAME tick — don't wait 0.2s
                # for the next poll cycle. Saves a round-trip per round.
                if state.max_rounds is None or state.current_round < int(state.max_rounds):
                    next_round = state.current_round
                    next_idx = self.bucket.uri_for_key(
                        paths.jobs_index_key(self.run_id, next_round)
                    )
                    if not self.bucket.exists(next_idx) and self._coverage_ok(pin):
                        self._emit_initial_round(next_round, workers_info, pin)
                        self._round_started_at[next_round] = time.time()

    def _emit_initial_round(
        self,
        round_id: int,
        workers_info: list[WorkerInfo],
        pin: dict[int, list[str]],
    ) -> None:
        worker_ids = [w.worker_id for w in workers_info]
        critical = self._pick_critical_worker(workers_info)
        forward_worker = critical
        eval_worker = critical
        # Forward pool: the M lowest-cost workers, so M concurrent forwards
        # spread across the H200 fleet.
        scored = sorted(workers_info, key=lambda w: _worker_cost(w.worker_id, [w]))
        forward_pool = [w.worker_id for w in scored[: max(self.params.n_microbatches, 1)]]
        jobs, graphs_to_write = build_initial_round_jobs(
            bucket=self.bucket.bucket,
            run_id=self.run_id,
            round_id=round_id,
            n_unique_blocks=self.params.n_unique_blocks,
            inner_replicas_per_ub=self.params.inner_replicas_per_ub,
            workers=worker_ids,
            pin_table=pin,
            forward_worker=forward_worker,
            eval_worker=eval_worker,
            graphs=self.graphs,
            common_params=self.params.common_params,
            inner_params=self.params.inner_params,
            outer_params=self.params.outer_params,
            eval_params=self.params.eval_params,
            forward_extra_outputs=self.params.forward_extra_outputs,
            inner_extra_outputs=self.params.inner_extra_outputs,
            outer_extra_state=self.params.outer_extra_state,
            n_microbatches=self.params.n_microbatches,
            forward_workers=forward_pool,
        )
        for sha, g in graphs_to_write.items():
            uri = self.bucket.uri_for_key(paths.graph_key(self.run_id, sha))
            if not self.bucket.exists(uri):
                self.bucket.put(uri, g.to_canonical_json())
        for j in jobs:
            self.bucket.put_json(
                self.bucket.uri_for_key(
                    paths.job_manifest_key(self.run_id, round_id, j.job_id)
                ),
                j.to_dict(),
            )
        index = [j.job_id for j in jobs]
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.jobs_index_key(self.run_id, round_id)),
            index,
        )
        log.info("orchestrator emitted round %s with %s jobs", round_id, len(jobs))

    def _maybe_emit_reduces(
        self,
        round_id: int,
        worker_ids: list[str],
        pin: dict[int, list[str]],
    ) -> None:
        # With M microbatches we have M*R deltas per UB instead of R.
        n_mb = max(1, self.params.n_microbatches)
        per_replica_target = self.params.m_target if self.params.m_target is not None else self.params.inner_replicas_per_ub
        per_replica_min = self.params.m_min if self.params.m_min is not None else max(
            1, (self.params.inner_replicas_per_ub + 1) // 2
        )
        m_target = per_replica_target * n_mb
        m_min = per_replica_min * n_mb
        t_max = self.params.t_max_sec if self.params.t_max_sec is not None else config.DEFAULT_T_MAX_SEC
        started = self._round_started_at.get(round_id, time.time())
        elapsed = time.time() - started
        for ub in range(self.params.n_unique_blocks):
            marker_uri = self.bucket.uri_for_key(
                paths.reduce_done_marker_key(self.run_id, round_id, ub)
            )
            if self.bucket.exists(marker_uri):
                continue
            prefix = self.bucket.uri_for_key(
                paths.output_delta_prefix(self.run_id, round_id, ub)
            )
            uris = [u for u in self.bucket.list(prefix) if u.endswith("/delta.bin")]
            count = len(uris)
            quorum = (count >= m_target) or (elapsed >= t_max and count >= m_min)
            if not quorum:
                continue
            reduce_graph = self.graphs.reduce_for_n(count)
            pinned = pin.get(ub) or list(worker_ids)
            # Co-locate the reduce on a worker that actually produced one of
            # the input deltas — saves N delta-sized GETs per round at scale.
            producers = [
                u.split("worker=")[1].split("/")[0]
                for u in uris if "worker=" in u
            ]
            producers_in_pin = [w for w in pinned if w in producers]
            if producers_in_pin:
                assignee = producers_in_pin[0]
            elif producers:
                assignee = producers[0]
            else:
                assignee = pinned[(round_id + ub) % len(pinned)]
            job, gmap = build_reduce_job(
                bucket=self.bucket.bucket,
                run_id=self.run_id,
                round_id=round_id,
                ub=ub,
                delta_uris=uris,
                reduce_graph=reduce_graph,
                assigned_to=assignee,
                common_params=self.params.common_params,
                reduce_params=self.params.reduce_params,
            )
            for sha, g in gmap.items():
                guri = self.bucket.uri_for_key(paths.graph_key(self.run_id, sha))
                if not self.bucket.exists(guri):
                    self.bucket.put(guri, g.to_canonical_json())
            self.bucket.put_json(
                self.bucket.uri_for_key(
                    paths.job_manifest_key(self.run_id, round_id, job.job_id)
                ),
                job.to_dict(),
            )
            self.bucket.put_json(marker_uri, {"emitted_unix": int(time.time()), "n": count, "assignee": assignee})
            log.info(
                "orch emitted reduce r=%s ub=%s assignee=%s deltas=%s",
                round_id, ub, assignee, count,
            )

            # Emit outer_step on the same worker — it has the reduced delta in
            # its local output cache, saving a GET. (The worker won't pick up
            # outer until reduce completes because outer's `reduced_delta`
            # input doesn't exist yet — the gating is automatic.)
            outer_graph = (
                self.graphs.outer_per_ub[ub]
                if self.graphs.outer_per_ub is not None
                else self.graphs.outer
            )
            outer_job, outer_gmap = build_outer_job(
                bucket=self.bucket.bucket,
                run_id=self.run_id,
                round_id=round_id,
                ub=ub,
                outer_graph=outer_graph,
                assigned_to=assignee,
                common_params=self.params.common_params,
                outer_params=self.params.outer_params,
                forward_extra_outputs=self.params.forward_extra_outputs,
                outer_extra_state=self.params.outer_extra_state,
            )
            for sha, g in outer_gmap.items():
                guri = self.bucket.uri_for_key(paths.graph_key(self.run_id, sha))
                if not self.bucket.exists(guri):
                    self.bucket.put(guri, g.to_canonical_json())
            self.bucket.put_json(
                self.bucket.uri_for_key(
                    paths.job_manifest_key(self.run_id, round_id, outer_job.job_id)
                ),
                outer_job.to_dict(),
            )

    def _maybe_emit_eval(self, round_id: int, workers_info: list[WorkerInfo]) -> None:
        """Emit eval for round R once weights/round=R+1/UB-* all exist.

        Eval is OFF the round's critical path. We deliberately do NOT assign
        it to the lowest-cost worker, because that worker is also the
        forward/outer specialist and gets immediately re-tasked by the
        speculative round R+1 forward emission. Instead we round-robin eval
        across the worker pool so different rounds' evals land on different
        workers.
        """
        if not workers_info:
            return
        marker_uri = self.bucket.uri_for_key(
            f"runs/{self.run_id}/eval_emitted/round={round_id}.json"
        )
        if self.bucket.exists(marker_uri):
            return
        # Wait for all next-round weights to exist.
        for ub in range(self.params.n_unique_blocks):
            wuri = self.bucket.uri_for_key(paths.weights_key(self.run_id, round_id + 1, ub))
            if not self.bucket.exists(wuri):
                return
        # Round-robin eval across workers, skipping the critical worker.
        critical = self._pick_critical_worker(workers_info)
        non_critical = [w for w in workers_info if w.worker_id != critical]
        pool = non_critical or workers_info
        eval_assignee = pool[round_id % len(pool)].worker_id
        eval_job, gmap = build_eval_job(
            bucket=self.bucket.bucket,
            run_id=self.run_id,
            round_id=round_id,
            n_unique_blocks=self.params.n_unique_blocks,
            eval_graph=self.graphs.eval,
            assigned_to=eval_assignee,
            common_params=self.params.common_params,
            eval_params=self.params.eval_params,
        )
        for sha, g in gmap.items():
            guri = self.bucket.uri_for_key(paths.graph_key(self.run_id, sha))
            if not self.bucket.exists(guri):
                self.bucket.put(guri, g.to_canonical_json())
        self.bucket.put_json(
            self.bucket.uri_for_key(
                paths.job_manifest_key(self.run_id, round_id, eval_job.job_id)
            ),
            eval_job.to_dict(),
        )
        self.bucket.put_json(marker_uri, {"emitted_unix": int(time.time()), "assignee": eval_assignee})
        log.info("orch emitted eval r=%s assignee=%s", round_id, eval_assignee)

    def _round_complete(self, round_id: int) -> bool:
        uris = round_completion_uris(
            bucket=self.bucket.bucket,
            run_id=self.run_id,
            round_id=round_id,
            n_unique_blocks=self.params.n_unique_blocks,
        )
        return all(self.bucket.exists(u) for u in uris)

    def _aggregate_round_telemetry(self, round_id: int) -> None:
        """Count bytes / files / per-worker activity for a completed round.

        Also reads per-job timing stamps (written by worker._execute) to
        compute the headline utilization metric:
            utilization = pool_busy_seconds / (wallclock_seconds * n_workers)
        which is "what fraction of pool capacity actually did useful work?"
        """
        try:
            run_prefix = self.bucket.uri_for_key(f"runs/{self.run_id}/")
            all_objs = self.bucket.list(run_prefix)
            tag = f"round={round_id}"
            relevant = [u for u in all_objs if tag in u]
            total_bytes = 0
            kinds: dict[str, dict[str, int]] = {}
            workers_bytes: dict[str, int] = {}
            for u in relevant:
                h = self.bucket.head(u)
                size = int(h["size_bytes"]) if h else 0
                total_bytes += size
                rest = u.split(f"runs/{self.run_id}/", 1)[1]
                kind = rest.split("/", 1)[0]
                slot = kinds.setdefault(kind, {"n": 0, "bytes": 0})
                slot["n"] += 1
                slot["bytes"] += size
                if "worker=" in u:
                    wid = u.split("worker=")[1].split("/")[0]
                    workers_bytes[wid] = workers_bytes.get(wid, 0) + size
            elapsed = (
                time.time() - self._round_started_at[round_id]
                if round_id in self._round_started_at else None
            )

            # Read per-job timing stamps for this round.
            stamps_prefix = self.bucket.uri_for_key(
                f"runs/{self.run_id}/jobtimes/round={round_id}/"
            )
            jobtimes: list[dict[str, Any]] = []
            for u in self.bucket.list(stamps_prefix):
                if not u.endswith(".json"):
                    continue
                try:
                    jobtimes.append(self.bucket.get_json(u))
                except Exception:
                    continue
            pool_busy_sec = sum(float(j.get("total_sec", 0)) for j in jobtimes)
            pool_compute_sec = sum(float(j.get("compute_sec", 0)) for j in jobtimes)
            pool_io_sec = sum(float(j.get("fetch_inputs_sec", 0)) for j in jobtimes)
            workers_total = self._n_pool_workers()
            wallclock = elapsed or 1.0
            utilization = pool_busy_sec / (wallclock * max(workers_total, 1))
            compute_share = pool_compute_sec / max(pool_busy_sec, 1e-9)

            # Per-worker per-job table (for heatmap).
            per_worker: dict[str, list[dict[str, Any]]] = {}
            for j in jobtimes:
                per_worker.setdefault(j.get("worker_id", "?"), []).append({
                    "kind": j.get("kind"),
                    "start": j.get("start_unix"),
                    "end": j.get("end_unix"),
                    "compute": j.get("compute_sec"),
                })
            for wid in per_worker:
                per_worker[wid].sort(key=lambda r: r.get("start") or 0)

            tele = {
                "round": round_id,
                "n_objects": len(relevant),
                "total_bytes": total_bytes,
                "wallclock_sec": elapsed,
                "by_kind": kinds,
                "by_worker_bytes": workers_bytes,
                "active_workers": len(workers_bytes),
                "n_jobs": len(jobtimes),
                "pool_size": workers_total,
                "pool_busy_sec": round(pool_busy_sec, 3),
                "pool_compute_sec": round(pool_compute_sec, 3),
                "pool_io_sec": round(pool_io_sec, 3),
                "utilization": round(utilization, 4),
                "compute_share_of_busy": round(compute_share, 4),
                "per_worker_jobs": per_worker,
            }
            uri = self.bucket.uri_for_key(
                f"runs/{self.run_id}/telemetry/round={round_id}.json"
            )
            self.bucket.put_json(uri, tele)
            log.info(
                "telemetry r=%s: wall=%.1fs busy=%.1fs util=%.1f%% (compute=%.1fs of busy) "
                "%.2f MB %d jobs %d/%d workers",
                round_id, elapsed or 0.0, pool_busy_sec, utilization * 100.0,
                pool_compute_sec, total_bytes / 1e6, len(jobtimes),
                len(per_worker), workers_total,
            )
        except Exception:
            log.exception("telemetry aggregate failed for r=%s", round_id)

    def _n_pool_workers(self) -> int:
        """Best-effort current pool size = number of fresh worker heartbeats."""
        try:
            workers = self._list_workers()
            return len(workers)
        except Exception:
            return 0
