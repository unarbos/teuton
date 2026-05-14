"""Streaming-pipeline orchestrator (alongside the sync-round Orchestrator).

The synchronous round model is fundamentally limited by phase-gates:
forward → inner → reduce → outer → eval. Even with M microbatches in flight,
each phase has to wait for the previous one to finish before any worker
can move on. This caps utilization in the 5-10% range (measured Steps 0-2).

The streaming model breaks the round abstraction. Workers are organized as
STAGES (each holds one layer of the model). A microbatch flows through
the stages forward-then-backward. Many microbatches are in flight
simultaneously; in steady state, every stage is processing a different
microbatch at any moment.

Concrete protocol:

  - Orchestrator emits, at startup, all M * S * 2 jobs (M microbatches × S
    stages × {forward, backward}). Each job's input depends on the previous
    stage/direction's output URI. Workers can claim a job only when its
    inputs already exist on S3.

  - Workers run the existing _maybe_pickup loop with no changes — they pull
    any job assigned to them whose inputs are ready.

  - Steady state: stage K worker, while computing forward for mb=M, also has
    forward for mb=M+1 sitting in its input queue (stage K-1 just finished).

  - Outer step (parameter update) happens every M microbatches: orchestrator
    emits an outer_step job per stage that reads accumulated gradients.

For Step 3 we test utilization on a forward-only "ping pipeline" task. Once
that's measured, full forward+backward+outer is straightforward.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import config, paths
from .ir import Graph
from .schedule import TaskGraphs
from .storage import LocalBucket
from .types import GraphRef, IORef, JobManifest, RunState, WorkerInfo


log = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Configuration of one pipeline stage.

    For pure inference (forward-only): set forward_graph + forward_input_specs
    + forward_output_specs.

    For training: also set backward_graph + outer_graph + per-epoch weight
    URIs handled by the orchestrator.
    """
    stage_id: int
    forward_graph: Graph
    backward_graph: Graph | None = None
    outer_graph: Graph | None = None
    forward_input_specs: list[tuple[str, list[int], str]] = field(default_factory=list)
    forward_output_specs: list[tuple[str, list[int], str]] = field(default_factory=list)
    # If set, this stage's W is loaded as job input from the per-epoch path.
    weights_input_name: str | None = None
    weights_shape: list[int] = field(default_factory=list)
    weights_dtype: str = "float32"
    # Backward-job specs (for training).
    backward_takes_loss_target: bool = False     # tail stage reads `target` static
    backward_emits_dx_in: bool = True            # non-head stages emit dL_dx_in


@dataclass
class StreamingParams:
    n_stages: int
    n_microbatches: int
    max_epochs: int                  # 1 epoch = M microbatches all complete
    common_params: dict[str, Any] = field(default_factory=dict)
    forward_params: dict[str, Any] = field(default_factory=dict)
    deadline_seconds: int = 1200
    # Training mode toggles
    training: bool = False
    target_static_uri: str | None = None        # full URI of the loss target
    lr: float = 0.01
    # Generalized static inputs: name -> (uri, stage_filter)
    # stage_filter is one of: "tail", "head", "head_and_tail", "all".
    # Used to inject e.g. the tokens corpus into both head and tail stages
    # (head reads input ids, tail reads target ids in cross-entropy bwd).
    static_inputs: dict[str, tuple[str, str]] = field(default_factory=dict)
    # tokens-per-microbatch (B * T). Used by cost telemetry to compute
    # tokens/sec and $/B-tokens. Defaults to 0 which disables cost reporting.
    tokens_per_mb: int = 0
    # gpu_class -> dollars/hour map for the fleet, used by cost telemetry.
    # Workers report their gpu_class capability; we look it up here.
    # Default empty: cost telemetry disabled.
    gpu_class_price_per_hour: dict[str, float] = field(default_factory=dict)


class StreamingOrchestrator:
    """Continuous-pipeline orchestrator.

    Differs from the sync-round Orchestrator in three key ways:
      1. No "rounds". One run = N epochs of M microbatches each.
      2. Jobs emitted in bulk at start of each epoch with chained input URIs.
      3. State.json tracks (epoch, mb_completed_count) instead of round_id.
    """

    def __init__(
        self,
        *,
        bucket: LocalBucket,
        run_id: str,
        stages: list[PipelineStage],
        params: StreamingParams,
        poll_interval: float = 0.2,
        startup_grace_sec: float = 0.5,
        retry_after_sec: float = 60.0,
        max_retries: int = 2,
        max_retries_per_tick: int = 8,
        resume_from_epoch: int | None = None,
    ) -> None:
        self.bucket = bucket
        self.run_id = run_id
        self.stages = stages
        self.params = params
        self.poll_interval = poll_interval
        self.startup_grace_sec = startup_grace_sec
        # If set, start emission from this epoch instead of 0. Bootstrap is
        # skipped; we expect `weights/epoch=E/stage_K_W.bin` for all stages
        # to already exist in the bucket (from an earlier crashed run).
        self.resume_from_epoch = resume_from_epoch
        # Stale-job retry knobs:
        # - retry_after_sec: don't retry until the epoch has been running this
        #   long (gives the original assignee a chance to complete).
        # - max_retries: per-job hard cap on retry attempts.
        # - max_retries_per_tick: avoid stampedes by capping retries/tick.
        self.retry_after_sec = retry_after_sec
        self.max_retries = max_retries
        self.max_retries_per_tick = max_retries_per_tick
        self._stop = threading.Event()
        self._epoch_started_at: dict[int, float] = {}
        self._started_at = time.time()
        # Dead-worker set: workers that have failed max_retries on at least
        # one job. They're excluded from future pin-table builds. Persisted
        # to S3 so they stay evicted across orch restarts.
        self._evicted_workers: set[str] = set()
        # worker_id -> hostname map, populated by _list_workers_by_stage. Used
        # by Phase 3.2 co-located stage assignment.
        self._worker_hostnames: dict[str, str] = {}

    def stop(self) -> None:
        self._stop.set()

    def loop(self) -> None:
        log.info("streaming orch starting (run=%s, stages=%d, mb=%d, epochs=%d)",
                 self.run_id, self.params.n_stages, self.params.n_microbatches,
                 self.params.max_epochs)
        self._load_evicted_workers()
        if self._evicted_workers:
            log.info("streaming orch loaded %d previously-evicted workers: %s",
                     len(self._evicted_workers), sorted(self._evicted_workers))
        if self.resume_from_epoch is not None:
            self._apply_resume()
        state = self._load_or_init_state()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("streaming orch tick failure")
            state = self._load_state()
            if state.current_round >= self.params.max_epochs:
                log.info("streaming orch reached max_epochs=%s; exiting",
                         self.params.max_epochs)
                break
            time.sleep(self.poll_interval)
        log.info("streaming orch stopped")

    def _load_or_init_state(self) -> RunState:
        uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        if self.bucket.exists(uri):
            return RunState.from_dict(self.bucket.get_json(uri))
        state = RunState(run_id=self.run_id, current_round=0,
                          max_rounds=self.params.max_epochs)
        self._save_state(state)
        return state

    def _apply_resume(self) -> None:
        """Apply --resume-from-epoch=E. Verifies that the per-stage weights
        for epoch E exist in the bucket, then writes state.json so the loop
        starts emission from epoch E. Re-emission for epoch E is idempotent
        (we skip if jobs_index.json already exists)."""
        e = int(self.resume_from_epoch or 0)
        if e <= 0:
            log.info("resume_from_epoch=%s is non-positive; treating as fresh start", e)
            return
        # Verify all stage weights present.
        missing = []
        if self.params.training:
            for stage in self.stages:
                if stage.outer_graph is None:
                    continue
                w = self._epoch_weights_uri(e, stage.stage_id)
                if not self.bucket.exists(w):
                    missing.append(w)
        if missing:
            raise RuntimeError(
                f"resume requested at epoch={e} but missing weight files: {missing[:3]}"
                + (f" (and {len(missing) - 3} more)" if len(missing) > 3 else "")
            )
        state = RunState(
            run_id=self.run_id,
            current_round=e,
            max_rounds=self.params.max_epochs,
            completed_rounds=list(range(e)),
        )
        self._save_state(state)
        log.info("streaming orch resuming from epoch=%s", e)

    def _load_state(self) -> RunState:
        uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        return RunState.from_dict(self.bucket.get_json(uri))

    def _save_state(self, state: RunState) -> None:
        uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        self.bucket.put_json(uri, state.to_dict())

    def _list_workers_by_stage(self) -> dict[int, list[str]]:
        """Group live workers by their `stage_id` capability.
        Workers pinned to multiple stages appear in each.

        Sorts each stage's worker list by RTT-to-bucket (ascending) and then
        worker_id (alphabetical tiebreak). This means lower-RTT workers
        (typically US-based) get the lowest job indices in each stage and
        therefore are picked FIRST for assignments. The orchestrator's
        round-robin in `_emit_epoch_jobs` (mb % len(pinned)) then spreads
        load — but the lowest-index workers (lowest RTT) take the lion's
        share of any uneven distribution, which is what we want for
        S3-bound jobs.

        Excludes evicted workers (those that failed max_retries on prior jobs).

        Side effect: also populates `self._worker_hostnames` for use by
        co-location-aware assignment.
        """
        prefix = self.bucket.uri_for_key(paths.workers_prefix(self.run_id))
        cutoff = time.time() - config.WORKER_STALE_SEC
        by_stage_with_rtt: dict[int, list[tuple[float, str]]] = {
            s: [] for s in range(self.params.n_stages)
        }
        hostnames: dict[str, str] = {}
        for u in self.bucket.list(prefix):
            if not u.endswith(".json"):
                continue
            try:
                wi = WorkerInfo.from_dict(self.bucket.get_json(u))
            except Exception:
                continue
            if wi.last_seen_unix < cutoff:
                continue
            if wi.worker_id in self._evicted_workers:
                continue
            stages = wi.capabilities.get("pipe_stage") if wi.capabilities else None
            if stages is None:
                continue
            stages = stages if isinstance(stages, (list, tuple)) else [stages]
            rtt = float(wi.capabilities.get("rtt_to_bucket_ms", 1e9))
            host = str(wi.capabilities.get("hostname", "unknown"))
            hostnames[wi.worker_id] = host
            for s in stages:
                s = int(s)
                if 0 <= s < self.params.n_stages:
                    by_stage_with_rtt[s].append((rtt, wi.worker_id))
        by_stage: dict[int, list[str]] = {}
        for s, lst in by_stage_with_rtt.items():
            lst.sort(key=lambda t: (t[0], t[1]))
            by_stage[s] = [wid for _rtt, wid in lst]
        self._worker_hostnames = hostnames
        return by_stage

    def _eviction_state_uri(self) -> str:
        return self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/evicted_workers.json"
        )

    def _load_evicted_workers(self) -> None:
        uri = self._eviction_state_uri()
        if not self.bucket.exists(uri):
            return
        try:
            data = self.bucket.get_json(uri) or {}
            self._evicted_workers = set(data.get("evicted", []))
        except Exception:
            pass

    def _save_evicted_workers(self) -> None:
        try:
            self.bucket.put_json(
                self._eviction_state_uri(),
                {"evicted": sorted(self._evicted_workers),
                 "updated_unix": int(time.time())},
            )
        except Exception:
            pass

    def _evict_worker(self, worker_id: str, reason: str) -> None:
        if worker_id in self._evicted_workers:
            return
        self._evicted_workers.add(worker_id)
        log.warning("streaming orch evicting worker %s (%s)", worker_id, reason)
        self._save_evicted_workers()

    def _pick_colocated_assignee(
        self,
        pinned: list[str],
        prefer_host: str | None,
        mb: int,
        n_pinned: int,
    ) -> str:
        """Pick a worker from `pinned` for microbatch `mb`. If prefer_host
        is given (host of the previous stage's assignee for this mb), prefer
        a worker on the same host. Falls back to round-robin if none.

        This is Phase 3.2: co-located stage placement. With S3 as the only
        substrate (no peer TCP), an intra-box S3 PUT/GET round-trip is
        typically 50-100ms (cached / nearby AZ) vs 800-1500ms cross-region.
        Putting stage K+1 of microbatch M on the same box as stage K of
        microbatch M turns one 1500ms cross-region GET into a ~50ms
        intra-region GET.
        """
        if not pinned:
            raise RuntimeError("_pick_colocated_assignee: empty pinned list")
        if prefer_host is None or not self._worker_hostnames:
            return pinned[mb % n_pinned]
        # Find candidates on the preferred host
        same_host = [w for w in pinned if self._worker_hostnames.get(w) == prefer_host]
        if same_host:
            # Round-robin among same-host candidates so we don't overload
            # one worker if multiple mbs all want the same host.
            return same_host[mb % len(same_host)]
        # No same-host worker available; default round-robin.
        return pinned[mb % n_pinned]

    def _coverage_ok(self, by_stage: dict[int, list[str]]) -> bool:
        return all(by_stage.get(s) for s in range(self.params.n_stages))

    def _tick(self) -> None:
        state = self._load_state()
        epoch = state.current_round
        if epoch >= self.params.max_epochs:
            return

        by_stage = self._list_workers_by_stage()
        if epoch == 0 and (time.time() - self._started_at) < self.startup_grace_sec:
            return
        if not self._coverage_ok(by_stage):
            log.info("streaming orch waiting for stage coverage: %s",
                     {s: len(by_stage[s]) for s in range(self.params.n_stages)})
            return

        # Emit jobs for current epoch if not already done.
        idx_uri = self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/jobs_index.json"
        )
        if not self.bucket.exists(idx_uri):
            self._emit_epoch_jobs(epoch, by_stage)
            self._epoch_started_at[epoch] = time.time()

        # Stale-job retry: re-emit jobs whose original assignee hasn't produced
        # output within `retry_after_sec` AND whose inputs DO exist (so the
        # worker should have been able to start).
        self._maybe_retry_stale_jobs(epoch, by_stage)

        # Epoch-complete check:
        # - Inference mode: all M done.json markers at tail stage exist
        # - Training mode: also requires next-epoch weights to exist for every stage
        tail = self.params.n_stages - 1
        done_count = self._count_done(epoch, tail)
        epoch_done = done_count >= self.params.n_microbatches
        if epoch_done and self.params.training:
            for stage in self.stages:
                if stage.outer_graph is None:
                    continue
                w_uri = self._epoch_weights_uri(epoch + 1, stage.stage_id)
                if not self.bucket.exists(w_uri):
                    epoch_done = False
                    break

        if epoch_done:
            state = self._load_state()
            if state.current_round == epoch:
                state.current_round = epoch + 1
                state.completed_rounds = list(state.completed_rounds) + [epoch]
                self._save_state(state)
                self._aggregate_epoch_telemetry(epoch)
                log.info("streaming orch advanced to epoch %s", state.current_round)

    def _maybe_retry_stale_jobs(self, epoch: int, by_stage: dict[int, list[str]]) -> None:
        """Find jobs that should have completed by now (inputs exist, output
        doesn't, original assignee hasn't responded) and reassign to a
        different worker on the same stage.

        We walk the index for the epoch, look for outputs that don't exist
        yet AND whose inputs DO exist, and check the assignee's heartbeat
        recency. If stale, re-write the manifest with a new assignee.
        """
        retry_state_uri = self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/retries.json"
        )
        # Load retry state (which jobs have been retried and how many times).
        retried: dict[str, dict[str, Any]] = {}
        if self.bucket.exists(retry_state_uri):
            try:
                retried = self.bucket.get_json(retry_state_uri) or {}
            except Exception:
                retried = {}

        # Need the index of jobs for this epoch.
        idx_uri = self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/jobs_index.json"
        )
        if not self.bucket.exists(idx_uri):
            return
        try:
            jids = self.bucket.get_json(idx_uri)
        except Exception:
            return

        # Pre-compute stale-worker set: workers whose heartbeat is missing/old.
        all_alive_workers = {wid for ids in by_stage.values() for wid in ids}

        epoch_started = self._epoch_started_at.get(epoch, time.time())
        epoch_age = time.time() - epoch_started
        # Don't even try to retry until the epoch has been running for a while.
        if epoch_age < self.retry_after_sec:
            return

        retried_now = 0
        for jid in jids:
            # Skip if already retried max times.
            r_info = retried.get(jid, {"attempts": 0, "assignees": []})
            if r_info["attempts"] >= self.max_retries:
                continue

            mfst_uri = self.bucket.uri_for_key(
                paths.job_manifest_key(self.run_id, epoch, jid)
            )
            if not self.bucket.exists(mfst_uri):
                continue
            try:
                m = JobManifest.from_dict(self.bucket.get_json(mfst_uri))
            except Exception:
                continue

            # If outputs already exist, nothing to retry.
            if all(self.bucket.exists(o.uri) for o in m.outputs):
                continue
            # If inputs DON'T exist yet, the worker can't run yet — not stale.
            if m.inputs and not all(self.bucket.exists(i.uri) for i in m.inputs):
                continue
            # If the assigned worker is still alive AND the manifest is fresh,
            # give it more time. (But also: if this is the LAST allowed retry
            # attempt and the worker is alive but somehow not progressing, we
            # still consider it for eviction below.)
            if m.assigned_to in all_alive_workers and r_info["attempts"] + 1 < self.max_retries:
                continue
            # If we're about to do the LAST retry and the assignee is still
            # alive, that worker is the prime suspect for being broken — evict.
            if r_info["attempts"] + 1 >= self.max_retries:
                self._evict_worker(m.assigned_to,
                                   f"failed {self.max_retries} retry attempts on job {jid}")

            # Retry: pick a different worker on the same stage.
            stage_id = int(m.params.get("stage", -1))
            if stage_id < 0 or stage_id >= self.params.n_stages:
                continue
            candidates = [w for w in by_stage.get(stage_id, []) if w != m.assigned_to]
            tried = set(r_info.get("assignees", [m.assigned_to]))
            candidates = [w for w in candidates if w not in tried] or candidates
            if not candidates:
                continue
            new_assignee = candidates[0]
            new_m = JobManifest(
                schema_version=m.schema_version,
                job_id=m.job_id,
                run_id=m.run_id,
                round_id=m.round_id,
                kind=m.kind,
                graph_ref=m.graph_ref,
                params=m.params,
                inputs=m.inputs,
                outputs=m.outputs,
                assigned_to=new_assignee,
                deadline_unix=int(time.time()) + self.params.deadline_seconds,
                created_unix=int(time.time()),
            )
            self.bucket.put_json(mfst_uri, new_m.to_dict())
            r_info["attempts"] = int(r_info.get("attempts", 0)) + 1
            r_info["assignees"] = list(tried | {new_assignee, m.assigned_to})
            retried[jid] = r_info
            log.warning(
                "streaming retry r=%s job=%s old_assignee=%s -> %s (attempt %d)",
                epoch, jid, m.assigned_to, new_assignee, r_info["attempts"],
            )
            retried_now += 1
            if retried_now >= self.max_retries_per_tick:
                break

        if retried:
            try:
                self.bucket.put_json(retry_state_uri, retried)
            except Exception:
                pass

    def _count_done(self, epoch: int, tail_stage: int) -> int:
        prefix = self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/stage={tail_stage}/outputs/"
        )
        return sum(1 for u in self.bucket.list(prefix) if u.endswith("/done.json"))

    def _epoch_weights_uri(self, epoch: int, stage_id: int) -> str:
        return self.bucket.uri_for_key(
            f"runs/{self.run_id}/weights/epoch={epoch}/stage_{stage_id}_W.bin"
        )

    def _fwd_output_uri(self, epoch: int, stage_id: int, mb: int, name: str) -> str:
        return self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/stage={stage_id}/outputs/mb={mb}/{name}.bin"
        )

    def _bwd_output_uri(self, epoch: int, stage_id: int, mb: int, name: str) -> str:
        return self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/stage={stage_id}/bwd/mb={mb}/{name}.bin"
        )

    def _outer_marker_uri(self, epoch: int, stage_id: int) -> str:
        return self.bucket.uri_for_key(
            f"runs/{self.run_id}/streaming/epoch={epoch}/outer/stage={stage_id}/applied.json"
        )

    def _emit_epoch_jobs(self, epoch: int, by_stage: dict[int, list[str]]) -> None:
        """Emit forward jobs (and backward + outer if training). Each job's
        inputs chain to the prerequisite outputs; workers pick up as inputs land."""
        all_jobs: list[JobManifest] = []
        all_graphs: dict[str, Graph] = {}
        now = int(time.time())
        deadline = now + self.params.deadline_seconds

        for stage in self.stages:
            sha = stage.forward_graph.graph_id()
            all_graphs[sha] = stage.forward_graph
            if self.params.training:
                if stage.backward_graph is not None:
                    all_graphs[stage.backward_graph.graph_id()] = stage.backward_graph
                if stage.outer_graph is not None:
                    all_graphs[stage.outer_graph.graph_id()] = stage.outer_graph

        # ---- Forward jobs ----
        # Track the assignee chosen for stage K of microbatch M, so we can
        # prefer the same physical host for stage K+1 of the same mb (Phase
        # 3.2 co-location: intra-box S3 RTT is ~50-100ms vs ~800-1500ms
        # cross-region, so co-locating consecutive stages of one mb on the
        # same box collapses inter-stage activation latency).
        fwd_assignee_by_stage_mb: dict[tuple[int, int], str] = {}
        for mb in range(self.params.n_microbatches):
            prev_assignee_host: str | None = None
            for stage in self.stages:
                s = stage.stage_id
                pinned = by_stage[s]
                assignee = self._pick_colocated_assignee(
                    pinned, prev_assignee_host, mb, len(pinned),
                )
                fwd_assignee_by_stage_mb[(s, mb)] = assignee
                prev_assignee_host = self._worker_hostnames.get(assignee)
                sha = stage.forward_graph.graph_id()
                graph_ref = GraphRef(
                    sha256=sha,
                    uri=self.bucket.uri_for_key(paths.graph_key(self.run_id, sha)),
                )

                inputs: list[IORef] = []
                if s != 0:
                    for name, _, _ in stage.forward_input_specs:
                        inputs.append(IORef(
                            name=name,
                            uri=self._fwd_output_uri(epoch, s - 1, mb, name),
                        ))
                # Inject epoch-versioned weights as a job input.
                if stage.weights_input_name and self.params.training:
                    inputs.append(IORef(
                        name=stage.weights_input_name,
                        uri=self._epoch_weights_uri(epoch, s),
                    ))
                # Tail stage forward also reads the loss target.
                is_tail = (s == self.params.n_stages - 1)
                is_head_fwd = (s == 0)
                if is_tail and self.params.training and self.params.target_static_uri:
                    inputs.append(IORef(name="target", uri=self.params.target_static_uri))
                # Generalized static inputs: head/tail/head_and_tail/all
                for name, (uri, scope) in self.params.static_inputs.items():
                    needed = (
                        scope == "all"
                        or (scope == "head" and is_head_fwd)
                        or (scope == "tail" and is_tail)
                        or (scope == "head_and_tail" and (is_head_fwd or is_tail))
                    )
                    if needed:
                        inputs.append(IORef(name=name, uri=uri))

                outputs: list[IORef] = [
                    IORef(name=name,
                          uri=self._fwd_output_uri(epoch, s, mb, name))
                    for (name, _shape, _dtype) in stage.forward_output_specs
                ]
                # Tail stage forward writes loss + done marker.
                if is_tail:
                    if self.params.training:
                        # In training mode, `loss` IS the done marker —
                        # we write it to a .json URI so the worker JSON-encodes
                        # the tensor (1-element loss scalar).
                        outputs.append(IORef(
                            name="loss",
                            uri=self.bucket.uri_for_key(
                                f"runs/{self.run_id}/streaming/epoch={epoch}/stage={s}/outputs/mb={mb}/done.json"
                            ),
                        ))
                    else:
                        outputs.append(IORef(
                            name="done",
                            uri=self.bucket.uri_for_key(
                                f"runs/{self.run_id}/streaming/epoch={epoch}/stage={s}/outputs/mb={mb}/done.json"
                            ),
                        ))
                jid = f"j-e{epoch}-s{s}-mb{mb}-fwd"
                all_jobs.append(JobManifest(
                    job_id=jid,
                    run_id=self.run_id,
                    round_id=epoch,
                    kind="pipe_forward",
                    graph_ref=graph_ref,
                    params={
                        **self.params.common_params,
                        **self.params.forward_params,
                        "epoch": epoch, "stage": s, "mb": mb,
                        "mb_seed": epoch * 10000 + mb,
                    },
                    inputs=inputs,
                    outputs=outputs,
                    assigned_to=assignee,
                    deadline_unix=deadline,
                    created_unix=now,
                ))

        # ---- Backward jobs (training only) ----
        if self.params.training:
            for mb in range(self.params.n_microbatches):
                # Reverse order so that bwd of stage S-1 is first.
                for stage in reversed(self.stages):
                    s = stage.stage_id
                    if stage.backward_graph is None:
                        continue
                    pinned = by_stage[s]
                    # Prefer the same worker that handled this (mb, s)'s forward
                    # so the output cache (LRU on the worker) skips a GET on the
                    # weights and possibly x_in. If the forward assignee is
                    # still in the pinned set, use it; else fall back to a
                    # round-robin same-host pick.
                    fwd_pick = fwd_assignee_by_stage_mb.get((s, mb))
                    if fwd_pick and fwd_pick in pinned:
                        assignee = fwd_pick
                    else:
                        prev_host = (
                            self._worker_hostnames.get(fwd_pick) if fwd_pick else None
                        )
                        assignee = self._pick_colocated_assignee(
                            pinned, prev_host, mb, len(pinned),
                        )
                    sha = stage.backward_graph.graph_id()
                    graph_ref = GraphRef(
                        sha256=sha,
                        uri=self.bucket.uri_for_key(paths.graph_key(self.run_id, sha)),
                    )
                    is_tail = (s == self.params.n_stages - 1)
                    is_head = (s == 0)
                    inputs: list[IORef] = [
                        IORef(name=stage.weights_input_name or "W",
                              uri=self._epoch_weights_uri(epoch, s)),
                    ]
                    # x_in: stage 0's input is synthetic — we re-synth in bwd graph.
                    # For non-stage-0, x_in = stage s-1's forward output.
                    if not is_head:
                        inputs.append(IORef(
                            name="x_in",
                            uri=self._fwd_output_uri(epoch, s - 1, mb, "x"),
                        ))
                    else:
                        # Head stage: regenerate input from seed via a fwd-style
                        # input. For our pipe_train, we mark this with a special
                        # name and the bwd graph for stage 0 is built differently.
                        # Simplification: for stage 0 we let the graph re-run forward
                        # internally from mb_seed (same seed param).
                        pass
                    if is_tail and self.params.target_static_uri:
                        inputs.append(IORef(name="target",
                                            uri=self.params.target_static_uri))
                    # Generalized static inputs in bwd path too
                    for name, (uri, scope) in self.params.static_inputs.items():
                        needed = (
                            scope == "all"
                            or (scope == "head" and is_head)
                            or (scope == "tail" and is_tail)
                            or (scope == "head_and_tail" and (is_head or is_tail))
                        )
                        if needed:
                            inputs.append(IORef(name=name, uri=uri))
                    if not is_tail:
                        # dL_dx_out comes from the next stage's bwd.
                        inputs.append(IORef(
                            name="dL_dx_out",
                            uri=self._bwd_output_uri(epoch, s + 1, mb, "dL_dx_in"),
                        ))
                    outputs: list[IORef] = [
                        IORef(name="dW",
                              uri=self._bwd_output_uri(epoch, s, mb, "dW")),
                    ]
                    if not is_head and stage.backward_emits_dx_in:
                        outputs.append(IORef(
                            name="dL_dx_in",
                            uri=self._bwd_output_uri(epoch, s, mb, "dL_dx_in"),
                        ))
                    jid = f"j-e{epoch}-s{s}-mb{mb}-bwd"
                    all_jobs.append(JobManifest(
                        job_id=jid,
                        run_id=self.run_id,
                        round_id=epoch,
                        kind="pipe_backward",
                        graph_ref=graph_ref,
                        params={
                            **self.params.common_params,
                            "epoch": epoch, "stage": s, "mb": mb,
                            "mb_seed": epoch * 10000 + mb,
                        },
                        inputs=inputs,
                        outputs=outputs,
                        assigned_to=assignee,
                        deadline_unix=deadline,
                        created_unix=now,
                    ))

            # ---- Outer step jobs (one per stage; consumes all M dW) ----
            for stage in self.stages:
                s = stage.stage_id
                if stage.outer_graph is None:
                    continue
                pinned = by_stage[s]
                # Pick the FIRST pinned worker for outer (deterministic).
                assignee = pinned[0]
                sha = stage.outer_graph.graph_id()
                graph_ref = GraphRef(
                    sha256=sha,
                    uri=self.bucket.uri_for_key(paths.graph_key(self.run_id, sha)),
                )
                inputs = [IORef(name=stage.weights_input_name or "W",
                                uri=self._epoch_weights_uri(epoch, s))]
                for mb in range(self.params.n_microbatches):
                    inputs.append(IORef(
                        name=f"dW_{mb}",
                        uri=self._bwd_output_uri(epoch, s, mb, "dW"),
                    ))
                outputs = [
                    IORef(name="W_new",
                          uri=self._epoch_weights_uri(epoch + 1, s)),
                ]
                jid = f"j-e{epoch}-s{s}-outer"
                all_jobs.append(JobManifest(
                    job_id=jid,
                    run_id=self.run_id,
                    round_id=epoch,
                    kind="pipe_outer",
                    graph_ref=graph_ref,
                    params={
                        **self.params.common_params,
                        "epoch": epoch, "stage": s,
                    },
                    inputs=inputs,
                    outputs=outputs,
                    assigned_to=assignee,
                    deadline_unix=deadline,
                    created_unix=now,
                ))

        # Write graphs (deduped) + manifests + index.
        for sha, g in all_graphs.items():
            guri = self.bucket.uri_for_key(paths.graph_key(self.run_id, sha))
            if not self.bucket.exists(guri):
                self.bucket.put(guri, g.to_canonical_json())
        for j in all_jobs:
            self.bucket.put_json(
                self.bucket.uri_for_key(
                    paths.job_manifest_key(self.run_id, epoch, j.job_id)
                ),
                j.to_dict(),
            )
        self.bucket.put_json(
            self.bucket.uri_for_key(
                f"runs/{self.run_id}/streaming/epoch={epoch}/jobs_index.json"
            ),
            [j.job_id for j in all_jobs],
        )
        # Also write a normal jobs_index so worker._handle_round picks them up.
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.jobs_index_key(self.run_id, epoch)),
            [j.job_id for j in all_jobs],
        )
        log.info("streaming orch emitted epoch %s with %s jobs across %s stages",
                 epoch, len(all_jobs), self.params.n_stages)

    def _aggregate_epoch_telemetry(self, epoch: int) -> None:
        """Same shape as Orchestrator._aggregate_round_telemetry but per epoch."""
        try:
            run_prefix = self.bucket.uri_for_key(f"runs/{self.run_id}/")
            all_objs = self.bucket.list(run_prefix)
            tag = f"round={epoch}"
            relevant = [u for u in all_objs if tag in u or f"epoch={epoch}" in u]
            total_bytes = 0
            kinds: dict[str, dict[str, int]] = {}
            for u in relevant:
                h = self.bucket.head(u)
                size = int(h["size_bytes"]) if h else 0
                total_bytes += size
                rest = u.split(f"runs/{self.run_id}/", 1)[1]
                kind = rest.split("/", 1)[0]
                slot = kinds.setdefault(kind, {"n": 0, "bytes": 0})
                slot["n"] += 1
                slot["bytes"] += size

            stamps_prefix = self.bucket.uri_for_key(
                f"runs/{self.run_id}/jobtimes/round={epoch}/"
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
            elapsed = time.time() - self._epoch_started_at.get(epoch, time.time())
            wallclock = elapsed or 1.0
            utilization = pool_busy_sec / (wallclock * max(workers_total, 1))

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

            # ---- cost telemetry ----
            # We approximate cost by attributing each ACTIVE worker's portion
            # of the elapsed wallclock to its gpu_class price. This double-
            # counts when one box runs N workers (each priced at the per-GPU
            # share), but the sum across the box equals the box's hourly
            # rate when fully populated.
            dollars_spent = 0.0
            tokens_processed = self.params.tokens_per_mb * self.params.n_microbatches
            cost_by_gpu_class: dict[str, dict[str, float]] = {}
            if self.params.gpu_class_price_per_hour:
                # Walk live workers (since we can't reliably read past
                # heartbeats for workers that died). Note: this approximates
                # by ASSUMING the active workers were alive for the full
                # epoch, which overcounts during ramp-up but is close after
                # the first epoch.
                worker_caps: dict[str, dict[str, Any]] = {}
                try:
                    prefix = self.bucket.uri_for_key(paths.workers_prefix(self.run_id))
                    for u in self.bucket.list(prefix):
                        if not u.endswith(".json"):
                            continue
                        try:
                            wi = WorkerInfo.from_dict(self.bucket.get_json(u))
                        except Exception:
                            continue
                        worker_caps[wi.worker_id] = wi.capabilities or {}
                except Exception:
                    pass

                for wid in per_worker:
                    caps = worker_caps.get(wid, {})
                    gpu_class = str(caps.get("gpu_class", "unknown"))
                    rate = float(self.params.gpu_class_price_per_hour.get(gpu_class, 0.0))
                    n_gpus_on_box = max(int(caps.get("n_gpus", 1)), 1)
                    # rate is per-BOX/hour; per-GPU/hour = rate / n_gpus_on_box
                    per_worker_rate = rate / n_gpus_on_box
                    cost = per_worker_rate * (elapsed / 3600.0)
                    dollars_spent += cost
                    slot = cost_by_gpu_class.setdefault(gpu_class, {"n": 0, "cost": 0.0, "rate_per_hour": rate})
                    slot["n"] += 1
                    slot["cost"] += cost

            dollars_per_b_tokens = (
                (dollars_spent / max(tokens_processed, 1)) * 1e9
                if tokens_processed > 0 else 0.0
            )
            tokens_per_sec = (
                tokens_processed / max(elapsed, 1e-9) if tokens_processed > 0 else 0.0
            )

            tele = {
                "round": epoch,
                "epoch": epoch,
                "n_objects": len(relevant),
                "total_bytes": total_bytes,
                "wallclock_sec": elapsed,
                "by_kind": kinds,
                "active_workers": len(per_worker),
                "n_jobs": len(jobtimes),
                "pool_size": workers_total,
                "pool_busy_sec": round(pool_busy_sec, 3),
                "pool_compute_sec": round(pool_compute_sec, 3),
                "pool_io_sec": round(pool_io_sec, 3),
                "utilization": round(utilization, 4),
                "compute_share_of_busy": round(pool_compute_sec / max(pool_busy_sec, 1e-9), 4),
                "per_worker_jobs": per_worker,
                # Cost telemetry (Cross-cutting from the plan)
                "tokens_processed": tokens_processed,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "dollars_spent": round(dollars_spent, 4),
                "dollars_per_b_tokens": round(dollars_per_b_tokens, 2),
                "cost_by_gpu_class": cost_by_gpu_class,
            }
            uri = self.bucket.uri_for_key(
                f"runs/{self.run_id}/telemetry/round={epoch}.json"
            )
            self.bucket.put_json(uri, tele)
            log.info(
                "streaming telemetry e=%s: wall=%.1fs busy=%.1fs util=%.1f%% "
                "%.2f MB %d jobs %d workers cost=$%.2f tokens/sec=%.0f $/B-tok=$%.2f",
                epoch, elapsed, pool_busy_sec, utilization * 100.0,
                total_bytes / 1e6, len(jobtimes), len(per_worker),
                dollars_spent, tokens_per_sec, dollars_per_b_tokens,
            )
        except Exception:
            log.exception("streaming telemetry aggregate failed for e=%s", epoch)

    def _n_pool_workers(self) -> int:
        try:
            by_stage = self._list_workers_by_stage()
            return len({wid for ids in by_stage.values() for wid in ids})
        except Exception:
            return 0
