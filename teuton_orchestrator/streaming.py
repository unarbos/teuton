"""V3 streaming scheduler for GPT-style tasks.

This ports the important v2 streaming idea into the v3 manifest/receipt world:
the task may still use v2 graph builders and v2 artifact URIs internally, but
job assignment, signatures, receipts, and validation are v3-native.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from teuton_core import paths
from teuton_core.metagraph import BtcliMetagraphHotkeyResolver, MetagraphHotkeyResolver
from teuton_core.protocol import ArtifactCryptoPolicy, ArtifactRef, AssignmentGrantV3, GraphRef, JobManifestV3, ResourceRequirements, VerificationPolicy, WorkerIdentity
from teuton_core.signatures import Signer
from teuton_core.telemetry import TelemetryWriter
from teuton_core.wallet_crypto import AssignmentEncryptor, DevAssignmentCrypto, Ed25519SealedBoxAssignmentCrypto
from teuton_runtime.discovery import build_discovery_backend
from teuton_runtime.grants import broker_for_mode
from teuton_runtime.queue import OrchestratorQueue, QueueEntry, scan_recent_receipt_job_ids
from teuton_runtime.storage import ObjectStore
from teuton_tasks import load_streaming_task
from .scheduler import QuotaBook


# Per-hotkey backpressure: orchestrator refuses to emit more work for a miner
# that already has this many outstanding entries in the queue. Replaces the
# legacy ``TEUTON_BASE_QUOTA=50000`` workaround that defeated all flow
# control. Override via env if a particular run wants a deeper queue.
_DEFAULT_MAX_INFLIGHT_PER_HOTKEY = int(os.environ.get("TEUTON_MAX_INFLIGHT_PER_HOTKEY", "8"))


@dataclass
class StreamingRunConfig:
    netuid: int
    run_id: str
    task: str = "gpt_pipe"
    max_epochs: int = 1
    owner_secret: str = "owner-dev-secret"
    owner_signer: Signer | None = None
    crypto_policy: ArtifactCryptoPolicy | None = None
    grant_mode: str = "direct"
    grant_ttl_sec: int = 600
    assignment_secret: str = "teuton-dev-assignment"
    assignment_crypto: str = "dev"
    network: str = "finney"
    discovery_backend: str = "bucket"
    discovery_heartbeat_ttl_sec: float | None = 30.0
    stress_emit: bool = False
    stress_emit_interval: float = 0.0
    stress_epoch_base: int = 1_000_000
    stress_pin_weights_epoch: int = 0
    stress_skip_bootstrap_if_present: bool = True
    stress_max_iterations: int = 0
    epoch_timeout_sec: float = 300.0
    # Queue-depth backpressure. ``max_inflight_per_hotkey`` is the maximum
    # number of outstanding queue entries any one miner may have at once;
    # ``emit_*`` blocks (or skips) when the limit is hit so slow miners don't
    # accumulate work. ``flush_interval_sec`` controls how often the queue
    # snapshot is published to the bucket.
    max_inflight_per_hotkey: int = _DEFAULT_MAX_INFLIGHT_PER_HOTKEY
    queue_flush_interval_sec: float = 0.5


class StreamingRunManager:
    def __init__(self, *, bucket: ObjectStore, config: StreamingRunConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.task = load_streaming_task(config.task)
        self.quota = QuotaBook()
        self.emitted: list[str] = []
        self.jobs: dict[str, JobManifestV3] = {}
        self.discovery = build_discovery_backend(
            config.discovery_backend,
            bucket=bucket,
            netuid=config.netuid,
            run_id=config.run_id,
            heartbeat_ttl_sec=config.discovery_heartbeat_ttl_sec,
        )
        self.grant_broker = broker_for_mode(config.grant_mode, bucket)
        self.assignment_crypto: AssignmentEncryptor = (
            Ed25519SealedBoxAssignmentCrypto()
            if config.assignment_crypto == "ed25519"
            else DevAssignmentCrypto(config.assignment_secret)
        )
        self.hotkey_resolver: MetagraphHotkeyResolver | None = (
            BtcliMetagraphHotkeyResolver(netuid=config.netuid, network=config.network)
            if config.assignment_crypto == "ed25519"
            else None
        )
        self.telemetry = TelemetryWriter(
            bucket=bucket,
            netuid=config.netuid,
            run_id=config.run_id,
            component="orchestrator",
        )
        # Authoritative outstanding-work queue. Replaces the per-emit
        # ``index.json`` / ``hotkey={hk}/index.json`` / ``step={n}/index.json``
        # writes. Reconcile from the bucket on startup so a restarted
        # orchestrator picks up where the previous one left off.
        self.queue = OrchestratorQueue(
            bucket=bucket,
            netuid=config.netuid,
            run_id=config.run_id,
            role="train",
            flush_interval_sec=config.queue_flush_interval_sec,
        )
        self.queue.reconcile_from_bucket()
        self.queue.start_background_flush()
        # Tracks the receipt-prefix mtime cursor for incremental scans in
        # ``_drain_queue_via_receipts``. Initialised to "now" so the cold
        # scan ignores stale receipts from previous runs that share
        # epoch-based job_ids (e.g. stress mode emits ``j-e1000000-...``
        # every restart, and yesterday's receipts for the same job_id
        # would otherwise drain today's queue entries before miners can
        # process them).
        self._receipt_scan_cursor: float = time.time()

    def bootstrap(self) -> None:
        if not (self.config.stress_emit and self._bootstrap_artifacts_present()):
            self.task.bootstrap(bucket=self.bucket, run_id=self.config.run_id, max_rounds=self.config.max_epochs)
        stages, params = self.task.build_streaming_inputs(bucket=self.bucket, run_id=self.config.run_id)
        params.max_epochs = self.config.max_epochs
        self.stages = stages
        self.params = params

    def _bootstrap_artifacts_present(self) -> bool:
        """Return True when an existing run already has the static artifacts we
        would otherwise rebuild. Lets the stress emitter attach to a live run
        without reseeding tokens/weights, which is critical for compatibility
        with miners that have already pinned to those URIs.
        """
        if not self.config.stress_skip_bootstrap_if_present:
            return False
        manifest_uri = self.bucket.uri_for_key(
            paths.manifest_config_key(self.config.netuid, self.config.run_id)
        )
        try:
            return bool(self.bucket.exists(manifest_uri))
        except Exception:
            return False

    def discover_workers(self) -> list[WorkerIdentity]:
        records = self.discovery.discover_workers()
        workers = [record.worker for record in records]
        self.quota.update_workers([record.miner for record in records], workers)
        return workers

    def run_loop(self, *, poll_interval: float = 0.1, timeout_sec: float = 600.0) -> None:
        self.bootstrap()
        try:
            if self.config.stress_emit:
                self._run_stress_emit_loop(poll_interval=poll_interval, timeout_sec=timeout_sec)
                return
            deadline = time.time() + timeout_sec
            for epoch in range(self.config.max_epochs):
                while time.time() < deadline and not self.discover_workers():
                    time.sleep(poll_interval)
                if time.time() >= deadline:
                    raise TimeoutError("no miners available for streaming run")
                t_emit = time.time()
                self.emit_epoch(epoch)
                emit_seconds = time.time() - t_emit
                t_wait = time.time()
                outcome = "ok"
                epoch_deadline = min(deadline, t_wait + max(0.0, self.config.epoch_timeout_sec))
                try:
                    self.wait_epoch(epoch, deadline=epoch_deadline, poll_interval=poll_interval)
                except TimeoutError:
                    outcome = "timeout"
                    self._emit_epoch_telemetry(
                        epoch=epoch,
                        emit_seconds=emit_seconds,
                        wait_seconds=time.time() - t_wait,
                        outcome=outcome,
                    )
                    raise
                self._emit_epoch_telemetry(
                    epoch=epoch,
                    emit_seconds=emit_seconds,
                    wait_seconds=time.time() - t_wait,
                    outcome=outcome,
                )
        finally:
            # Flush the final queue snapshot and stop the background thread
            # so the bucket reflects the actual outstanding state on exit.
            self.queue.stop()

    def stop(self) -> None:
        """Stop the queue's background flush thread (idempotent)."""
        self.queue.stop()

    def _run_stress_emit_loop(self, *, poll_interval: float, timeout_sec: float) -> None:
        """Continuously publish work for network load testing.

        - Iterates a synthetic epoch counter starting at ``stress_epoch_base``
          so emitted ``j-e{epoch}-...`` IDs never collide with the historic
          range any orchestrator already used for this run.
        - Pins all per-epoch weight URIs to ``stress_pin_weights_epoch`` so
          miners always read the bootstrap-time weights instead of waiting on
          outer steps that we deliberately do not emit.
        - Emits forward jobs only (training=False) so there is no
          inter-epoch dependency that can stall the pipeline.
        - Periodically rediscovers live workers so newly-joined miners pick up
          work without an orchestrator restart.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self.discover_workers():
            time.sleep(poll_interval)
        if time.time() >= deadline:
            raise TimeoutError("no miners available for streaming stress run")

        iteration = 0
        while time.time() < deadline:
            if self.config.stress_max_iterations and iteration >= self.config.stress_max_iterations:
                return
            epoch = self.config.stress_epoch_base + iteration
            iteration += 1
            try:
                self.discover_workers()
            except Exception:
                pass
            # Drain receipts -> queue BEFORE emit so backpressure reflects
            # currently-completed work. Without this, every epoch sees the
            # queue at "post-last-emit" depth and refuses to emit.
            self._drain_queue_via_receipts()
            t_emit = time.time()
            before = len(self.emitted)
            try:
                self.emit_epoch(epoch, forward_only=True)
            except RuntimeError as exc:
                # `pick_worker` raises when no live miner has queue room or
                # quota; back off and rediscover.
                print(
                    f"[orchestrator:stress] epoch={epoch} pick_worker err: {exc!r}; "
                    f"queue_depth={self.queue.depth()} sleeping 5s",
                    flush=True,
                )
                self._drop_epoch_jobs(epoch)
                time.sleep(5.0)
                continue
            emit_seconds = time.time() - t_emit
            self._release_epoch_quota(epoch)
            try:
                self._emit_epoch_telemetry(
                    epoch=epoch,
                    emit_seconds=emit_seconds,
                    wait_seconds=0.0,
                    outcome="stress_emitted",
                )
            except Exception:
                pass
            emitted = len(self.emitted) - before
            # NOTE: do NOT drop epoch jobs from the queue here. In the legacy
            # design we cleared local bookkeeping immediately because the
            # quota accounting was emit-time only; with the queue we need
            # entries to stay until receipts land (or the deadline expires)
            # so workers can pick them up.
            self.jobs = {
                job_id: manifest
                for job_id, manifest in self.jobs.items()
                if not job_id.startswith(f"j-e{epoch}-")
            }
            print(
                f"[orchestrator:stress] iter={iteration} epoch={epoch} "
                f"emitted_jobs={emitted} emit_seconds={emit_seconds:.3f} "
                f"queue_depth={self.queue.depth()} "
                f"workers={sum(len(a.workers) for a in self.quota.accounts.values())}",
                flush=True,
            )
            if self.config.stress_emit_interval > 0:
                time.sleep(self.config.stress_emit_interval)

    def epoch_weights_uri(self, epoch: int, stage_id: int) -> str:
        effective = self.config.stress_pin_weights_epoch if self.config.stress_emit else epoch
        return self.bucket.uri_for_key(
            f"runs/{self.config.run_id}/weights/epoch={effective}/stage_{stage_id}_W.bin"
        )

    def _release_epoch_quota(self, epoch: int) -> None:
        """Release legacy QuotaBook reservations for an epoch.

        Queue depth is now the authoritative backpressure signal; the
        QuotaBook is retained only for ``pick_worker`` priority math
        (``priority = trust * (1 + #workers) / inflight_cu``). Releasing
        reservations here keeps the priority denominator from drifting.
        """
        epoch_prefix = f"j-e{epoch}-"
        for job_id, manifest in self.jobs.items():
            if job_id.startswith(epoch_prefix):
                self.quota.release(manifest.assigned_hotkey)

    def _drop_epoch_jobs(self, epoch: int) -> None:
        """Remove epoch entries from local bookkeeping AND the queue.

        The queue is the source of truth, so dropping here also unblocks
        ``max_inflight_per_hotkey`` budget for the assigned miners.
        """
        epoch_prefix = f"j-e{epoch}-"
        drop_ids = [job_id for job_id in self.jobs if job_id.startswith(epoch_prefix)]
        self.queue.remove_many(drop_ids)
        self.jobs = {
            job_id: manifest
            for job_id, manifest in self.jobs.items()
            if not job_id.startswith(epoch_prefix)
        }

    def _emit_epoch_telemetry(
        self,
        *,
        epoch: int,
        emit_seconds: float,
        wait_seconds: float,
        outcome: str,
    ) -> None:
        try:
            manifest_cfg = self.bucket.get_json(
                self.bucket.uri_for_key(paths.manifest_config_key(self.config.netuid, self.config.run_id))
            )
        except Exception:
            manifest_cfg = {}
        epoch_prefix = f"j-e{epoch}-"
        by_kind: dict[str, int] = {}
        n_jobs = 0
        for job_id, mani in self.jobs.items():
            if not job_id.startswith(epoch_prefix):
                continue
            n_jobs += 1
            by_kind[mani.kind] = by_kind.get(mani.kind, 0) + 1
        n_mb = int(getattr(self.params, "n_microbatches", 0) or 0)
        b = int(manifest_cfg.get("B", 0) or 0)
        t = int(manifest_cfg.get("T", 0) or 0)
        tokens_in_epoch = b * t * n_mb if (b and t and n_mb) else 0
        wall = emit_seconds + wait_seconds
        try:
            self.telemetry.orchestrator(
                {
                    "epoch": int(epoch),
                    "outcome": outcome,
                    "emit_seconds": round(emit_seconds, 3),
                    "wait_seconds": round(wait_seconds, 3),
                    "wall_seconds": round(wall, 3),
                    "tokens_in_epoch": tokens_in_epoch,
                    "tokens_per_sec": round(tokens_in_epoch / wall, 2) if wall > 0 else None,
                    "n_jobs_emitted": n_jobs,
                    "n_microbatches": n_mb,
                    "n_stages": int(getattr(self.params, "n_stages", 0) or 0),
                    "by_kind": by_kind,
                    "queue": self._queue_telemetry(),
                    "manifest_config": {
                        k: manifest_cfg.get(k)
                        for k in (
                            "task", "d", "n_head", "d_ff", "B", "T",
                            "n_stages", "n_microbatches", "n_blocks_per_stage", "tokens_uri",
                        )
                    },
                }
            )
        except Exception:
            pass

    def _queue_telemetry(self) -> dict:
        """Snapshot of queue depth + per-hotkey distribution.

        Cheap (in-memory only) and reported on every epoch telemetry emit
        so dashboards can plot queue_depth over time and detect "queue
        getting huge and never draining" in seconds.
        """
        per_hotkey: dict[str, int] = {}
        for entry in list(self.queue):
            per_hotkey[entry.assigned_hotkey] = per_hotkey.get(entry.assigned_hotkey, 0) + 1
        depth = sum(per_hotkey.values())
        depths = sorted(per_hotkey.values())
        if depths:
            p99 = depths[min(len(depths) - 1, int(0.99 * len(depths)))]
            mx = depths[-1]
        else:
            p99 = 0
            mx = 0
        return {
            "depth": depth,
            "n_hotkeys_with_work": len(per_hotkey),
            "depth_p99_per_hotkey": p99,
            "depth_max_per_hotkey": mx,
            "max_inflight_per_hotkey": self.config.max_inflight_per_hotkey,
        }

    def emit_epoch(self, epoch: int, *, forward_only: bool = False) -> None:
        for stage in self.stages:
            for graph in (stage.forward_graph, stage.backward_graph, stage.outer_graph):
                if graph is None:
                    continue
                sha = graph.graph_id()
                uri = self.bucket.uri_for_key(paths.graph_key(self.config.netuid, self.config.run_id, sha))
                if not self.bucket.exists(uri):
                    self.bucket.put(uri, graph.to_canonical_json())

        for mb in range(self.params.n_microbatches):
            for stage in self.stages:
                self.emit_forward(epoch, mb, stage)
        if forward_only:
            return
        if self.params.training:
            for mb in range(self.params.n_microbatches):
                for stage in reversed(self.stages):
                    if stage.backward_graph is not None:
                        self.emit_backward(epoch, mb, stage)
            for stage in self.stages:
                if stage.outer_graph is not None:
                    self.emit_outer(epoch, stage)

    def emit_forward(self, epoch: int, mb: int, stage, *, force_worker: WorkerIdentity | None = None) -> JobManifestV3:
        s = stage.stage_id
        inputs: list[ArtifactRef] = []
        if s != 0:
            for name, _shape, _dtype in stage.forward_input_specs:
                upstream = self.jobs[f"j-e{epoch}-s{s - 1}-mb{mb}-fwd"].outputs[0]
                inputs.append(ArtifactRef(name=name, uri=upstream.uri, crypto=upstream.crypto))
        if stage.weights_input_name and self.params.training:
            inputs.append(ArtifactRef(name=stage.weights_input_name, uri=self.epoch_weights_uri(epoch, s)))
        is_tail = s == self.params.n_stages - 1
        is_head = s == 0
        if is_tail and self.params.training and self.params.target_static_uri:
            inputs.append(ArtifactRef(name="target", uri=self.params.target_static_uri))
        for name, (uri, scope) in self.params.static_inputs.items():
            if scope == "all" or (scope == "head" and is_head) or (scope == "tail" and is_tail) or (scope == "head_and_tail" and (is_head or is_tail)):
                inputs.append(ArtifactRef(name=name, uri=uri))
        outputs = [self.output_ref(name=name, uri=self.fwd_output_uri(epoch, s, mb, name)) for name, _shape, _dtype in stage.forward_output_specs]
        if is_tail:
            outputs.append(self.output_ref(name="loss" if self.params.training else "done", uri=self.fwd_done_uri(epoch, s, mb)))
        return self.emit_job(
            job_id=f"j-e{epoch}-s{s}-mb{mb}-fwd",
            kind="pipe_forward",
            step_id=epoch,
            graph=stage.forward_graph,
            params={**self.params.common_params, **self.params.forward_params, "epoch": epoch, "stage": s, "mb": mb, "mb_seed": epoch * 10000 + mb},
            inputs=inputs,
            outputs=outputs,
            force_worker=force_worker,
        )

    def emit_backward(self, epoch: int, mb: int, stage) -> JobManifestV3:
        s = stage.stage_id
        is_tail = s == self.params.n_stages - 1
        is_head = s == 0
        inputs = [ArtifactRef(name=stage.weights_input_name or "W", uri=self.epoch_weights_uri(epoch, s))]
        if not is_head:
            upstream = self.jobs[f"j-e{epoch}-s{s - 1}-mb{mb}-fwd"].outputs[0]
            inputs.append(ArtifactRef(name="x_in", uri=upstream.uri, crypto=upstream.crypto))
        if is_tail and self.params.target_static_uri:
            inputs.append(ArtifactRef(name="target", uri=self.params.target_static_uri))
        for name, (uri, scope) in self.params.static_inputs.items():
            if scope == "all" or (scope == "head" and is_head) or (scope == "tail" and is_tail) or (scope == "head_and_tail" and (is_head or is_tail)):
                inputs.append(ArtifactRef(name=name, uri=uri))
        if not is_tail:
            upstream = self.jobs[f"j-e{epoch}-s{s + 1}-mb{mb}-bwd"]
            dx = next(ref for ref in upstream.outputs if ref.name == "dL_dx_in")
            inputs.append(ArtifactRef(name="dL_dx_out", uri=dx.uri, crypto=dx.crypto))
        outputs = [self.output_ref(name="dW", uri=self.bwd_output_uri(epoch, s, mb, "dW"))]
        if not is_head and stage.backward_emits_dx_in:
            outputs.append(self.output_ref(name="dL_dx_in", uri=self.bwd_output_uri(epoch, s, mb, "dL_dx_in")))
        return self.emit_job(
            job_id=f"j-e{epoch}-s{s}-mb{mb}-bwd",
            kind="pipe_backward",
            step_id=epoch,
            graph=stage.backward_graph,
            params={**self.params.common_params, "epoch": epoch, "stage": s, "mb": mb, "mb_seed": epoch * 10000 + mb},
            inputs=inputs,
            outputs=outputs,
        )

    def emit_outer(self, epoch: int, stage) -> JobManifestV3:
        s = stage.stage_id
        inputs = [ArtifactRef(name=stage.weights_input_name or "W", uri=self.epoch_weights_uri(epoch, s))]
        for mb in range(self.params.n_microbatches):
            upstream = self.jobs[f"j-e{epoch}-s{s}-mb{mb}-bwd"]
            dw = next(ref for ref in upstream.outputs if ref.name == "dW")
            inputs.append(ArtifactRef(name=f"dW_{mb}", uri=dw.uri, crypto=dw.crypto))
        outputs = [self.output_ref(name="W_new", uri=self.epoch_weights_uri(epoch + 1, s))]
        return self.emit_job(
            job_id=f"j-e{epoch}-s{s}-outer",
            kind="pipe_outer",
            step_id=epoch,
            graph=stage.outer_graph,
            params={**self.params.common_params, "epoch": epoch, "stage": s},
            inputs=inputs,
            outputs=outputs,
        )

    def emit_job(
        self,
        *,
        job_id: str,
        kind: str,
        step_id: int,
        graph,
        params: dict,
        inputs: list[ArtifactRef],
        outputs: list[ArtifactRef],
        force_worker: WorkerIdentity | None = None,
    ) -> JobManifestV3:
        requirements = self.resource_requirements(params)
        # ``force_worker`` lets external callers (e.g. the ``teuton-v3 send-job``
        # CLI) pin a manifest to a specific miner/worker without going through
        # the QuotaBook. Default behaviour is unchanged.
        if force_worker is not None:
            worker = force_worker
        else:
            max_inflight = self.config.max_inflight_per_hotkey
            worker = self.quota.pick_worker(
                requirements=requirements,
                hotkey_filter=lambda hk: self.queue.depth(hk) < max_inflight,
            )
        now = int(time.time())
        sha = graph.graph_id()
        outputs = [self.resolve_output_crypto(ref, worker) for ref in outputs]
        manifest = JobManifestV3(
            job_id=job_id,
            run_id=self.config.run_id,
            step_id=step_id,
            kind=kind,
            graph_ref=GraphRef(sha256=sha, uri=self.bucket.uri_for_key(paths.graph_key(self.config.netuid, self.config.run_id, sha))),
            params=params,
            inputs=inputs,
            outputs=outputs,
            assigned_hotkey=worker.hotkey_ss58,
            assigned_worker=worker.worker_id,
            attempt=0,
            deadline_unix=now + int(self.params.deadline_seconds),
            created_unix=now,
            resource_requirements=requirements,
            verification_policy=VerificationPolicy(critical=kind == "pipe_outer"),
        ).sign(self.config.owner_signer or self.config.owner_secret)
        manifest_uri = self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, self.config.run_id, job_id))
        self.bucket.put_json(manifest_uri, manifest.to_dict())
        grant_uri = self.emit_assignment_grant(manifest)
        self.emitted.append(job_id)
        self.jobs[job_id] = manifest
        self.queue.add(QueueEntry.from_manifest(manifest, manifest_uri=manifest_uri, grant_uri=grant_uri))
        return manifest

    def wait_epoch(self, epoch: int, *, deadline: float, poll_interval: float) -> None:
        """Block until this epoch's tail outputs and (for training) outer weights
        all exist on the bucket.

        Drains the queue concurrently via :meth:`_drain_queue_via_receipts`
        so completed jobs are removed from the published snapshot promptly.
        """
        tail = self.params.n_stages - 1
        released: set[str] = set()
        epoch_prefix = f"j-e{epoch}-"
        while time.time() < deadline:
            self._drain_queue_via_receipts()
            done = sum(
                1
                for mb in range(self.params.n_microbatches)
                if self.bucket.exists(self.fwd_done_uri(epoch, tail, mb))
            )
            outer_done = True
            if self.params.training:
                for stage in self.stages:
                    if (
                        stage.outer_graph is not None
                        and not self.bucket.exists(self.epoch_weights_uri(epoch + 1, stage.stage_id))
                    ):
                        outer_done = False
                        break

            # Walk this epoch's jobs and release legacy QuotaBook reservations
            # for any whose terminal output exists. Queue depth is the
            # authoritative backpressure signal; this is just the priority
            # math feed.
            for job_id, manifest in self.jobs.items():
                if job_id in released:
                    continue
                if not job_id.startswith(epoch_prefix):
                    continue
                if self._job_outputs_present(manifest):
                    self.quota.release(manifest.assigned_hotkey)
                    released.add(job_id)

            if done >= self.params.n_microbatches and outer_done:
                # Drain any remaining unreleased jobs for this epoch.
                for job_id, manifest in self.jobs.items():
                    if job_id in released or not job_id.startswith(epoch_prefix):
                        continue
                    self.quota.release(manifest.assigned_hotkey)
                    released.add(job_id)
                # Final reconcile of the queue so completed work disappears.
                self._drain_queue_via_receipts()
                return
            time.sleep(poll_interval)
        for job_id, manifest in self.jobs.items():
            if job_id in released or not job_id.startswith(epoch_prefix):
                continue
            self.quota.release(manifest.assigned_hotkey)
            released.add(job_id)
        raise TimeoutError(f"streaming epoch {epoch} timed out")

    def _job_outputs_present(self, manifest: JobManifestV3) -> bool:
        """Treat a job as 'done' when ALL of its declared output URIs exist."""
        try:
            for ref in manifest.outputs:
                if not self.bucket.exists(ref.uri):
                    return False
            return True
        except Exception:
            return False

    def _drain_queue_via_receipts(self) -> int:
        """Remove queue entries whose receipts have landed.

        Scans the receipts prefix incrementally (mtime > last cursor) so the
        cost is O(receipts since last drain), not O(receipts in the run).
        Called from ``wait_epoch`` and the stress loop. Also expires
        deadline-passed entries so misbehaving miners don't permanently hold
        their per-hotkey budget.
        """
        # Slight backdate so a clock skew between bucket and orchestrator
        # doesn't cause us to miss a receipt that landed at exactly the
        # cursor time.
        since = max(0.0, self._receipt_scan_cursor - 5.0)
        try:
            done_ids = scan_recent_receipt_job_ids(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                since_unix=since if since > 0 else None,
            )
        except Exception:
            done_ids = set()
        removed = self.queue.remove_many(done_ids) if done_ids else 0
        # Drop deadline-expired entries so per-hotkey backpressure releases.
        expired = self.queue.prune_expired()
        self._receipt_scan_cursor = time.time()
        return removed + len(expired)

    def fwd_output_uri(self, epoch: int, stage_id: int, mb: int, name: str) -> str:
        return self.bucket.uri_for_key(f"runs/{self.config.run_id}/streaming/epoch={epoch}/stage={stage_id}/outputs/mb={mb}/{name}.bin")

    def fwd_done_uri(self, epoch: int, stage_id: int, mb: int) -> str:
        return self.bucket.uri_for_key(f"runs/{self.config.run_id}/streaming/epoch={epoch}/stage={stage_id}/outputs/mb={mb}/done.json")

    def bwd_output_uri(self, epoch: int, stage_id: int, mb: int, name: str) -> str:
        return self.bucket.uri_for_key(f"runs/{self.config.run_id}/streaming/epoch={epoch}/stage={stage_id}/bwd/mb={mb}/{name}.bin")

    def output_ref(self, *, name: str, uri: str) -> ArtifactRef:
        return ArtifactRef(name=name, uri=uri, crypto=self.config.crypto_policy)

    @staticmethod
    def resolve_output_crypto(ref: ArtifactRef, worker: WorkerIdentity) -> ArtifactRef:
        if ref.crypto is None or ref.crypto.required_signer != "assigned_hotkey":
            return ref
        crypto = ArtifactCryptoPolicy.from_dict(ref.crypto.to_dict())
        crypto.required_signer = worker.hotkey_ss58
        return ArtifactRef(name=ref.name, uri=ref.uri, sha256=ref.sha256, size_bytes=ref.size_bytes, crypto=crypto)

    @staticmethod
    def resource_requirements(params: dict) -> ResourceRequirements:
        raw = params.get("resource_requirements") or {}
        if not raw and params.get("distributed_runner") == "gpt_tensor_parallel_v1":
            parallelism = dict(params.get("parallelism") or {})
            world_size = int(parallelism.get("tensor_parallel_size") or params.get("tp_world_size") or 1)
            raw = {"min_gpus": world_size, "placement": "single_host", "parallelism": parallelism}
        return ResourceRequirements.from_dict(raw)

    def emit_assignment_grant(self, manifest: JobManifestV3) -> str | None:
        """Write the encrypted assignment grant; return its bucket URI (or None).

        ``None`` is returned when ``grant_mode == "direct"`` (no grant broker)
        so the queue entry's ``grant_uri`` field is left null and the miner
        falls back to direct bucket access. The returned URI lets
        :meth:`emit_job` record the grant location on the queue entry.
        """
        if self.grant_broker is None:
            return None
        now = int(time.time())
        receipt_uri = self.bucket.uri_for_key(
            paths.receipt_key(
                self.config.netuid,
                manifest.run_id,
                manifest.assigned_hotkey,
                manifest.job_id,
                manifest.attempt,
            )
        )
        grant = AssignmentGrantV3(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            assigned_hotkey=manifest.assigned_hotkey,
            input_gets=[self.grant_broker.get_grant(ref.uri, expires_in=self.config.grant_ttl_sec) for ref in manifest.inputs],
            output_puts=[self.grant_broker.put_grant(uri, expires_in=self.config.grant_ttl_sec) for uri in self.output_grant_uris(manifest)],
            receipt_put=self.grant_broker.put_grant(receipt_uri, expires_in=self.config.grant_ttl_sec),
            created_unix=now,
            expires_unix=now + int(self.config.grant_ttl_sec),
        )
        if self.hotkey_resolver is not None:
            hotkey_info = self.hotkey_resolver.resolve(manifest.assigned_hotkey)
            encrypted = self.assignment_crypto.encrypt_for_hotkey(
                grant,
                recipient_hotkey=manifest.assigned_hotkey,
                recipient_uid=hotkey_info.uid,
                metagraph_block=hotkey_info.block,
                metagraph_hash=hotkey_info.metagraph_hash,
                recipient_public_key=hotkey_info.public_key,
            )
        else:
            encrypted = self.assignment_crypto.encrypt_for_hotkey(grant, recipient_hotkey=manifest.assigned_hotkey)
        grant_uri = self.bucket.uri_for_key(paths.assignment_key(self.config.netuid, manifest.run_id, manifest.job_id, manifest.assigned_hotkey))
        self.bucket.put_json(grant_uri, encrypted.to_dict())
        return grant_uri

    @staticmethod
    def output_grant_uris(manifest: JobManifestV3) -> list[str]:
        out = [ref.uri for ref in manifest.outputs]
        sharding = dict(manifest.params.get("output_sharding") or {})
        for ref in manifest.outputs:
            cfg = sharding.get(ref.name)
            if not cfg:
                continue
            base = ref.uri[:-5] if ref.uri.endswith(".json") else ref.uri
            for rank in range(int(cfg.get("world_size", manifest.resource_requirements.min_gpus))):
                out.append(f"{base}.rank{rank}.bin")
        return out
