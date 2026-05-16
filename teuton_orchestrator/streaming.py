"""V3 streaming scheduler for GPT-style tasks.

This ports the important v2 streaming idea into the v3 manifest/receipt world:
the task may still use v2 graph builders and v2 artifact URIs internally, but
job assignment, signatures, receipts, and validation are v3-native.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from teuton_core import paths
from teuton_core.metagraph import BtcliMetagraphHotkeyResolver, MetagraphHotkeyResolver
from teuton_core.protocol import ArtifactCryptoPolicy, ArtifactRef, AssignmentGrantV3, GraphRef, JobManifestV3, ResourceRequirements, VerificationPolicy, WorkerIdentity
from teuton_core.telemetry import TelemetryWriter
from teuton_core.wallet_crypto import AssignmentEncryptor, DevAssignmentCrypto, Ed25519SealedBoxAssignmentCrypto
from teuton_runtime.discovery import build_discovery_backend
from teuton_runtime.grants import broker_for_mode
from teuton_runtime.storage import ObjectStore
from teuton_tasks import load_streaming_task
from .scheduler import QuotaBook


@dataclass
class StreamingRunConfig:
    netuid: int
    run_id: str
    task: str = "gpt_pipe"
    max_epochs: int = 1
    owner_secret: str = "owner-dev-secret"
    crypto_policy: ArtifactCryptoPolicy | None = None
    grant_mode: str = "direct"
    grant_ttl_sec: int = 600
    assignment_secret: str = "teuton-dev-assignment"
    assignment_crypto: str = "dev"
    network: str = "finney"
    discovery_backend: str = "bucket"
    discovery_heartbeat_ttl_sec: float | None = 30.0


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

    def bootstrap(self) -> None:
        self.task.bootstrap(bucket=self.bucket, run_id=self.config.run_id, max_rounds=self.config.max_epochs)
        stages, params = self.task.build_streaming_inputs(bucket=self.bucket, run_id=self.config.run_id)
        params.max_epochs = self.config.max_epochs
        self.stages = stages
        self.params = params

    def discover_workers(self) -> list[WorkerIdentity]:
        records = self.discovery.discover_workers()
        workers = [record.worker for record in records]
        self.quota.update_workers([record.miner for record in records], workers)
        return workers

    def run_loop(self, *, poll_interval: float = 0.1, timeout_sec: float = 600.0) -> None:
        self.bootstrap()
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
            try:
                self.wait_epoch(epoch, deadline=deadline, poll_interval=poll_interval)
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

    def emit_epoch(self, epoch: int) -> None:
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
        if self.params.training:
            for mb in range(self.params.n_microbatches):
                for stage in reversed(self.stages):
                    if stage.backward_graph is not None:
                        self.emit_backward(epoch, mb, stage)
            for stage in self.stages:
                if stage.outer_graph is not None:
                    self.emit_outer(epoch, stage)

    def emit_forward(self, epoch: int, mb: int, stage) -> JobManifestV3:
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

    def emit_job(self, *, job_id: str, kind: str, step_id: int, graph, params: dict, inputs: list[ArtifactRef], outputs: list[ArtifactRef]) -> JobManifestV3:
        requirements = self.resource_requirements(params)
        worker = self.quota.pick_worker(requirements=requirements)
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
        ).sign(self.config.owner_secret)
        self.bucket.put_json(self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, self.config.run_id, job_id)), manifest.to_dict())
        self.emit_assignment_grant(manifest)
        self.emitted.append(job_id)
        self.jobs[job_id] = manifest
        self.bucket.put_json(self.bucket.uri_for_key(paths.job_index_key(self.config.netuid, self.config.run_id)), self.emitted)
        step_index = self.bucket.uri_for_key(paths.job_step_index_key(self.config.netuid, self.config.run_id, step_id))
        current = self.bucket.get_json(step_index) if self.bucket.exists(step_index) else []
        if job_id not in current:
            current.append(job_id)
        self.bucket.put_json(step_index, current)
        return manifest

    def wait_epoch(self, epoch: int, *, deadline: float, poll_interval: float) -> None:
        """Block until this epoch's tail outputs and (for training) outer weights
        all exist on the bucket.

        Also releases QuotaBook entries as each emitted job's terminal output
        appears. Prior to this we leaked quota forever, requiring
        TEUTON_BASE_QUOTA=1000 as a workaround.
        """
        tail = self.params.n_stages - 1
        released: set[str] = set()
        epoch_prefix = f"j-e{epoch}-"
        while time.time() < deadline:
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

            # Walk this epoch's jobs and release quota for any whose terminal
            # output exists.
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
                return
            time.sleep(poll_interval)
        # Timeout - release whatever we have so we don't trap quota forever.
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

    def epoch_weights_uri(self, epoch: int, stage_id: int) -> str:
        return self.bucket.uri_for_key(f"runs/{self.config.run_id}/weights/epoch={epoch}/stage_{stage_id}_W.bin")

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

    def emit_assignment_grant(self, manifest: JobManifestV3) -> None:
        if self.grant_broker is None:
            return
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
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.assignment_key(self.config.netuid, manifest.run_id, manifest.job_id, manifest.assigned_hotkey)),
            encrypted.to_dict(),
        )

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
