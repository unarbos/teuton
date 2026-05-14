"""V3 streaming scheduler for GPT-style tasks.

This ports the important v2 streaming idea into the v3 manifest/receipt world:
the task may still use v2 graph builders and v2 artifact URIs internally, but
job assignment, signatures, receipts, and validation are v3-native.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from locus_core import paths
from locus_core.protocol import ArtifactCryptoPolicy, ArtifactRef, AssignmentGrantV3, GraphRef, JobManifestV3, MinerIdentity, VerificationPolicy, WorkerIdentity
from locus_core.wallet_crypto import DevAssignmentCrypto
from locus_runtime.grants import broker_for_mode
from locus_runtime.storage import ObjectStore
from locus_tasks import load_streaming_task
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
    assignment_secret: str = "locus-dev-assignment"


class StreamingRunManager:
    def __init__(self, *, bucket: ObjectStore, config: StreamingRunConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.task = load_streaming_task(config.task)
        self.quota = QuotaBook()
        self.emitted: list[str] = []
        self.jobs: dict[str, JobManifestV3] = {}
        self.grant_broker = broker_for_mode(config.grant_mode, bucket)
        self.assignment_crypto = DevAssignmentCrypto(config.assignment_secret)

    def bootstrap(self) -> None:
        self.task.bootstrap(bucket=self.bucket, run_id=self.config.run_id, max_rounds=self.config.max_epochs)
        stages, params = self.task.build_streaming_inputs(bucket=self.bucket, run_id=self.config.run_id)
        params.max_epochs = self.config.max_epochs
        self.stages = stages
        self.params = params

    def discover_workers(self) -> list[WorkerIdentity]:
        prefix = self.bucket.uri_for_key(paths.miners_prefix(self.config.netuid))
        out: list[WorkerIdentity] = []
        for uri in self.bucket.list(prefix):
            if not uri.endswith("/heartbeat.json"):
                continue
            try:
                data = self.bucket.get_json(uri)
                if data.get("run_id") != self.config.run_id:
                    continue
                out.append(WorkerIdentity.from_dict(data["worker"]))
            except Exception:
                continue
        self.quota.update_workers(
            [MinerIdentity(netuid=self.config.netuid, hotkey_ss58=w.hotkey_ss58) for w in out],
            out,
        )
        return out

    def run_loop(self, *, poll_interval: float = 0.1, timeout_sec: float = 600.0) -> None:
        self.bootstrap()
        deadline = time.time() + timeout_sec
        for epoch in range(self.config.max_epochs):
            while time.time() < deadline and not self.discover_workers():
                time.sleep(poll_interval)
            if time.time() >= deadline:
                raise TimeoutError("no miners available for streaming run")
            self.emit_epoch(epoch)
            self.wait_epoch(epoch, deadline=deadline, poll_interval=poll_interval)

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
        worker = self.quota.pick_worker()
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
        tail = self.params.n_stages - 1
        while time.time() < deadline:
            done = sum(1 for mb in range(self.params.n_microbatches) if self.bucket.exists(self.fwd_done_uri(epoch, tail, mb)))
            outer_done = True
            if self.params.training:
                for stage in self.stages:
                    if stage.outer_graph is not None and not self.bucket.exists(self.epoch_weights_uri(epoch + 1, stage.stage_id)):
                        outer_done = False
                        break
            if done >= self.params.n_microbatches and outer_done:
                return
            time.sleep(poll_interval)
        raise TimeoutError(f"streaming epoch {epoch} timed out")

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
            output_puts=[self.grant_broker.put_grant(ref.uri, expires_in=self.config.grant_ttl_sec) for ref in manifest.outputs],
            receipt_put=self.grant_broker.put_grant(receipt_uri, expires_in=self.config.grant_ttl_sec),
            created_unix=now,
            expires_unix=now + int(self.config.grant_ttl_sec),
        )
        encrypted = self.assignment_crypto.encrypt_for_hotkey(grant, recipient_hotkey=manifest.assigned_hotkey)
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.assignment_key(self.config.netuid, manifest.run_id, manifest.job_id, manifest.assigned_hotkey)),
            encrypted.to_dict(),
        )
