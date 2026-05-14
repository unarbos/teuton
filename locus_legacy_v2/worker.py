"""Worker poll loop.

A worker is a stateless interpreter of the IR:

  while not stopped:
      heartbeat()
      job = pick_assigned_job_for_me()
      if job is None: sleep; continue
      graph = fetch_graph(job.graph_ref)
      inputs = {ref.name: decode_tensor(get(ref.uri)) for ref in job.inputs}
      outputs = evaluate(graph, inputs, job.params, bucket=bucket)
      for spec in job.outputs:
          put(spec.uri, encode(outputs[spec.name]))

The worker writes a heartbeat at `manifest/workers/<worker_id>.json` every
poll loop. The orchestrator reads these to populate its pin table.

Eval-job outputs are JSON, not tensors: when an output URI ends in `.json`
we serialize the tensor's values as a JSON document.
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
import random
import socket
import threading
import time
from typing import Any

import torch

from . import config, paths, tensor_io
from .ir import Graph
from .storage import LocalBucket
from .types import ArtifactDigest, IORef, JobManifest, JobReceipt, VerificationSpec, WorkerInfo


log = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        *,
        bucket: LocalBucket,
        run_id: str,
        worker_id: str,
        poll_interval: float | None = None,
        heartbeat_interval: float | None = None,
        capabilities: dict[str, Any] | None = None,
        max_idle_iters: int | None = None,
        device: str = "cpu",
        fault_mode: str | None = None,
        fault_rate: float | None = None,
    ) -> None:
        self.bucket = bucket
        self.run_id = run_id
        self.worker_id = worker_id
        self.poll_interval = (
            poll_interval if poll_interval is not None else config.env_poll_interval()
        )
        self.heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else config.DEFAULT_HEARTBEAT_INTERVAL_SEC
        )
        self.capabilities = dict(capabilities or {})
        # Auto-detect any missing capabilities (rtt to bucket, gpu class, vram).
        # Skipped if the caller already specified them.
        if "gpu_class" not in self.capabilities or "vram_mb" not in self.capabilities:
            try:
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    vram = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                    self.capabilities.setdefault("gpu_class", name.replace("NVIDIA ", "").replace("GeForce ", "").strip())
                    self.capabilities.setdefault("vram_mb", vram)
                    self.capabilities.setdefault("n_gpus", torch.cuda.device_count())
                else:
                    self.capabilities.setdefault("gpu_class", "cpu")
            except Exception:
                self.capabilities.setdefault("gpu_class", "cpu")
        if "rtt_to_bucket_ms" not in self.capabilities:
            try:
                t0 = time.time()
                probe = self.bucket.uri_for_key(
                    f"runs/{self.run_id}/manifest/_rtt_probe_{self.worker_id}.txt"
                )
                self.bucket.put(probe, b"x")
                self.bucket.exists(probe)
                self.bucket.delete(probe)
                self.capabilities["rtt_to_bucket_ms"] = round((time.time() - t0) * 1000.0, 1)
            except Exception:
                self.capabilities["rtt_to_bucket_ms"] = 1000.0
        # Hostname (used by orchestrator's Phase 3.2 co-location: prefer
        # placing adjacent stages of the same microbatch on the same physical
        # box so intermediate activations don't have to round-trip the wire
        # across countries).
        if "hostname" not in self.capabilities:
            try:
                self.capabilities["hostname"] = socket.gethostname()
            except Exception:
                self.capabilities["hostname"] = "unknown"
        self.device = device
        self.max_idle_iters = max_idle_iters
        self.fault_mode = (fault_mode or os.environ.get("LOCUS_WORKER_FAULT_MODE", "")).strip()
        self.fault_rate = float(
            fault_rate if fault_rate is not None
            else os.environ.get("LOCUS_WORKER_FAULT_RATE", "1.0")
        )
        self._stale_outputs: dict[str, Any] = {}

        # GPU self-probe: do a tiny matmul on the requested device. If it
        # fails (the cuda:N-broken-hardware bug we hit on the megafleet),
        # raise so this worker process exits before it heartbeats — the
        # orchestrator then never adds it to the pin table. Without this,
        # one broken GPU bricks every run.
        if device != "cpu":
            try:
                with torch.no_grad():
                    a = torch.randn(8, 8, device=device)
                    b = torch.randn(8, 8, device=device)
                    c = a @ b
                    _ = c.sum().item()
            except Exception as e:
                raise RuntimeError(
                    f"worker {worker_id}: GPU self-probe failed on device "
                    f"{device!r} ({e}); refusing to register"
                ) from e

        self._stop = threading.Event()
        self._first_seen: int | None = None
        self._last_heartbeat: float = 0.0
        self._graph_cache: dict[str, Graph] = {}
        self._idle_iters = 0
        self._in_progress: dict[str, float] = {}
        # LRU cache of recently written outputs (keyed by URI). Lets a
        # subsequent job assigned to this worker (e.g. outer_step right
        # after reduce, when the orchestrator co-locates) skip the GET on
        # `reduced_delta`. Bounded by total bytes, default 256 MB.
        self._output_cache: dict[str, torch.Tensor] = {}
        self._output_cache_order: list[str] = []
        self._output_cache_bytes_max = int(os.environ.get("LOCUS_OUT_CACHE_MB", "256")) * 1024 * 1024
        self._output_cache_bytes = 0

        # Input cache: tensors fetched from S3, keyed by URI. Lets a stage-
        # pinned worker download its weight blob ONCE per epoch instead of
        # once per job (~33 jobs × 84 MB → 1 × 84 MB on a stage-0 worker).
        # We treat any URI containing "/weights/", "/static/", or
        # "/data/tokens.bin" as cacheable; activations and gradients are NOT
        # cached (they're per-mb and we shouldn't pin them).
        self._input_cache: dict[str, torch.Tensor] = {}
        self._input_cache_order: list[str] = []
        self._input_cache_bytes_max = int(os.environ.get("LOCUS_IN_CACHE_MB", "1024")) * 1024 * 1024
        self._input_cache_bytes = 0

        # Write initial heartbeat eagerly so the orchestrator's pin table sees
        # this worker before its loop thread starts. Without this the
        # orchestrator can race ahead and assign all jobs to the first worker
        # that happened to heartbeat (observed under fast in-process tests).
        try:
            self._heartbeat()
        except Exception:
            log.exception("worker %s: initial heartbeat failed", self.worker_id)

    def stop(self) -> None:
        self._stop.set()

    def loop(self) -> None:
        log.info("worker %s starting (run=%s)", self.worker_id, self.run_id)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("worker tick failure")
            if self._stop.is_set():
                break
            time.sleep(self.poll_interval)
        log.info("worker %s stopped", self.worker_id)

    def _heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_interval:
            return
        self._last_heartbeat = now
        if self._first_seen is None:
            self._first_seen = int(now)
        info = WorkerInfo(
            worker_id=self.worker_id,
            last_seen_unix=int(now),
            first_seen_unix=int(self._first_seen),
            capabilities=dict(self.capabilities),
        )
        uri = self.bucket.uri_for_key(
            paths.worker_heartbeat_key(self.run_id, self.worker_id)
        )
        self.bucket.put_json(uri, info.to_dict())

    def _tick(self) -> None:
        self._heartbeat()
        state_uri = self.bucket.uri_for_key(paths.state_key(self.run_id))
        if not self.bucket.exists(state_uri):
            self._idle_iters += 1
            if self.max_idle_iters is not None and self._idle_iters >= self.max_idle_iters:
                self.stop()
            return
        try:
            state = self.bucket.get_json(state_uri)
        except FileNotFoundError:
            self._idle_iters += 1
            return
        cur_round = int(state.get("current_round", 0))
        candidate_rounds = (
            cur_round,
            cur_round + 1,
            cur_round - 1,
            cur_round - 2,
            cur_round - 3,
        )
        for r in candidate_rounds:
            if r < 0:
                continue
            if self._handle_round(r):
                self._idle_iters = 0
                return
        self._idle_iters += 1
        if self.max_idle_iters is not None and self._idle_iters >= self.max_idle_iters:
            self.stop()

    def _handle_round(self, round_id: int) -> bool:
        idx_uri = self.bucket.uri_for_key(paths.jobs_index_key(self.run_id, round_id))
        if not self.bucket.exists(idx_uri):
            return False
        try:
            idx = self.bucket.get_json(idx_uri)
        except FileNotFoundError:
            return False
        # 1F1B-style priority: when both fwd and bwd jobs are ready, prefer
        # backward + outer first. This lets a stage start a bwd of mb_i AS
        # SOON AS the downstream's dL_dx_in for mb_i is available, rather
        # than draining all M fwds first (which is the GPipe bubble).
        # Key idea: bwd jobs RELEASE bubble (they advance the pipeline drain),
        # while fwd jobs FILL the pipeline. Once steady-state, alternating
        # is optimal.
        # We do a stable two-pass scan: first all `bwd` and `outer` jobs in
        # idx order, then all remaining (fwd) jobs.
        bwd_first: list[str] = []
        fwd_after: list[str] = []
        for jid in idx:
            # Job-id convention: "...-bwd", "...-outer", or "...-fwd"
            if "-bwd" in jid or "-outer" in jid or "outer" in jid:
                bwd_first.append(jid)
            else:
                fwd_after.append(jid)
        for jid in (*bwd_first, *fwd_after):
            if not self._maybe_pickup(round_id, jid):
                continue
            return True
        prefix_uri = self.bucket.uri_for_key(
            f"runs/{self.run_id}/jobs/round={round_id}/"
        )
        for u in self.bucket.list(prefix_uri):
            if not u.endswith(".json"):
                continue
            if u.endswith("/index.json"):
                continue
            jid = os.path.splitext(os.path.basename(u))[0]
            if jid in idx:
                continue
            if self._maybe_pickup(round_id, jid):
                return True
        return False

    def _maybe_pickup(self, round_id: int, job_id: str) -> bool:
        manifest_uri = self.bucket.uri_for_key(
            paths.job_manifest_key(self.run_id, round_id, job_id)
        )
        try:
            manifest_dict = self.bucket.get_json(manifest_uri)
        except FileNotFoundError:
            return False
        manifest = JobManifest.from_dict(manifest_dict)
        if manifest.assigned_to != self.worker_id:
            return False
        if all(self.bucket.exists(o.uri) for o in manifest.outputs):
            return False
        if not all(self.bucket.exists(i.uri) for i in manifest.inputs):
            return False
        if manifest.deadline_unix and time.time() > manifest.deadline_unix:
            return False
        last = self._in_progress.get(job_id)
        now = time.time()
        if last is not None and now - last < 1.0:
            return False
        self._in_progress[job_id] = now
        try:
            self._execute(manifest)
        except Exception:
            log.exception("execute failed for job %s", job_id)
        return True

    def _fetch_graph(self, sha: str, uri: str) -> Graph:
        g = self._graph_cache.get(sha)
        if g is not None:
            return g
        body = self.bucket.get(uri).decode("utf-8")
        d = json.loads(body)
        g = Graph.from_dict(d)
        if g.graph_id() != sha:
            raise ValueError(
                f"graph hash mismatch: declared {sha}, computed {g.graph_id()}"
            )
        self._graph_cache[sha] = g
        return g

    def _decode_input(self, ref: IORef) -> torch.Tensor:
        # Output cache: we just wrote this URI (e.g. co-located outer follows
        # bwd on the same worker).
        cached_out = self._output_cache.get(ref.uri)
        if cached_out is not None:
            return cached_out
        # Input cache: weights / U_k / tokens that don't change within an
        # epoch (or across epochs for static blobs).
        cached_in = self._input_cache.get(ref.uri)
        if cached_in is not None:
            return cached_in
        body = self.bucket.get(ref.uri)
        t = tensor_io.decode_tensor(body)
        if self._is_input_cacheable(ref.uri):
            self._cache_input(ref.uri, t)
        return t

    def _is_input_cacheable(self, uri: str) -> bool:
        """URIs that don't change within an epoch (so caching is safe).
        Static blobs (U_k, tokens.bin) never change at all; weights change
        per epoch but a stage-pinned worker only ever needs ONE epoch's
        weight at a time, so caching across jobs in an epoch is correct.

        We also want to invalidate weight cache entries from PRIOR epochs
        if the cache pressure is high — handled by LRU eviction.
        """
        return ("/weights/" in uri or "/static/" in uri
                or "/data/tokens.bin" in uri)

    def _cache_input(self, uri: str, value: torch.Tensor) -> None:
        sz = value.element_size() * value.numel()
        if sz > self._input_cache_bytes_max // 2:
            return
        if uri in self._input_cache:
            old = self._input_cache.pop(uri)
            self._input_cache_bytes -= old.element_size() * old.numel()
            self._input_cache_order.remove(uri)
        while (self._input_cache_bytes + sz > self._input_cache_bytes_max
               and self._input_cache_order):
            evict = self._input_cache_order.pop(0)
            old = self._input_cache.pop(evict, None)
            if old is not None:
                self._input_cache_bytes -= old.element_size() * old.numel()
        self._input_cache[uri] = value
        self._input_cache_order.append(uri)
        self._input_cache_bytes += sz

    def _cache_output(self, uri: str, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        # Approximate byte size (element_size * numel).
        sz = value.element_size() * value.numel()
        if sz > self._output_cache_bytes_max // 2:
            # Single tensor too big to cache without thrashing.
            return
        if uri in self._output_cache:
            old = self._output_cache.pop(uri)
            self._output_cache_bytes -= old.element_size() * old.numel()
            self._output_cache_order.remove(uri)
        # Evict oldest until it fits.
        while self._output_cache_bytes + sz > self._output_cache_bytes_max and self._output_cache_order:
            evict = self._output_cache_order.pop(0)
            old = self._output_cache.pop(evict, None)
            if old is not None:
                self._output_cache_bytes -= old.element_size() * old.numel()
        self._output_cache[uri] = value.detach().cpu()
        self._output_cache_order.append(uri)
        self._output_cache_bytes += sz

    def _encode_output(self, value: Any) -> bytes:
        if isinstance(value, torch.Tensor):
            return tensor_io.encode_tensor(value)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        raise TypeError(f"cannot encode output of type {type(value).__name__}")

    @staticmethod
    def _digest_body(name: str, uri: str, body: bytes) -> ArtifactDigest:
        return ArtifactDigest(
            name=name,
            uri=uri,
            sha256=hashlib.sha256(body).hexdigest(),
            size_bytes=len(body),
        )

    def _digest_artifact(self, ref: IORef) -> ArtifactDigest:
        body = self.bucket.get(ref.uri)
        return self._digest_body(ref.name, ref.uri, body)

    def _fault_is_active(self, manifest: JobManifest) -> bool:
        if not self.fault_mode or self.fault_mode == "none":
            return False
        if self.fault_rate >= 1.0:
            return True
        seed = f"{self.worker_id}:{manifest.job_id}:{self.fault_mode}".encode("utf-8")
        x = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") / float(2**64)
        return x < max(0.0, self.fault_rate)

    def _random_like(self, value: torch.Tensor, manifest: JobManifest, name: str) -> torch.Tensor:
        seed_bytes = hashlib.sha256(
            f"{self.worker_id}:{manifest.job_id}:{name}:fault".encode("utf-8")
        ).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big") % (2**31 - 1)
        g = torch.Generator(device="cpu").manual_seed(seed)
        if value.dtype.is_floating_point:
            return torch.randn(value.shape, generator=g, dtype=torch.float32).to(value.dtype)
        if value.dtype is torch.bool:
            return torch.randint(0, 2, value.shape, generator=g, dtype=torch.uint8).to(torch.bool)
        return torch.randint(0, 127, value.shape, generator=g, dtype=value.dtype)

    def _apply_faults(self, manifest: JobManifest, outputs: dict[str, Any]) -> None:
        """Experiment-only adversarial modes. Disabled by default."""
        if not self._fault_is_active(manifest):
            return
        mode = self.fault_mode
        for name, value in list(outputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if mode in ("wrong_output", "skip_compute"):
                outputs[name] = self._random_like(value, manifest, name)
            elif mode == "partial_corrupt":
                corrupted = value.detach().clone()
                flat = corrupted.reshape(-1)
                if flat.numel() > 0:
                    idx = int(hashlib.sha256(
                        f"{manifest.job_id}:{name}:partial".encode("utf-8")
                    ).hexdigest(), 16) % flat.numel()
                    if corrupted.dtype.is_floating_point:
                        flat[idx] = flat[idx] + torch.as_tensor(1.0, dtype=corrupted.dtype)
                    elif corrupted.dtype is torch.bool:
                        flat[idx] = ~flat[idx]
                    else:
                        flat[idx] = flat[idx] + 1
                outputs[name] = corrupted
            elif mode == "stale_output":
                stale = self._stale_outputs.get(name)
                if isinstance(stale, torch.Tensor) and list(stale.shape) == list(value.shape):
                    outputs[name] = stale.detach().clone().to(value.dtype)
                else:
                    outputs[name] = self._random_like(value, manifest, name)

    def _write_receipt(
        self,
        manifest: JobManifest,
        *,
        t_start: float,
        t_fetch_inputs: float,
        t_compute_start: float,
        t_compute_end: float,
        t_final: float,
        output_digests: list[ArtifactDigest],
    ) -> None:
        try:
            input_digests = [self._digest_artifact(ref) for ref in manifest.inputs]
            compute_sec = t_compute_end - t_compute_start
            claimed_compute_sec = compute_sec
            if self.fault_mode == "timing_lie" and self._fault_is_active(manifest):
                claimed_compute_sec = compute_sec * 10.0 + 30.0
            receipt = JobReceipt(
                receipt_id=f"{manifest.run_id}:{manifest.round_id}:{manifest.job_id}:{self.worker_id}",
                job_id=manifest.job_id,
                run_id=manifest.run_id,
                round_id=manifest.round_id,
                kind=manifest.kind,
                worker_id=self.worker_id,
                assigned_to=manifest.assigned_to,
                manifest_uri=self.bucket.uri_for_key(
                    paths.job_manifest_key(manifest.run_id, manifest.round_id, manifest.job_id)
                ),
                graph_sha256=manifest.graph_ref.sha256,
                input_digests=input_digests,
                output_digests=output_digests,
                started_unix=t_start,
                finished_unix=t_final,
                fetch_inputs_sec=t_compute_start - t_fetch_inputs,
                compute_sec=compute_sec,
                total_sec=t_final - t_start,
                gpu_class=str(self.capabilities.get("gpu_class", "unknown")),
                device=self.device,
                claimed_compute_sec=claimed_compute_sec,
                claimed_bytes_read=sum(d.size_bytes for d in input_digests),
                claimed_bytes_written=sum(d.size_bytes for d in output_digests),
                verification=VerificationSpec(
                    sample_seed=int(hashlib.sha256(manifest.job_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
                ),
            )
            self.bucket.put_json(
                self.bucket.uri_for_key(
                    paths.job_receipt_key(
                        manifest.run_id, manifest.round_id, manifest.job_id, self.worker_id
                    )
                ),
                receipt.to_dict(),
            )
        except Exception:
            log.exception("failed to write receipt for job %s", manifest.job_id)

    def _execute(self, manifest: JobManifest) -> None:
        log.debug("worker %s executing %s", self.worker_id, manifest.job_id)
        t_start = time.time()
        graph = self._fetch_graph(manifest.graph_ref.sha256, manifest.graph_ref.uri)
        inputs: dict[str, torch.Tensor] = {}
        t_fetch_inputs = time.time()
        # Parallelize cache-miss S3 GETs. Cache-hit refs are resolved
        # synchronously (no S3 call) — only refs that need a fresh GET go
        # through the thread pool. Big win when a job has multiple uncached
        # inputs (e.g. a fwd job needs W + tokens + x_in: 3 calls in serial
        # = 3 * S3_RTT, but in parallel = 1 * S3_RTT).
        cached_refs: list[IORef] = []
        miss_refs: list[IORef] = []
        for ref in manifest.inputs:
            if (ref.uri in self._output_cache or ref.uri in self._input_cache):
                cached_refs.append(ref)
            else:
                miss_refs.append(ref)
        for ref in cached_refs:
            inputs[ref.name] = self._decode_input(ref)
        if miss_refs:
            from concurrent.futures import ThreadPoolExecutor
            n_threads = min(len(miss_refs), 8)
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                fetched = list(ex.map(self._decode_input, miss_refs))
            for ref, val in zip(miss_refs, fetched):
                inputs[ref.name] = val
        t_compute_start = time.time()
        from .eval import evaluate
        outputs = evaluate(graph, inputs, manifest.params, bucket=self.bucket, device=self.device)
        self._apply_faults(manifest, outputs)
        t_compute_end = time.time()

        # Encode all outputs SERIALLY (cheap CPU work), then PUT in PARALLEL
        # (S3 latency-bound). Big win when a job has multiple outputs (a bwd
        # job has dW + dL_dx_in + a jobtime stamp = 3 PUTs in serial = 3*RTT
        # ~= 3-4s; in parallel = 1 RTT ~= 1s).
        put_jobs: list[tuple[str, bytes]] = []
        outputs_to_cache: list[tuple[str, Any]] = []
        output_digests: list[ArtifactDigest] = []
        for ref in manifest.outputs:
            if ref.name not in outputs:
                raise KeyError(
                    f"job {manifest.job_id} graph did not produce output {ref.name!r}"
                )
            value = outputs[ref.name]
            if ref.uri.endswith(".json"):
                if isinstance(value, torch.Tensor):
                    body = json.dumps(
                        {"value": value.detach().to(torch.float64).cpu().tolist()},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                else:
                    body = self._encode_output(value)
            else:
                body = self._encode_output(value)
                outputs_to_cache.append((ref.uri, value))
            put_jobs.append((ref.uri, body))
            output_digests.append(self._digest_body(ref.name, ref.uri, body))

        # Also build the jobtime stamp PUT (fire-and-forget alongside outputs).
        t_end_pre_put = time.time()
        try:
            stamp_uri = self.bucket.uri_for_key(
                f"runs/{self.run_id}/jobtimes/round={manifest.round_id}/{manifest.job_id}.json"
            )
            stamp_body = json.dumps({
                "worker_id": self.worker_id,
                "job_id": manifest.job_id,
                "kind": manifest.kind,
                "round": manifest.round_id,
                "start_unix": t_start,
                "end_unix_pre_put": t_end_pre_put,
                "fetch_inputs_sec": round(t_compute_start - t_fetch_inputs, 4),
                "compute_sec": round(t_compute_end - t_compute_start, 4),
                # total_sec patched in below as we know the end time only
                # after the parallel PUTs complete.
                "total_sec": round(t_end_pre_put - t_start, 4),
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")
            put_jobs.append((stamp_uri, stamp_body))
        except Exception:
            log.exception("failed to encode jobtime stamp for %s", manifest.job_id)

        if put_jobs:
            from concurrent.futures import ThreadPoolExecutor
            n_threads = min(len(put_jobs), 8)
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                list(ex.map(lambda j: self.bucket.put(j[0], j[1]), put_jobs))
        t_final = time.time()

        # Cache outputs AFTER puts succeed (so a co-located follow-up job
        # can short-circuit the GET).
        for uri, value in outputs_to_cache:
            self._cache_output(uri, value)
        for name, value in outputs.items():
            if isinstance(value, torch.Tensor):
                self._stale_outputs[name] = value.detach().cpu()

        self._write_receipt(
            manifest,
            t_start=t_start,
            t_fetch_inputs=t_fetch_inputs,
            t_compute_start=t_compute_start,
            t_compute_end=t_compute_end,
            t_final=t_final,
            output_digests=output_digests,
        )

        t_end = time.time()
