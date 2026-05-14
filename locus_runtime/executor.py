"""Pure job execution for Locus v3 manifests."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from locus_core.ir import Graph
from locus_core.protocol import ArtifactDigest, ArtifactRef, JobManifestV3, JobReceiptV3, WorkerIdentity
from locus_core.signatures import HmacSigner
from . import tensor_io
from .crypto import decode_envelope, encode_envelope, artifact_digest_from_blob, DrandTimelockProvider
from .eval import evaluate
from .storage import ObjectStore
from .transport import ArtifactTransport, DirectArtifactTransport


class JobExecutor:
    def __init__(
        self,
        *,
        bucket: ObjectStore,
        device: str = "cpu",
        input_cache_mb: int = 1024,
        output_cache_mb: int = 256,
        encryption_secret: str = "locus-dev-encryption",
        timelock_provider: DrandTimelockProvider | None = None,
        transport: ArtifactTransport | None = None,
    ) -> None:
        self.bucket = bucket
        self.device = device
        self._graph_cache: dict[str, Graph] = {}
        self._input_cache: dict[str, torch.Tensor] = {}
        self._input_order: list[str] = []
        self._input_bytes = 0
        self._input_max = input_cache_mb * 1024 * 1024
        self._output_cache: dict[str, torch.Tensor] = {}
        self._output_order: list[str] = []
        self._output_bytes = 0
        self._output_max = output_cache_mb * 1024 * 1024
        self.encryption_secret = encryption_secret
        self.timelock_provider = timelock_provider
        self.transport = transport or DirectArtifactTransport(bucket)

    @staticmethod
    def digest_body(name: str, uri: str, body: bytes, policy=None) -> ArtifactDigest:
        data = artifact_digest_from_blob(name, uri, body, policy)
        return ArtifactDigest(name=name, uri=uri, **data)

    def digest_artifact(self, ref: ArtifactRef) -> ArtifactDigest:
        return self.digest_body(ref.name, ref.uri, self.transport.get(ref.uri), ref.crypto)

    def fetch_graph(self, graph_sha: str, graph_uri: str) -> Graph:
        cached = self._graph_cache.get(graph_sha)
        if cached is not None:
            return cached
        graph = Graph.from_dict(json.loads(self.bucket.get(graph_uri).decode("utf-8")))
        actual = graph.graph_id()
        if actual != graph_sha:
            raise ValueError(f"graph hash mismatch: declared {graph_sha}, computed {actual}")
        self._graph_cache[graph_sha] = graph
        return graph

    def decode_input(self, ref: ArtifactRef, grants: dict[str, Any] | None = None) -> torch.Tensor:
        cached_out = self._output_cache.get(ref.uri)
        if cached_out is not None:
            return cached_out.to(self.device)
        cached_in = self._input_cache.get(ref.uri)
        if cached_in is not None:
            return cached_in.to(self.device)
        body = self.transport.get(ref.uri, (grants or {}).get(ref.uri))
        body = decode_envelope(
            body,
            ref.crypto,
            verifier=HmacSigner("miner-dev-secret"),
            encryption_secret=self.encryption_secret,
            timelock_provider=self.timelock_provider,
        )
        value = tensor_io.decode_tensor(body)
        if self.is_input_cacheable(ref.uri):
            self._cache_input(ref.uri, value)
        return value.to(self.device)

    @staticmethod
    def is_input_cacheable(uri: str) -> bool:
        return "/weights/" in uri or "/static/" in uri or "/data/tokens.bin" in uri

    def _cache_input(self, uri: str, value: torch.Tensor) -> None:
        self._cache_tensor(uri, value.detach().cpu(), self._input_cache, self._input_order, "_input_bytes", self._input_max)

    def _cache_output(self, uri: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self._cache_tensor(uri, value.detach().cpu(), self._output_cache, self._output_order, "_output_bytes", self._output_max)

    def _cache_tensor(
        self,
        uri: str,
        value: torch.Tensor,
        cache: dict[str, torch.Tensor],
        order: list[str],
        size_attr: str,
        max_bytes: int,
    ) -> None:
        size = value.element_size() * value.numel()
        if size > max_bytes // 2:
            return
        if uri in cache:
            old = cache.pop(uri)
            setattr(self, size_attr, getattr(self, size_attr) - old.element_size() * old.numel())
            order.remove(uri)
        while getattr(self, size_attr) + size > max_bytes and order:
            evict = order.pop(0)
            old = cache.pop(evict, None)
            if old is not None:
                setattr(self, size_attr, getattr(self, size_attr) - old.element_size() * old.numel())
        cache[uri] = value
        order.append(uri)
        setattr(self, size_attr, getattr(self, size_attr) + size)

    @staticmethod
    def encode_output(value: Any, *, json_uri: bool = False) -> bytes:
        if json_uri and isinstance(value, torch.Tensor):
            return json.dumps(
                {"value": value.detach().to(torch.float64).cpu().tolist()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        if isinstance(value, torch.Tensor):
            return tensor_io.encode_tensor(value)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        raise TypeError(f"cannot encode output type {type(value).__name__}")

    def execute(
        self,
        manifest: JobManifestV3,
        *,
        worker: WorkerIdentity,
        miner_secret: str,
        fault_mode: str = "",
        fault_rate: float = 1.0,
        grants: dict[str, Any] | None = None,
    ) -> JobReceiptV3:
        t0 = time.time()
        graph = self.fetch_graph(manifest.graph_ref.sha256, manifest.graph_ref.uri)
        cached_refs: list[ArtifactRef] = []
        miss_refs: list[ArtifactRef] = []
        for ref in manifest.inputs:
            if ref.uri in self._input_cache or ref.uri in self._output_cache:
                cached_refs.append(ref)
            else:
                miss_refs.append(ref)
        inputs: dict[str, torch.Tensor] = {}
        for ref in cached_refs:
            inputs[ref.name] = self.decode_input(ref, grants=grants)
        if miss_refs:
            with ThreadPoolExecutor(max_workers=min(len(miss_refs), 8)) as ex:
                values = list(ex.map(lambda r: self.decode_input(r, grants=grants), miss_refs))
            for ref, value in zip(miss_refs, values):
                inputs[ref.name] = value

        t_compute = time.time()
        outputs = evaluate(graph, inputs, manifest.params, bucket=self.bucket, device=self.device)
        self.apply_faults(manifest, outputs, worker.worker_id, fault_mode, fault_rate)
        t_done = time.time()

        put_jobs: list[tuple[str, bytes]] = []
        output_digests: list[ArtifactDigest] = []
        artifact_signer = HmacSigner(miner_secret, identity=worker.hotkey_ss58)
        for ref in manifest.outputs:
            if ref.name not in outputs:
                raise KeyError(f"graph did not produce output {ref.name!r}")
            body = self.encode_output(outputs[ref.name], json_uri=ref.uri.endswith(".json"))
            body = encode_envelope(
                body,
                ref.crypto,
                signer=artifact_signer,
                encryption_secret=self.encryption_secret,
                timelock_provider=self.timelock_provider,
            )
            put_jobs.append((ref.uri, body))
            output_digests.append(self.digest_body(ref.name, ref.uri, body, ref.crypto))
        with ThreadPoolExecutor(max_workers=min(len(put_jobs), 8) or 1) as ex:
            list(ex.map(lambda item: self.transport.put(item[0], item[1], (grants or {}).get(item[0])), put_jobs))
        for ref in manifest.outputs:
            self._cache_output(ref.uri, outputs[ref.name])
        t_final = time.time()

        input_digests = [
            self.digest_body(ref.name, ref.uri, self.transport.get(ref.uri, (grants or {}).get(ref.uri)), ref.crypto)
            for ref in manifest.inputs
        ]
        receipt = JobReceiptV3(
            receipt_id=f"{manifest.run_id}:{manifest.job_id}:{worker.hotkey_ss58}:{worker.worker_id}:{manifest.attempt}",
            manifest_hash=manifest.manifest_hash(),
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            step_id=manifest.step_id,
            kind=manifest.kind,
            worker=worker,
            input_digests=input_digests,
            output_digests=output_digests,
            started_unix=t0,
            finished_unix=t_final,
            compute_sec=t_done - t_compute,
            claimed_bytes_read=sum(d.size_bytes for d in input_digests),
            claimed_bytes_written=sum(d.size_bytes for d in output_digests),
        )
        return receipt.sign(miner_secret)

    @staticmethod
    def apply_faults(
        manifest: JobManifestV3,
        outputs: dict[str, Any],
        worker_id: str,
        fault_mode: str,
        fault_rate: float,
    ) -> None:
        if not fault_mode or fault_mode == "none":
            return
        seed = f"{worker_id}:{manifest.job_id}:{fault_mode}".encode("utf-8")
        active = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") / float(2**64) < fault_rate
        if not active:
            return
        for name, value in list(outputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if fault_mode in ("wrong_output", "skip_compute"):
                g = torch.Generator(device="cpu").manual_seed(
                    int(hashlib.sha256(f"{manifest.job_id}:{name}".encode()).hexdigest(), 16) % (2**31 - 1)
                )
                if value.dtype.is_floating_point:
                    outputs[name] = torch.randn(value.shape, generator=g).to(value.dtype)
                else:
                    outputs[name] = torch.zeros_like(value)
            elif fault_mode == "partial_corrupt":
                corrupted = value.detach().clone()
                flat = corrupted.reshape(-1)
                if flat.numel() > 0:
                    idx = int(hashlib.sha256(f"{manifest.job_id}:{name}:partial".encode()).hexdigest(), 16) % flat.numel()
                    if corrupted.dtype.is_floating_point:
                        flat[idx] = flat[idx] + torch.as_tensor(1.0, dtype=corrupted.dtype)
                    else:
                        flat[idx] = flat[idx] + 1
                outputs[name] = corrupted
