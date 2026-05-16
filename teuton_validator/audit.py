"""Reusable audit replay computation for validators and auditor jobs."""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from teuton_core.ir import Graph
from teuton_core.protocol import ArtifactDigest, ArtifactRef, AuditResultV3, JobManifestV3, JobReceiptV3
from teuton_core.signatures import HmacSigner, verify_dict
from teuton_runtime import tensor_io
from teuton_runtime.crypto import decode_envelope, DrandTimelockProvider, TimelockPending
from teuton_runtime.distributed_gpt import GPTTensorParallelRunner, TensorParallelContext
from teuton_runtime.eval import evaluate
from teuton_runtime.sharded_tensor import get_sharded_tensor
from teuton_runtime.storage import ObjectStore
from teuton_runtime.transport import ArtifactTransport, DirectArtifactTransport


@dataclass
class AuditReplayConfig:
    owner_secret: str = "owner-dev-secret"
    miner_secret: str = "miner-dev-secret"
    device: str = "cpu"
    max_sample_elements: int = 4096
    encryption_secret: str = "teuton-dev-encryption"
    timelock_provider: DrandTimelockProvider | None = None


class AuditReplayRunner:
    def __init__(
        self,
        *,
        bucket: ObjectStore,
        config: AuditReplayConfig,
        transport: ArtifactTransport | None = None,
        grants: dict[str, Any] | None = None,
    ) -> None:
        self.bucket = bucket
        self.config = config
        self.transport = transport or DirectArtifactTransport(bucket)
        self.grants = grants or {}
        self.graph_cache: dict[str, Graph] = {}

    def run(
        self,
        *,
        receipt_uri: str,
        manifest: JobManifestV3,
        receipt: JobReceiptV3,
        auditor_hotkey: str,
    ) -> AuditResultV3:
        t0 = time.time()
        comparison: dict[str, Any] = {"receipt_uri": receipt_uri, "inputs": {}, "outputs": {}}
        try:
            if not self.verify_manifest_signature(manifest):
                return self.result(receipt, auditor_hotkey, "fail", "bad owner signature", 0.0, comparison, t0)
            if not self.verify_receipt_signature(receipt):
                return self.result(receipt, auditor_hotkey, "fail", "bad miner signature", 0.0, comparison, t0)
            if manifest.manifest_hash() != receipt.manifest_hash:
                return self.result(receipt, auditor_hotkey, "fail", "manifest hash mismatch", 0.0, comparison, t0)
            graph = self.fetch_graph(manifest.graph_ref.sha256, manifest.graph_ref.uri)
            inputs = self.load_inputs(manifest.inputs, receipt.input_digests, comparison, receipt)
            if (manifest.params.get("distributed_runner") or manifest.params.get("runner")) == "gpt_tensor_parallel_v1":
                dev = torch.device(self.config.device)
                outputs = GPTTensorParallelRunner(TensorParallelContext(rank=0, world_size=1, device=dev)).run(inputs, manifest.params)
                comparison["distributed_replay"] = {
                    "mode": "single_rank_functional",
                    "receipt_world_size": receipt.execution.get("world_size"),
                    "sharding_plan_hash": receipt.execution.get("sharding_plan_hash"),
                }
            else:
                outputs = evaluate(graph, inputs, manifest.params, bucket=self.bucket, device=self.config.device)
            replay_compute = time.time() - t0
            ok = True
            reasons: list[str] = []
            for ref in manifest.outputs:
                out_ok, out_cmp = self.compare_output(ref, outputs.get(ref.name), manifest, receipt)
                comparison["outputs"][ref.name] = out_cmp
                if not out_ok:
                    ok = False
                    reasons.append(f"{ref.name}: {out_cmp.get('reason')}")
            status = "pass" if ok else "fail"
            reason = "all outputs matched" if ok else "; ".join(reasons[:4])
            return self.result(receipt, auditor_hotkey, status, reason, replay_compute, comparison, t0)
        except TimelockPending as e:
            comparison["crypto_pending"] = str(e)
            return self.result(receipt, auditor_hotkey, "inconclusive", str(e), 0.0, comparison, t0)
        except ValueError as e:
            comparison["verification_error"] = str(e)
            return self.result(receipt, auditor_hotkey, "fail", str(e), 0.0, comparison, t0)
        except Exception as e:
            comparison["error"] = repr(e)
            return self.result(receipt, auditor_hotkey, "inconclusive", str(e), 0.0, comparison, t0)

    def verify_manifest_signature(self, manifest: JobManifestV3) -> bool:
        if self.config.owner_secret == "skip":
            return True
        if not manifest.owner_signature:
            return False
        return verify_dict(manifest.unsigned_dict(), self.config.owner_secret, manifest.owner_signature)

    def miner_signing_secret(self, receipt: JobReceiptV3) -> str:
        return receipt.worker.hotkey_ss58 if self.config.miner_secret == "hotkey" else self.config.miner_secret

    def verify_receipt_signature(self, receipt: JobReceiptV3) -> bool:
        if not receipt.miner_signature:
            return False
        return verify_dict(receipt.unsigned_dict(), self.miner_signing_secret(receipt), receipt.miner_signature)

    def fetch_graph(self, sha: str, uri: str) -> Graph:
        cached = self.graph_cache.get(sha)
        if cached is not None:
            return cached
        graph = Graph.from_dict(json.loads(self.bucket.get(uri).decode("utf-8")))
        if graph.graph_id() != sha:
            raise ValueError("graph hash mismatch")
        self.graph_cache[sha] = graph
        return graph

    def load_inputs(
        self,
        refs: list[ArtifactRef],
        digests: list[ArtifactDigest],
        comparison: dict[str, Any],
        receipt: JobReceiptV3,
    ) -> dict[str, torch.Tensor]:
        expected = {d.name: d for d in digests}
        inputs: dict[str, torch.Tensor] = {}
        for ref in refs:
            body = self.transport.get(ref.uri, self.grants.get(ref.uri))
            sha = hashlib.sha256(body).hexdigest()
            exp = expected.get(ref.name)
            comparison["inputs"][ref.name] = {
                "sha256": sha,
                "size_bytes": len(body),
                "matches_receipt": exp is None or exp.sha256 == sha or exp.envelope_sha256 == sha,
                "crypto_mode": ref.crypto.mode if ref.crypto else "none",
            }
            if exp is not None and exp.sha256 != sha and exp.envelope_sha256 != sha:
                raise ValueError(f"input changed: {ref.name}")
            if exp is not None and exp.sharded is not None:
                inputs[ref.name] = get_sharded_tensor(self.transport, ref.uri, grants=self.grants, device=self.config.device)
                continue
            body = decode_envelope(
                body,
                ref.crypto,
                verifier=HmacSigner(self.miner_signing_secret(receipt)),
                encryption_secret=self.config.encryption_secret,
                timelock_provider=self.config.timelock_provider,
            )
            inputs[ref.name] = tensor_io.decode_tensor(body).to(self.config.device)
        return inputs

    def compare_output(
        self,
        ref: ArtifactRef,
        expected_value: Any,
        manifest: JobManifestV3,
        receipt: JobReceiptV3,
    ) -> tuple[bool, dict[str, Any]]:
        if expected_value is None:
            return False, {"status": "fail", "reason": "missing replay output"}
        output_digest = next((d for d in receipt.output_digests if d.name == ref.name and d.uri == ref.uri), None)
        if output_digest is not None and output_digest.sharded is not None:
            observed = get_sharded_tensor(self.transport, ref.uri, grants=self.grants, device=self.config.device)
        else:
            observed_body = decode_envelope(
                self.transport.get(ref.uri, self.grants.get(ref.uri)),
                ref.crypto,
                verifier=HmacSigner(self.miner_signing_secret(receipt)),
                encryption_secret=self.config.encryption_secret,
                timelock_provider=self.config.timelock_provider,
            )
            if ref.uri.endswith(".json"):
                observed_json = json.loads(observed_body.decode("utf-8"))
                if "shards" in observed_json and "world_size" in observed_json:
                    observed = get_sharded_tensor(self.transport, ref.uri, grants=self.grants, device=self.config.device)
                else:
                    observed = torch.as_tensor(observed_json.get("value"), device=self.config.device)
            else:
                observed = tensor_io.decode_tensor(observed_body).to(self.config.device)
        if not isinstance(expected_value, torch.Tensor):
            return False, {"status": "fail", "reason": "non-tensor replay output"}
        expected = expected_value.detach().to(self.config.device)
        if list(observed.shape) != list(expected.shape):
            return False, {"status": "fail", "reason": "shape mismatch"}
        policy = manifest.verification_policy
        obs = observed.reshape(-1)
        exp = expected.reshape(-1)
        total = int(obs.numel())
        if 0 < policy.max_sample_elements < total:
            rng = random.Random(policy.sample_seed)
            idx = torch.as_tensor(rng.sample(range(total), policy.max_sample_elements), device=self.config.device)
            obs = obs.index_select(0, idx)
            exp = exp.index_select(0, idx)
        comparator = policy.comparator
        if comparator == "auto":
            comparator = "allclose" if expected.dtype.is_floating_point else "exact"
        if comparator == "exact":
            ok = bool(torch.equal(obs.cpu(), exp.cpu()))
            max_abs = 0.0 if ok else float((obs.to(torch.float32) - exp.to(torch.float32)).abs().max().item())
        else:
            ok = bool(torch.allclose(obs.to(torch.float32), exp.to(torch.float32), rtol=policy.rtol, atol=policy.atol))
            max_abs = 0.0 if obs.numel() == 0 else float((obs.to(torch.float32) - exp.to(torch.float32)).abs().max().item())
        return ok, {
            "status": "pass" if ok else "fail",
            "reason": "matched" if ok else "tensor mismatch",
            "comparator": comparator,
            "checked_elements": int(obs.numel()),
            "total_elements": total,
            "max_abs_error": max_abs,
            "crypto_mode": ref.crypto.mode if ref.crypto else "none",
            "sharded": output_digest.sharded.to_dict() if output_digest and output_digest.sharded else None,
        }

    @staticmethod
    def result(
        receipt: JobReceiptV3,
        auditor_hotkey: str,
        status: str,
        reason: str,
        replay_compute_sec: float,
        comparison: dict[str, Any],
        t0: float,
    ) -> AuditResultV3:
        return AuditResultV3(
            audit_id=f"{auditor_hotkey}:{receipt.receipt_id}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=receipt.run_id,
            miner_hotkey=receipt.worker.hotkey_ss58,
            auditor_hotkey=auditor_hotkey,
            status=status,
            reason=reason,
            replay_compute_sec=replay_compute_sec,
            checked_unix=time.time(),
            comparison=comparison,
        )
