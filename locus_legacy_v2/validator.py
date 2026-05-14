"""Replay-based validators for Locus job receipts.

The first verifier method is intentionally simple: sample completed
JobReceipt records, reload the exact JobManifest, replay its IR graph on one
device, and compare replayed outputs against the worker's committed outputs.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from . import paths, tensor_io
from .eval import evaluate
from .ir import Graph
from .types import (
    ArtifactDigest,
    IORef,
    JobManifest,
    JobReceipt,
    VerificationSpec,
    VerificationVerdict,
)


@dataclass
class LedgerSummary:
    receipts: int
    verdicts: int
    passed: int
    failed: int
    inconclusive: int
    claimed_compute_sec: float
    claimed_bytes_read: int
    claimed_bytes_written: int
    estimated_cu: float
    payable_cu: float
    by_worker: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipts": self.receipts,
            "verdicts": self.verdicts,
            "passed": self.passed,
            "failed": self.failed,
            "inconclusive": self.inconclusive,
            "claimed_compute_sec": self.claimed_compute_sec,
            "claimed_bytes_read": self.claimed_bytes_read,
            "claimed_bytes_written": self.claimed_bytes_written,
            "estimated_cu": self.estimated_cu,
            "payable_cu": self.payable_cu,
            "by_worker": self.by_worker,
        }


class ReplayValidator:
    def __init__(
        self,
        *,
        bucket,
        validator_id: str,
        run_id: str | None = None,
        device: str = "cpu",
        sample_rate: float = 1.0,
        poll_interval: float = 2.0,
        max_sample_elements: int = 4096,
    ) -> None:
        self.bucket = bucket
        self.validator_id = validator_id
        self.run_id = run_id if run_id not in ("", "all", "*") else None
        self.device = device
        self.sample_rate = float(sample_rate)
        self.poll_interval = float(poll_interval)
        self.max_sample_elements = int(max_sample_elements)
        self._graph_cache: dict[str, Graph] = {}

    def loop(self, *, max_jobs: int | None = None, timeout_sec: float | None = None) -> int:
        deadline = time.time() + timeout_sec if timeout_sec is not None else None
        checked = 0
        while True:
            n = self.run_once(max_jobs=None if max_jobs is None else max_jobs - checked)
            checked += n
            if max_jobs is not None and checked >= max_jobs:
                return checked
            if deadline is not None and time.time() >= deadline:
                return checked
            time.sleep(self.poll_interval)

    def run_once(self, *, max_jobs: int | None = None) -> int:
        receipts = self._sample_receipts()
        checked = 0
        for uri, receipt in receipts:
            if max_jobs is not None and checked >= max_jobs:
                break
            if self._has_verdict(receipt):
                continue
            verdict = self.verify_receipt(uri, receipt)
            self.bucket.put_json(
                self.bucket.uri_for_key(
                    paths.verdict_key(receipt.run_id, self.validator_id, receipt.receipt_id)
                ),
                verdict.to_dict(),
            )
            checked += 1
        return checked

    def verify_receipt(self, receipt_uri: str, receipt: JobReceipt) -> VerificationVerdict:
        t0 = time.time()
        comparison: dict[str, Any] = {
            "receipt_uri": receipt_uri,
            "outputs": {},
            "inputs": {},
        }
        try:
            manifest = JobManifest.from_dict(self.bucket.get_json(receipt.manifest_uri))
            if manifest.graph_ref.sha256 != receipt.graph_sha256:
                return self._verdict(receipt, "fail", "manifest graph hash changed", t0, comparison)

            graph = self._fetch_graph(manifest.graph_ref.sha256, manifest.graph_ref.uri)
            inputs = self._load_inputs(manifest.inputs, receipt.input_digests, comparison)
            outputs = evaluate(graph, inputs, manifest.params, bucket=self.bucket, device=self.device)
            t_compute_end = time.time()

            spec = receipt.verification
            if spec.max_sample_elements <= 0:
                spec.max_sample_elements = self.max_sample_elements
            ok = True
            reasons: list[str] = []
            for ref in manifest.outputs:
                if ref.name not in outputs:
                    ok = False
                    reasons.append(f"missing replay output {ref.name}")
                    comparison["outputs"][ref.name] = {"status": "fail", "reason": "missing replay output"}
                    continue
                out_ok, out_cmp = self._compare_output(ref, outputs[ref.name], spec)
                comparison["outputs"][ref.name] = out_cmp
                if not out_ok:
                    ok = False
                    reasons.append(f"{ref.name}: {out_cmp.get('reason', 'mismatch')}")

            status = "pass" if ok else "fail"
            reason = "all outputs matched" if ok else "; ".join(reasons[:4])
            verdict = self._verdict(receipt, status, reason, t0, comparison)
            verdict.replay_compute_sec = t_compute_end - t0
            return verdict
        except Exception as e:
            comparison["error"] = repr(e)
            return self._verdict(receipt, "inconclusive", str(e), t0, comparison)

    def _receipt_uris(self) -> list[str]:
        if self.run_id is not None:
            prefix = self.bucket.uri_for_key(paths.receipts_prefix(self.run_id))
            return [u for u in self.bucket.list(prefix) if u.endswith(".json")]
        prefix = self.bucket.uri_for_key("runs/")
        return [u for u in self.bucket.list(prefix) if "/receipts/" in u and u.endswith(".json")]

    def _sample_receipts(self) -> list[tuple[str, JobReceipt]]:
        out: list[tuple[str, JobReceipt]] = []
        for uri in self._receipt_uris():
            try:
                receipt = JobReceipt.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if self.sample_rate < 1.0:
                h = hashlib.sha256(
                    f"{self.validator_id}:{receipt.receipt_id}".encode("utf-8")
                ).digest()
                x = int.from_bytes(h[:8], "big") / float(2**64)
                if x >= self.sample_rate:
                    continue
            out.append((uri, receipt))
        random.Random(17).shuffle(out)
        return out

    def _has_verdict(self, receipt: JobReceipt) -> bool:
        uri = self.bucket.uri_for_key(
            paths.verdict_key(receipt.run_id, self.validator_id, receipt.receipt_id)
        )
        return self.bucket.exists(uri)

    def _fetch_graph(self, sha: str, uri: str) -> Graph:
        cached = self._graph_cache.get(sha)
        if cached is not None:
            return cached
        graph = Graph.from_dict(json.loads(self.bucket.get(uri).decode("utf-8")))
        actual = graph.graph_id()
        if actual != sha:
            raise ValueError(f"graph hash mismatch: declared {sha}, computed {actual}")
        self._graph_cache[sha] = graph
        return graph

    def _load_inputs(
        self,
        refs: list[IORef],
        digests: list[ArtifactDigest],
        comparison: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        expected = {d.name: d for d in digests}
        inputs: dict[str, torch.Tensor] = {}
        for ref in refs:
            body = self.bucket.get(ref.uri)
            sha = hashlib.sha256(body).hexdigest()
            exp = expected.get(ref.name)
            comparison["inputs"][ref.name] = {
                "uri": ref.uri,
                "sha256": sha,
                "size_bytes": len(body),
                "matches_receipt": exp is None or (exp.sha256 == sha and exp.size_bytes == len(body)),
            }
            if exp is not None and (exp.sha256 != sha or exp.size_bytes != len(body)):
                raise ValueError(f"input digest changed for {ref.name}")
            inputs[ref.name] = tensor_io.decode_tensor(body).to(self.device)
        return inputs

    def _compare_output(
        self,
        ref: IORef,
        expected_value: Any,
        spec: VerificationSpec,
    ) -> tuple[bool, dict[str, Any]]:
        if ref.uri.endswith(".json"):
            return self._compare_json_output(ref, expected_value, spec)
        observed = tensor_io.decode_tensor(self.bucket.get(ref.uri)).to(self.device)
        if not isinstance(expected_value, torch.Tensor):
            return False, {"status": "fail", "reason": "replay output was not a tensor"}
        expected = expected_value.detach().to(self.device)
        self._write_scratch(ref.uri, ref.name, expected)
        return self._compare_tensors(observed, expected, spec)

    def _compare_json_output(
        self,
        ref: IORef,
        expected_value: Any,
        spec: VerificationSpec,
    ) -> tuple[bool, dict[str, Any]]:
        observed = json.loads(self.bucket.get(ref.uri).decode("utf-8"))
        if isinstance(expected_value, torch.Tensor):
            obs_tensor = torch.as_tensor(observed.get("value"), device=self.device)
            exp_tensor = expected_value.detach().to(self.device)
            return self._compare_tensors(obs_tensor, exp_tensor, spec)
        ok = observed == expected_value
        return ok, {
            "status": "pass" if ok else "fail",
            "comparator": "exact_json",
            "reason": "matched" if ok else "json mismatch",
        }

    def _compare_tensors(
        self,
        observed: torch.Tensor,
        expected: torch.Tensor,
        spec: VerificationSpec,
    ) -> tuple[bool, dict[str, Any]]:
        if list(observed.shape) != list(expected.shape):
            return False, {
                "status": "fail",
                "reason": f"shape mismatch observed={list(observed.shape)} expected={list(expected.shape)}",
            }
        if observed.dtype != expected.dtype and spec.comparator == "exact":
            return False, {
                "status": "fail",
                "reason": f"dtype mismatch observed={observed.dtype} expected={expected.dtype}",
            }

        obs = observed.reshape(-1)
        exp = expected.reshape(-1)
        n = int(obs.numel())
        max_sample = min(int(spec.max_sample_elements or self.max_sample_elements), n)
        if 0 < max_sample < n:
            rng = random.Random(spec.sample_seed)
            idx = torch.as_tensor(rng.sample(range(n), max_sample), device=self.device)
            obs = obs.index_select(0, idx)
            exp = exp.index_select(0, idx)

        comparator = spec.comparator
        if comparator == "auto":
            comparator = "allclose" if expected.dtype.is_floating_point else "exact"

        if comparator == "exact":
            ok = bool(torch.equal(obs.cpu(), exp.cpu()))
            max_abs = 0.0 if ok or obs.numel() == 0 else float((obs.to(torch.float32) - exp.to(torch.float32)).abs().max().item())
        else:
            obs_f = obs.to(torch.float32)
            exp_f = exp.to(torch.float32)
            ok = bool(torch.allclose(obs_f, exp_f, rtol=spec.rtol, atol=spec.atol))
            max_abs = 0.0 if obs.numel() == 0 else float((obs_f - exp_f).abs().max().item())

        return ok, {
            "status": "pass" if ok else "fail",
            "comparator": comparator,
            "shape": list(expected.shape),
            "dtype": str(expected.dtype).replace("torch.", ""),
            "checked_elements": int(obs.numel()),
            "total_elements": n,
            "max_abs_error": max_abs,
            "rtol": spec.rtol,
            "atol": spec.atol,
            "reason": "matched" if ok else "tensor mismatch",
        }

    def _write_scratch(self, output_uri: str, output_name: str, value: torch.Tensor) -> None:
        # Scratch output is useful when debugging a failing verifier, but best
        # effort only because validation should not fail on debug artifact PUTs.
        try:
            uri = self.bucket.uri_for_key(
                paths.verifier_scratch_key(
                    self.run_id or "_all",
                    self.validator_id,
                    hashlib.sha256(output_uri.encode("utf-8")).hexdigest(),
                    f"{output_name}.bin",
                )
            )
            self.bucket.put(uri, tensor_io.encode_tensor(value))
        except Exception:
            pass

    def _verdict(
        self,
        receipt: JobReceipt,
        status: str,
        reason: str,
        t0: float,
        comparison: dict[str, Any],
    ) -> VerificationVerdict:
        verdict_id = f"{self.validator_id}:{receipt.receipt_id}"
        return VerificationVerdict(
            verdict_id=verdict_id,
            receipt_id=receipt.receipt_id,
            job_id=receipt.job_id,
            run_id=receipt.run_id,
            round_id=receipt.round_id,
            kind=receipt.kind,
            worker_id=receipt.worker_id,
            validator_id=self.validator_id,
            status=status,
            reason=reason,
            checked_unix=time.time(),
            replay_compute_sec=time.time() - t0,
            comparison=comparison,
        )


def summarize_ledger(bucket, *, run_id: str | None = None) -> LedgerSummary:
    receipt_uris = _list_receipts(bucket, run_id)
    receipts: list[JobReceipt] = []
    for uri in receipt_uris:
        try:
            receipts.append(JobReceipt.from_dict(bucket.get_json(uri)))
        except Exception:
            continue

    verdicts = _list_verdicts(bucket, run_id)
    latest_by_receipt: dict[str, VerificationVerdict] = {}
    for uri in verdicts:
        try:
            v = VerificationVerdict.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        old = latest_by_receipt.get(v.receipt_id)
        if old is None or v.checked_unix > old.checked_unix:
            latest_by_receipt[v.receipt_id] = v

    by_worker: dict[str, dict[str, Any]] = {}
    worker_receipts: dict[str, list[tuple[JobReceipt, float, VerificationVerdict | None]]] = {}
    total_cu = 0.0
    for r in receipts:
        worker = by_worker.setdefault(r.worker_id, {
            "receipts": 0,
            "verdicts": 0,
            "passed": 0,
            "failed": 0,
            "inconclusive": 0,
            "claimed_compute_sec": 0.0,
            "claimed_bytes_read": 0,
            "claimed_bytes_written": 0,
            "estimated_cu": 0.0,
            "payable_cu": 0.0,
            "trust_multiplier": 1.0,
        })
        cu = estimate_receipt_cu(r)
        verdict = latest_by_receipt.get(r.receipt_id)
        if verdict is not None:
            worker["verdicts"] += 1
            worker[{"pass": "passed", "fail": "failed"}.get(verdict.status, "inconclusive")] += 1
        worker["receipts"] += 1
        worker["claimed_compute_sec"] += r.claimed_compute_sec
        worker["claimed_bytes_read"] += r.claimed_bytes_read
        worker["claimed_bytes_written"] += r.claimed_bytes_written
        worker["estimated_cu"] += cu
        worker_receipts.setdefault(r.worker_id, []).append((r, cu, verdict))
        total_cu += cu

    payable_cu = 0.0
    for worker_id, items in worker_receipts.items():
        worker = by_worker[worker_id]
        passed = int(worker["passed"])
        failed = int(worker["failed"])
        inconclusive = int(worker["inconclusive"])
        checked = passed + failed + inconclusive
        if checked == 0:
            trust_multiplier = 1.0
        elif failed > 0:
            # A sampled failure is evidence about the worker, not just the
            # individual job. Discount all unsampled CU from that worker.
            trust_multiplier = max(0.0, (passed - 2.0 * failed) / checked)
        else:
            trust_multiplier = max(0.0, passed / checked - 0.25 * inconclusive / checked)
        worker["trust_multiplier"] = trust_multiplier
        worker_payable = 0.0
        for _r, cu, verdict in items:
            if verdict is not None and verdict.status == "fail":
                job_multiplier = 0.0
            elif verdict is not None and verdict.status == "inconclusive":
                job_multiplier = min(0.5, trust_multiplier)
            else:
                job_multiplier = trust_multiplier
            worker_payable += cu * job_multiplier
        worker["payable_cu"] = worker_payable
        payable_cu += worker_payable

    passed = sum(1 for v in latest_by_receipt.values() if v.status == "pass")
    failed = sum(1 for v in latest_by_receipt.values() if v.status == "fail")
    inconclusive = sum(1 for v in latest_by_receipt.values() if v.status == "inconclusive")
    return LedgerSummary(
        receipts=len(receipts),
        verdicts=len(latest_by_receipt),
        passed=passed,
        failed=failed,
        inconclusive=inconclusive,
        claimed_compute_sec=sum(r.claimed_compute_sec for r in receipts),
        claimed_bytes_read=sum(r.claimed_bytes_read for r in receipts),
        claimed_bytes_written=sum(r.claimed_bytes_written for r in receipts),
        estimated_cu=total_cu,
        payable_cu=payable_cu,
        by_worker=by_worker,
    )


def estimate_receipt_cu(receipt: JobReceipt) -> float:
    # Local harness metric: one CU ~= one GPU-second plus light bandwidth cost.
    bytes_total = receipt.claimed_bytes_read + receipt.claimed_bytes_written
    return float(receipt.claimed_compute_sec) + bytes_total / 1_000_000_000.0


def _list_receipts(bucket, run_id: str | None) -> list[str]:
    if run_id and run_id not in ("all", "*"):
        return [u for u in bucket.list(bucket.uri_for_key(paths.receipts_prefix(run_id))) if u.endswith(".json")]
    return [u for u in bucket.list(bucket.uri_for_key("runs/")) if "/receipts/" in u and u.endswith(".json")]


def _list_verdicts(bucket, run_id: str | None) -> list[str]:
    if run_id and run_id not in ("all", "*"):
        return [u for u in bucket.list(bucket.uri_for_key(paths.verdicts_prefix(run_id))) if u.endswith(".json")]
    return [u for u in bucket.list(bucket.uri_for_key("runs/")) if "/verdicts/" in u and u.endswith(".json")]
