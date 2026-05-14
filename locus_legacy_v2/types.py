"""Wire types for the Locus protocol.

These are the on-the-bucket schemas: `JobManifest`, `GraphRef`, `IORef`,
`JobReceipt`, `VerificationVerdict`, `RunState`, `WorkerInfo`. Everything is
JSON-serializable; the `to_dict` / `from_dict` helpers form the canonical wire
codec.

The IR's own types (`Graph`, `Op`, `Ref`, etc.) live in `ir.py`; they share a
similar shape but are kept separate because the IR is shipped as a separate
content-addressed artifact from the manifests.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Graph / artifact references
# --------------------------------------------------------------------------- #


@dataclass
class GraphRef:
    """Content-addressed pointer to a graph stored in the bucket."""

    sha256: str
    uri: str

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "uri": self.uri}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GraphRef":
        return GraphRef(sha256=d["sha256"], uri=d["uri"])


# --------------------------------------------------------------------------- #
# Job-manifest sub-types
# --------------------------------------------------------------------------- #


@dataclass
class IORef:
    """A named input or output URI for a job."""

    name: str
    uri: str
    sha256: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "uri": self.uri}
        if self.sha256 is not None:
            d["sha256"] = self.sha256
        if self.size_bytes is not None:
            d["size_bytes"] = self.size_bytes
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "IORef":
        return IORef(
            name=d["name"],
            uri=d["uri"],
            sha256=d.get("sha256"),
            size_bytes=d.get("size_bytes"),
        )


# --------------------------------------------------------------------------- #
# JobManifest
# --------------------------------------------------------------------------- #


@dataclass
class JobManifest:
    """The wire format of a single unit of work.

    Workers select a job iff `assigned_to == worker_id`. They evaluate
    `graph_ref`'s graph with `inputs` and `params`, writing outputs to the
    URIs declared in `outputs`.
    """

    job_id: str
    run_id: str
    round_id: int
    kind: str  # informational: "forward_pass" | "inner_step" | "reduce" | "outer_step" | "eval"
    graph_ref: GraphRef
    params: dict[str, Any]
    inputs: list[IORef]
    outputs: list[IORef]
    assigned_to: str
    deadline_unix: int
    created_unix: int
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "round_id": self.round_id,
            "kind": self.kind,
            "graph_ref": self.graph_ref.to_dict(),
            "params": dict(self.params),
            "inputs": [r.to_dict() for r in self.inputs],
            "outputs": [r.to_dict() for r in self.outputs],
            "assigned_to": self.assigned_to,
            "deadline_unix": int(self.deadline_unix),
            "created_unix": int(self.created_unix),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JobManifest":
        return JobManifest(
            schema_version=int(d.get("schema_version", 2)),
            job_id=d["job_id"],
            run_id=d["run_id"],
            round_id=int(d["round_id"]),
            kind=d["kind"],
            graph_ref=GraphRef.from_dict(d["graph_ref"]),
            params=dict(d.get("params") or {}),
            inputs=[IORef.from_dict(x) for x in d["inputs"]],
            outputs=[IORef.from_dict(x) for x in d["outputs"]],
            assigned_to=d["assigned_to"],
            deadline_unix=int(d["deadline_unix"]),
            created_unix=int(d["created_unix"]),
        )


# --------------------------------------------------------------------------- #
# Verification / settlement records
# --------------------------------------------------------------------------- #


@dataclass
class ArtifactDigest:
    """Digest and size for one named input/output artifact."""

    name: str
    uri: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArtifactDigest":
        return ArtifactDigest(
            name=d["name"],
            uri=d["uri"],
            sha256=d["sha256"],
            size_bytes=int(d.get("size_bytes", 0)),
        )


@dataclass
class VerificationSpec:
    """How a validator should compare replayed outputs for a job."""

    method: str = "replay_ir_v1"
    comparator: str = "auto"  # "auto" | "exact" | "allclose"
    rtol: float = 1e-3
    atol: float = 1e-4
    max_sample_elements: int = 4096
    sample_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "comparator": self.comparator,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "max_sample_elements": int(self.max_sample_elements),
            "sample_seed": int(self.sample_seed),
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "VerificationSpec":
        d = dict(d or {})
        return VerificationSpec(
            method=d.get("method", "replay_ir_v1"),
            comparator=d.get("comparator", "auto"),
            rtol=float(d.get("rtol", 1e-3)),
            atol=float(d.get("atol", 1e-4)),
            max_sample_elements=int(d.get("max_sample_elements", 4096)),
            sample_seed=int(d.get("sample_seed", 0)),
        )


@dataclass
class JobReceipt:
    """Worker's claim that a manifest was executed and artifacts were written."""

    receipt_id: str
    job_id: str
    run_id: str
    round_id: int
    kind: str
    worker_id: str
    assigned_to: str
    manifest_uri: str
    graph_sha256: str
    input_digests: list[ArtifactDigest]
    output_digests: list[ArtifactDigest]
    started_unix: float
    finished_unix: float
    fetch_inputs_sec: float
    compute_sec: float
    total_sec: float
    gpu_class: str
    device: str
    claimed_compute_sec: float
    claimed_bytes_read: int
    claimed_bytes_written: int
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "receipt_id": self.receipt_id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "round_id": int(self.round_id),
            "kind": self.kind,
            "worker_id": self.worker_id,
            "assigned_to": self.assigned_to,
            "manifest_uri": self.manifest_uri,
            "graph_sha256": self.graph_sha256,
            "input_digests": [d.to_dict() for d in self.input_digests],
            "output_digests": [d.to_dict() for d in self.output_digests],
            "started_unix": float(self.started_unix),
            "finished_unix": float(self.finished_unix),
            "fetch_inputs_sec": float(self.fetch_inputs_sec),
            "compute_sec": float(self.compute_sec),
            "total_sec": float(self.total_sec),
            "gpu_class": self.gpu_class,
            "device": self.device,
            "claimed_compute_sec": float(self.claimed_compute_sec),
            "claimed_bytes_read": int(self.claimed_bytes_read),
            "claimed_bytes_written": int(self.claimed_bytes_written),
            "verification": self.verification.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JobReceipt":
        return JobReceipt(
            schema_version=int(d.get("schema_version", 1)),
            receipt_id=d["receipt_id"],
            job_id=d["job_id"],
            run_id=d["run_id"],
            round_id=int(d["round_id"]),
            kind=d["kind"],
            worker_id=d["worker_id"],
            assigned_to=d.get("assigned_to", d["worker_id"]),
            manifest_uri=d["manifest_uri"],
            graph_sha256=d["graph_sha256"],
            input_digests=[ArtifactDigest.from_dict(x) for x in d.get("input_digests", [])],
            output_digests=[ArtifactDigest.from_dict(x) for x in d.get("output_digests", [])],
            started_unix=float(d.get("started_unix", 0.0)),
            finished_unix=float(d.get("finished_unix", 0.0)),
            fetch_inputs_sec=float(d.get("fetch_inputs_sec", 0.0)),
            compute_sec=float(d.get("compute_sec", 0.0)),
            total_sec=float(d.get("total_sec", 0.0)),
            gpu_class=d.get("gpu_class", "unknown"),
            device=d.get("device", "unknown"),
            claimed_compute_sec=float(d.get("claimed_compute_sec", d.get("compute_sec", 0.0))),
            claimed_bytes_read=int(d.get("claimed_bytes_read", 0)),
            claimed_bytes_written=int(d.get("claimed_bytes_written", 0)),
            verification=VerificationSpec.from_dict(d.get("verification")),
        )


@dataclass
class VerificationVerdict:
    """Validator's signed-ish replay result for one receipt.

    Signatures are intentionally out-of-scope for the local fleet harness; the
    record contains enough stable fields to sign later.
    """

    verdict_id: str
    receipt_id: str
    job_id: str
    run_id: str
    round_id: int
    kind: str
    worker_id: str
    validator_id: str
    status: str  # "pass" | "fail" | "inconclusive"
    reason: str
    checked_unix: float
    replay_compute_sec: float
    comparison: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "verdict_id": self.verdict_id,
            "receipt_id": self.receipt_id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "round_id": int(self.round_id),
            "kind": self.kind,
            "worker_id": self.worker_id,
            "validator_id": self.validator_id,
            "status": self.status,
            "reason": self.reason,
            "checked_unix": float(self.checked_unix),
            "replay_compute_sec": float(self.replay_compute_sec),
            "comparison": dict(self.comparison),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VerificationVerdict":
        return VerificationVerdict(
            schema_version=int(d.get("schema_version", 1)),
            verdict_id=d["verdict_id"],
            receipt_id=d["receipt_id"],
            job_id=d["job_id"],
            run_id=d["run_id"],
            round_id=int(d["round_id"]),
            kind=d["kind"],
            worker_id=d["worker_id"],
            validator_id=d["validator_id"],
            status=d["status"],
            reason=d.get("reason", ""),
            checked_unix=float(d.get("checked_unix", 0.0)),
            replay_compute_sec=float(d.get("replay_compute_sec", 0.0)),
            comparison=dict(d.get("comparison") or {}),
        )


# --------------------------------------------------------------------------- #
# RunState (the orchestrator's monotonic cursor)
# --------------------------------------------------------------------------- #


@dataclass
class RunState:
    run_id: str
    current_round: int = 0
    max_rounds: int | None = None
    completed_rounds: list[int] = field(default_factory=list)
    failed_rounds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "current_round": int(self.current_round),
            "max_rounds": self.max_rounds,
            "completed_rounds": list(self.completed_rounds),
            "failed_rounds": list(self.failed_rounds),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RunState":
        return RunState(
            run_id=d["run_id"],
            current_round=int(d.get("current_round", 0)),
            max_rounds=d.get("max_rounds"),
            completed_rounds=list(d.get("completed_rounds") or []),
            failed_rounds=list(d.get("failed_rounds") or []),
        )


# --------------------------------------------------------------------------- #
# WorkerInfo (heartbeat + capabilities)
# --------------------------------------------------------------------------- #


@dataclass
class WorkerInfo:
    worker_id: str
    last_seen_unix: int
    first_seen_unix: int
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "last_seen_unix": int(self.last_seen_unix),
            "first_seen_unix": int(self.first_seen_unix),
            "capabilities": dict(self.capabilities),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "WorkerInfo":
        return WorkerInfo(
            worker_id=d["worker_id"],
            last_seen_unix=int(d.get("last_seen_unix", 0)),
            first_seen_unix=int(d.get("first_seen_unix", 0)),
            capabilities=dict(d.get("capabilities") or {}),
        )
