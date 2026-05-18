"""Pydantic models for every API payload.

Models are the single source of truth for the FastAPI ``response_model=`` (which
drives the OpenAPI schema, which drives the SvelteKit ``openapi-typescript``
codegen). Adding a field here propagates to the frontend's TypeScript types on
the next build.

Names + shapes preserve parity with the legacy ``teuton_core.dashboard_backend``
endpoints so the cutover is a drop-in.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class QueueEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str
    assigned_hotkey: str
    assigned_worker: Optional[str] = None
    manifest_uri: str
    grant_uri: Optional[str] = None
    deadline_unix: int = 0
    attempt: int = 0
    created_unix: int = 0


class QueueHistoryPoint(BaseModel):
    ts: int
    depth_total: int
    at_cap_count: int


class QueueSnapshot(BaseModel):
    """One ``(run_id, role)`` queue view.

    Carries the outstanding entries directly so SSE consumers can render the
    queue panel + outstanding-jobs table without a follow-up bucket read.
    Size scales as ``depth_total`` (bounded by ``max_inflight * n_miners``);
    at typical Teuton scale (8 * ~30) that's ~50KB per snapshot, fine for SSE.
    """

    run_id: str
    role: str
    snapshot_unix: int
    snapshot_id: int
    depth_total: int
    depth_by_hotkey: dict[str, int]
    max_inflight_per_hotkey: int
    at_cap_count: int
    at_cap_hotkeys: list[str]
    oldest_entry_age_sec: Optional[float] = None
    oldest_job_id: Optional[str] = None
    outstanding: list[QueueEntry] = Field(default_factory=list)
    history: list[QueueHistoryPoint] = Field(default_factory=list)


class QueueResponse(BaseModel):
    queue: Optional[QueueSnapshot]
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Jobs (outstanding from queue, completed from receipts)
# ---------------------------------------------------------------------------


class OutstandingJobRow(BaseModel):
    job_id: str
    kind: str = ""
    assigned_hotkey: str
    assigned_worker: Optional[str] = None
    attempt: int = 0
    created_unix: int = 0
    deadline_unix: int = 0
    age_sec: Optional[float] = None
    deadline_sec: Optional[float] = None
    manifest_uri: Optional[str] = None
    grant_uri: Optional[str] = None
    role: str = "train"


class CompletedJobRow(BaseModel):
    job_id: Optional[str] = None
    kind: str = ""
    status: str
    assigned_hotkey: Optional[str] = None
    assigned_worker: Optional[str] = None
    finished_unix: int = 0
    started_unix: Optional[int] = None
    duration_sec: Optional[float] = None
    checked_unix: Optional[int] = None
    compute_sec: float = 0.0
    bytes_read: int = 0
    bytes_written: int = 0
    receipt_id: str
    verdict: Optional[dict[str, Any]] = None
    audit: Optional[dict[str, Any]] = None


class JobsSplit(BaseModel):
    outstanding: list[OutstandingJobRow] = Field(default_factory=list)
    completed: list[CompletedJobRow] = Field(default_factory=list)
    audit_outstanding: list[OutstandingJobRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Miners / discovery
# ---------------------------------------------------------------------------


class ChainSummary(BaseModel):
    uid: Optional[int] = None
    stake: Optional[float] = None
    incentive: Optional[float] = None
    emission: Optional[float] = None
    validator_permit: Optional[bool] = None
    last_update_block: Optional[int] = None
    observed_block: Optional[int] = None
    blocks_since_last_update: Optional[int] = None
    observed_unix: Optional[int] = None


class WorkerRow(BaseModel):
    role: str
    status: str
    miner: dict[str, Any]
    worker: dict[str, Any]
    chain: Optional[ChainSummary] = None
    last_seen_unix: Optional[int] = None
    age_sec: Optional[float] = None
    n_receipts: int = 0
    queue_depth: int = 0
    queue_cap: int = 0
    at_cap: bool = False
    sources: list[str] = Field(default_factory=list)


class Machine(BaseModel):
    host_id: str
    roles: list[str] = Field(default_factory=list)
    hotkeys: list[str] = Field(default_factory=list)
    workers: list[WorkerRow] = Field(default_factory=list)
    last_seen_unix: Optional[int] = None
    age_sec: Optional[float] = None


class DiscoveryRecord(BaseModel):
    miner: dict[str, Any]
    worker: dict[str, Any]
    run_id: Optional[str] = None
    role: str
    last_seen_unix: Optional[int] = None
    age_sec: Optional[float] = None


class DiscoveryResponse(BaseModel):
    meta: dict[str, Any]
    records: list[DiscoveryRecord]


# ---------------------------------------------------------------------------
# Runs / chain / health
# ---------------------------------------------------------------------------


class RunsResponse(BaseModel):
    runs: list[str]
    default_run_id: str = ""


class ChainMeta(BaseModel):
    netuid: Optional[int] = None
    current_block: Optional[int] = None
    tempo: Optional[int] = None
    weights_set_rate_limit: Optional[int] = None
    observed_unix: Optional[int] = None
    error: Optional[str] = None


class ChainMetaResponse(BaseModel):
    chain: Optional[ChainMeta]


class ChainHotkey(BaseModel):
    uid: Optional[int] = None
    hotkey: Optional[str] = None
    stake: Optional[float] = None
    incentive: Optional[float] = None
    emission: Optional[float] = None
    validator_permit: Optional[bool] = None
    last_update_block: Optional[int] = None
    observed_block: Optional[int] = None
    blocks_since_last_update: Optional[int] = None
    observed_unix: Optional[int] = None


class ChainHotkeysResponse(BaseModel):
    run_id: str = "all"
    hotkeys: list[ChainHotkey]


class IndexerState(BaseModel):
    name: str
    cursor_json: Optional[str] = None
    updated_unix: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool = True
    netuid: int
    run_id: Optional[str] = None
    states: dict[str, IndexerState] = Field(default_factory=dict)
    chain: Optional[ChainMeta] = None


# ---------------------------------------------------------------------------
# Snapshot (top-level)
# ---------------------------------------------------------------------------


class SnapshotMeta(BaseModel):
    bucket: str = ""
    netuid: int
    run_id: str = "all"
    generated_unix: int
    max_jobs: int
    max_inflight_per_hotkey: int
    heartbeat_ttl_sec: Optional[float] = None
    source: str = "sqlite"
    health: Optional[HealthResponse] = None


class SnapshotResponse(BaseModel):
    meta: SnapshotMeta
    run: dict[str, str]
    queue: Optional[QueueSnapshot] = None
    audit_queue: Optional[QueueSnapshot] = None
    machines: list[Machine]
    jobs: JobsSplit


# ---------------------------------------------------------------------------
# Single job detail
# ---------------------------------------------------------------------------


class JobDetailResponse(BaseModel):
    meta: dict[str, Any]
    job: Optional[CompletedJobRow] = None
    manifest: Optional[dict[str, Any]] = None
