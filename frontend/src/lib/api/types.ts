/**
 * Hand-mirrored TS types for the FastAPI backend in
 * teuton_dashboard/src/teuton_dashboard/models.py.
 *
 * Kept in lockstep manually rather than auto-generated from openapi.json so
 * the build doesn't require a running backend. If you change the pydantic
 * models, mirror the change here.
 */

export interface QueueEntry {
    job_id: string;
    assigned_hotkey: string;
    assigned_worker: string | null;
    manifest_uri: string;
    grant_uri: string | null;
    deadline_unix: number;
    attempt: number;
    created_unix: number;
}

export interface QueueHistoryPoint {
    ts: number;
    depth_total: number;
    at_cap_count: number;
}

export interface QueueSnapshot {
    run_id: string;
    role: string;
    snapshot_unix: number;
    snapshot_id: number;
    depth_total: number;
    depth_by_hotkey: Record<string, number>;
    max_inflight_per_hotkey: number;
    at_cap_count: number;
    at_cap_hotkeys: string[];
    oldest_entry_age_sec: number | null;
    oldest_job_id: string | null;
    outstanding: QueueEntry[];
    history: QueueHistoryPoint[];
}

export interface QueueResponse {
    queue: QueueSnapshot | null;
    meta: Record<string, unknown>;
}

export interface OutstandingJobRow {
    job_id: string;
    kind: string;
    assigned_hotkey: string;
    assigned_worker: string | null;
    attempt: number;
    created_unix: number;
    deadline_unix: number;
    age_sec: number | null;
    deadline_sec: number | null;
    manifest_uri: string | null;
    grant_uri: string | null;
    role: string;
}

export interface CompletedJobRow {
    job_id: string | null;
    kind: string;
    status: 'completed' | 'verified' | 'failed' | string;
    assigned_hotkey: string | null;
    assigned_worker: string | null;
    finished_unix: number;
    started_unix: number | null;
    duration_sec: number | null;
    checked_unix: number | null;
    compute_sec: number;
    bytes_read: number;
    bytes_written: number;
    receipt_id: string;
    verdict: Record<string, unknown> | null;
    audit: Record<string, unknown> | null;
}

export interface JobsSplit {
    outstanding: OutstandingJobRow[];
    completed: CompletedJobRow[];
    audit_outstanding: OutstandingJobRow[];
}

export interface ChainSummary {
    uid: number | null;
    stake: number | null;
    incentive: number | null;
    emission: number | null;
    validator_permit: boolean | null;
    last_update_block: number | null;
    observed_block: number | null;
    blocks_since_last_update: number | null;
    observed_unix: number | null;
}

export interface WorkerRow {
    role: string;
    status: 'live' | 'stale' | string;
    miner: Record<string, unknown>;
    worker: Record<string, unknown>;
    chain: ChainSummary | null;
    last_seen_unix: number | null;
    age_sec: number | null;
    n_receipts: number;
    queue_depth: number;
    queue_cap: number;
    at_cap: boolean;
    sources: string[];
}

export interface Machine {
    host_id: string;
    roles: string[];
    hotkeys: string[];
    workers: WorkerRow[];
    last_seen_unix: number | null;
    age_sec: number | null;
}

export interface RunsResponse {
    runs: string[];
    default_run_id: string;
}

export interface ChainMeta {
    netuid: number | null;
    current_block: number | null;
    tempo: number | null;
    weights_set_rate_limit: number | null;
    observed_unix: number | null;
    error: string | null;
}

export interface IndexerState {
    name: string;
    cursor_json: string | null;
    updated_unix: number | null;
    error: string | null;
}

export interface HealthResponse {
    ok: boolean;
    netuid: number;
    run_id: string | null;
    states: Record<string, IndexerState>;
    chain: ChainMeta | null;
}

export interface SnapshotMeta {
    bucket: string;
    netuid: number;
    run_id: string;
    generated_unix: number;
    max_jobs: number;
    max_inflight_per_hotkey: number;
    heartbeat_ttl_sec: number | null;
    source: string;
    health: HealthResponse | null;
}

export interface SnapshotResponse {
    meta: SnapshotMeta;
    run: { run_id: string };
    queue: QueueSnapshot | null;
    audit_queue: QueueSnapshot | null;
    machines: Machine[];
    jobs: JobsSplit;
}
