"""Bucket key conventions.

All keys are relative to the bucket root. URIs are constructed via
`storage.join_uri(bucket, key)` at call sites that need a fully-qualified
locator.
"""
from __future__ import annotations


def run_root(run_id: str) -> str:
    return f"runs/{run_id}"


def manifest_config_key(run_id: str) -> str:
    return f"{run_root(run_id)}/manifest/config.json"


def manifest_workers_key(run_id: str) -> str:
    return f"{run_root(run_id)}/manifest/workers.json"


def state_key(run_id: str) -> str:
    return f"{run_root(run_id)}/state.json"


def graph_key(run_id: str, sha256: str) -> str:
    return f"{run_root(run_id)}/graphs/{sha256}.json"


def jobs_index_key(run_id: str, round_id: int) -> str:
    return f"{run_root(run_id)}/jobs/round={round_id}/index.json"


def job_manifest_key(run_id: str, round_id: int, job_id: str) -> str:
    return f"{run_root(run_id)}/jobs/round={round_id}/{job_id}.json"


def receipts_prefix(run_id: str) -> str:
    return f"{run_root(run_id)}/receipts/"


def receipt_round_prefix(run_id: str, round_id: int) -> str:
    return f"{receipts_prefix(run_id)}round={round_id}/"


def job_receipt_key(run_id: str, round_id: int, job_id: str, worker_id: str) -> str:
    return f"{receipt_round_prefix(run_id, round_id)}{job_id}/{worker_id}.json"


def verdicts_prefix(run_id: str) -> str:
    return f"{run_root(run_id)}/verdicts/"


def verdict_key(run_id: str, validator_id: str, receipt_id: str) -> str:
    safe_receipt = receipt_id.replace("/", "_")
    return f"{verdicts_prefix(run_id)}validator={validator_id}/{safe_receipt}.json"


def validator_heartbeat_key(run_id: str, validator_id: str) -> str:
    return f"{run_root(run_id)}/manifest/validators/{validator_id}.json"


def verifier_scratch_key(
    run_id: str,
    validator_id: str,
    receipt_id: str,
    output_name: str,
) -> str:
    safe_receipt = receipt_id.replace("/", "_")
    return f"{run_root(run_id)}/verifier_scratch/{validator_id}/{safe_receipt}/{output_name}"


def weights_key(run_id: str, round_id: int, ub: int) -> str:
    return f"{run_root(run_id)}/weights/round={round_id}/UB-{ub}.bin"


def target_key(run_id: str, round_id: int, cut: int, mb: int = 0) -> str:
    """Target tensor for (round, microbatch, cut). `mb` defaults to 0 so
    existing single-microbatch tasks/tests keep their old paths."""
    return f"{run_root(run_id)}/targets/round={round_id}/mb={mb}/cut-{cut}.bin"


def output_delta_key(run_id: str, round_id: int, ub: int, worker_id: str, mb: int = 0) -> str:
    """Per-replica delta for (round, ub, microbatch, worker)."""
    return f"{run_root(run_id)}/outputs/round={round_id}/ub={ub}/mb={mb}/worker={worker_id}/delta.bin"


def output_delta_prefix(run_id: str, round_id: int, ub: int) -> str:
    """All deltas for a UB this round, across all microbatches and workers."""
    return f"{run_root(run_id)}/outputs/round={round_id}/ub={ub}/"


def reduced_delta_key(run_id: str, round_id: int, ub: int) -> str:
    return f"{run_root(run_id)}/reduced/round={round_id}/ub={ub}.bin"


def metrics_key(run_id: str, round_id: int) -> str:
    return f"{run_root(run_id)}/metrics/round={round_id}.json"


def workers_prefix(run_id: str) -> str:
    return f"{run_root(run_id)}/manifest/workers/"


def worker_heartbeat_key(run_id: str, worker_id: str) -> str:
    return f"{run_root(run_id)}/manifest/workers/{worker_id}.json"


def reduce_done_marker_key(run_id: str, round_id: int, ub: int) -> str:
    """Marker that the reduce job has been emitted for (round, ub)."""
    return f"{run_root(run_id)}/reduce_emitted/round={round_id}/ub={ub}.json"


def optim_state_key(run_id: str, round_id: int, ub: int, name: str) -> str:
    """Path for per-UB outer-optimizer state tensors carried across rounds.

    `name` is the state-tensor's identifier (e.g., "m", "v", "ef").
    """
    return f"{run_root(run_id)}/optim_state/round={round_id}/UB-{ub}/{name}.bin"


def static_blob_key(run_id: str, name: str) -> str:
    """Path for run-static tensors: U_k basis, DCT bases, fixed teachers."""
    return f"{run_root(run_id)}/static/{name}.bin"


def grassmann_accum_key(run_id: str, round_id: int) -> str:
    return f"{run_root(run_id)}/grassmann/round={round_id}.bin"
