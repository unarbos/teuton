"""Versioned bucket layout for Locus v3."""
from __future__ import annotations


def root(netuid: int) -> str:
    return f"v3/netuid={int(netuid)}"


def run_root(netuid: int, run_id: str) -> str:
    return f"{root(netuid)}/runs/{run_id}"


def graphs_prefix(netuid: int, run_id: str) -> str:
    return f"{run_root(netuid, run_id)}/graphs/"


def graph_key(netuid: int, run_id: str, sha256: str) -> str:
    return f"{graphs_prefix(netuid, run_id)}{sha256}.json"


def state_key(netuid: int, run_id: str) -> str:
    return f"{run_root(netuid, run_id)}/state.json"


def manifest_config_key(netuid: int, run_id: str) -> str:
    return f"{run_root(netuid, run_id)}/manifest/config.json"


def miners_prefix(netuid: int) -> str:
    return f"{root(netuid)}/miners/"


def worker_heartbeat_key(netuid: int, hotkey: str, worker_id: str) -> str:
    return f"{miners_prefix(netuid)}{hotkey}/workers/{worker_id}/heartbeat.json"


def jobs_prefix(netuid: int, run_id: str) -> str:
    return f"{root(netuid)}/jobs/{run_id}/"


def job_manifest_key(netuid: int, run_id: str, job_id: str) -> str:
    return f"{jobs_prefix(netuid, run_id)}{job_id}/manifest.json"


def job_index_key(netuid: int, run_id: str) -> str:
    return f"{jobs_prefix(netuid, run_id)}index.json"


def job_step_index_key(netuid: int, run_id: str, step_id: int) -> str:
    return f"{jobs_prefix(netuid, run_id)}step={step_id}/index.json"


def assignment_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return f"{root(netuid)}/assignments/{run_id}/{job_id}/hotkey={hotkey}.json"


def weights_key(netuid: int, run_id: str, step_id: int, ub: int) -> str:
    return f"{run_root(netuid, run_id)}/weights/step={step_id}/UB-{ub}.bin"


def target_key(netuid: int, run_id: str, step_id: int, ub: int) -> str:
    return f"{run_root(netuid, run_id)}/targets/step={step_id}/cut-{ub}.bin"


def artifact_prefix(
    netuid: int,
    run_id: str,
    job_id: str,
    hotkey: str,
    worker_id: str,
    attempt: int,
) -> str:
    return (
        f"{root(netuid)}/artifacts/{run_id}/{job_id}/"
        f"hotkey={hotkey}/worker={worker_id}/attempt={attempt}/"
    )


def artifact_key(
    netuid: int,
    run_id: str,
    job_id: str,
    hotkey: str,
    worker_id: str,
    attempt: int,
    name: str,
    suffix: str = ".bin",
) -> str:
    return f"{artifact_prefix(netuid, run_id, job_id, hotkey, worker_id, attempt)}{name}{suffix}"


def receipts_prefix(netuid: int, run_id: str) -> str:
    return f"{root(netuid)}/receipts/{run_id}/"


def receipt_key(netuid: int, run_id: str, hotkey: str, job_id: str, attempt: int) -> str:
    return f"{receipts_prefix(netuid, run_id)}hotkey={hotkey}/{job_id}/attempt={attempt}.json"


def verdicts_prefix(netuid: int, run_id: str) -> str:
    return f"{root(netuid)}/verdicts/{run_id}/"


def verdict_key(netuid: int, run_id: str, validator_hotkey: str, receipt_id: str) -> str:
    safe = receipt_id.replace("/", "_").replace(":", "_")
    return f"{verdicts_prefix(netuid, run_id)}validator={validator_hotkey}/{safe}.json"


def scores_key(netuid: int, window_id: str) -> str:
    return f"{root(netuid)}/scores/window={window_id}/scores.json"


def run_prefixes(netuid: int, run_id: str) -> list[str]:
    """All v3 prefixes owned by a run, for teardown and lifecycle tooling."""
    return [
        run_root(netuid, run_id) + "/",
        jobs_prefix(netuid, run_id),
        f"{root(netuid)}/artifacts/{run_id}/",
        receipts_prefix(netuid, run_id),
        verdicts_prefix(netuid, run_id),
    ]
