"""Versioned bucket layout for Teuton v3."""
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


def queue_key(netuid: int, run_id: str, role: str = "train") -> str:
    """Path of the orchestrator-owned outstanding-work queue.

    There is one queue per ``(netuid, run_id, role)``: ``role="train"`` is
    written by the streaming/run-manager orchestrator; ``role="audit"`` is
    written by ``AuditJobManager``. The file body is a small JSON document
    (see :mod:`teuton_runtime.queue` for the schema) containing only the
    currently-outstanding entries, so the file is bounded by active work
    rather than by emission history.

    Lives under ``jobs/`` (not ``runs/``) so it lands inside the
    publicly-readable bucket-policy allowlist (``v3/netuid=*/jobs/*`` and
    ``v3/netuid=*/audits/*/jobs/*``). Miners running without S3 credentials
    therefore read it via the bucket's public GET path.
    """
    if role == "train":
        return f"{jobs_prefix(netuid, run_id)}queue.json"
    if role == "audit":
        return f"{audit_jobs_prefix(netuid, run_id)}queue.json"
    raise ValueError(f"queue_key: unknown role {role!r}")


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


def auditors_prefix(netuid: int) -> str:
    return f"{root(netuid)}/auditors/"


def auditor_heartbeat_key(netuid: int, auditor_hotkey: str, worker_id: str) -> str:
    return f"{auditors_prefix(netuid)}{auditor_hotkey}/workers/{worker_id}/heartbeat.json"


def audit_root(netuid: int, run_id: str) -> str:
    return f"{root(netuid)}/audits/{run_id}"


def audit_jobs_prefix(netuid: int, run_id: str) -> str:
    return f"{audit_root(netuid, run_id)}/jobs/"


def audit_job_manifest_key(netuid: int, run_id: str, job_id: str) -> str:
    return f"{audit_jobs_prefix(netuid, run_id)}{job_id}/manifest.json"


def audit_assignment_key(netuid: int, run_id: str, job_id: str, auditor_hotkey: str) -> str:
    return f"{audit_root(netuid, run_id)}/assignments/{job_id}/hotkey={auditor_hotkey}.json"


def audit_result_key(netuid: int, run_id: str, auditor_hotkey: str, receipt_id: str) -> str:
    safe = receipt_id.replace("/", "_").replace(":", "_")
    return f"{audit_root(netuid, run_id)}/results/auditor={auditor_hotkey}/{safe}.json"


def audit_results_prefix(netuid: int, run_id: str) -> str:
    return f"{audit_root(netuid, run_id)}/results/"


def scores_key(netuid: int, window_id: str) -> str:
    return f"{root(netuid)}/scores/window={window_id}/scores.json"


def run_prefixes(netuid: int, run_id: str) -> list[str]:
    """All v3 prefixes owned by a run, for teardown and lifecycle tooling.

    ``run_root(...) + "/"`` already covers the per-run queue files at
    ``runs/{run_id}/queue/{role}.json`` so we don't list them separately.
    """
    return [
        run_root(netuid, run_id) + "/",
        jobs_prefix(netuid, run_id),
        f"{root(netuid)}/artifacts/{run_id}/",
        receipts_prefix(netuid, run_id),
        verdicts_prefix(netuid, run_id),
        audit_root(netuid, run_id) + "/",
        telemetry_run_prefix(netuid, run_id),
    ]


# -------------------------------------------------------------------
# Telemetry layout (v3/netuid=N/telemetry/<run_id>/<stream>/<unix>.json)
#
# Operator-facing observability stream. Components emit small JSON
# blobs here on top of the existing protocol objects. Readers tail the
# prefix by sorting by LastModified.
# -------------------------------------------------------------------


def telemetry_run_prefix(netuid: int, run_id: str) -> str:
    return f"{root(netuid)}/telemetry/{run_id}/"


def telemetry_stream_prefix(netuid: int, run_id: str, stream: str) -> str:
    return f"{telemetry_run_prefix(netuid, run_id)}{stream}/"


def telemetry_event_key(netuid: int, run_id: str, stream: str, when_unix: int) -> str:
    return f"{telemetry_stream_prefix(netuid, run_id, stream)}{int(when_unix):010d}.json"


def telemetry_chain_key(netuid: int, run_id: str, when_unix: int) -> str:
    return telemetry_event_key(netuid, run_id, "chain", when_unix)


def telemetry_scores_key(netuid: int, run_id: str, when_unix: int) -> str:
    return telemetry_event_key(netuid, run_id, "scores", when_unix)


def telemetry_audits_key(netuid: int, run_id: str, when_unix: int) -> str:
    return telemetry_event_key(netuid, run_id, "audits", when_unix)


def telemetry_orchestrator_key(netuid: int, run_id: str, when_unix: int) -> str:
    return telemetry_event_key(netuid, run_id, "orchestrator", when_unix)


def telemetry_monitor_key(netuid: int, run_id: str, when_unix: int) -> str:
    return telemetry_event_key(netuid, run_id, "monitor", when_unix)
