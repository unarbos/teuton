"""Read-only viewers for the bucket-backed Teuton fleet.

The CLI ``ls`` subcommand group calls into ``fetch_*`` to collect data and
``render_*`` to pretty-print it. Everything reads from the same paths the
dashboard indexer scans, so the output reflects the source of truth, not a
cached projection.

Designed to stay stdlib-only: no ``rich``/``tabulate`` dependency, no HTTP
client, no background threads. One shot, one S3 client per invocation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from teuton_core import paths
from teuton_core.protocol import (
    AuditResultV3,
    JobManifestV3,
    JobReceiptV3,
    VerificationVerdictV3,
)
from teuton_runtime.discovery import scan_bucket_discovery_records
from teuton_runtime.queue import read_queue
from teuton_runtime.storage import ObjectStore


_SS58_HEAD = 5
_SS58_TAIL = 4


def _short_ss58(value: str | None) -> str:
    if not value:
        return "-"
    if len(value) <= _SS58_HEAD + _SS58_TAIL + 3:
        return value
    return f"{value[:_SS58_HEAD]}...{value[-_SS58_TAIL:]}"


def _short_worker(value: str | None) -> str:
    """``{hotkey}-gpu0`` -> ``gpu0``; otherwise truncate from the tail."""
    if not value:
        return "-"
    if "-" in value:
        return value.rsplit("-", 1)[-1]
    return _short_ss58(value)


def age_str(unix_ts: float | int | None, *, now: float | None = None) -> str:
    """Render ``unix_ts`` as ``5m 32s ago``. ``None`` becomes ``-``."""
    if not unix_ts:
        return "-"
    now = now or time.time()
    delta = max(0.0, now - float(unix_ts))
    if delta < 1.0:
        return "now"
    if delta < 60.0:
        return f"{int(delta)}s ago"
    if delta < 3600.0:
        m, s = divmod(int(delta), 60)
        return f"{m}m {s}s ago"
    if delta < 86400.0:
        h, rem = divmod(int(delta), 3600)
        m = rem // 60
        return f"{h}h {m}m ago"
    d, rem = divmod(int(delta), 86400)
    h = rem // 3600
    return f"{d}d {h}h ago"


def _short_age(unix_ts: float | int | None, *, now: float | None = None) -> str:
    """Compact form of ``age_str`` for tight columns (drops the trailing 'ago')."""
    rendered = age_str(unix_ts, now=now)
    if rendered.endswith(" ago"):
        return rendered[:-4]
    return rendered


# ---------------------------------------------------------------------------
# Data row dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunRow:
    run_id: str
    netuid: int
    manifests: int = 0
    receipts: int = 0
    verdicts: int = 0
    miners: int = 0
    latest_manifest_unix: float = 0.0
    latest_receipt_unix: float = 0.0
    latest_verdict_unix: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "netuid": self.netuid,
            "manifests": self.manifests,
            "receipts": self.receipts,
            "verdicts": self.verdicts,
            "miners": self.miners,
            "latest_manifest_unix": self.latest_manifest_unix,
            "latest_receipt_unix": self.latest_receipt_unix,
            "latest_verdict_unix": self.latest_verdict_unix,
        }


@dataclass
class MinerRow:
    hotkey_ss58: str
    worker_id: str
    run_id: str
    host_id: str
    gpu_class: str
    status: str
    last_seen_unix: float
    jobs: int = 0
    receipts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey_ss58": self.hotkey_ss58,
            "worker_id": self.worker_id,
            "run_id": self.run_id,
            "host_id": self.host_id,
            "gpu_class": self.gpu_class,
            "status": self.status,
            "last_seen_unix": self.last_seen_unix,
            "jobs": self.jobs,
            "receipts": self.receipts,
        }


@dataclass
class JobRow:
    job_id: str
    run_id: str
    kind: str
    status: str
    assigned_hotkey: str
    assigned_worker: str | None
    created_unix: int
    deadline_unix: int
    finished_unix: float = 0.0
    checked_unix: float = 0.0
    verdict_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "assigned_hotkey": self.assigned_hotkey,
            "assigned_worker": self.assigned_worker,
            "created_unix": self.created_unix,
            "deadline_unix": self.deadline_unix,
            "finished_unix": self.finished_unix,
            "checked_unix": self.checked_unix,
            "verdict_status": self.verdict_status,
        }


@dataclass
class JobDetail:
    job_id: str
    run_id: str
    manifest: dict[str, Any] | None
    receipts: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    audits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "manifest": self.manifest,
            "receipts": self.receipts,
            "verdicts": self.verdicts,
            "audits": self.audits,
        }


# ---------------------------------------------------------------------------
# Bucket scan helpers (all bounded; no unbounded I/O)
# ---------------------------------------------------------------------------


def _list_keys(bucket: ObjectStore, prefix: str) -> list[str]:
    return list(bucket.list(bucket.uri_for_key(prefix)))


def _safe_get_json(bucket: ObjectStore, uri: str) -> dict | list | None:
    try:
        return bucket.get_json(uri)
    except Exception:
        return None


def _safe_head_mtime(bucket: ObjectStore, uri: str) -> int:
    head = getattr(bucket, "head", None)
    if head is None:
        return 0
    try:
        result = head(uri)
    except Exception:
        return 0
    if not result:
        return 0
    return int(result.get("mtime_unix") or 0)


def _segment_value(uri: str, marker: str) -> str | None:
    for part in uri.split("/"):
        if part.startswith(marker):
            return part[len(marker) :]
    return None


def _run_id_from_uri(uri: str, prefix: str) -> str | None:
    marker = prefix.rstrip("/") + "/"
    if marker not in uri:
        return None
    rest = uri.split(marker, 1)[1]
    return rest.split("/", 1)[0] if rest else None


def _job_id_from_receipt_uri(uri: str) -> str | None:
    parts = uri.split("/")
    for i, part in enumerate(parts):
        if part.startswith("hotkey=") and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _job_id_from_verdict_uri(uri: str) -> str | None:
    parts = uri.split("/")
    if not parts:
        return None
    last = parts[-1]
    if not last.endswith(".json"):
        return None
    name = last[: -len(".json")]
    # Verdict key format from teuton_core/paths.py:
    #   "{verdict_key_prefix}/validator={hk}/{receipt_id_safe}.json"
    # where receipt_id_safe is "{run_id}_{job_id}_{hotkey}_{worker}_{attempt}"
    # (the ":" separators were replaced with "_"). Reconstruct the job_id by
    # taking the second token after splitting on "_".
    bits = name.split("_")
    if len(bits) >= 2:
        return bits[1]
    return None


# ---------------------------------------------------------------------------
# Run-level scan
# ---------------------------------------------------------------------------


def _discover_run_ids(bucket: ObjectStore, *, netuid: int) -> list[str]:
    seen: set[str] = set()
    root = paths.root(netuid)
    for sub in ("jobs", "receipts", "verdicts", "runs", "audits"):
        prefix = f"{root}/{sub}/"
        for uri in _list_keys(bucket, prefix):
            rid = _run_id_from_uri(uri, prefix)
            if rid:
                seen.add(rid)
    return sorted(seen, reverse=True)


def fetch_runs(bucket: ObjectStore, *, netuid: int) -> list[RunRow]:
    rows: list[RunRow] = []
    run_ids = _discover_run_ids(bucket, netuid=netuid)
    miners_by_run = _miners_by_run(bucket, netuid=netuid)
    for rid in run_ids:
        row = RunRow(run_id=rid, netuid=netuid)
        row.miners = miners_by_run.get(rid, 0)
        manifest_uris = _list_keys(bucket, paths.jobs_prefix(netuid, rid))
        for uri in manifest_uris:
            if uri.endswith("/manifest.json"):
                row.manifests += 1
        if manifest_uris:
            row.latest_manifest_unix = _safe_head_mtime(bucket, manifest_uris[-1])
        receipt_uris = _list_keys(bucket, paths.receipts_prefix(netuid, rid))
        receipt_uris = [u for u in receipt_uris if u.endswith(".json")]
        row.receipts = len(receipt_uris)
        if receipt_uris:
            row.latest_receipt_unix = _safe_head_mtime(bucket, receipt_uris[-1])
        verdict_uris = _list_keys(bucket, paths.verdicts_prefix(netuid, rid))
        verdict_uris = [u for u in verdict_uris if u.endswith(".json")]
        row.verdicts = len(verdict_uris)
        if verdict_uris:
            row.latest_verdict_unix = _safe_head_mtime(bucket, verdict_uris[-1])
        rows.append(row)
    rows.sort(
        key=lambda r: max(r.latest_manifest_unix, r.latest_receipt_unix, r.latest_verdict_unix),
        reverse=True,
    )
    return rows


def _miners_by_run(bucket: ObjectStore, *, netuid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in scan_bucket_discovery_records(bucket, netuid=netuid, heartbeat_ttl_sec=None):
        if not record.run_id:
            continue
        out[record.run_id] = out.get(record.run_id, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Miner-level scan
# ---------------------------------------------------------------------------


def _summarise_miner_workloads(
    bucket: ObjectStore, *, netuid: int, run_id: str
) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """Return (jobs_by_(hotkey, worker), receipts_by_hotkey) for one run.

    Outstanding jobs come from the orchestrator's queue snapshot (bounded
    by ``max_inflight * n_miners``); receipts come from the receipts prefix.
    """
    jobs_by_worker: dict[tuple[str, str], int] = {}
    receipts_by_hotkey: dict[str, int] = {}

    state = read_queue(bucket, netuid=netuid, run_id=run_id, role="train")
    if state is not None:
        for entry in state.outstanding:
            hk = entry.assigned_hotkey or "-"
            worker = entry.assigned_worker or "-"
            jobs_by_worker[(hk, worker)] = jobs_by_worker.get((hk, worker), 0) + 1

    for uri in _list_keys(bucket, paths.receipts_prefix(netuid, run_id)):
        if not uri.endswith(".json"):
            continue
        hk = _segment_value(uri, "hotkey=")
        if not hk:
            continue
        receipts_by_hotkey[hk] = receipts_by_hotkey.get(hk, 0) + 1

    return jobs_by_worker, receipts_by_hotkey


def fetch_miners(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str | None = None,
    heartbeat_ttl_sec: float = 30.0,
    include_stale: bool = True,
    limit: int | None = None,
) -> list[MinerRow]:
    records = scan_bucket_discovery_records(
        bucket,
        netuid=netuid,
        run_id=run_id,
        heartbeat_ttl_sec=None,
    )
    now = time.time()
    jobs_by_worker: dict[tuple[str, str], int] = {}
    receipts_by_hotkey: dict[str, int] = {}
    if run_id is not None:
        jobs_by_worker, receipts_by_hotkey = _summarise_miner_workloads(
            bucket, netuid=netuid, run_id=run_id
        )

    rows: list[MinerRow] = []
    for record in records:
        last_seen = float(record.last_seen_unix or 0)
        age = now - last_seen if last_seen else float("inf")
        status = "live" if age <= heartbeat_ttl_sec else "stale"
        if status == "stale" and not include_stale:
            continue
        worker = record.worker
        gpu_class = str(
            worker.capabilities.get("gpu_class")
            or worker.capabilities.get("gpu_model")
            or worker.capabilities.get("gpu_name")
            or worker.capabilities.get("device")
            or "-"
        )
        rows.append(
            MinerRow(
                hotkey_ss58=worker.hotkey_ss58,
                worker_id=worker.worker_id,
                run_id=record.run_id,
                host_id=worker.host_id or "unknown",
                gpu_class=gpu_class,
                status=status,
                last_seen_unix=last_seen,
                jobs=jobs_by_worker.get((worker.hotkey_ss58, worker.worker_id), 0),
                receipts=receipts_by_hotkey.get(worker.hotkey_ss58, 0),
            )
        )

    rows.sort(key=lambda r: (r.status != "live", -r.jobs, -r.last_seen_unix))
    if limit is not None:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------------
# Job-level scan
# ---------------------------------------------------------------------------


def _index_jobs_for_run(bucket: ObjectStore, *, netuid: int, run_id: str) -> list[str]:
    """Job ids currently outstanding on the orchestrator queue.

    For "what jobs were ever emitted", callers also walk receipts/verdicts
    (which are durable). The queue only tells you what's still in flight.
    """
    state = read_queue(bucket, netuid=netuid, run_id=run_id, role="train")
    if state is None:
        return []
    return [entry.job_id for entry in state.outstanding]


def _receipts_for_job(
    bucket: ObjectStore, *, netuid: int, run_id: str, job_id: str
) -> list[tuple[str, JobReceiptV3]]:
    """All receipts (across hotkeys/attempts) that reference ``job_id``."""
    out: list[tuple[str, JobReceiptV3]] = []
    for uri in _list_keys(bucket, paths.receipts_prefix(netuid, run_id)):
        if not uri.endswith(".json"):
            continue
        if _job_id_from_receipt_uri(uri) != job_id:
            continue
        payload = _safe_get_json(bucket, uri)
        if not isinstance(payload, dict):
            continue
        try:
            out.append((uri, JobReceiptV3.from_dict(payload)))
        except Exception:
            continue
    return out


def _verdicts_for_job(
    bucket: ObjectStore, *, netuid: int, run_id: str, job_id: str
) -> list[tuple[str, VerificationVerdictV3]]:
    out: list[tuple[str, VerificationVerdictV3]] = []
    for uri in _list_keys(bucket, paths.verdicts_prefix(netuid, run_id)):
        if not uri.endswith(".json"):
            continue
        if _job_id_from_verdict_uri(uri) != job_id:
            continue
        payload = _safe_get_json(bucket, uri)
        if not isinstance(payload, dict):
            continue
        try:
            out.append((uri, VerificationVerdictV3.from_dict(payload)))
        except Exception:
            continue
    return out


def _audits_for_job(
    bucket: ObjectStore, *, netuid: int, run_id: str, job_id: str
) -> list[tuple[str, AuditResultV3]]:
    out: list[tuple[str, AuditResultV3]] = []
    for uri in _list_keys(bucket, paths.audit_results_prefix(netuid, run_id)):
        if not uri.endswith(".json"):
            continue
        payload = _safe_get_json(bucket, uri)
        if not isinstance(payload, dict):
            continue
        if payload.get("job_id") != job_id:
            continue
        try:
            out.append((uri, AuditResultV3.from_dict(payload)))
        except Exception:
            continue
    return out


def _job_status(
    *,
    has_receipt: bool,
    verdict_status: str,
    deadline_unix: int,
    now: float,
) -> str:
    if verdict_status == "fail":
        return "failed"
    if verdict_status == "pass":
        return "verified"
    if has_receipt:
        return "completed"
    if deadline_unix and now > deadline_unix:
        return "stale"
    return "created"


_POST_EMISSION_STATUSES = {"completed", "verified", "failed", "stale"}


def fetch_jobs(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[JobRow]:
    """Return the most recent ``limit`` jobs for ``run_id`` after filtering.

    Walk strategy depends on the requested status filter:

    - No status, or status == "created": walk the tail of the job index
      (newest emissions first). Cheap when miners are slow.
    - status in {completed, verified, failed, stale}: walk the receipts /
      verdicts prefixes instead so we always surface processed work, not the
      freshest unfinished manifest tail.
    """
    # ``ids`` is the current outstanding queue (bounded). Completed jobs are
    # found via the receipts/verdicts walks below. We never early-return on
    # an empty queue because there may still be useful completed work to show.
    ids = _index_jobs_for_run(bucket, netuid=netuid, run_id=run_id)
    now = time.time()

    post_emission = status in _POST_EMISSION_STATUSES
    sort_by_finished = post_emission

    receipt_window = max(limit * 6, 500)
    verdict_window = max(limit * 6, 500)
    receipt_uris = [u for u in _list_keys(bucket, paths.receipts_prefix(netuid, run_id)) if u.endswith(".json")]
    receipt_uris = receipt_uris[-receipt_window:]
    receipt_by_job: dict[str, dict[str, Any]] = {}
    for uri in receipt_uris:
        jid = _job_id_from_receipt_uri(uri)
        if not jid:
            continue
        payload = _safe_get_json(bucket, uri)
        if not isinstance(payload, dict):
            continue
        finished = float(payload.get("finished_unix") or 0)
        prev = receipt_by_job.get(jid)
        if prev is None or finished >= float(prev.get("finished_unix") or 0):
            receipt_by_job[jid] = payload

    verdict_uris = [u for u in _list_keys(bucket, paths.verdicts_prefix(netuid, run_id)) if u.endswith(".json")]
    verdict_uris = verdict_uris[-verdict_window:]
    verdict_by_job: dict[str, dict[str, Any]] = {}
    for uri in verdict_uris:
        jid = _job_id_from_verdict_uri(uri)
        if not jid:
            continue
        payload = _safe_get_json(bucket, uri)
        if not isinstance(payload, dict):
            continue
        checked = float(payload.get("checked_unix") or 0)
        prev = verdict_by_job.get(jid)
        if prev is None or checked >= float(prev.get("checked_unix") or 0):
            verdict_by_job[jid] = payload

    if post_emission:
        # Order candidate jobs by (finished_unix, checked_unix) descending so
        # we surface the freshest processed work first.
        candidates: list[tuple[float, str]] = []
        for jid, receipt in receipt_by_job.items():
            ts = max(
                float(receipt.get("finished_unix") or 0),
                float((verdict_by_job.get(jid) or {}).get("checked_unix") or 0),
            )
            candidates.append((ts, jid))
        # ``stale`` = jobs with no receipt yet but past their deadline. Only
        # outstanding entries can be stale (a completed job has a receipt by
        # definition), so use the queue snapshot here.
        if status == "stale":
            for jid in ids:
                if jid not in receipt_by_job:
                    candidates.append((0.0, jid))
        candidates.sort(reverse=True)
        ordered_ids: list[str] = [jid for _, jid in candidates]
    else:
        # Default / status="created": surface outstanding queue (newest
        # first), plus any recent receipts that landed but haven't been
        # cleared from the queue yet, so callers see a continuous timeline
        # rather than a hole around the queue->receipt boundary.
        seen: set[str] = set()
        ordered_ids = []
        for jid in reversed(ids):
            if jid in seen:
                continue
            seen.add(jid)
            ordered_ids.append(jid)
        recent_receipt_ids = sorted(
            receipt_by_job.keys(),
            key=lambda j: float(receipt_by_job[j].get("finished_unix") or 0),
            reverse=True,
        )
        for jid in recent_receipt_ids:
            if jid in seen:
                continue
            seen.add(jid)
            ordered_ids.append(jid)

    rows: list[JobRow] = []
    for jid in ordered_ids:
        manifest_uri = bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, jid))
        payload = _safe_get_json(bucket, manifest_uri)
        if not isinstance(payload, dict):
            continue
        try:
            manifest = JobManifestV3.from_dict(payload)
        except Exception:
            continue
        if kind and manifest.kind != kind:
            continue
        receipt = receipt_by_job.get(jid)
        verdict = verdict_by_job.get(jid)
        finished = float((receipt or {}).get("finished_unix") or 0)
        checked = float((verdict or {}).get("checked_unix") or 0)
        verdict_status = (verdict or {}).get("status") or ""
        job_status = _job_status(
            has_receipt=receipt is not None,
            verdict_status=verdict_status,
            deadline_unix=int(manifest.deadline_unix or 0),
            now=now,
        )
        if status and job_status != status:
            continue
        rows.append(
            JobRow(
                job_id=manifest.job_id,
                run_id=manifest.run_id,
                kind=manifest.kind,
                status=job_status,
                assigned_hotkey=manifest.assigned_hotkey,
                assigned_worker=manifest.assigned_worker,
                created_unix=int(manifest.created_unix or 0),
                deadline_unix=int(manifest.deadline_unix or 0),
                finished_unix=finished,
                checked_unix=checked,
                verdict_status=verdict_status,
            )
        )
        if len(rows) >= limit:
            break
    if sort_by_finished:
        rows.sort(key=lambda r: max(r.finished_unix, r.checked_unix, r.created_unix), reverse=True)
    return rows


def fetch_job_detail(
    bucket: ObjectStore, *, netuid: int, run_id: str, job_id: str
) -> JobDetail:
    manifest_payload = _safe_get_json(
        bucket, bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id))
    )
    manifest_dict = manifest_payload if isinstance(manifest_payload, dict) else None
    receipts = [r.to_dict() for _uri, r in _receipts_for_job(bucket, netuid=netuid, run_id=run_id, job_id=job_id)]
    verdicts = [v.to_dict() for _uri, v in _verdicts_for_job(bucket, netuid=netuid, run_id=run_id, job_id=job_id)]
    audits = [a.to_dict() for _uri, a in _audits_for_job(bucket, netuid=netuid, run_id=run_id, job_id=job_id)]
    return JobDetail(
        job_id=job_id,
        run_id=run_id,
        manifest=manifest_dict,
        receipts=receipts,
        verdicts=verdicts,
        audits=audits,
    )


# ---------------------------------------------------------------------------
# Table rendering (stdlib only)
# ---------------------------------------------------------------------------


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "  ".join(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    out_lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    for row in rows:
        out_lines.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out_lines)


def render_runs_table(rows: list[RunRow]) -> str:
    headers = ["RUN_ID", "MINERS", "JOBS", "RCPTS", "VERDICTS", "LAST_RECEIPT", "LAST_VERDICT"]
    table = _format_table(
        headers,
        [
            [
                r.run_id,
                str(r.miners),
                str(r.manifests),
                str(r.receipts),
                str(r.verdicts),
                age_str(r.latest_receipt_unix),
                age_str(r.latest_verdict_unix),
            ]
            for r in rows
        ],
    )
    summary = f"\n{len(rows)} run{'s' if len(rows) != 1 else ''}" if rows else ""
    return table + summary


def render_miners_table(rows: list[MinerRow]) -> str:
    headers = ["HOTKEY", "WORKER", "GPU", "STATUS", "JOBS", "RCPTS", "RUN_ID", "PING"]
    table = _format_table(
        headers,
        [
            [
                _short_ss58(r.hotkey_ss58),
                _short_worker(r.worker_id),
                r.gpu_class,
                r.status,
                str(r.jobs),
                str(r.receipts),
                r.run_id or "-",
                _short_age(r.last_seen_unix),
            ]
            for r in rows
        ],
    )
    if not rows:
        return table
    live = sum(1 for r in rows if r.status == "live")
    stale = len(rows) - live
    return f"{table}\n{len(rows)} miner{'s' if len(rows) != 1 else ''} ({live} live, {stale} stale)"


def render_jobs_table(rows: list[JobRow]) -> str:
    headers = ["JOB_ID", "KIND", "STATUS", "ASSIGNED", "CREATED", "FINISHED", "CHECKED"]
    table = _format_table(
        headers,
        [
            [
                r.job_id,
                r.kind,
                r.status,
                _short_ss58(r.assigned_hotkey),
                age_str(r.created_unix),
                age_str(r.finished_unix) if r.finished_unix else "-",
                age_str(r.checked_unix) if r.checked_unix else "-",
            ]
            for r in rows
        ],
    )
    if not rows:
        return table
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    return f"{table}\n{len(rows)} jobs ({summary})"


def render_job_detail(detail: JobDetail) -> str:
    lines: list[str] = []
    lines.append(f"job_id   : {detail.job_id}")
    lines.append(f"run_id   : {detail.run_id}")
    if detail.manifest is None:
        lines.append("manifest : <missing>")
    else:
        m = detail.manifest
        lines.append("manifest :")
        lines.append(f"  kind         : {m.get('kind')}")
        lines.append(f"  step_id      : {m.get('step_id')}")
        lines.append(f"  assigned     : {m.get('assigned_hotkey')} / {m.get('assigned_worker')}")
        lines.append(f"  created      : {age_str(m.get('created_unix'))}")
        lines.append(f"  deadline     : {age_str(m.get('deadline_unix'))}")
        lines.append(f"  attempt      : {m.get('attempt')}")
        lines.append(f"  inputs       : {len(m.get('inputs') or [])}")
        lines.append(f"  outputs      : {len(m.get('outputs') or [])}")
        lines.append(f"  graph_sha    : {(m.get('graph_ref') or {}).get('sha256')}")
    if detail.receipts:
        lines.append("receipts :")
        for r in detail.receipts:
            worker = r.get("worker") or {}
            lines.append(
                "  - {hk} attempt={a} finished {age} compute={c:.3f}s read={ri}B written={wo}B".format(
                    hk=_short_ss58(worker.get("hotkey_ss58")),
                    a=r.get("attempt", 0),
                    age=age_str(r.get("finished_unix")),
                    c=float(r.get("compute_sec") or 0.0),
                    ri=int(r.get("claimed_bytes_read") or 0),
                    wo=int(r.get("claimed_bytes_written") or 0),
                )
            )
    else:
        lines.append("receipts : none")
    if detail.verdicts:
        lines.append("verdicts :")
        for v in detail.verdicts:
            lines.append(
                "  - {hk} status={st} checked {age} reason={r}".format(
                    hk=_short_ss58(v.get("validator_hotkey")),
                    st=v.get("status"),
                    age=age_str(v.get("checked_unix")),
                    r=(v.get("reason") or "")[:60],
                )
            )
    else:
        lines.append("verdicts : none")
    if detail.audits:
        lines.append("audits :")
        for a in detail.audits:
            lines.append(
                "  - {hk} status={st} checked {age}".format(
                    hk=_short_ss58(a.get("auditor_hotkey")),
                    st=a.get("status"),
                    age=age_str(a.get("checked_unix")),
                )
            )
    else:
        lines.append("audits   : none")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON helpers used by the CLI
# ---------------------------------------------------------------------------


def to_json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        return json.dumps(value.to_dict(), sort_keys=True, indent=2)
    if isinstance(value, list):
        return json.dumps([item.to_dict() if hasattr(item, "to_dict") else item for item in value], sort_keys=True, indent=2)
    return json.dumps(value, sort_keys=True, indent=2)
