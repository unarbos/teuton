"""Read-only bucket snapshots for the Locus visualizer."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from locus_core import paths
from locus_core.protocol import ArtifactRef, AuditResultV3, JobManifestV3, JobReceiptV3, VerificationVerdictV3
from locus_runtime.discovery import (
    BucketMinerObservation,
    DiscoveryRecord,
    derive_miners_from_bucket,
    scan_bucket_discovery_records,
)
from locus_runtime.storage import ObjectStore


@dataclass(frozen=True)
class VisualizerConfig:
    max_jobs: int = 500
    max_artifacts: int = 300
    heartbeat_ttl_sec: float | None = 30.0


def discover_run_ids(bucket: ObjectStore, *, netuid: int, limit: int = 200) -> list[str]:
    """Discover run IDs from heartbeats, job indexes, receipts, and run state."""
    run_ids: set[str] = set()
    for role in ("train", "audit"):
        for record in scan_bucket_discovery_records(bucket, netuid=netuid, role=role, heartbeat_ttl_sec=None):
            if record.run_id:
                run_ids.add(record.run_id)
    root = paths.root(netuid)
    for prefix in (f"{root}/jobs/", f"{root}/receipts/", f"{root}/audits/", f"{root}/runs/"):
        for uri in bucket.list(bucket.uri_for_key(prefix))[: max(limit * 10, limit)]:
            run_id = _run_id_from_uri(uri, prefix)
            if run_id:
                run_ids.add(run_id)
            if len(run_ids) >= limit:
                break
    return sorted(run_ids, reverse=True)[:limit]


def visualizer_snapshot(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    config: VisualizerConfig | None = None,
) -> dict[str, Any]:
    config = config or VisualizerConfig()
    now = time.time()
    machines = _machines(bucket, netuid=netuid, run_id=run_id, config=config, now=now)
    manifests = _load_manifests(bucket, netuid=netuid, run_id=run_id, max_jobs=config.max_jobs)
    receipts = _load_receipts(bucket, netuid=netuid, run_id=run_id)
    verdicts = _load_verdicts(bucket, netuid=netuid, run_id=run_id)
    audit_results = _load_audit_results(bucket, netuid=netuid, run_id=run_id)
    jobs: list[dict[str, Any]] = []
    artifact_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for manifest, role in manifests:
        receipt = receipts.get(manifest.job_id)
        job_verdicts = verdicts.get(manifest.job_id, [])
        job_audits = audit_results.get(manifest.job_id, [])
        status = _job_status(manifest, receipt, job_verdicts, now=now, bucket=bucket)
        job = _job_row(manifest, role=role, status=status, receipt=receipt, verdicts=job_verdicts, audits=job_audits, now=now)
        jobs.append(job)
        worker_key = manifest.assigned_worker or manifest.assigned_hotkey
        edges.append({"kind": "assignment", "from": f"worker:{worker_key}", "to": f"job:{manifest.job_id}"})
        _index_refs(
            artifact_map,
            edges,
            manifest=manifest,
            refs=manifest.inputs,
            direction="input",
            max_artifacts=config.max_artifacts,
            bucket=bucket,
        )
        _index_refs(
            artifact_map,
            edges,
            manifest=manifest,
            refs=manifest.outputs,
            direction="output",
            max_artifacts=config.max_artifacts,
            bucket=bucket,
        )

    jobs.sort(key=lambda j: (j["created_unix"], j["job_id"]))
    artifacts = sorted(artifact_map.values(), key=lambda a: (a["exists"] is False, a["name"], a["uri"]))
    return {
        "meta": {
            "bucket": bucket.bucket,
            "netuid": int(netuid),
            "run_id": run_id,
            "generated_unix": int(now),
            "max_jobs": config.max_jobs,
            "max_artifacts": config.max_artifacts,
            "heartbeat_ttl_sec": config.heartbeat_ttl_sec,
        },
        "run": _run_info(bucket, netuid=netuid, run_id=run_id),
        "machines": machines,
        "jobs": jobs,
        "artifacts": artifacts,
        "edges": edges[: max(config.max_jobs * 6, 100)],
        "summary": _summary(machines=machines, jobs=jobs, artifacts=artifacts),
    }


def job_detail(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    job_id: str,
    config: VisualizerConfig | None = None,
) -> dict[str, Any]:
    snapshot = visualizer_snapshot(bucket, netuid=netuid, run_id=run_id, config=config)
    job = next((j for j in snapshot["jobs"] if j["job_id"] == job_id), None)
    manifest = _find_manifest(bucket, netuid=netuid, run_id=run_id, job_id=job_id)
    return {
        "meta": snapshot["meta"],
        "job": job,
        "manifest": manifest.to_dict() if manifest is not None else None,
        "artifacts": [
            a for a in snapshot["artifacts"]
            if a.get("producer_job") == job_id or job_id in a.get("consumer_jobs", [])
        ],
    }


def artifact_metadata(bucket: ObjectStore, *, uri: str) -> dict[str, Any]:
    head = _head(bucket, uri)
    return {"uri": uri, "exists": head is not None, **(head or {})}


def _machines(bucket: ObjectStore, *, netuid: int, run_id: str, config: VisualizerConfig, now: float) -> list[dict[str, Any]]:
    """Build the miner table by merging two discovery planes.

    1. Live heartbeats from ``v3/netuid=N/{miners,auditors}/.../heartbeat.json``
       (carries gpu/host/rtt capabilities, but only while a worker process is up).
    2. Bucket-derived observations from receipts + manifests (durable; survives a
       miner restart, lets us show miners that were registered + worked on this
       run even if their worker is currently offline).

    Each worker row gets a ``status`` of ``live``/``stale``/``seen``/``assigned``
    so the UI can distinguish "online now" from "did work but offline now" from
    "scheduler assigned but no output yet".
    """
    records: list[DiscoveryRecord] = []
    for role in ("train", "audit"):
        records.extend(
            scan_bucket_discovery_records(
                bucket,
                netuid=netuid,
                run_id=run_id,
                role=role,
                heartbeat_ttl_sec=None,
            )
        )

    derived = sorted(
        derive_miners_from_bucket(bucket, netuid=netuid, run_id=run_id),
        key=lambda o: (o.worker_id is None, o.hotkey_ss58, o.worker_id or ""),
    )

    by_host: dict[str, dict[str, Any]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    ttl = config.heartbeat_ttl_sec

    def _machine(host_id: str) -> dict[str, Any]:
        return by_host.setdefault(
            host_id,
            {
                "host_id": host_id,
                "roles": [],
                "hotkeys": [],
                "workers": [],
                "last_seen_unix": 0,
                "age_sec": 0.0,
            },
        )

    def _bump_machine(machine: dict[str, Any], *, role: str | None, hotkey: str, ts: int) -> None:
        if role and role not in machine["roles"]:
            machine["roles"].append(role)
        if hotkey not in machine["hotkeys"]:
            machine["hotkeys"].append(hotkey)
        machine["last_seen_unix"] = max(machine["last_seen_unix"], int(ts or 0))
        machine["age_sec"] = max(0.0, now - machine["last_seen_unix"])

    for record in records:
        age = max(0.0, now - record.last_seen_unix)
        status = "live" if (ttl is None or age <= ttl) else "stale"
        machine = _machine(record.worker.host_id)
        _bump_machine(machine, role=record.role, hotkey=record.worker.hotkey_ss58, ts=record.last_seen_unix)
        machine["workers"].append(
            {
                "role": record.role,
                "status": status,
                "miner": record.miner.to_dict(),
                "worker": record.worker.to_dict(),
                "last_seen_unix": record.last_seen_unix,
                "age_sec": age,
                "n_jobs": 0,
                "n_receipts": 0,
                "sources": ["heartbeat"],
            }
        )
        seen_pairs.add((record.worker.hotkey_ss58, record.worker.worker_id or ""))

    def _all_workers_for_hotkey(hotkey: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for machine in by_host.values():
            for w in machine["workers"]:
                if w["worker"].get("hotkey_ss58") == hotkey:
                    out.append(w)
        return out

    def _exact_worker(hotkey: str, worker_id: str | None) -> dict[str, Any] | None:
        wid = worker_id or ""
        for w in _all_workers_for_hotkey(hotkey):
            if (w["worker"].get("worker_id") or "") == wid:
                return w
        return None

    def _merge_into(target: dict[str, Any], obs: BucketMinerObservation) -> None:
        target["n_jobs"] += obs.n_jobs
        target["n_receipts"] += obs.n_receipts
        for s in obs.sources:
            if s not in target["sources"]:
                target["sources"].append(s)
        target["last_seen_unix"] = max(target.get("last_seen_unix") or 0, obs.last_seen_unix)
        if obs.last_seen_unix:
            target["age_sec"] = max(0.0, now - target["last_seen_unix"])
        if target["status"] == "assigned" and obs.n_receipts > 0:
            target["status"] = "seen"

    for obs in derived:
        existing = _exact_worker(obs.hotkey_ss58, obs.worker_id)
        if existing is not None:
            _merge_into(existing, obs)
            continue

        if obs.worker_id is None:
            siblings = _all_workers_for_hotkey(obs.hotkey_ss58)
            if siblings:
                _merge_into(siblings[0], obs)
                continue

        host_id = "(offline)"
        machine = _machine(host_id)
        worker_dict = {
            "hotkey_ss58": obs.hotkey_ss58,
            "worker_id": obs.worker_id or "",
            "host_id": host_id,
            "gpu_index": None,
            "session_nonce": "",
            "software_hash": "",
            "device_group": [],
            "capabilities": {},
        }
        miner_dict = {
            "netuid": int(netuid),
            "hotkey_ss58": obs.hotkey_ss58,
            "uid": None,
            "endpoint": None,
            "commitment_hash": None,
            "capabilities": {},
        }
        status = "seen" if obs.n_receipts > 0 else "assigned"
        _bump_machine(machine, role="train", hotkey=obs.hotkey_ss58, ts=obs.last_seen_unix)
        machine["workers"].append(
            {
                "role": "train",
                "status": status,
                "miner": miner_dict,
                "worker": worker_dict,
                "last_seen_unix": obs.last_seen_unix,
                "age_sec": max(0.0, now - obs.last_seen_unix) if obs.last_seen_unix else None,
                "n_jobs": obs.n_jobs,
                "n_receipts": obs.n_receipts,
                "sources": sorted(obs.sources),
            }
        )

    def _machine_sort_key(m: dict[str, Any]) -> tuple[int, str]:
        return (1 if m["host_id"] == "(offline)" else 0, m["host_id"])

    return sorted(by_host.values(), key=_machine_sort_key)


def _load_manifests(bucket: ObjectStore, *, netuid: int, run_id: str, max_jobs: int) -> list[tuple[JobManifestV3, str]]:
    out: list[tuple[JobManifestV3, str]] = []
    seen: set[str] = set()
    for job_id in _job_ids(bucket, paths.job_index_key(netuid, run_id), paths.jobs_prefix(netuid, run_id)):
        manifest = _load_manifest_uri(bucket, bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id)))
        if manifest is not None and manifest.job_id not in seen:
            out.append((manifest, "train"))
            seen.add(manifest.job_id)
        if len(out) >= max_jobs:
            return out
    for job_id in _job_ids(bucket, paths.audit_job_index_key(netuid, run_id), paths.audit_jobs_prefix(netuid, run_id)):
        manifest = _load_manifest_uri(bucket, bucket.uri_for_key(paths.audit_job_manifest_key(netuid, run_id, job_id)))
        if manifest is not None and manifest.job_id not in seen:
            out.append((manifest, "audit"))
            seen.add(manifest.job_id)
        if len(out) >= max_jobs:
            return out
    return out


def _job_ids(bucket: ObjectStore, index_key: str, prefix: str) -> list[str]:
    ids: list[str] = []
    index_uri = bucket.uri_for_key(index_key)
    try:
        if bucket.exists(index_uri):
            ids.extend(str(x) for x in bucket.get_json(index_uri))
    except Exception:
        pass
    if ids:
        return ids
    for uri in bucket.list(bucket.uri_for_key(prefix)):
        if uri.endswith("/manifest.json"):
            ids.append(uri.rsplit("/", 2)[-2])
    return ids


def _load_manifest_uri(bucket: ObjectStore, uri: str) -> JobManifestV3 | None:
    try:
        return JobManifestV3.from_dict(bucket.get_json(uri))
    except Exception:
        return None


def _find_manifest(bucket: ObjectStore, *, netuid: int, run_id: str, job_id: str) -> JobManifestV3 | None:
    for uri in (
        bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id)),
        bucket.uri_for_key(paths.audit_job_manifest_key(netuid, run_id, job_id)),
    ):
        manifest = _load_manifest_uri(bucket, uri)
        if manifest is not None:
            return manifest
    return None


def _load_receipts(bucket: ObjectStore, *, netuid: int, run_id: str) -> dict[str, JobReceiptV3]:
    out: dict[str, JobReceiptV3] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))):
        if not uri.endswith(".json"):
            continue
        try:
            receipt = JobReceiptV3.from_dict(bucket.get_json(uri))
            out[receipt.job_id] = receipt
        except Exception:
            continue
    return out


def _load_verdicts(bucket: ObjectStore, *, netuid: int, run_id: str) -> dict[str, list[VerificationVerdictV3]]:
    out: dict[str, list[VerificationVerdictV3]] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.verdicts_prefix(netuid, run_id))):
        if not uri.endswith(".json"):
            continue
        try:
            verdict = VerificationVerdictV3.from_dict(bucket.get_json(uri))
            out.setdefault(verdict.job_id, []).append(verdict)
        except Exception:
            continue
    return out


def _load_audit_results(bucket: ObjectStore, *, netuid: int, run_id: str) -> dict[str, list[AuditResultV3]]:
    out: dict[str, list[AuditResultV3]] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.audit_results_prefix(netuid, run_id))):
        if not uri.endswith(".json"):
            continue
        try:
            audit = AuditResultV3.from_dict(bucket.get_json(uri))
            out.setdefault(audit.job_id, []).append(audit)
        except Exception:
            continue
    return out


def _job_status(
    manifest: JobManifestV3,
    receipt: JobReceiptV3 | None,
    verdicts: list[VerificationVerdictV3],
    *,
    now: float,
    bucket: ObjectStore,
) -> str:
    if any(v.status == "fail" for v in verdicts):
        return "failed"
    if any(v.status == "pass" for v in verdicts):
        return "verified"
    if receipt is not None:
        return "completed"
    outputs = [_head(bucket, ref.uri) is not None for ref in manifest.outputs]
    if outputs and all(outputs):
        return "outputs_written"
    if now > manifest.deadline_unix:
        return "stale"
    if outputs and any(outputs):
        return "running"
    return "created"


def _job_row(
    manifest: JobManifestV3,
    *,
    role: str,
    status: str,
    receipt: JobReceiptV3 | None,
    verdicts: list[VerificationVerdictV3],
    audits: list[AuditResultV3],
    now: float,
) -> dict[str, Any]:
    started = receipt.started_unix if receipt is not None else None
    finished = receipt.finished_unix if receipt is not None else None
    duration = (finished - started) if started is not None and finished is not None else max(0.0, now - manifest.created_unix)
    return {
        "job_id": manifest.job_id,
        "role": role,
        "kind": manifest.kind,
        "status": status,
        "run_id": manifest.run_id,
        "step_id": manifest.step_id,
        "created_unix": manifest.created_unix,
        "deadline_unix": manifest.deadline_unix,
        "assigned_hotkey": manifest.assigned_hotkey,
        "assigned_worker": manifest.assigned_worker,
        "attempt": manifest.attempt,
        "critical": manifest.verification_policy.critical,
        "input_count": len(manifest.inputs),
        "output_count": len(manifest.outputs),
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": duration,
        "bytes_read": receipt.claimed_bytes_read if receipt is not None else 0,
        "bytes_written": receipt.claimed_bytes_written if receipt is not None else 0,
        "compute_sec": receipt.compute_sec if receipt is not None else 0.0,
        "receipt_id": receipt.receipt_id if receipt is not None else None,
        "verdicts": [v.to_dict() for v in verdicts],
        "audit_results": [a.to_dict() for a in audits],
        "params": dict(manifest.params),
    }


def _index_refs(
    artifact_map: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    manifest: JobManifestV3,
    refs: list[ArtifactRef],
    direction: str,
    max_artifacts: int,
    bucket: ObjectStore,
) -> None:
    for ref in refs:
        item = artifact_map.get(ref.uri)
        if item is None:
            head = _head(bucket, ref.uri) if len(artifact_map) < max_artifacts else None
            item = {
                "name": ref.name,
                "uri": ref.uri,
                "exists": head is not None,
                "size_bytes": (head or {}).get("size_bytes") or ref.size_bytes,
                "mtime_unix": (head or {}).get("mtime_unix"),
                "sha256": ref.sha256,
                "crypto": ref.crypto.to_dict() if ref.crypto is not None else None,
                "producer_job": None,
                "consumer_jobs": [],
            }
            artifact_map[ref.uri] = item
        if direction == "output":
            item["producer_job"] = manifest.job_id
            edges.append({"kind": "artifact_output", "from": f"job:{manifest.job_id}", "to": f"artifact:{ref.uri}"})
        else:
            if manifest.job_id not in item["consumer_jobs"]:
                item["consumer_jobs"].append(manifest.job_id)
            edges.append({"kind": "artifact_input", "from": f"artifact:{ref.uri}", "to": f"job:{manifest.job_id}"})


def _run_info(bucket: ObjectStore, *, netuid: int, run_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": run_id}
    for name, key in (
        ("state", paths.state_key(netuid, run_id)),
        ("config", paths.manifest_config_key(netuid, run_id)),
    ):
        uri = bucket.uri_for_key(key)
        try:
            out[name] = bucket.get_json(uri) if bucket.exists(uri) else None
        except Exception:
            out[name] = None
    return out


def _summary(*, machines: list[dict[str, Any]], jobs: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_worker: dict[str, int] = {}
    audits = {"pending": 0, "pass": 0, "fail": 0}
    for job in jobs:
        by_status[job["status"]] = by_status.get(job["status"], 0) + 1
        by_kind[job["kind"]] = by_kind.get(job["kind"], 0) + 1
        worker = job.get("assigned_worker") or job.get("assigned_hotkey") or "unknown"
        by_worker[worker] = by_worker.get(worker, 0) + 1
        for audit in job.get("audit_results", []):
            if audit.get("status") in ("pass", "fail"):
                audits[audit["status"]] += 1
            else:
                audits["pending"] += 1
    return {
        "machines": len(machines),
        "workers": sum(len(m["workers"]) for m in machines),
        "jobs": len(jobs),
        "in_flight_jobs": sum(1 for j in jobs if j["status"] in {"created", "running", "outputs_written"}),
        "completed_jobs": sum(1 for j in jobs if j["status"] in {"completed", "verified"}),
        "failed_or_stale_jobs": sum(1 for j in jobs if j["status"] in {"failed", "stale"}),
        "bytes_read": sum(int(j.get("bytes_read") or 0) for j in jobs),
        "bytes_written": sum(int(j.get("bytes_written") or 0) for j in jobs),
        "artifacts": len(artifacts),
        "present_artifacts": sum(1 for a in artifacts if a.get("exists")),
        "missing_artifacts": sum(1 for a in artifacts if not a.get("exists")),
        "audits": audits,
        "by_status": by_status,
        "by_kind": by_kind,
        "by_worker": by_worker,
    }


def _head(bucket: ObjectStore, uri: str) -> dict[str, int] | None:
    head = getattr(bucket, "head", None)
    if head is not None:
        try:
            return head(uri)
        except Exception:
            return None
    try:
        return {"size_bytes": len(bucket.get(uri)), "mtime_unix": 0} if bucket.exists(uri) else None
    except Exception:
        return None


def _run_id_from_uri(uri: str, prefix: str) -> str | None:
    marker = prefix.rstrip("/") + "/"
    if marker not in uri:
        return None
    rest = uri.split(marker, 1)[1]
    return rest.split("/", 1)[0] if rest else None
