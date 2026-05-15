"""Worker discovery backends for Locus runtimes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from locus_core import paths
from locus_core.protocol import JobManifestV3, MinerIdentity, WorkerIdentity
from .storage import ObjectStore


@dataclass(frozen=True)
class DiscoveryRecord:
    miner: MinerIdentity
    worker: WorkerIdentity
    run_id: str
    last_seen_unix: int
    role: str = "train"


@dataclass
class BucketMinerObservation:
    """A miner inferred from durable bucket artifacts (no live heartbeat needed)."""

    hotkey_ss58: str
    worker_id: str | None = None
    last_seen_unix: int = 0
    n_receipts: int = 0
    n_jobs: int = 0
    sources: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "hotkey_ss58": self.hotkey_ss58,
            "worker_id": self.worker_id,
            "last_seen_unix": int(self.last_seen_unix),
            "n_receipts": int(self.n_receipts),
            "n_jobs": int(self.n_jobs),
            "sources": sorted(self.sources),
        }


class DiscoveryBackend(Protocol):
    def advertise_worker(self, *, miner: MinerIdentity, worker: WorkerIdentity) -> None:
        """Publish a worker identity to the selected discovery plane."""

    def discover_workers(self) -> list[DiscoveryRecord]:
        """Return currently discoverable workers for this backend."""


class BucketDiscoveryBackend:
    """Discovery backend that uses bucket heartbeat objects as the registry."""

    def __init__(
        self,
        *,
        bucket: ObjectStore,
        netuid: int,
        run_id: str,
        role: str = "train",
        heartbeat_ttl_sec: float | None = None,
    ) -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.run_id = run_id
        self.role = role
        self.heartbeat_ttl_sec = heartbeat_ttl_sec

    def advertise_worker(self, *, miner: MinerIdentity, worker: WorkerIdentity) -> None:
        self.bucket.put_json(
            self.bucket.uri_for_key(self._heartbeat_key(worker)),
            {
                "miner": miner.to_dict(),
                "worker": worker.to_dict(),
                "run_id": self.run_id,
                "role": self.role,
                "last_seen_unix": int(time.time()),
            },
        )

    def discover_workers(self) -> list[DiscoveryRecord]:
        return scan_bucket_discovery_records(
            self.bucket,
            netuid=self.netuid,
            run_id=self.run_id,
            role=self.role,
            heartbeat_ttl_sec=self.heartbeat_ttl_sec,
        )

    def _prefix(self) -> str:
        if self.role == "audit":
            return paths.auditors_prefix(self.netuid)
        return paths.miners_prefix(self.netuid)

    def _heartbeat_key(self, worker: WorkerIdentity) -> str:
        if self.role == "audit":
            return paths.auditor_heartbeat_key(self.netuid, worker.hotkey_ss58, worker.worker_id)
        return paths.worker_heartbeat_key(self.netuid, worker.hotkey_ss58, worker.worker_id)


def scan_bucket_discovery_records(
    bucket: ObjectStore,
    *,
    netuid: int,
    role: str = "train",
    run_id: str | None = None,
    heartbeat_ttl_sec: float | None = None,
) -> list[DiscoveryRecord]:
    """Scan bucket heartbeats for active discovery records.

    If run_id is omitted, records for all runs under the selected netuid/role are
    returned. This is useful for fleet dashboards that need a network-wide view.
    """
    records: list[DiscoveryRecord] = []
    now = time.time()
    prefix = paths.auditors_prefix(netuid) if role == "audit" else paths.miners_prefix(netuid)
    for uri in bucket.list(bucket.uri_for_key(prefix)):
        if not uri.endswith("/heartbeat.json"):
            continue
        try:
            record = discovery_record_from_dict(bucket.get_json(uri), netuid=netuid, role=role)
        except Exception:
            continue
        if run_id is not None and record.run_id != run_id:
            continue
        if heartbeat_ttl_sec is not None and now - record.last_seen_unix > heartbeat_ttl_sec:
            continue
        records.append(record)
    records.sort(key=lambda r: (r.role, r.run_id, r.worker.hotkey_ss58, r.worker.worker_id))
    return records


def discovery_record_from_dict(data: dict, *, netuid: int, role: str = "train") -> DiscoveryRecord:
    worker = WorkerIdentity.from_dict(data["worker"])
    miner_data = data.get("miner") or {
        "netuid": netuid,
        "hotkey_ss58": worker.hotkey_ss58,
        "capabilities": dict(worker.capabilities),
    }
    return DiscoveryRecord(
        miner=MinerIdentity.from_dict(miner_data),
        worker=worker,
        run_id=data.get("run_id", ""),
        last_seen_unix=int(data.get("last_seen_unix") or 0),
        role=data.get("role") or role,
    )


def build_discovery_backend(
    backend: str,
    *,
    bucket: ObjectStore,
    netuid: int,
    run_id: str,
    role: str = "train",
    heartbeat_ttl_sec: float | None = None,
) -> DiscoveryBackend:
    if backend == "bucket":
        return BucketDiscoveryBackend(
            bucket=bucket,
            netuid=netuid,
            run_id=run_id,
            role=role,
            heartbeat_ttl_sec=heartbeat_ttl_sec,
        )
    raise ValueError(f"unknown discovery backend: {backend}")


def derive_miners_from_bucket(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str | None,
) -> list[BucketMinerObservation]:
    """Reconstruct miner identities from durable bucket artifacts.

    Pulls hotkey/worker pairs out of two places that don't require a running
    miner process:

    - ``v3/netuid=N/receipts/{run_id}/hotkey={hk}/{job_id}/attempt=K.json``
      (mtime is the best "last did real work" timestamp)
    - ``v3/netuid=N/jobs/{run_id}/{job_id}/manifest.json``
      (orchestrator's intended assignment - hotkey + worker_id)

    Scoped to a single run when ``run_id`` is given; otherwise scans every run
    that exists under the netuid.
    """
    by_key: dict[tuple[str, str | None], BucketMinerObservation] = {}

    def _bump(hotkey: str, worker_id: str | None, *, source: str, ts: int = 0, jobs: int = 0, receipts: int = 0) -> None:
        key = (hotkey, worker_id)
        obs = by_key.get(key)
        if obs is None:
            obs = BucketMinerObservation(hotkey_ss58=hotkey, worker_id=worker_id)
            by_key[key] = obs
        obs.sources.add(source)
        obs.last_seen_unix = max(obs.last_seen_unix, int(ts or 0))
        obs.n_jobs += jobs
        obs.n_receipts += receipts

    run_ids = [run_id] if run_id else _list_run_ids_from_receipts(bucket, netuid=netuid)
    for rid in run_ids:
        if not rid:
            continue
        for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(netuid, rid))):
            if not uri.endswith(".json"):
                continue
            hotkey = _segment_value(uri, "hotkey=")
            if not hotkey:
                continue
            head = _safe_head(bucket, uri)
            ts = int((head or {}).get("mtime_unix") or 0)
            _bump(hotkey, None, source="receipts", ts=ts, receipts=1)
        for uri in bucket.list(bucket.uri_for_key(paths.jobs_prefix(netuid, rid))):
            if not uri.endswith("/manifest.json"):
                continue
            try:
                manifest = JobManifestV3.from_dict(bucket.get_json(uri))
            except Exception:
                continue
            head = _safe_head(bucket, uri)
            ts = max(int((head or {}).get("mtime_unix") or 0), int(manifest.created_unix or 0))
            _bump(
                manifest.assigned_hotkey,
                manifest.assigned_worker,
                source="manifests",
                ts=ts,
                jobs=1,
            )
    return sorted(by_key.values(), key=lambda o: (o.hotkey_ss58, o.worker_id or ""))


def _list_run_ids_from_receipts(bucket: ObjectStore, *, netuid: int) -> list[str]:
    seen: set[str] = set()
    receipts_root = f"{paths.root(netuid)}/receipts/"
    for uri in bucket.list(bucket.uri_for_key(receipts_root)):
        rest = uri.split(receipts_root, 1)[-1]
        rid = rest.split("/", 1)[0]
        if rid:
            seen.add(rid)
    return sorted(seen)


def _segment_value(uri: str, marker: str) -> str | None:
    for part in uri.split("/"):
        if part.startswith(marker):
            return part[len(marker):]
    return None


def _safe_head(bucket: ObjectStore, uri: str) -> dict | None:
    head = getattr(bucket, "head", None)
    if head is None:
        return None
    try:
        return head(uri)
    except Exception:
        return None
