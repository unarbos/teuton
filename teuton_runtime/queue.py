"""Orchestrator-owned outstanding-work queue.

This module defines the single source of truth for "what jobs are currently
outstanding" on the bucket. The orchestrator is the unique writer per
``(run_id, role)`` and treats its in-memory ``OrchestratorQueue`` as the
authoritative state; the bucket file is just a published snapshot.

Workers read the snapshot via :func:`read_queue` (with ``If-None-Match`` so
unchanged snapshots cost a 304) and execute the entries assigned to them.

Why a queue file instead of per-job indexes
-------------------------------------------

The old design wrote three append-only index files per emit
(``job_index.json``, ``step={n}/index.json``, ``hotkey={hk}/index.json``).
Their bodies grew O(jobs ever emitted in this run); under stress mode they
crossed MB territory in minutes, blowing up emit cost and read amplification
on miners.

The queue file is bounded by O(outstanding work) instead. Entries are removed
when the orchestrator observes the job's receipt or its deadline expires, so
size stays at roughly ``max_inflight_per_hotkey * n_miners`` regardless of
how long the run has been emitting.

Concurrency
-----------

Single-writer per ``(run_id, role)``: the orchestrator for ``role="train"``,
``AuditJobManager`` for ``role="audit"``. We still pass ``If-Match`` on every
PUT for defence-in-depth: if a second orchestrator accidentally points at the
same run, the second writer's PUT fails with :class:`PreconditionFailed` and
it can refuse to start instead of silently scrambling the queue.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

from teuton_core import paths
from teuton_core.protocol import JobManifestV3, JobReceiptV3
from teuton_runtime.storage import ObjectStore, PreconditionFailed


LOG = logging.getLogger(__name__)


_QUEUE_VERSION = 1


@dataclass(frozen=True)
class QueueEntry:
    """One outstanding job in the queue.

    ``manifest_uri`` and ``grant_uri`` are absolute ``s3://...`` URIs so the
    miner can fetch the signed manifest and encrypted grant directly without
    re-deriving the key from ``paths.py``.
    """

    job_id: str
    assigned_hotkey: str
    assigned_worker: str | None
    manifest_uri: str
    grant_uri: str | None
    deadline_unix: int
    attempt: int
    created_unix: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QueueEntry":
        return cls(
            job_id=str(data["job_id"]),
            assigned_hotkey=str(data["assigned_hotkey"]),
            assigned_worker=data.get("assigned_worker"),
            manifest_uri=str(data["manifest_uri"]),
            grant_uri=data.get("grant_uri"),
            deadline_unix=int(data.get("deadline_unix") or 0),
            attempt=int(data.get("attempt") or 0),
            created_unix=int(data.get("created_unix") or 0),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: JobManifestV3,
        *,
        manifest_uri: str,
        grant_uri: str | None,
    ) -> "QueueEntry":
        return cls(
            job_id=manifest.job_id,
            assigned_hotkey=manifest.assigned_hotkey,
            assigned_worker=manifest.assigned_worker,
            manifest_uri=manifest_uri,
            grant_uri=grant_uri,
            deadline_unix=int(manifest.deadline_unix or 0),
            attempt=int(manifest.attempt or 0),
            created_unix=int(manifest.created_unix or 0),
        )


@dataclass
class QueueState:
    """Decoded queue snapshot with the etag the bucket returned.

    The etag is opaque; pass it back to :func:`read_queue` as
    ``if_none_match`` to skip the GET when the snapshot hasn't changed.
    """

    role: str
    snapshot_unix: int
    snapshot_id: int
    outstanding: list[QueueEntry]
    etag: str | None = None
    version: int = _QUEUE_VERSION

    def filter_for_worker(self, *, hotkey_ss58: str, worker_id: str | None) -> list[QueueEntry]:
        """Return entries assigned to ``(hotkey_ss58, worker_id)``."""
        out: list[QueueEntry] = []
        for entry in self.outstanding:
            if entry.assigned_hotkey != hotkey_ss58:
                continue
            if entry.assigned_worker not in (None, "", worker_id):
                continue
            out.append(entry)
        return out

    def to_payload(self) -> dict:
        return {
            "version": self.version,
            "role": self.role,
            "snapshot_unix": int(self.snapshot_unix),
            "snapshot_id": int(self.snapshot_id),
            "outstanding": [e.to_dict() for e in self.outstanding],
        }

    @classmethod
    def from_payload(cls, payload: dict, *, etag: str | None) -> "QueueState":
        outstanding = [QueueEntry.from_dict(item) for item in payload.get("outstanding") or []]
        return cls(
            role=str(payload.get("role") or "train"),
            snapshot_unix=int(payload.get("snapshot_unix") or 0),
            snapshot_id=int(payload.get("snapshot_id") or 0),
            outstanding=outstanding,
            etag=etag,
            version=int(payload.get("version") or _QUEUE_VERSION),
        )


def queue_uri(bucket: ObjectStore, *, netuid: int, run_id: str, role: str) -> str:
    return bucket.uri_for_key(paths.queue_key(netuid, run_id, role))


def read_queue(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    role: str = "train",
    if_none_match: str | None = None,
) -> QueueState | None:
    """Fetch the published queue snapshot.

    Returns ``None`` when the bucket object is missing or when the
    ``if_none_match`` etag matches the current bucket etag (i.e. the snapshot
    hasn't moved since the caller's last read).
    """
    uri = queue_uri(bucket, netuid=netuid, run_id=run_id, role=role)
    body, etag = bucket.get_with_etag(uri, if_none_match=if_none_match)
    if body is None:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        LOG.warning("queue snapshot at %s is not valid JSON", uri)
        return None
    if not isinstance(payload, dict):
        return None
    return QueueState.from_payload(payload, etag=etag)


class OrchestratorQueue:
    """Authoritative in-memory state owned by the orchestrator.

    Mutations (``add``/``remove``/``replay``/``prune_expired``) update the
    in-memory ``_outstanding`` dict immediately and mark the queue dirty.
    A coalesced ``flush(...)`` is what publishes the snapshot to the bucket;
    a background thread (started by :meth:`start_background_flush`) calls it
    every ``flush_interval_sec`` so callers don't need to.

    Thread safety: all mutators take ``self._lock``. ``flush()`` snapshots
    the outstanding map under the lock, then writes outside it so the lock
    is held only for the local copy.
    """

    def __init__(
        self,
        *,
        bucket: ObjectStore,
        netuid: int,
        run_id: str,
        role: str = "train",
        flush_interval_sec: float = 0.5,
        flush_max_changes: int = 32,
    ) -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.run_id = run_id
        self.role = role
        self.flush_interval_sec = float(flush_interval_sec)
        self.flush_max_changes = int(flush_max_changes)

        self._lock = threading.RLock()
        self._outstanding: dict[str, QueueEntry] = {}
        self._snapshot_id = 0
        self._etag: str | None = None
        self._dirty = False
        self._pending_changes = 0
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, entry: QueueEntry) -> None:
        """Insert ``entry`` if not already present.

        Idempotent on ``job_id``: re-emitting the same job_id replaces the
        existing entry (so attempt/deadline get refreshed after replay).
        """
        with self._lock:
            self._outstanding[entry.job_id] = entry
            self._mark_dirty(1)

    def remove(self, job_id: str) -> bool:
        """Drop ``job_id`` from the queue. Returns ``True`` if it was present."""
        with self._lock:
            if job_id in self._outstanding:
                del self._outstanding[job_id]
                self._mark_dirty(1)
                return True
            return False

    def remove_many(self, job_ids: Iterable[str]) -> int:
        removed = 0
        with self._lock:
            for jid in job_ids:
                if jid in self._outstanding:
                    del self._outstanding[jid]
                    removed += 1
            if removed:
                self._mark_dirty(removed)
        return removed

    def replay(self, job_id: str, *, attempt: int, deadline_unix: int) -> None:
        """Re-add ``job_id`` with a higher attempt number.

        No-op if the entry was already removed (e.g. receipt landed in the
        same tick). The caller is responsible for actually re-emitting the
        manifest if the body needs to change.
        """
        with self._lock:
            entry = self._outstanding.get(job_id)
            if entry is None:
                return
            self._outstanding[job_id] = QueueEntry(
                job_id=entry.job_id,
                assigned_hotkey=entry.assigned_hotkey,
                assigned_worker=entry.assigned_worker,
                manifest_uri=entry.manifest_uri,
                grant_uri=entry.grant_uri,
                deadline_unix=int(deadline_unix),
                attempt=int(attempt),
                created_unix=entry.created_unix,
            )
            self._mark_dirty(1)

    def prune_expired(self, *, now: float | None = None) -> list[QueueEntry]:
        """Remove entries whose ``deadline_unix`` is in the past. Returns the dropped entries."""
        now = float(now if now is not None else time.time())
        dropped: list[QueueEntry] = []
        with self._lock:
            for job_id, entry in list(self._outstanding.items()):
                if entry.deadline_unix and entry.deadline_unix < now:
                    del self._outstanding[job_id]
                    dropped.append(entry)
            if dropped:
                self._mark_dirty(len(dropped))
        return dropped

    def reconcile_from_bucket(
        self,
        *,
        recent_receipt_job_ids: Iterable[str] = (),
    ) -> None:
        """Load the published queue from the bucket (used on restart).

        Removes entries whose receipts have already landed (caller scans
        recent receipts and passes their ``job_id`` set). Subsequent
        ``flush()`` calls will publish the reconciled snapshot.
        """
        state = read_queue(
            self.bucket,
            netuid=self.netuid,
            run_id=self.run_id,
            role=self.role,
        )
        with self._lock:
            self._outstanding.clear()
            if state is not None:
                for entry in state.outstanding:
                    self._outstanding[entry.job_id] = entry
                self._snapshot_id = max(self._snapshot_id, state.snapshot_id)
                self._etag = state.etag
            else:
                self._etag = None
            for jid in recent_receipt_job_ids:
                self._outstanding.pop(jid, None)
            self._dirty = state is None or bool(recent_receipt_job_ids)
            self._pending_changes = 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def depth(self, hotkey: str | None = None) -> int:
        """Outstanding count, total or filtered by ``hotkey``."""
        with self._lock:
            if hotkey is None:
                return len(self._outstanding)
            return sum(1 for e in self._outstanding.values() if e.assigned_hotkey == hotkey)

    def outstanding_job_ids(self) -> list[str]:
        with self._lock:
            return list(self._outstanding.keys())

    def __len__(self) -> int:
        return self.depth()

    def __contains__(self, job_id: object) -> bool:
        if not isinstance(job_id, str):
            return False
        with self._lock:
            return job_id in self._outstanding

    def __iter__(self) -> Iterator[QueueEntry]:
        with self._lock:
            return iter(list(self._outstanding.values()))

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self, *, force: bool = False) -> bool:
        """Publish the current state to the bucket.

        Returns ``True`` when a write actually happened. ``force=True``
        publishes even when nothing has changed (useful on shutdown).

        On precondition failure (another writer raced ahead) we merge the
        bucket's outstanding list back into ours and retry once. ``add`` is
        idempotent on ``job_id`` so the merge is safe: locally-added entries
        survive, bucket-only entries get pulled in.
        """
        with self._lock:
            if not (self._dirty or force):
                return False
            self._snapshot_id += 1
            payload = QueueState(
                role=self.role,
                snapshot_unix=int(time.time()),
                snapshot_id=self._snapshot_id,
                outstanding=list(self._outstanding.values()),
            ).to_payload()
            prior_etag = self._etag

        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        uri = queue_uri(self.bucket, netuid=self.netuid, run_id=self.run_id, role=self.role)
        if_match = "" if prior_etag is None else prior_etag
        try:
            new_etag = self.bucket.put_with_etag(uri, body, if_match=if_match)
        except PreconditionFailed as exc:
            LOG.warning(
                "queue flush precondition failed for %s; merging and retrying: %s",
                uri,
                exc,
            )
            self._merge_from_bucket()
            with self._lock:
                self._snapshot_id += 1
                payload = QueueState(
                    role=self.role,
                    snapshot_unix=int(time.time()),
                    snapshot_id=self._snapshot_id,
                    outstanding=list(self._outstanding.values()),
                ).to_payload()
                prior_etag = self._etag
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            new_etag = self.bucket.put_with_etag(uri, body, if_match=("" if prior_etag is None else prior_etag))

        with self._lock:
            self._etag = new_etag
            self._dirty = False
            self._pending_changes = 0
        return True

    def _merge_from_bucket(self) -> None:
        """Merge bucket-side state into ours WITHOUT clobbering pending entries.

        Used by ``flush()`` when a concurrent writer beat us. ``reconcile_…``
        is for restart (where we want to reload from durable state); merge
        is for conflict resolution (where we want to add new bucket entries
        but keep our own additions).
        """
        state = read_queue(
            self.bucket,
            netuid=self.netuid,
            run_id=self.run_id,
            role=self.role,
        )
        with self._lock:
            if state is not None:
                for entry in state.outstanding:
                    self._outstanding.setdefault(entry.job_id, entry)
                self._snapshot_id = max(self._snapshot_id, state.snapshot_id)
                self._etag = state.etag
            else:
                self._etag = None

    def _mark_dirty(self, n: int) -> None:
        # Caller already holds self._lock.
        self._dirty = True
        self._pending_changes += int(n)

    # ------------------------------------------------------------------
    # Background flush thread
    # ------------------------------------------------------------------

    def start_background_flush(self) -> None:
        """Spawn a daemon thread that periodically calls :meth:`flush`."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name=f"queue-flush-{self.role}",
            daemon=True,
        )
        self._flush_thread.start()

    def stop(self, *, final_flush: bool = True) -> None:
        """Stop the background flush thread; flush once more on the way out."""
        self._stop_event.set()
        thread = self._flush_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._flush_thread = None
        if final_flush:
            try:
                self.flush(force=False)
            except Exception as exc:
                LOG.debug("queue final flush failed: %r", exc)

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            should_flush = False
            with self._lock:
                if self._dirty:
                    should_flush = (
                        self._pending_changes >= self.flush_max_changes
                        or self._pending_changes > 0
                    )
            if should_flush:
                try:
                    self.flush()
                except Exception as exc:
                    LOG.warning("queue background flush failed: %r", exc)
            self._stop_event.wait(self.flush_interval_sec)


# ----------------------------------------------------------------------
# Receipt -> queue reconciliation helper for the orchestrator side
# ----------------------------------------------------------------------


def scan_recent_receipt_job_ids(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    since_unix: float | None = None,
    limit: int = 50000,
) -> set[str]:
    """Return job_ids of receipts written after ``since_unix``.

    Used by the orchestrator to remove entries from the queue once miners
    finish their work. We rely on two cheap operations:

    1. ``ListObjectsV2`` returns ``LastModified`` per object in the page
       response itself, so :meth:`ObjectStore.list_with_meta` gives us
       ``(uri, mtime_unix)`` without any per-object HEAD.
    2. The receipt URI carries ``job_id`` directly
       (``.../hotkey={hk}/{job_id}/attempt={k}.json``) so we extract it
       from the path without ``get_object``-ing the body.

    Net cost per drain: O(LIST pages), not O(receipts × HEAD). With ~10K
    receipts that's a handful of LISTs (a couple of seconds), not minutes.
    """
    out: set[str] = set()
    receipts_uri = bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))
    list_meta = getattr(bucket, "list_with_meta", None)
    if list_meta is not None:
        entries = list_meta(receipts_uri)
        count = 0
        for uri, mtime, _size in entries:
            if count >= limit:
                break
            if not uri.endswith(".json"):
                continue
            if since_unix is not None and mtime < int(since_unix):
                continue
            job_id = _job_id_from_receipt_uri(uri)
            if job_id is None:
                continue
            out.add(job_id)
            count += 1
        return out
    # Backend without list_with_meta: fall back to URI-only scan, no time
    # filter. Lifecycle bounds the prefix size so this stays cheap.
    count = 0
    for uri in bucket.list(receipts_uri):
        if count >= limit:
            break
        if not uri.endswith(".json"):
            continue
        job_id = _job_id_from_receipt_uri(uri)
        if job_id is None:
            continue
        out.add(job_id)
        count += 1
    return out


def _job_id_from_receipt_uri(uri: str) -> str | None:
    """Extract ``job_id`` from a receipt URI without fetching its body.

    Receipt keys are
    ``{receipts_prefix}hotkey={hk}/{job_id}/attempt={k}.json``; the segment
    after ``hotkey={hk}/`` is the job_id.
    """
    parts = uri.split("/")
    for i, part in enumerate(parts):
        if part.startswith("hotkey=") and i + 1 < len(parts):
            return parts[i + 1] or None
    return None
