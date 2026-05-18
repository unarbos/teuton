"""In-process pub/sub for queue snapshots.

The queue_sampler indexer publishes a :class:`QueueSnapshot` whenever the
orchestrator's snapshot_id advances. SSE handlers subscribe by ``(run_id, role)``
and receive every subsequent snapshot via a per-subscriber ``asyncio.Queue``.

A small ``_latest`` cache holds the most recent snapshot per key so new
subscribers get an immediate first event instead of waiting for the next
publish cycle (good UX after a page refresh / SSE reconnect).

Backpressure: each subscriber's queue is bounded (``maxsize``). When full
we drop the OLDEST snapshot, not the new one, because the dashboard only
cares about the freshest state.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncIterator

from .models import QueueHistoryPoint, QueueSnapshot


LOG = logging.getLogger(__name__)


_DEFAULT_SUBSCRIBER_QUEUE = 8
_DEFAULT_HISTORY_SECONDS = 30 * 60
_DEFAULT_HISTORY_MAX = 400


class QueueBus:
    """Per-process pub/sub keyed by ``(run_id, role)``.

    Also owns the rolling queue-depth history each subscriber's snapshot
    embeds; storing it here keeps the sampler stateless (just publishes the
    latest snapshot + a history sample point) and avoids duplicating the
    ring buffer across multiple subscribers.
    """

    def __init__(
        self,
        *,
        history_seconds: int = _DEFAULT_HISTORY_SECONDS,
        history_max_points: int = _DEFAULT_HISTORY_MAX,
        subscriber_queue_size: int = _DEFAULT_SUBSCRIBER_QUEUE,
    ) -> None:
        self._latest: dict[tuple[str, str], QueueSnapshot] = {}
        self._subscribers: dict[tuple[str, str], set[asyncio.Queue[QueueSnapshot]]] = {}
        self._history: dict[tuple[str, str], deque[QueueHistoryPoint]] = {}
        self._lock = asyncio.Lock()
        self._history_seconds = int(history_seconds)
        self._history_max = int(history_max_points)
        self._subscriber_queue_size = int(subscriber_queue_size)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def record_history_point(self, run_id: str, role: str, point: QueueHistoryPoint) -> None:
        """Append a history sample. Cheap; no lock needed (deque append is atomic)."""
        key = (run_id, role)
        buf = self._history.get(key)
        if buf is None:
            buf = deque(maxlen=self._history_max)
            self._history[key] = buf
        buf.append(point)

    def history_for(self, run_id: str, role: str, *, now_unix: int) -> list[QueueHistoryPoint]:
        cutoff = int(now_unix) - self._history_seconds
        buf = self._history.get((run_id, role))
        if not buf:
            return []
        return [p for p in buf if p.ts >= cutoff]

    # ------------------------------------------------------------------
    # Pub / sub
    # ------------------------------------------------------------------

    async def publish(self, snapshot: QueueSnapshot) -> int:
        """Broadcast ``snapshot`` to every subscriber for its (run_id, role).

        Returns the number of subscribers notified. Snapshots are immutable
        from the publisher's point of view so we can share the same object
        across queues.
        """
        key = (snapshot.run_id, snapshot.role)
        delivered = 0
        async with self._lock:
            self._latest[key] = snapshot
            subs = list(self._subscribers.get(key, ()))
        for q in subs:
            try:
                q.put_nowait(snapshot)
                delivered += 1
            except asyncio.QueueFull:
                # Drop the oldest, then put the fresh snapshot. Subscribers
                # only care about the latest state; slow consumers shouldn't
                # block the publisher.
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(snapshot)
                    delivered += 1
                except asyncio.QueueFull:
                    LOG.debug("queue_bus: subscriber still full after drop_oldest; skipping")
        return delivered

    def latest(self, run_id: str, role: str) -> QueueSnapshot | None:
        return self._latest.get((run_id, role))

    async def subscribe(self, run_id: str, role: str) -> AsyncIterator[QueueSnapshot]:
        """Yield snapshots for ``(run_id, role)`` until the consumer stops.

        Emits the cached latest snapshot first (if any) so the SSE client
        gets an immediate update instead of waiting for the next publish.
        """
        key = (run_id, role)
        q: asyncio.Queue[QueueSnapshot] = asyncio.Queue(maxsize=self._subscriber_queue_size)
        async with self._lock:
            self._subscribers.setdefault(key, set()).add(q)
            seed = self._latest.get(key)
        if seed is not None:
            await q.put(seed)
        try:
            while True:
                snap = await q.get()
                yield snap
        finally:
            async with self._lock:
                subs = self._subscribers.get(key)
                if subs is not None:
                    subs.discard(q)
                    if not subs:
                        self._subscribers.pop(key, None)

    def subscriber_count(self, run_id: str | None = None, role: str | None = None) -> int:
        if run_id is None or role is None:
            return sum(len(s) for s in self._subscribers.values())
        return len(self._subscribers.get((run_id, role), ()))
