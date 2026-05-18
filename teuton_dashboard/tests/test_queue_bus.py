"""QueueBus unit tests: publish, subscribe seed, fan-out, drop-oldest."""
from __future__ import annotations

import asyncio

import pytest

from teuton_dashboard.models import QueueHistoryPoint, QueueSnapshot
from teuton_dashboard.queue_bus import QueueBus


def _snap(snapshot_id: int = 1, depth_total: int = 1) -> QueueSnapshot:
    return QueueSnapshot(
        run_id="r",
        role="train",
        snapshot_unix=100 + snapshot_id,
        snapshot_id=snapshot_id,
        depth_total=depth_total,
        depth_by_hotkey={"hk": depth_total},
        max_inflight_per_hotkey=8,
        at_cap_count=0,
        at_cap_hotkeys=[],
    )


async def test_publish_then_subscribe_seeds_latest():
    bus = QueueBus()
    snap = _snap(1)
    await bus.publish(snap)
    it = bus.subscribe("r", "train")
    first = await asyncio.wait_for(anext(it), timeout=1.0)
    assert first.snapshot_id == 1
    await it.aclose()


async def _start_subscriber(bus: QueueBus, run_id: str, role: str):
    """Kick the generator past its registration step.

    ``QueueBus.subscribe`` is an async generator: the body that registers
    the per-subscriber queue only runs on the first ``__anext__``. Tests
    that want a registered subscriber before publishing must drive at least
    one ``__anext__`` first (and consume any seeded latest snapshot).
    """
    it = bus.subscribe(run_id, role)
    started = asyncio.create_task(anext(it))
    # Yield control until the generator body has run far enough to register.
    for _ in range(10):
        await asyncio.sleep(0)
        if bus.subscriber_count(run_id, role) >= 1:
            break
    return it, started


async def test_subscribe_then_publish_delivers():
    bus = QueueBus()
    it, started = await _start_subscriber(bus, "r", "train")
    await bus.publish(_snap(2))
    first = await asyncio.wait_for(started, timeout=1.0)
    assert first.snapshot_id == 2
    await it.aclose()


async def test_multiple_subscribers_get_same_snapshot():
    bus = QueueBus(subscriber_queue_size=4)
    it_a, started_a = await _start_subscriber(bus, "r", "train")
    it_b, started_b = await _start_subscriber(bus, "r", "train")
    await bus.publish(_snap(7))
    snap_a, snap_b = await asyncio.wait_for(asyncio.gather(started_a, started_b), timeout=1.0)
    assert snap_a.snapshot_id == 7
    assert snap_b.snapshot_id == 7
    await it_a.aclose()
    await it_b.aclose()


async def test_drop_oldest_when_subscriber_slow():
    bus = QueueBus(subscriber_queue_size=2)
    it, started = await _start_subscriber(bus, "r", "train")
    # Three publishes, no consumer drain. Queue size is 2; the oldest must drop.
    for i in range(1, 4):
        await bus.publish(_snap(i))
    # The first await consumes whatever the pre-started anext got.
    seen = [(await asyncio.wait_for(started, timeout=1.0)).snapshot_id]
    while len(seen) < 3:
        try:
            snap = await asyncio.wait_for(anext(it), timeout=0.2)
        except asyncio.TimeoutError:
            break
        seen.append(snap.snapshot_id)
    assert 3 in seen, f"newest snapshot must survive; saw {seen}"
    await it.aclose()


def test_history_buffer_records_and_trims():
    bus = QueueBus(history_seconds=10, history_max_points=5)
    for i in range(7):
        bus.record_history_point("r", "train", QueueHistoryPoint(ts=100 + i, depth_total=i, at_cap_count=0))
    points = bus.history_for("r", "train", now_unix=110)
    # max_points caps at 5; only the last 5 survive.
    assert [p.depth_total for p in points] == [2, 3, 4, 5, 6]


def test_history_window_cuts_old_points():
    bus = QueueBus(history_seconds=5, history_max_points=100)
    for ts in range(100, 120):
        bus.record_history_point("r", "train", QueueHistoryPoint(ts=ts, depth_total=ts, at_cap_count=0))
    points = bus.history_for("r", "train", now_unix=120)
    # window=5s, now=120 -> cutoff=115; only ts>=115 survive.
    assert all(p.ts >= 115 for p in points)
    assert len(points) == 5
