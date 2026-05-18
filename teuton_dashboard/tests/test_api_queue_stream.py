"""SSE end-to-end: publishing a snapshot triggers a stream event."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from teuton_dashboard.models import QueueSnapshot


async def test_sse_emits_event_on_publish(client, app):
    bus = app.state.bus
    # The fixture seeds a queue.json on the bucket, so the sampler may have
    # already published it. We publish our own snapshot AFTER opening the
    # stream so the assertion is deterministic on the publish-after path.
    target = QueueSnapshot(
        run_id="test-run", role="train",
        snapshot_unix=int(time.time()), snapshot_id=4242,
        depth_total=2, depth_by_hotkey={"hk-a": 2},
        max_inflight_per_hotkey=4, at_cap_count=0, at_cap_hotkeys=[],
    )

    # Schedule the publish to fire shortly AFTER we start consuming, in
    # parallel, so the handler's subscriber is registered first.
    async def publish_after_delay():
        await asyncio.sleep(0.15)
        await bus.publish(target)

    publish_task = asyncio.create_task(publish_after_delay())

    async with client.stream("GET", "/api/queue/stream?role=train") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        deadline = asyncio.get_event_loop().time() + 3.0
        buf = ""
        payload = None
        seen_ids: list[int] = []
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                for line in block.splitlines():
                    if line.startswith("data:"):
                        cand = json.loads(line[5:].strip())
                        seen_ids.append(int(cand.get("snapshot_id", -1)))
                        if int(cand.get("snapshot_id", -1)) == 4242:
                            payload = cand
                            break
                if payload is not None:
                    break
            if payload is not None:
                break
            if asyncio.get_event_loop().time() > deadline:
                pytest.fail(f"no SSE event with snapshot_id=4242 within 3s; saw {seen_ids}")
        assert payload is not None
        assert payload["depth_total"] == 2

    await publish_task
