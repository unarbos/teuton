"""/api/queue and /api/queue/stream (SSE)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from teuton_runtime.queue import read_queue
from teuton_runtime.storage import ObjectStore

from ..indexers.queue_sampler import project_state
from ..models import QueueResponse, QueueSnapshot
from ..queue_bus import QueueBus
from ..settings import Settings
from .deps import get_bucket, get_bus, get_settings


router = APIRouter()
LOG = logging.getLogger(__name__)


def _resolve_run_id(settings: Settings, run_id: Optional[str]) -> str:
    if run_id and run_id not in {"all", "*", "network"}:
        return run_id
    return settings.run_id or ""


def _read_snapshot(
    *,
    bucket: ObjectStore,
    settings: Settings,
    bus: QueueBus,
    run_id: str,
    role: str,
) -> QueueSnapshot:
    """Cache-first read with bucket fallback (used for /api/queue + /api/snapshot)."""
    if not run_id:
        return project_state(
            None, run_id="", role=role, cap=settings.max_inflight_per_hotkey, bus=bus, now_unix=int(time.time())
        )
    cached = bus.latest(run_id, role)
    if cached is not None:
        return cached
    state = read_queue(bucket, netuid=settings.netuid, run_id=run_id, role=role)
    return project_state(
        state, run_id=run_id, role=role, cap=settings.max_inflight_per_hotkey, bus=bus, now_unix=int(time.time())
    )


@router.get("/api/queue", response_model=QueueResponse)
async def queue(
    bucket: ObjectStore = Depends(get_bucket),
    settings: Settings = Depends(get_settings),
    bus: QueueBus = Depends(get_bus),
    run_id: Optional[str] = Query(default=None),
    role: str = Query(default="train"),
) -> QueueResponse:
    if role not in ("train", "audit"):
        role = "train"
    resolved = _resolve_run_id(settings, run_id)
    if not resolved:
        return QueueResponse(
            queue=None,
            meta={"netuid": settings.netuid, "run_id": "", "role": role, "generated_unix": int(time.time())},
        )
    snap = await asyncio.to_thread(
        _read_snapshot, bucket=bucket, settings=settings, bus=bus, run_id=resolved, role=role
    )
    return QueueResponse(
        queue=snap,
        meta={"netuid": settings.netuid, "run_id": resolved, "role": role, "generated_unix": int(time.time())},
    )


@router.get("/api/queue/stream")
async def queue_stream(
    request: Request,
    bus: QueueBus = Depends(get_bus),
    settings: Settings = Depends(get_settings),
    run_id: Optional[str] = Query(default=None),
    role: str = Query(default="train"),
) -> StreamingResponse:
    """Server-Sent Events stream of QueueSnapshot updates.

    Emits one ``data: {...}`` event on every snapshot_id advance. Heartbeats
    every ``sse_keepalive_sec`` keep the connection alive through Cloudflare
    Tunnel's idle timeout. Disconnects cleanly when the client goes away.
    """
    if role not in ("train", "audit"):
        role = "train"
    resolved = _resolve_run_id(settings, run_id)
    keepalive = float(settings.sse_keepalive_sec)

    async def event_stream():
        if not resolved:
            yield b": no run selected\n\n"
            return
        # Two-source loop: race a fresh snapshot against the keepalive timer
        # with ``asyncio.wait``. Whichever fires first wins; the loser is
        # cancelled. ``GeneratorExit`` from a disconnected client unwinds
        # through the ``finally`` so we de-register from the bus cleanly.
        sub_iter = bus.subscribe(resolved, role)
        try:
            while True:
                snap_task = asyncio.create_task(_safe_anext(sub_iter))
                keepalive_task = asyncio.create_task(asyncio.sleep(keepalive))
                done, pending = await asyncio.wait(
                    {snap_task, keepalive_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except (asyncio.CancelledError, Exception):
                        pass
                if snap_task in done:
                    snap = snap_task.result()
                    if snap is None:
                        return
                    yield _format_data_event(snap)
                else:
                    yield b": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            return
        except Exception as exc:
            LOG.warning("SSE stream error: %r", exc)
            return
        finally:
            try:
                await sub_iter.aclose()
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-store",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _format_data_event(snap: QueueSnapshot) -> bytes:
    return f"event: queue\ndata: {snap.model_dump_json()}\n\n".encode("utf-8")


async def _safe_anext(iterator):
    """Wrapper that returns ``None`` instead of raising ``StopAsyncIteration``.

    Makes the ``asyncio.wait`` race in the SSE loop simpler: a None means the
    subscription closed; the loop returns.
    """
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None
