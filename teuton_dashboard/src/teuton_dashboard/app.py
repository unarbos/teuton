"""FastAPI app factory.

``create_app(settings)`` builds the app, registers every router, and starts
the background indexers + queue sampler on lifespan-startup. ``main.py`` is
just the uvicorn entrypoint around this.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import chain as chain_routes
from .api import discovery as discovery_routes
from .api import health as health_routes
from .api import jobs as jobs_routes
from .api import queue as queue_routes
from .api import snapshot as snapshot_routes
from .bucket_factory import build_bucket
from .db import DashboardDB
from .indexers.bucket import run_bucket_indexer_loop
from .indexers.chain import run_chain_indexer_loop
from .indexers.queue_sampler import run_queue_sampler_loop
from .queue_bus import QueueBus
from .settings import Settings, get_settings


LOG = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Singletons live on app.state; deps.py reads them.
        app.state.settings = settings
        app.state.bucket = build_bucket(settings)
        app.state.db = DashboardDB(settings.db_path)
        await app.state.db.init()
        app.state.bus = QueueBus(
            history_seconds=settings.queue_history_seconds,
            history_max_points=settings.queue_history_max_points,
        )

        stop = asyncio.Event()
        app.state.stop_event = stop
        tasks: list[asyncio.Task] = []

        tasks.append(asyncio.create_task(
            run_bucket_indexer_loop(
                bucket=app.state.bucket,
                db=app.state.db,
                settings=settings,
                stop_event=stop,
            ),
            name="bucket-indexer",
        ))
        tasks.append(asyncio.create_task(
            run_queue_sampler_loop(
                bucket=app.state.bucket,
                db=app.state.db,
                bus=app.state.bus,
                settings=settings,
                stop_event=stop,
            ),
            name="queue-sampler",
        ))
        if settings.enable_chain_indexer:
            tasks.append(asyncio.create_task(
                run_chain_indexer_loop(db=app.state.db, settings=settings, stop_event=stop),
                name="chain-indexer",
            ))

        try:
            yield
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    app = FastAPI(
        title="Teuton Dashboard",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # API routers
    app.include_router(health_routes.router, tags=["health"])
    app.include_router(snapshot_routes.router, tags=["snapshot"])
    app.include_router(queue_routes.router, tags=["queue"])
    app.include_router(jobs_routes.router, tags=["jobs"])
    app.include_router(discovery_routes.router, tags=["discovery"])
    app.include_router(chain_routes.router, tags=["chain"])

    # Static frontend (SPA). Falls through to index.html for unknown paths so
    # SvelteKit's client-side router handles deep links.
    _mount_static(app, settings)
    return app


def _mount_static(app: FastAPI, settings: Settings) -> None:
    static_dir = Path(settings.static_dir)
    if not static_dir.is_dir():
        LOG.warning(
            "static dir %s not found; only /api/* and /openapi.json are served",
            static_dir,
        )
        return

    index_html = static_dir / "index.html"

    # Mount built assets (Vite emits /_app/...).
    assets_dir = static_dir / "_app"
    if assets_dir.is_dir():
        app.mount("/_app", StaticFiles(directory=str(assets_dir)), name="sveltekit-assets")

    # Catch-all for static + SPA fallback. Registered LAST so /api/* always wins.
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        # Prefer a real file under static_dir if it exists (favicon, robots, etc.).
        candidate = static_dir / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        # Otherwise serve the SPA shell so client-side routing works.
        if index_html.is_file():
            return FileResponse(str(index_html))
        return FileResponse(str(static_dir / "404.html")) if (static_dir / "404.html").is_file() else FileResponse(str(index_html))


# Module-level app for ``uvicorn teuton_dashboard.app:app`` style entry.
app = create_app()
