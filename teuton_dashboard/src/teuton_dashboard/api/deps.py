"""Shared FastAPI dependencies.

Singletons (the DB, the QueueBus, the ObjectStore, the parsed Settings) live
on ``app.state`` and are reachable from any handler via these getter
functions. Keeping them small + explicit makes tests trivial: instantiate
``app.state.*`` to fakes and call the handler directly.
"""
from __future__ import annotations

from fastapi import Request

from teuton_runtime.storage import ObjectStore

from ..db import DashboardDB
from ..queue_bus import QueueBus
from ..settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> DashboardDB:
    return request.app.state.db


def get_bus(request: Request) -> QueueBus:
    return request.app.state.bus


def get_bucket(request: Request) -> ObjectStore:
    return request.app.state.bucket
