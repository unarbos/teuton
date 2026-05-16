"""Runtime configuration knobs (environment-variable driven, with defaults).

These are the few global knobs that the runtime consults independently of any
per-run config. Per-run knobs (model dims, lr, replica count, etc.) live in
`runs/<run_id>/manifest/config.json`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_LOCAL_ROOT = "./teuton_data"
DEFAULT_BUCKET = "teuton-test"
DEFAULT_RUN_ID = "demo-run"

DEFAULT_POLL_INTERVAL_SEC = 0.2
DEFAULT_T_MAX_SEC = 60
DEFAULT_TRIM_FRAC = 0.0
DEFAULT_DEADLINE_SEC = 600
DEFAULT_HEARTBEAT_INTERVAL_SEC = 1.0

# A worker is considered "active" if its heartbeat is no older than this.
# Beyond this, the orchestrator drops it from the pin table; failure-retry
# (Phase 4) re-emits jobs that were assigned to it.
WORKER_STALE_SEC = 30.0


# --------------------------------------------------------------------------- #
# Storage config
# --------------------------------------------------------------------------- #


@dataclass
class StorageConfig:
    """Sufficient to construct a `LocalBucket`."""

    root: str
    bucket: str


def load_storage_config(
    *,
    root: str | None = None,
    bucket: str | None = None,
) -> StorageConfig:
    return StorageConfig(
        root=root or os.environ.get("TEUTON_LOCAL_ROOT", DEFAULT_LOCAL_ROOT),
        bucket=bucket or os.environ.get("TEUTON_BUCKET", DEFAULT_BUCKET),
    )


def env_run_id(default: str | None = None) -> str:
    return os.environ.get("TEUTON_RUN_ID", default or DEFAULT_RUN_ID)


def env_poll_interval(default: float = DEFAULT_POLL_INTERVAL_SEC) -> float:
    try:
        return float(os.environ.get("TEUTON_POLL_INTERVAL_SEC", default))
    except ValueError:
        return default


def env_t_max(default: float = DEFAULT_T_MAX_SEC) -> float:
    try:
        return float(os.environ.get("TEUTON_T_MAX_SEC", default))
    except ValueError:
        return default


def env_trim_frac(default: float = DEFAULT_TRIM_FRAC) -> float:
    try:
        return float(os.environ.get("TEUTON_TRIM_FRAC", default))
    except ValueError:
        return default
