"""Env-driven settings for the dashboard.

Mirrors the env contract the legacy ``teuton_core.cli dashboard-backend``
honoured so the existing deploy_dashboard.sh + compose file flow stays
identical from the operator's perspective.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Bucket
    s3_bucket: str = Field(default="", alias="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")
    s3_endpoint_url: str = Field(default="", alias="S3_ENDPOINT_URL")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    local_bucket_root: Optional[str] = Field(default=None, alias="TEUTON_LOCAL_BUCKET_ROOT")
    local_bucket_name: Optional[str] = Field(default=None, alias="TEUTON_LOCAL_BUCKET_NAME")

    # Chain / netuid
    netuid: int = Field(default=3, alias="TEUTON_NETUID")
    run_id: str = Field(default="", alias="TEUTON_RUN_ID")
    bt_network: str = Field(default="finney", alias="BT_NETWORK")

    # Queue cap (mirrors orchestrator backpressure setting)
    max_inflight_per_hotkey: int = Field(default=8, alias="TEUTON_MAX_INFLIGHT_PER_HOTKEY")

    # Dashboard runtime
    db_path: str = Field(
        default="/var/lib/teuton-dashboard/dashboard.sqlite3",
        alias="TEUTON_DASHBOARD_DB_PATH",
    )
    host: str = Field(default="0.0.0.0", alias="TEUTON_DASHBOARD_HOST")
    port: int = Field(default=8765, alias="TEUTON_DASHBOARD_PORT")
    bucket_poll_sec: float = Field(default=5.0, alias="TEUTON_DASHBOARD_BUCKET_POLL_SEC")
    chain_poll_sec: float = Field(default=30.0, alias="TEUTON_DASHBOARD_CHAIN_POLL_SEC")
    queue_sample_sec: float = Field(default=0.5, alias="TEUTON_DASHBOARD_QUEUE_SAMPLE_SEC")
    heartbeat_ttl_sec: Optional[float] = Field(default=30.0, alias="TEUTON_DASHBOARD_HEARTBEAT_TTL_SEC")
    max_completed_jobs: int = Field(default=200, alias="TEUTON_DASHBOARD_MAX_JOBS")
    queue_history_seconds: int = Field(default=30 * 60, alias="TEUTON_DASHBOARD_QUEUE_HISTORY_SEC")
    queue_history_max_points: int = Field(default=400, alias="TEUTON_DASHBOARD_QUEUE_HISTORY_MAX")

    # Frontend static assets directory. The Docker image bakes the SvelteKit
    # build into /app/static; for dev runs point this at frontend/build.
    static_dir: str = Field(default="static", alias="TEUTON_DASHBOARD_STATIC_DIR")

    # SSE keepalive cadence (seconds). Cloudflare's idle timeout is ~100s by
    # default; 15s keeps us comfortably below that.
    sse_keepalive_sec: float = Field(default=15.0, alias="TEUTON_DASHBOARD_SSE_KEEPALIVE_SEC")

    # Chain indexing is optional; flipped to False in tests / local dev where
    # bittensor isn't installed.
    enable_chain_indexer: bool = Field(default=True, alias="TEUTON_DASHBOARD_ENABLE_CHAIN")


_singleton: Optional[Settings] = None


def get_settings() -> Settings:
    """Lazily construct + cache the env-driven settings singleton."""
    global _singleton
    if _singleton is None:
        _singleton = Settings()
    return _singleton


def override_settings(**kwargs) -> Settings:
    """Test-only: replace the singleton with a fresh instance built from kwargs."""
    global _singleton
    _singleton = Settings(**kwargs)
    return _singleton
