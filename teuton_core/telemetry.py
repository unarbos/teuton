"""Telemetry writer for Teuton components.

Emits small JSON files to v3/netuid=N/telemetry/<run_id>/<stream>/<unix>.json.
Best-effort: never raises on the caller's path, retries briefly, no-ops on
bucket failure.

Streams in use:
  chain        - validator set_weights outcomes (one per validator loop pass)
  scores       - rolling per-pass score snapshots
  audits       - per-auditor rolled-up timing + pass rate
  orchestrator - per-epoch summary (job counts, manifest config, wall time)
  monitor      - dev-box health beacon
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from teuton_core import paths


LOG = logging.getLogger(__name__)


class TelemetryWriter:
    def __init__(self, *, bucket, netuid: int, run_id: str, component: str = "unknown") -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.run_id = run_id
        self.component = component

    def emit(self, stream: str, payload: dict, *, when_unix: Optional[int] = None) -> bool:
        ts = int(when_unix if when_unix is not None else time.time())
        body = dict(payload)
        body.setdefault("ts", ts)
        body.setdefault("component", self.component)
        body.setdefault("run_id", self.run_id)
        body.setdefault("netuid", self.netuid)
        key = paths.telemetry_event_key(self.netuid, self.run_id, stream, ts)
        uri = self.bucket.uri_for_key(key)
        for attempt in range(3):
            try:
                self.bucket.put_json(uri, body)
                return True
            except Exception as e:
                LOG.debug("telemetry %s emit failed (attempt %s): %r", stream, attempt + 1, e)
                time.sleep(0.5 * (attempt + 1))
        LOG.warning("telemetry %s drop after retries", stream)
        return False

    def chain(self, payload: dict) -> bool:
        return self.emit("chain", payload)

    def scores(self, payload: dict) -> bool:
        return self.emit("scores", payload)

    def audits(self, payload: dict) -> bool:
        return self.emit("audits", payload)

    def orchestrator(self, payload: dict) -> bool:
        return self.emit("orchestrator", payload)

    def monitor(self, payload: dict) -> bool:
        return self.emit("monitor", payload)


def make_writer(bucket, *, netuid: int, run_id: str, component: str) -> Optional[TelemetryWriter]:
    if not run_id or not bucket:
        return None
    return TelemetryWriter(bucket=bucket, netuid=netuid, run_id=run_id, component=component)
