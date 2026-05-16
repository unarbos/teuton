"""Telemetry helpers for v3 streaming and legacy fleet runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamingTelemetrySummary:
    n_jobs: int = 0
    active_workers: int = 0
    pool_busy_sec: float = 0.0
    pool_compute_sec: float = 0.0
    pool_io_sec: float = 0.0
    claimed_bytes_read: int = 0
    claimed_bytes_written: int = 0
    per_worker_jobs: dict[str, int] = field(default_factory=dict)

    @property
    def compute_share_of_busy(self) -> float:
        return self.pool_compute_sec / max(self.pool_busy_sec, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_jobs": self.n_jobs,
            "active_workers": self.active_workers,
            "pool_busy_sec": round(self.pool_busy_sec, 6),
            "pool_compute_sec": round(self.pool_compute_sec, 6),
            "pool_io_sec": round(self.pool_io_sec, 6),
            "compute_share_of_busy": round(self.compute_share_of_busy, 6),
            "claimed_bytes_read": int(self.claimed_bytes_read),
            "claimed_bytes_written": int(self.claimed_bytes_written),
            "per_worker_jobs": dict(sorted(self.per_worker_jobs.items())),
        }


def summarize_receipts(receipts) -> StreamingTelemetrySummary:
    summary = StreamingTelemetrySummary()
    for receipt in receipts:
        worker = receipt.worker.worker_id
        elapsed = max(0.0, float(receipt.finished_unix) - float(receipt.started_unix))
        summary.n_jobs += 1
        summary.pool_busy_sec += elapsed
        summary.pool_compute_sec += float(receipt.compute_sec)
        summary.pool_io_sec += max(0.0, elapsed - float(receipt.compute_sec))
        summary.claimed_bytes_read += int(receipt.claimed_bytes_read)
        summary.claimed_bytes_written += int(receipt.claimed_bytes_written)
        summary.per_worker_jobs[worker] = summary.per_worker_jobs.get(worker, 0) + 1
    summary.active_workers = len(summary.per_worker_jobs)
    return summary
