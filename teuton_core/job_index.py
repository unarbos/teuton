"""Canonical job-id discovery for bucket-backed runs.

Orchestrators append job IDs to a single aggregate JSON file (see
``teuton_core.paths.job_index_key`` for training,
``teuton_core.paths.audit_job_index_key`` for audit). Callers should always
resolve IDs through :func:`list_job_ids` so miners, dashboards, and tools share
one contract: one ``get_json`` when the index is non-empty, and an optional
bounded prefix scan of ``*/manifest.json`` otherwise.
"""
from __future__ import annotations

from typing import Any

from teuton_runtime.storage import ObjectStore


def list_job_ids(
    bucket: ObjectStore,
    *,
    index_key: str,
    jobs_prefix_key: str,
    manifest_list_max_uris: int | None = None,
) -> list[str]:
    """Return job IDs for the given aggregate index and jobs prefix.

    Parameters
    ----------
    index_key
        Storage key for the JSON array of job IDs (not a full URI).
    jobs_prefix_key
        Storage key prefix under which ``{job_id}/manifest.json`` objects live.
    manifest_list_max_uris
        When the aggregate index cannot be used, cap how many URIs from
        ``bucket.list`` are scanned before extracting ``manifest.json`` paths.
        ``None`` means no cap (full prefix list). UIs may pass a finite cap to
        bound work on huge buckets.
    """
    index_uri = bucket.uri_for_key(index_key)
    try:
        raw: Any = bucket.get_json(index_uri)
    except Exception:
        raw = None

    if isinstance(raw, list) and len(raw) > 0:
        return [str(x) for x in raw]

    prefix_uri = bucket.uri_for_key(jobs_prefix_key)
    try:
        uris = bucket.list(prefix_uri)
    except Exception:
        return []

    if manifest_list_max_uris is not None:
        uris = uris[:manifest_list_max_uris]

    out: list[str] = []
    for uri in uris:
        if uri.endswith("/manifest.json"):
            out.append(uri.rsplit("/", 2)[-2])
    return out
