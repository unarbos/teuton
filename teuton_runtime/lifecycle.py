"""Bucket lifecycle helpers for v3 run prefixes."""
from __future__ import annotations

from teuton_core import paths
from .storage import ObjectStore


def wipe_run(bucket: ObjectStore, *, netuid: int, run_id: str) -> int:
    deleted = 0
    delete = getattr(bucket, "delete", None)
    if delete is None:
        raise TypeError("bucket does not support delete(uri)")
    seen: set[str] = set()
    for prefix_key in paths.run_prefixes(netuid, run_id):
        for uri in bucket.list(bucket.uri_for_key(prefix_key)):
            if uri in seen:
                continue
            delete(uri)
            seen.add(uri)
            deleted += 1
    return deleted
