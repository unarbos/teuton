"""Build the ``ObjectStore`` the dashboard reads from.

Honors the same env names the legacy teuton-v3 CLI does so deployments can
drop in without changing variables.
"""
from __future__ import annotations

import os

from teuton_runtime.storage import LocalBucket, ObjectStore, S3Bucket

from .settings import Settings


def build_bucket(settings: Settings) -> ObjectStore:
    if settings.s3_bucket:
        return S3Bucket(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            access_key=settings.aws_access_key_id,
            secret_key=settings.aws_secret_access_key,
            endpoint_url=settings.s3_endpoint_url or None,
        )
    root = settings.local_bucket_root or os.environ.get("TEUTON_LOCAL_ROOT") or "."
    name = settings.local_bucket_name or os.environ.get("TEUTON_LOCAL_BUCKET") or "local-bucket"
    return LocalBucket(root=root, bucket=name)
