"""Presigned URL grant brokers."""
from __future__ import annotations

import time

from teuton_core.protocol import PresignedUrlGrant
from .storage import ObjectStore, parse_uri


class PresignedUrlBroker:
    def get_grant(self, uri: str, *, expires_in: int) -> PresignedUrlGrant:
        raise NotImplementedError

    def put_grant(self, uri: str, *, expires_in: int, content_sha256: str | None = None) -> PresignedUrlGrant:
        raise NotImplementedError


class LocalGrantBroker(PresignedUrlBroker):
    """Local/direct test broker. The URL is the canonical URI."""

    def get_grant(self, uri: str, *, expires_in: int) -> PresignedUrlGrant:
        return PresignedUrlGrant("GET", uri, uri, int(time.time()) + int(expires_in))

    def put_grant(self, uri: str, *, expires_in: int, content_sha256: str | None = None) -> PresignedUrlGrant:
        return PresignedUrlGrant("PUT", uri, uri, int(time.time()) + int(expires_in), content_sha256)


class S3PresignedUrlBroker(PresignedUrlBroker):
    def __init__(self, bucket: ObjectStore) -> None:
        client = getattr(bucket, "_client", None)
        if client is None:
            raise TypeError("S3PresignedUrlBroker requires an S3Bucket with a boto3 client")
        self.client = client

    def get_grant(self, uri: str, *, expires_in: int) -> PresignedUrlGrant:
        bucket, key = parse_uri(uri)
        url = self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires_in),
        )
        return PresignedUrlGrant("GET", uri, url, int(time.time()) + int(expires_in))

    def put_grant(self, uri: str, *, expires_in: int, content_sha256: str | None = None) -> PresignedUrlGrant:
        bucket, key = parse_uri(uri)
        url = self.client.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires_in),
        )
        return PresignedUrlGrant("PUT", uri, url, int(time.time()) + int(expires_in), content_sha256)


def broker_for_mode(mode: str, bucket: ObjectStore) -> PresignedUrlBroker | None:
    if mode == "direct":
        return None
    if mode == "local":
        return LocalGrantBroker()
    if mode == "presigned":
        return S3PresignedUrlBroker(bucket)
    raise ValueError(f"unknown grant mode: {mode}")
