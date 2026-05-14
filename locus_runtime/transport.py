"""Artifact transports for direct bucket I/O and presigned URL I/O."""
from __future__ import annotations

import time
import urllib.request

from locus_core.protocol import PresignedUrlGrant
from .storage import ObjectStore


class ArtifactTransport:
    def get(self, uri: str, grant: PresignedUrlGrant | None = None) -> bytes:
        raise NotImplementedError

    def put(self, uri: str, data: bytes, grant: PresignedUrlGrant | None = None) -> None:
        raise NotImplementedError


class DirectArtifactTransport(ArtifactTransport):
    def __init__(self, bucket: ObjectStore) -> None:
        self.bucket = bucket

    def get(self, uri: str, grant: PresignedUrlGrant | None = None) -> bytes:
        return self.bucket.get(uri)

    def put(self, uri: str, data: bytes, grant: PresignedUrlGrant | None = None) -> None:
        self.bucket.put(uri, data)


class PresignedArtifactTransport(ArtifactTransport):
    def __init__(self, bucket: ObjectStore | None = None) -> None:
        # bucket is used for local-grant URLs where url == canonical s3:// uri.
        self.bucket = bucket

    def get(self, uri: str, grant: PresignedUrlGrant | None = None) -> bytes:
        grant = self._validate(grant, "GET", uri)
        if grant.url.startswith("s3://"):
            if self.bucket is None:
                raise ValueError("local presigned grant requires bucket transport fallback")
            return self.bucket.get(grant.url)
        with urllib.request.urlopen(grant.url) as resp:
            return resp.read()

    def put(self, uri: str, data: bytes, grant: PresignedUrlGrant | None = None) -> None:
        grant = self._validate(grant, "PUT", uri)
        if grant.content_sha256 is not None:
            import hashlib
            if hashlib.sha256(data).hexdigest() != grant.content_sha256:
                raise ValueError("presigned grant content sha256 mismatch")
        if grant.url.startswith("s3://"):
            if self.bucket is None:
                raise ValueError("local presigned grant requires bucket transport fallback")
            self.bucket.put(grant.url, data)
            return
        req = urllib.request.Request(grant.url, data=data, method="PUT")
        with urllib.request.urlopen(req) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"PUT failed with status {resp.status}")

    @staticmethod
    def _validate(grant: PresignedUrlGrant | None, method: str, uri: str) -> PresignedUrlGrant:
        if grant is None:
            raise ValueError(f"{method} grant required for {uri}")
        if grant.method != method:
            raise ValueError(f"grant method mismatch: {grant.method} != {method}")
        if grant.canonical_uri != uri:
            raise ValueError("grant canonical URI mismatch")
        if grant.expires_unix < int(time.time()):
            raise ValueError("grant expired")
        return grant
