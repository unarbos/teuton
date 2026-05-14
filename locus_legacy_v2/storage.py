"""Bucket abstraction.

Two backends:
  - `LocalBucket` — flat-file under <root>/<bucket>/<key>; default for tests.
  - `S3Bucket`    — boto3-backed; used for runs against real S3 / MinIO / R2.

URIs are always written `s3://<bucket>/<key>`. The storage backend interprets
the URI accordingly. The orchestrator + worker code paths only depend on the
methods exposed here (get/put/head/exists/list/...), so swapping backends is
purely a construction-time choice.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from typing import Iterable


_S3_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def parse_uri(uri: str) -> tuple[str, str]:
    m = _S3_RE.match(uri)
    if not m:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return m.group(1), m.group(2)


def join_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


class LocalBucket:
    """A flat-file bucket: <root>/<bucket>/<key>."""

    def __init__(self, root: str, bucket: str) -> None:
        self.root = os.path.abspath(root)
        self.bucket = bucket
        os.makedirs(os.path.join(self.root, self.bucket), exist_ok=True)

    # --- path / URI helpers ---

    def _path_for_uri(self, uri: str) -> str:
        b, k = parse_uri(uri)
        return os.path.join(self.root, b, k)

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return join_uri(bucket or self.bucket, key)

    # --- core ops (URI-keyed) ---

    def put(self, uri: str, data: bytes) -> None:
        path = self._path_for_uri(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def get(self, uri: str) -> bytes:
        path = self._path_for_uri(uri)
        with open(path, "rb") as f:
            return f.read()

    def exists(self, uri: str) -> bool:
        return os.path.exists(self._path_for_uri(uri))

    def head(self, uri: str) -> dict[str, int] | None:
        path = self._path_for_uri(uri)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return None
        return {"size_bytes": st.st_size, "mtime_unix": int(st.st_mtime)}

    def delete(self, uri: str) -> None:
        path = self._path_for_uri(uri)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def list(self, prefix_uri: str) -> list[str]:
        try:
            b, k = parse_uri(prefix_uri)
        except ValueError:
            b, k = self.bucket, prefix_uri
        base = os.path.join(self.root, b)
        prefix_path = os.path.join(base, k)
        out: list[str] = []
        if os.path.isdir(prefix_path):
            for dp, _dn, files in os.walk(prefix_path):
                for fn in files:
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, base)
                    out.append(join_uri(b, rel.replace(os.sep, "/")))
        else:
            walk_root = os.path.dirname(prefix_path) or base
            if os.path.isdir(walk_root):
                for dp, _dn, files in os.walk(walk_root):
                    for fn in files:
                        full = os.path.join(dp, fn)
                        rel = os.path.relpath(full, base).replace(os.sep, "/")
                        if rel.startswith(k):
                            out.append(join_uri(b, rel))
        out.sort()
        return out

    # --- JSON convenience ---

    def get_json(self, uri: str) -> dict:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: dict | list) -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.put(uri, body)

    # --- maintenance ---

    def ensure_bucket(self) -> None:
        os.makedirs(os.path.join(self.root, self.bucket), exist_ok=True)

    def wipe(self) -> None:
        path = os.path.join(self.root, self.bucket)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def wipe_run(self, run_id: str) -> int:
        """Remove all files under runs/<run_id>/. Returns count deleted."""
        prefix = os.path.join(self.root, self.bucket, "runs", run_id)
        if not os.path.exists(prefix):
            return 0
        n = sum(len(files) for _, _, files in os.walk(prefix))
        shutil.rmtree(prefix)
        return n


# --------------------------------------------------------------------------- #
# S3 backend
# --------------------------------------------------------------------------- #


class S3Bucket:
    """Boto3-backed bucket. Same interface as LocalBucket.

    boto3 is imported lazily so the rest of the package works without it.
    """

    def __init__(
        self,
        *,
        bucket: str,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise ImportError(
                "boto3 is required for S3Bucket. Install with `uv pip install boto3`."
            ) from e

        cfg = Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "adaptive"},
            region_name=region,
            connect_timeout=10,
            read_timeout=30,
        )
        self._client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=cfg,
        )
        self.bucket = bucket
        self.region = region

    # --- helpers ---

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return join_uri(bucket or self.bucket, key)

    @staticmethod
    def _is_404(err) -> bool:
        from botocore.exceptions import ClientError
        if not isinstance(err, ClientError):
            return False
        code = err.response.get("Error", {}).get("Code", "")
        status = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in ("NoSuchKey", "NotFound", "404") or status == 404

    # --- core ops ---

    def put(self, uri: str, data: bytes) -> None:
        b, k = parse_uri(uri)
        self._client.put_object(
            Bucket=b,
            Key=k,
            Body=data,
            ServerSideEncryption="AES256",
        )

    def get(self, uri: str) -> bytes:
        b, k = parse_uri(uri)
        from botocore.exceptions import ClientError
        try:
            resp = self._client.get_object(Bucket=b, Key=k)
        except ClientError as e:
            if self._is_404(e):
                raise FileNotFoundError(uri) from e
            raise
        return resp["Body"].read()

    def exists(self, uri: str) -> bool:
        b, k = parse_uri(uri)
        from botocore.exceptions import ClientError
        try:
            self._client.head_object(Bucket=b, Key=k)
            return True
        except ClientError as e:
            if self._is_404(e):
                return False
            raise

    def head(self, uri: str) -> dict[str, int] | None:
        b, k = parse_uri(uri)
        from botocore.exceptions import ClientError
        try:
            resp = self._client.head_object(Bucket=b, Key=k)
        except ClientError as e:
            if self._is_404(e):
                return None
            raise
        return {
            "size_bytes": int(resp["ContentLength"]),
            "mtime_unix": int(resp["LastModified"].timestamp()),
        }

    def delete(self, uri: str) -> None:
        b, k = parse_uri(uri)
        from botocore.exceptions import ClientError
        try:
            self._client.delete_object(Bucket=b, Key=k)
        except ClientError:
            pass

    def list(self, prefix_uri: str) -> list[str]:
        try:
            b, k = parse_uri(prefix_uri)
        except ValueError:
            b, k = self.bucket, prefix_uri
        out: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=b, Prefix=k):
            for obj in page.get("Contents", []) or []:
                out.append(join_uri(b, obj["Key"]))
        out.sort()
        return out

    # --- JSON convenience ---

    def get_json(self, uri: str) -> dict:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: dict | list) -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.put(uri, body)

    # --- maintenance ---

    def ensure_bucket(self) -> None:
        # Bucket should already exist; this is a no-op for S3.
        # We could do a HeadBucket here for verification but skip for speed.
        pass

    def wipe_run(self, run_id: str) -> int:
        """Batch-delete all objects under runs/<run_id>/. Returns count deleted."""
        prefix = f"runs/{run_id}/"
        keys: list[dict[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append({"Key": obj["Key"]})
        total = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": batch, "Quiet": True},
            )
            total += len(batch)
        return total

    def wipe(self) -> int:
        """Delete EVERY object in the bucket. Aggressive — for tests only."""
        keys: list[dict[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []) or []:
                keys.append({"Key": obj["Key"]})
        total = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": batch, "Quiet": True},
            )
            total += len(batch)
        return total


def open_bucket(root: str, bucket: str) -> LocalBucket:
    return LocalBucket(root=root, bucket=bucket)
