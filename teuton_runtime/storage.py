"""Object-store abstractions for Teuton v3."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from typing import Protocol


_S3_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def parse_uri(uri: str) -> tuple[str, str]:
    match = _S3_RE.match(uri)
    if not match:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return match.group(1), match.group(2)


def join_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


class LocalBucket:
    def __init__(self, root: str, bucket: str) -> None:
        self.root = os.path.abspath(root)
        self.bucket = bucket
        os.makedirs(os.path.join(self.root, self.bucket), exist_ok=True)

    def _path_for_uri(self, uri: str) -> str:
        bucket, key = parse_uri(uri)
        return os.path.join(self.root, bucket, key)

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return join_uri(bucket or self.bucket, key)

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
        with open(self._path_for_uri(uri), "rb") as f:
            return f.read()

    def exists(self, uri: str) -> bool:
        return os.path.exists(self._path_for_uri(uri))

    def head(self, uri: str) -> dict[str, int] | None:
        try:
            stat = os.stat(self._path_for_uri(uri))
        except FileNotFoundError:
            return None
        return {"size_bytes": stat.st_size, "mtime_unix": int(stat.st_mtime)}

    def delete(self, uri: str) -> None:
        try:
            os.unlink(self._path_for_uri(uri))
        except FileNotFoundError:
            pass

    def list(self, prefix_uri: str) -> list[str]:
        try:
            bucket, key = parse_uri(prefix_uri)
        except ValueError:
            bucket, key = self.bucket, prefix_uri
        base = os.path.join(self.root, bucket)
        prefix_path = os.path.join(base, key)
        out: list[str] = []
        if os.path.isdir(prefix_path):
            for dirpath, _dirnames, filenames in os.walk(prefix_path):
                for filename in filenames:
                    rel = os.path.relpath(os.path.join(dirpath, filename), base).replace(os.sep, "/")
                    out.append(join_uri(bucket, rel))
        else:
            walk_root = os.path.dirname(prefix_path) or base
            if os.path.isdir(walk_root):
                for dirpath, _dirnames, filenames in os.walk(walk_root):
                    for filename in filenames:
                        rel = os.path.relpath(os.path.join(dirpath, filename), base).replace(os.sep, "/")
                        if rel.startswith(key):
                            out.append(join_uri(bucket, rel))
        out.sort()
        return out

    def get_json(self, uri: str) -> dict:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: dict | list) -> None:
        self.put(uri, json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def ensure_bucket(self) -> None:
        os.makedirs(os.path.join(self.root, self.bucket), exist_ok=True)

    def wipe(self) -> None:
        path = os.path.join(self.root, self.bucket)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def wipe_run(self, run_id: str) -> int:
        prefix = os.path.join(self.root, self.bucket, "runs", run_id)
        if not os.path.exists(prefix):
            return 0
        count = sum(len(files) for _, _, files in os.walk(prefix))
        shutil.rmtree(prefix)
        return count


class S3Bucket:
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
            raise ImportError("boto3 is required for S3Bucket. Install with `uv pip install boto3`.") from e

        config = Config(
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
            config=config,
        )
        self.bucket = bucket
        self.region = region

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

    def put(self, uri: str, data: bytes) -> None:
        bucket, key = parse_uri(uri)
        self._client.put_object(Bucket=bucket, Key=key, Body=data, ServerSideEncryption="AES256")

    def get(self, uri: str) -> bytes:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if self._is_404(e):
                raise FileNotFoundError(uri) from e
            raise
        return response["Body"].read()

    def exists(self, uri: str) -> bool:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if self._is_404(e):
                return False
            raise

    def head(self, uri: str) -> dict[str, int] | None:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if self._is_404(e):
                return None
            raise
        return {"size_bytes": int(response["ContentLength"]), "mtime_unix": int(response["LastModified"].timestamp())}

    def delete(self, uri: str) -> None:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except ClientError:
            pass

    def list(self, prefix_uri: str) -> list[str]:
        try:
            bucket, key = parse_uri(prefix_uri)
        except ValueError:
            bucket, key = self.bucket, prefix_uri
        out: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            for obj in page.get("Contents", []) or []:
                out.append(join_uri(bucket, obj["Key"]))
        out.sort()
        return out

    def get_json(self, uri: str) -> dict:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: dict | list) -> None:
        self.put(uri, json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def ensure_bucket(self) -> None:
        pass

    def wipe_run(self, run_id: str) -> int:
        prefix = f"runs/{run_id}/"
        keys: list[dict[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append({"Key": obj["Key"]})
        total = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})
            total += len(batch)
        return total

    def wipe(self) -> int:
        keys: list[dict[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []) or []:
                keys.append({"Key": obj["Key"]})
        total = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})
            total += len(batch)
        return total


class ObjectStore(Protocol):
    bucket: str

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str: ...
    def put(self, uri: str, data: bytes) -> None: ...
    def get(self, uri: str) -> bytes: ...
    def exists(self, uri: str) -> bool: ...
    def delete(self, uri: str) -> None: ...
    def list(self, prefix_uri: str) -> list[str]: ...
    def get_json(self, uri: str) -> dict: ...
    def put_json(self, uri: str, value: dict | list) -> None: ...


def open_local_bucket(root: str, bucket: str) -> LocalBucket:
    return LocalBucket(root=root, bucket=bucket)
