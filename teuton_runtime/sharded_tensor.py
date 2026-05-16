"""Helpers for logical tensors stored as deterministic per-rank shards."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from teuton_core.protocol import ShardedTensorManifest, ShardSpec
from . import tensor_io
from .transport import ArtifactTransport


def shard_tensor(tensor: torch.Tensor, *, world_size: int, dim: int = 0) -> list[torch.Tensor]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if dim < 0:
        dim += tensor.dim()
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"invalid shard dim {dim} for tensor with {tensor.dim()} dims")
    return list(torch.tensor_split(tensor, world_size, dim=dim))


def join_shards(shards: list[torch.Tensor], *, dim: int = 0) -> torch.Tensor:
    if not shards:
        raise ValueError("cannot join empty shard list")
    return torch.cat(shards, dim=dim).contiguous()


def encode_manifest(manifest: ShardedTensorManifest) -> bytes:
    return json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_manifest(body: bytes) -> ShardedTensorManifest:
    return ShardedTensorManifest.from_dict(json.loads(body.decode("utf-8")))


def put_sharded_tensor(
    transport: ArtifactTransport,
    manifest_uri: str,
    tensor: torch.Tensor,
    *,
    name: str,
    world_size: int,
    partition_dim: int = 0,
    grants: dict[str, Any] | None = None,
) -> ShardedTensorManifest:
    grants = grants or {}
    shards: list[ShardSpec] = []
    base = manifest_uri[:-5] if manifest_uri.endswith(".json") else manifest_uri
    for rank, shard in enumerate(shard_tensor(tensor.detach().cpu(), world_size=world_size, dim=partition_dim)):
        shard_uri = f"{base}.rank{rank}.bin"
        body = tensor_io.encode_tensor(shard)
        transport.put(shard_uri, body, grants.get(shard_uri))
        shards.append(
            ShardSpec(
                rank=rank,
                uri=shard_uri,
                offset=0,
                length=int(shard.shape[partition_dim]) if shard.dim() else 1,
                sha256=hashlib.sha256(body).hexdigest(),
                size_bytes=len(body),
            )
        )
    manifest = ShardedTensorManifest(
        name=name,
        shape=list(tensor.shape),
        dtype=tensor_io.wire_dtype(tensor.detach().cpu()),
        partition_dim=partition_dim,
        world_size=world_size,
        shards=shards,
    )
    transport.put(manifest_uri, encode_manifest(manifest), grants.get(manifest_uri))
    return manifest


def get_sharded_tensor(
    transport: ArtifactTransport,
    manifest_uri: str,
    *,
    grants: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    grants = grants or {}
    manifest = decode_manifest(transport.get(manifest_uri, grants.get(manifest_uri)))
    shards: list[torch.Tensor] = []
    for shard in sorted(manifest.shards, key=lambda s: s.rank):
        body = transport.get(shard.uri, grants.get(shard.uri))
        sha = hashlib.sha256(body).hexdigest()
        if shard.sha256 is not None and shard.sha256 != sha:
            raise ValueError(f"shard digest mismatch for {shard.uri}")
        shards.append(tensor_io.decode_tensor(body))
    out = join_shards(shards, dim=manifest.partition_dim)
    if list(out.shape) != list(manifest.shape):
        raise ValueError(f"joined sharded tensor shape {list(out.shape)} != manifest shape {manifest.shape}")
    if device is not None:
        out = out.to(device)
    return out
