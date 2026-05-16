"""Shard-aware deterministic data sampler.

Resolves `(shard_idx, shard_seed, step, bsz, seq_len)` from a JobManifest
into concrete `(input_ids, target_ids)` microbatches. Both workers and
validators use this same module so a worker's training run is exactly
reproducible by a validator running spot-check.

Determinism contract:
- Same `(shard_idx, shard_seed, step, bsz, seq_len)` → bit-identical
  `(input_ids, target_ids)` regardless of host, given the same shard blob
  on the bucket and the same PyTorch CPU generator.
- The shard blob itself is content-addressed by sha256 in the manifest;
  `load_shard()` re-verifies the sha on every fetch by default so a
  bucket-level corruption can't desync workers from validators silently.

Caching:
- Manifest cached in-process after first read.
- Shard tensor cached on disk under `local_cache_dir` keyed by sha256.
  Subsequent fetches of the same shard hit the disk cache.

Usage:

    bucket = S3Bucket(...)
    cache  = ShardManifestCache(bucket)
    loader = ShardDataLoader(bucket, cache, local_cache_dir="/tmp/teuton_data/shard_cache")

    # In a worker doing H inner steps on shard 42 with seed 1234:
    for step, (x, y) in loader.iter_microbatches(
            shard_idx=42, shard_seed=1234, H=50, bsz=16, seq_len=2048):
        loss = model(x, y).backward()
        ...
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .. import tensor_io


# Path conventions (must match teuton.data.fineweb_shards) ------------------

DATASET_PREFIX = "data/fineweb-edu"
MANIFEST_KEY = f"{DATASET_PREFIX}/manifest.json"
SHARDS_PREFIX = f"{DATASET_PREFIX}/shards"


# ---------------------------------------------------------------------------
# Manifest cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardEntry:
    idx: int
    sha256: str
    n_tokens: int
    uri: str


class ShardManifestCache:
    """Lazy, thread-safe manifest fetch + lookup."""

    def __init__(self, bucket, *, refresh_after_sec: float | None = None) -> None:
        self._bucket = bucket
        self._lock = threading.Lock()
        self._raw: dict | None = None
        self._entries: list[ShardEntry] | None = None
        self._loaded_unix: float = 0.0
        self._refresh_after = refresh_after_sec

    def _maybe_load(self) -> None:
        with self._lock:
            stale = (
                self._raw is None
                or (self._refresh_after is not None
                    and time.time() - self._loaded_unix > self._refresh_after)
            )
            if not stale:
                return
            uri = self._bucket.uri_for_key(MANIFEST_KEY)
            self._raw = self._bucket.get_json(uri)
            self._entries = [
                ShardEntry(
                    idx=int(s["idx"]),
                    sha256=str(s["sha256"]),
                    n_tokens=int(s["n_tokens"]),
                    uri=str(s["uri"]),
                )
                for s in self._raw["shards"]
            ]
            for i, e in enumerate(self._entries):
                if e.idx != i:
                    raise RuntimeError(
                        f"manifest shard list out of order: position {i} has idx={e.idx}"
                    )
            self._loaded_unix = time.time()

    @property
    def manifest(self) -> dict:
        self._maybe_load()
        assert self._raw is not None
        return self._raw

    @property
    def n_shards(self) -> int:
        self._maybe_load()
        assert self._entries is not None
        return len(self._entries)

    @property
    def shard_n_tokens(self) -> int:
        return int(self.manifest["shard_n_tokens"])

    def entry(self, shard_idx: int) -> ShardEntry:
        self._maybe_load()
        assert self._entries is not None
        if not (0 <= shard_idx < len(self._entries)):
            raise IndexError(
                f"shard_idx={shard_idx} out of range [0, {len(self._entries)})"
            )
        return self._entries[shard_idx]


# ---------------------------------------------------------------------------
# Shard loader (with on-disk + in-process cache)
# ---------------------------------------------------------------------------


class ShardDataLoader:
    """Fetch + verify + cache shards; produce deterministic microbatches."""

    def __init__(
        self,
        bucket,
        manifest_cache: ShardManifestCache,
        *,
        local_cache_dir: str | os.PathLike | None = None,
        verify_sha: bool = True,
        in_memory_max_shards: int = 4,
    ) -> None:
        self._bucket = bucket
        self._manifest = manifest_cache
        self._local_cache_dir = (
            Path(local_cache_dir) if local_cache_dir is not None else None
        )
        if self._local_cache_dir is not None:
            self._local_cache_dir.mkdir(parents=True, exist_ok=True)
        self._verify_sha = verify_sha
        self._mem_lock = threading.Lock()
        self._mem_cache: dict[str, torch.Tensor] = {}
        self._mem_lru: list[str] = []
        self._mem_max = in_memory_max_shards

    # --- shard fetch ------------------------------------------------------

    def _disk_cache_path(self, sha256: str) -> Path | None:
        if self._local_cache_dir is None:
            return None
        return self._local_cache_dir / f"{sha256}.bin"

    def _put_mem(self, sha256: str, tensor: torch.Tensor) -> None:
        with self._mem_lock:
            if sha256 in self._mem_cache:
                self._mem_lru.remove(sha256)
            elif len(self._mem_lru) >= self._mem_max:
                ev = self._mem_lru.pop(0)
                self._mem_cache.pop(ev, None)
            self._mem_cache[sha256] = tensor
            self._mem_lru.append(sha256)

    def _get_mem(self, sha256: str) -> torch.Tensor | None:
        with self._mem_lock:
            t = self._mem_cache.get(sha256)
            if t is not None:
                self._mem_lru.remove(sha256)
                self._mem_lru.append(sha256)
            return t

    def load_shard(self, shard_idx: int) -> torch.Tensor:
        """Fetch shard tokens as a 1-D int32 tensor. Memo-cached + sha-verified."""
        entry = self._manifest.entry(shard_idx)
        cached = self._get_mem(entry.sha256)
        if cached is not None:
            return cached

        body: bytes | None = None
        disk = self._disk_cache_path(entry.sha256)
        if disk is not None and disk.exists():
            body = disk.read_bytes()

        if body is None:
            body = self._bucket.get(entry.uri)
            if disk is not None:
                tmp = disk.with_suffix(disk.suffix + ".tmp")
                tmp.write_bytes(body)
                os.replace(tmp, disk)

        if self._verify_sha:
            got_sha = hashlib.sha256(body).hexdigest()
            if got_sha != entry.sha256:
                raise RuntimeError(
                    f"shard idx={shard_idx} sha mismatch: "
                    f"manifest={entry.sha256[:16]}... bucket={got_sha[:16]}..."
                )

        tensor = tensor_io.decode_tensor(body)
        if tensor.dtype != torch.int32:
            tensor = tensor.to(torch.int32)
        if tensor.dim() != 1:
            raise RuntimeError(f"shard idx={shard_idx} expected 1-D, got shape {tuple(tensor.shape)}")
        if tensor.numel() != entry.n_tokens:
            raise RuntimeError(
                f"shard idx={shard_idx} expected {entry.n_tokens} tokens, got {tensor.numel()}"
            )

        self._put_mem(entry.sha256, tensor)
        return tensor

    # --- deterministic microbatch sampler --------------------------------

    @staticmethod
    def _step_seed(shard_seed: int, step: int) -> int:
        """Mix shard_seed and step into a 64-bit deterministic generator seed."""
        h = hashlib.blake2b(digest_size=8)
        h.update(int(shard_seed).to_bytes(8, "little", signed=False))
        h.update(int(step).to_bytes(8, "little", signed=False))
        return int.from_bytes(h.digest(), "little", signed=False) & ((1 << 63) - 1)

    def microbatch(
        self,
        *,
        shard_idx: int,
        shard_seed: int,
        step: int,
        bsz: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (input_ids, target_ids) microbatch.

        - input_ids:  long tensor of shape (bsz, seq_len)
        - target_ids: long tensor of shape (bsz, seq_len), shifted by 1

        Deterministic in (shard_idx, shard_seed, step, bsz, seq_len) given
        the same shard blob on the bucket.
        """
        if bsz <= 0 or seq_len <= 0:
            raise ValueError(f"bsz={bsz} seq_len={seq_len} must be positive")
        tokens = self.load_shard(shard_idx)
        n = int(tokens.numel())
        # Need seq_len + 1 contiguous tokens per row (for the shift).
        max_start_excl = n - seq_len - 1
        if max_start_excl <= 0:
            raise RuntimeError(
                f"shard idx={shard_idx} too small for seq_len={seq_len}: n={n}"
            )

        seed = self._step_seed(shard_seed, step)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        starts = torch.randint(
            low=0, high=max_start_excl, size=(bsz,), generator=gen, dtype=torch.int64
        )
        x_rows = []
        y_rows = []
        for s in starts.tolist():
            window = tokens[s : s + seq_len + 1]
            x_rows.append(window[:seq_len])
            y_rows.append(window[1 : seq_len + 1])
        x = torch.stack(x_rows, dim=0).to(torch.long).contiguous()
        y = torch.stack(y_rows, dim=0).to(torch.long).contiguous()
        return x, y

    def iter_microbatches(
        self,
        *,
        shard_idx: int,
        shard_seed: int,
        H: int,
        bsz: int,
        seq_len: int,
    ) -> Iterator[tuple[int, tuple[torch.Tensor, torch.Tensor]]]:
        """Yield (step, (x, y)) for step in 0..H-1."""
        for step in range(H):
            yield step, self.microbatch(
                shard_idx=shard_idx,
                shard_seed=shard_seed,
                step=step,
                bsz=bsz,
                seq_len=seq_len,
            )


__all__ = [
    "DATASET_PREFIX",
    "MANIFEST_KEY",
    "SHARDS_PREFIX",
    "ShardEntry",
    "ShardManifestCache",
    "ShardDataLoader",
]
