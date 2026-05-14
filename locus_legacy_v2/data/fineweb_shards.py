"""Content-addressed FineWeb shards on the bucket.

Tokenizes a slice of HuggingFaceFW/fineweb-edu (sample-10BT) with the GPT-2 BPE
tokenizer (tiktoken) and writes it to the bucket as a set of fixed-size shards.

Path layout (GLOBAL — shared across runs, not under any runs/<id>/ prefix):

    data/fineweb-edu/manifest.json
    data/fineweb-edu/shards/<sha256>.bin

The manifest is the source of truth for shard_idx -> URI. Each shard blob is
exactly `shard_n_tokens` int32 tokens encoded with tensor_io's wire format
(4-byte LE header_len || JSON header || raw bytes).

Workers reference shards by `shard_idx` in their JobManifest params; they
look up the URI in the cached manifest, fetch the (small) blob, and sample
microbatches from it deterministically using the round/job-supplied seed.

CLI:

    # Tokenize and upload (default 1M tokens/shard, 1B tokens total -> 1024 shards)
    python -m locus.data.fineweb_shards prepare \
        --target-tokens 1_000_000_000 \
        --shard-tokens  1_048_576

    # Inspect what's on the bucket
    python -m locus.data.fineweb_shards inspect

    # Re-upload only missing shards (idempotent)
    python -m locus.data.fineweb_shards prepare --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .. import tensor_io
from ..storage import LocalBucket, S3Bucket


# Path conventions ----------------------------------------------------------

DATASET_PREFIX = "data/fineweb-edu"
MANIFEST_KEY = f"{DATASET_PREFIX}/manifest.json"
SHARDS_PREFIX = f"{DATASET_PREFIX}/shards"

MANIFEST_VERSION = 1
DEFAULT_TOKENIZER = "tiktoken/gpt2"
DEFAULT_SOURCE = "HuggingFaceFW/fineweb-edu/sample-10BT"
EOT_ID = 50256
DTYPE = "int32"


# ---------------------------------------------------------------------------
# Bucket construction
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load .env from repo root if present (mirrors bench/dist.py)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for cand in [here.parent.parent.parent.parent / ".env",
                 here.parent.parent.parent / ".env",
                 Path("/root/.env"),
                 Path("/root/Locus/.env")]:
        if cand.exists():
            load_dotenv(cand, override=False)
            return


def _build_bucket() -> LocalBucket | S3Bucket:
    bucket_name = os.environ.get("S3_BUCKET") or os.environ.get("LOCUS_S3_BUCKET")
    if bucket_name:
        return S3Bucket(
            bucket=bucket_name,
            region=os.environ.get("S3_REGION", os.environ.get("AWS_REGION", "us-east-1")),
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.environ.get("S3_ENDPOINT") or None,
        )
    root = os.environ.get("LOCUS_LOCAL_BUCKET", "/tmp/locus_bucket")
    return LocalBucket(root=root, bucket=os.environ.get("LOCUS_BUCKET", "locus-test"))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _stream_fineweb_edu(target_tokens: int) -> Iterable[str]:
    import datasets
    target_chars = target_tokens * 5
    seen = 0
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue
        yield text
        seen += len(text)
        if seen >= target_chars:
            return


def _tokenize_streaming(target_tokens: int, *, eot_id: int = EOT_ID) -> np.ndarray:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    out = np.empty((target_tokens,), dtype=np.int32)
    n = 0
    docs = 0
    t0 = time.time()
    for text in _stream_fineweb_edu(target_tokens):
        ids = enc.encode_ordinary(text)
        ids.append(eot_id)
        room = target_tokens - n
        if len(ids) > room:
            ids = ids[:room]
        if not ids:
            break
        out[n : n + len(ids)] = ids
        n += len(ids)
        docs += 1
        if docs % 500 == 0:
            print(
                f"  tokenized {n / 1e6:.2f}M / {target_tokens / 1e6:.2f}M tokens "
                f"({docs} docs, {n / max(time.time() - t0, 1e-3):.0f} tok/sec)",
                flush=True,
            )
        if n >= target_tokens:
            break
    if n < target_tokens:
        print(f"  WARNING: only got {n} tokens, wanted {target_tokens}", flush=True)
        out = out[:n]
    print(f"  done: {n / 1e6:.2f}M tokens in {time.time() - t0:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Sharding + upload
# ---------------------------------------------------------------------------


@dataclass
class ShardEntry:
    idx: int
    sha256: str
    n_tokens: int
    uri: str


def _shard_uri(bucket: LocalBucket | S3Bucket, sha256: str) -> str:
    return bucket.uri_for_key(f"{SHARDS_PREFIX}/{sha256}.bin")


def _encode_shard(arr_slice: np.ndarray) -> tuple[bytes, str]:
    """Encode a 1-D int32 array as a tensor_io blob; return (body, sha256)."""
    if arr_slice.dtype != np.int32:
        arr_slice = arr_slice.astype(np.int32, copy=False)
    t = torch.from_numpy(arr_slice.copy())
    body = tensor_io.encode_tensor(t)
    sha = hashlib.sha256(body).hexdigest()
    return body, sha


def _existing_shard_shas(bucket) -> set[str]:
    out: set[str] = set()
    prefix_uri = bucket.uri_for_key(SHARDS_PREFIX + "/")
    try:
        uris = bucket.list(prefix_uri)
    except Exception:
        return out
    for u in uris:
        name = u.rsplit("/", 1)[-1]
        if name.endswith(".bin"):
            out.add(name[:-4])
    return out


def _upload_shard_if_needed(bucket, sha256: str, body: bytes,
                             skip_existing: bool, existing: set[str]) -> bool:
    if skip_existing and sha256 in existing:
        return False
    uri = _shard_uri(bucket, sha256)
    if skip_existing and bucket.exists(uri):
        existing.add(sha256)
        return False
    bucket.put(uri, body)
    return True


def _build_shard_tasks(arr: np.ndarray, shard_n_tokens: int) -> list[tuple[int, slice]]:
    n = arr.shape[0]
    tasks: list[tuple[int, slice]] = []
    idx = 0
    start = 0
    while start + shard_n_tokens <= n:
        tasks.append((idx, slice(start, start + shard_n_tokens)))
        start += shard_n_tokens
        idx += 1
    if start < n and (n - start) >= max(1024, shard_n_tokens // 64):
        tasks.append((idx, slice(start, n)))
    return tasks


def _shard_and_upload(
    arr: np.ndarray,
    *,
    bucket,
    shard_n_tokens: int,
    skip_existing: bool,
    n_workers: int,
) -> list[ShardEntry]:
    tasks = _build_shard_tasks(arr, shard_n_tokens)
    print(f"  encoding + uploading {len(tasks)} shards "
          f"({shard_n_tokens / 1e6:.2f}M tokens each, {n_workers} workers) ...",
          flush=True)

    existing = _existing_shard_shas(bucket) if skip_existing else set()
    if existing:
        print(f"  bucket has {len(existing)} existing shard blobs (will skip duplicates)",
              flush=True)

    entries: list[ShardEntry | None] = [None] * len(tasks)
    n_uploaded = 0
    n_skipped = 0
    bytes_uploaded = 0
    t0 = time.time()

    def _job(idx: int, sl: slice) -> tuple[int, ShardEntry, bool, int]:
        body, sha = _encode_shard(arr[sl])
        wrote = _upload_shard_if_needed(bucket, sha, body, skip_existing, existing)
        return idx, ShardEntry(
            idx=idx,
            sha256=sha,
            n_tokens=int(sl.stop - sl.start),
            uri=_shard_uri(bucket, sha),
        ), wrote, len(body)

    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_job, i, sl) for i, sl in tasks]
        for done_count, fut in enumerate(cf.as_completed(futures), start=1):
            idx, entry, wrote, body_size = fut.result()
            entries[idx] = entry
            if wrote:
                n_uploaded += 1
                bytes_uploaded += body_size
            else:
                n_skipped += 1
            if done_count % 50 == 0 or done_count == len(tasks):
                elapsed = max(time.time() - t0, 1e-3)
                print(
                    f"    {done_count:5d}/{len(tasks)}  "
                    f"uploaded={n_uploaded} skipped={n_skipped}  "
                    f"{bytes_uploaded / 1e6:.1f} MB  "
                    f"({bytes_uploaded / elapsed / 1e6:.1f} MB/s effective)",
                    flush=True,
                )

    out = [e for e in entries if e is not None]
    if len(out) != len(entries):
        raise RuntimeError("internal: not all shard tasks produced entries")
    return out


# ---------------------------------------------------------------------------
# Manifest read/write
# ---------------------------------------------------------------------------


def _build_manifest(
    *,
    entries: list[ShardEntry],
    shard_n_tokens: int,
    source: str,
    tokenizer: str,
) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "source": source,
        "tokenizer": tokenizer,
        "dtype": DTYPE,
        "shard_n_tokens": shard_n_tokens,
        "n_shards": len(entries),
        "total_tokens": sum(e.n_tokens for e in entries),
        "created_unix": int(time.time()),
        "shards": [
            {"idx": e.idx, "sha256": e.sha256, "n_tokens": e.n_tokens, "uri": e.uri}
            for e in entries
        ],
    }


def _write_manifest(bucket, manifest: dict) -> str:
    uri = bucket.uri_for_key(MANIFEST_KEY)
    bucket.put_json(uri, manifest)
    return uri


def _read_manifest(bucket) -> dict | None:
    uri = bucket.uri_for_key(MANIFEST_KEY)
    if not bucket.exists(uri):
        return None
    return bucket.get_json(uri)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    _load_dotenv()
    bucket = _build_bucket()
    bucket.ensure_bucket()
    print(f"  bucket: {bucket.uri_for_key('')}", flush=True)

    target_tokens = args.target_tokens
    shard_n_tokens = args.shard_tokens

    if args.resume:
        existing = _read_manifest(bucket)
        if existing is not None:
            print(f"  found existing manifest with {existing['n_shards']} shards "
                  f"({existing['total_tokens'] / 1e6:.1f}M tokens, "
                  f"shard={existing['shard_n_tokens'] / 1e6:.2f}M); resuming",
                  flush=True)
            if existing["shard_n_tokens"] != shard_n_tokens:
                print(f"  WARNING: --shard-tokens {shard_n_tokens} differs from "
                      f"manifest's {existing['shard_n_tokens']}; using manifest's",
                      flush=True)
                shard_n_tokens = existing["shard_n_tokens"]

    cache_path = Path(args.cache_path) if args.cache_path else None
    arr: np.ndarray | None = None
    if cache_path and cache_path.exists() and not args.force_tokenize:
        print(f"  loading cached tokens from {cache_path} ...", flush=True)
        arr = np.frombuffer(cache_path.read_bytes(), dtype=np.int32).copy()
        if arr.shape[0] < target_tokens:
            print(f"  cache has only {arr.shape[0]} tokens (< {target_tokens} requested); "
                  f"re-tokenizing", flush=True)
            arr = None
        else:
            arr = arr[:target_tokens]

    if arr is None:
        print(f"  tokenizing {target_tokens / 1e6:.1f}M tokens from FineWeb-edu ...",
              flush=True)
        arr = _tokenize_streaming(target_tokens)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(arr.tobytes())
            print(f"  cached raw tokens to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)",
                  flush=True)

    print(f"  arr: shape={arr.shape} dtype={arr.dtype} "
          f"min={int(arr.min())} max={int(arr.max())}", flush=True)

    entries = _shard_and_upload(
        arr,
        bucket=bucket,
        shard_n_tokens=shard_n_tokens,
        skip_existing=not args.no_skip_existing,
        n_workers=args.upload_workers,
    )

    manifest = _build_manifest(
        entries=entries,
        shard_n_tokens=shard_n_tokens,
        source=args.source,
        tokenizer=args.tokenizer,
    )
    manifest_uri = _write_manifest(bucket, manifest)

    print(f"  manifest -> {manifest_uri}", flush=True)
    print(f"  n_shards={manifest['n_shards']} "
          f"total_tokens={manifest['total_tokens'] / 1e6:.1f}M "
          f"shard_size={manifest['shard_n_tokens'] / 1e6:.2f}M", flush=True)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    _load_dotenv()
    bucket = _build_bucket()
    print(f"  bucket: {bucket.uri_for_key('')}", flush=True)
    manifest = _read_manifest(bucket)
    if manifest is None:
        print(f"  no manifest at {bucket.uri_for_key(MANIFEST_KEY)}", flush=True)
        return 1
    print(json.dumps(
        {k: v for k, v in manifest.items() if k != "shards"}, indent=2,
    ))
    print(f"  first 3 shards:")
    for s in manifest["shards"][:3]:
        print(f"    [{s['idx']:5d}] {s['sha256'][:16]}...  {s['n_tokens']:>8d} tok  -> {s['uri']}")
    if len(manifest["shards"]) > 3:
        print(f"    ... ({len(manifest['shards']) - 6} more) ...")
        for s in manifest["shards"][-3:]:
            print(f"    [{s['idx']:5d}] {s['sha256'][:16]}...  {s['n_tokens']:>8d} tok  -> {s['uri']}")
    if args.fetch:
        idx = max(0, min(args.fetch_idx, manifest["n_shards"] - 1))
        s = manifest["shards"][idx]
        print(f"  fetching shard {idx} from {s['uri']} ...", flush=True)
        body = bucket.get(s["uri"])
        sha = hashlib.sha256(body).hexdigest()
        ok = sha == s["sha256"]
        t = tensor_io.decode_tensor(body)
        arr = t.cpu().numpy()
        print(f"    bytes={len(body)} sha256_ok={ok} shape={t.shape} dtype={t.dtype}")
        print(f"    first 32 ids: {arr[:32].tolist()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fineweb_shards")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Tokenize FineWeb-edu and upload as shards")
    sp.add_argument("--target-tokens", type=int, default=int(1e9),
                    help="Total tokens to materialize (default: 1B)")
    sp.add_argument("--shard-tokens", type=int, default=1 << 20,
                    help="Tokens per shard (default: 1M = 4 MB int32)")
    sp.add_argument("--cache-path", default="/tmp/locus_data/fineweb_edu_tokens.i32",
                    help="Local raw-int32 token cache path; reused across runs")
    sp.add_argument("--force-tokenize", action="store_true",
                    help="Re-tokenize even if cache exists")
    sp.add_argument("--resume", action="store_true",
                    help="Skip shards already on the bucket; reuse manifest's shard size")
    sp.add_argument("--no-skip-existing", action="store_true",
                    help="Always re-upload shards even if their sha256 already exists")
    sp.add_argument("--upload-workers", type=int, default=16,
                    help="Parallel upload threads")
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    sp.set_defaults(fn=cmd_prepare)

    si = sub.add_parser("inspect")
    si.add_argument("--fetch", action="store_true",
                    help="Fetch one shard and verify its sha256")
    si.add_argument("--fetch-idx", type=int, default=0)
    si.set_defaults(fn=cmd_inspect)

    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
