"""FineWeb-edu data loader for Locus pipe-train runs.

Tokenizes a small slice of HuggingFaceFW/fineweb-edu (sample-10BT) with the
GPT-2 BPE tokenizer (via tiktoken) into a single int32 tensor, and uploads
it to S3 at `runs/<id>/data/tokens.bin`.

Workers then read it via the `data_indexer` IR op (declared with `mb_seed`,
`B`, `T`) which produces (input_ids, target_ids) deterministically.

The CLI:

    python -m locus.data.fineweb_loader prepare \\
        --run-id <id> \\
        --target-tokens 200_000_000 \\
        --upload-to-s3

Locally during dev, drop --upload-to-s3 to write only `tokens.bin` to disk.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .. import paths, tensor_io
from ..storage import LocalBucket


def _have_tiktoken():
    try:
        import tiktoken  # noqa: F401
        return True
    except ImportError:
        return False


def _have_datasets():
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def _stream_fineweb_edu(target_tokens: int) -> Iterable[str]:
    """Yield raw text shards from fineweb-edu (streaming) until we expect
    enough tokens. We assume ~5 chars/token on average, so request 5x."""
    import datasets

    target_chars = target_tokens * 5
    seen_chars = 0
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
        seen_chars += len(text)
        if seen_chars >= target_chars:
            return


def _tokenize_to_int32(target_tokens: int, eot_id: int = 50256) -> np.ndarray:
    """Tokenize streaming fineweb-edu into a single int32 array."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out = np.empty((target_tokens,), dtype=np.int32)
    n = 0
    docs = 0
    t0 = time.time()
    for text in _stream_fineweb_edu(target_tokens):
        ids = enc.encode_ordinary(text)
        # Append EOT separator so the model sees document boundaries.
        ids.append(eot_id)
        if n + len(ids) > target_tokens:
            ids = ids[: target_tokens - n]
        if not ids:
            break
        out[n : n + len(ids)] = ids
        n += len(ids)
        docs += 1
        if docs % 200 == 0:
            print(
                f"  tokenized {n / 1e6:.2f}M / {target_tokens / 1e6:.2f}M tokens "
                f"({docs} docs, {n / (time.time() - t0):.0f} tok/sec)",
                flush=True,
            )
        if n >= target_tokens:
            break
    if n < target_tokens:
        print(f"  WARNING: only got {n} tokens, wanted {target_tokens}", flush=True)
        out = out[:n]
    print(f"  done: {n} tokens, {time.time() - t0:.1f}s", flush=True)
    return out


def cmd_prepare(args: argparse.Namespace) -> int:
    if not _have_tiktoken():
        print("ERROR: tiktoken not installed. `uv pip install tiktoken`", file=sys.stderr)
        return 2
    if not _have_datasets():
        print("ERROR: datasets not installed. `uv pip install datasets`", file=sys.stderr)
        return 2

    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_bin = out_dir / "tokens.bin"
    local_meta = out_dir / "tokens.meta.json"

    if local_bin.exists() and not args.force:
        print(f"  found existing {local_bin} ({local_bin.stat().st_size} bytes); use --force to redownload",
              flush=True)
    else:
        print(f"  tokenizing {args.target_tokens / 1e6:.1f}M tokens to {local_bin} ...", flush=True)
        arr = _tokenize_to_int32(args.target_tokens)
        # Save raw int32 array as torch tensor on disk via tensor_io for
        # round-trip compatibility.
        t = torch.from_numpy(arr.copy())
        local_bin.write_bytes(tensor_io.encode_tensor(t))
        meta = {
            "n_tokens": int(t.numel()),
            "dtype": "int32",
            "tokenizer": "tiktoken/gpt2",
            "source": "HuggingFaceFW/fineweb-edu/sample-10BT",
            "encoded_unix": int(time.time()),
        }
        import json
        local_meta.write_text(json.dumps(meta, indent=2))
        print(f"  wrote {local_bin.stat().st_size / 1e6:.1f} MB", flush=True)

    if args.upload_to_s3:
        bucket = _build_bucket()
        key = f"runs/{args.run_id}/data/tokens.bin"
        meta_key = f"runs/{args.run_id}/data/tokens.meta.json"
        uri = bucket.uri_for_key(key)
        meta_uri = bucket.uri_for_key(meta_key)
        print(f"  uploading to {uri} ...", flush=True)
        body = local_bin.read_bytes()
        bucket.put(uri, body)
        if local_meta.exists():
            import json
            bucket.put_json(meta_uri, json.loads(local_meta.read_text()))
        print(f"  uploaded {len(body) / 1e6:.1f} MB to S3", flush=True)
    return 0


def _build_bucket() -> LocalBucket:
    """Build a bucket from env (matches bench/dist.py convention)."""
    # Try common env var names; matches bench/dist.py convention.
    bucket_name = os.environ.get("S3_BUCKET") or os.environ.get("LOCUS_S3_BUCKET")
    if bucket_name:
        from ..storage import S3Bucket  # noqa: WPS433
        return S3Bucket(
            bucket=bucket_name,
            region=os.environ.get("S3_REGION", os.environ.get("AWS_REGION", "us-east-1")),
        )
    root = os.environ.get("LOCUS_LOCAL_BUCKET", "/tmp/locus_bucket")
    return LocalBucket(root=root, bucket="local")


def cmd_inspect(args: argparse.Namespace) -> int:
    """Read back tokens.bin (locally or from S3) and print stats."""
    if args.from_s3:
        bucket = _build_bucket()
        uri = bucket.uri_for_key(f"runs/{args.run_id}/data/tokens.bin")
        body = bucket.get(uri)
    else:
        body = Path(args.local_dir, "tokens.bin").read_bytes()
    t = tensor_io.decode_tensor(body)
    arr = t.cpu().numpy()
    print(f"  shape={t.shape} dtype={t.dtype} bytes={len(body)}")
    print(f"  min={arr.min()} max={arr.max()} unique={len(np.unique(arr[:100000]))}")
    print(f"  first 64 ids: {arr[:64].tolist()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fineweb_loader")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Tokenize FineWeb-edu and (optionally) upload to S3")
    sp.add_argument("--run-id", default="data_global",
                    help="Run id for the S3 path (uses runs/<id>/data/tokens.bin)")
    sp.add_argument("--target-tokens", type=int, default=int(50e6),
                    help="Number of tokens to materialize (default 50M, ~150 MB int32)")
    sp.add_argument("--local-dir", default="/tmp/locus_data",
                    help="Where to cache tokens.bin locally")
    sp.add_argument("--upload-to-s3", action="store_true")
    sp.add_argument("--force", action="store_true",
                    help="Re-tokenize even if local tokens.bin exists")
    sp.set_defaults(fn=cmd_prepare)

    si = sub.add_parser("inspect")
    si.add_argument("--run-id", default="data_global")
    si.add_argument("--local-dir", default="/tmp/locus_data")
    si.add_argument("--from-s3", action="store_true")
    si.set_defaults(fn=cmd_inspect)

    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
