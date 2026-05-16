"""Stitch the existing fineweb-edu shards on the bucket into a single
`runs/<RUN_ID>/static/tokens.bin` in tensor_io wire format so the gpt_pipe
streaming task picks it up via `bucket.exists(tokens_uri)` and skips the
random-token bootstrap path.

Source: `data/fineweb-edu/manifest.json` + `data/fineweb-edu/shards/<sha>.bin`
Output: `runs/<RUN_ID>/static/tokens.bin`

Each shard is already tensor_io.encode_tensor(int32 tokens). We decode them
back to plain int32 tensors, concatenate, and re-encode the result as one
tensor.

Usage:
    python scripts/prepare_fineweb_tokens.py --run-id my-run --shards 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from urllib.parse import urlparse

import boto3
import torch

from teuton_runtime import tensor_io


def make_s3():
    return boto3.client(
        "s3",
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    ap.add_argument("--manifest-key", default="data/fineweb-edu/manifest.json")
    ap.add_argument(
        "--shards", type=int, default=50,
        help="number of shards to stitch (each is ~1.05M tokens)",
    )
    ap.add_argument(
        "--out-key", default=None,
        help="override output key; default runs/<RUN_ID>/static/tokens.bin",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.bucket:
        raise SystemExit("--bucket or S3_BUCKET required")
    out_key = args.out_key or f"runs/{args.run_id}/static/tokens.bin"

    s3 = make_s3()

    print(f"[prep] reading manifest s3://{args.bucket}/{args.manifest_key}")
    obj = s3.get_object(Bucket=args.bucket, Key=args.manifest_key)
    manifest = json.loads(obj["Body"].read())
    shard_entries = manifest.get("shards") or []
    if not shard_entries:
        raise SystemExit("manifest has no shards")
    target_shards = shard_entries[: args.shards]
    n_target = len(target_shards)
    shard_tokens = int(manifest.get("shard_n_tokens", 1048576))
    print(
        f"[prep] stitching {n_target} shards ~ {n_target * shard_tokens:,} tokens "
        f"(of {len(shard_entries)} total)"
    )

    if args.dry_run:
        print("[prep] dry-run, exiting.")
        return 0

    pieces: list[torch.Tensor] = []
    total_bytes = 0
    t0 = time.time()
    for i, entry in enumerate(target_shards):
        bkt, key = parse_s3_uri(entry["uri"])
        body = s3.get_object(Bucket=bkt, Key=key)["Body"].read()
        total_bytes += len(body)
        tensor = tensor_io.decode_tensor(body)
        if tensor.dtype != torch.int32:
            raise SystemExit(
                f"shard {entry['sha256']} has unexpected dtype {tensor.dtype}; aborting"
            )
        pieces.append(tensor)
        if (i + 1) % 10 == 0 or i + 1 == n_target:
            print(
                f"  [{i+1:>3}/{n_target}] downloaded {total_bytes/1e6:7.1f} MB "
                f"  elapsed={time.time()-t0:5.1f}s"
            )

    print("[prep] concatenating ...")
    tokens = torch.cat(pieces, dim=0)
    print(f"[prep] total tokens={tokens.numel():,}  dtype={tokens.dtype}")
    del pieces

    encoded = tensor_io.encode_tensor(tokens)
    print(f"[prep] encoded payload: {len(encoded)/1e6:.1f} MB")
    print(f"[prep] uploading to s3://{args.bucket}/{out_key} ...")
    s3.put_object(
        Bucket=args.bucket,
        Key=out_key,
        Body=encoded,
        ContentType="application/octet-stream",
    )

    meta = {
        "run_id": args.run_id,
        "source_manifest_key": args.manifest_key,
        "n_shards": n_target,
        "shard_n_tokens": shard_tokens,
        "total_tokens": int(tokens.numel()),
        "dtype": str(tokens.dtype),
        "encoded_bytes": len(encoded),
        "created_unix": int(time.time()),
    }
    meta_key = out_key + ".meta.json"
    s3.put_object(
        Bucket=args.bucket,
        Key=meta_key,
        Body=json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[prep] wrote meta s3://{args.bucket}/{meta_key}")
    print(f"[prep] DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
