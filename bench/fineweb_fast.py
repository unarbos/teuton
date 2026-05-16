"""Fast parallel FineWeb-edu tokenizer for scaling-experiment data.

Single-threaded tokenization (the original teuton.data.fineweb_loader)
maxes out around ~100k tokens/sec, so it takes ~28h for 10B tokens.
This script uses a multiprocessing pool: ~24 workers each pull HF
streaming rows, tokenize with tiktoken, and write to per-worker shard
files. The result is then concatenated into a single int32 tokens.bin
that openmythos_ddp.py can mmap directly.

Format: raw int32 little-endian, NO header (we use mmap path in DDP).

Usage:
    python bench/fineweb_fast.py --target-tokens 10_000_000_000 \\
        --workers 24 --out-dir /workspace/teuton_data_10B
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


EOT = 50256
SHARD_TOKENS = 50_000_000


def worker(args):
    worker_id, target_tokens, out_path, skip_rows = args
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    if skip_rows > 0:
        ds = ds.skip(skip_rows)

    buf = np.empty((target_tokens + 1024,), dtype=np.int32)
    n = 0
    t0 = time.time()
    docs = 0
    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(EOT)
        end = min(n + len(ids), target_tokens)
        buf[n:end] = ids[: end - n]
        n = end
        docs += 1
        if docs % 1000 == 0:
            dt = time.time() - t0
            print(f"  [w{worker_id}] {n/1e6:.1f}M tok ({n/dt/1000:.0f}k tok/s)", flush=True)
        if n >= target_tokens:
            break

    arr = buf[:n]
    arr.tofile(out_path)
    print(f"  [w{worker_id}] wrote {n/1e6:.1f}M tokens to {out_path}", flush=True)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-tokens", type=int, default=10_000_000_000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--out-dir", default="/workspace/teuton_data_10B")
    p.add_argument("--skip-stride", type=int, default=50_000,
                   help="rows to skip between workers (avoid duplicating data)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_worker = args.target_tokens // args.workers + 1
    jobs = []
    for i in range(args.workers):
        out_path = out_dir / f"shard_{i:03d}.i32"
        if out_path.exists() and out_path.stat().st_size >= per_worker * 4 * 0.9:
            print(f"  shard {i} already exists, skipping", flush=True)
            continue
        jobs.append((i, per_worker, str(out_path), i * args.skip_stride))

    if not jobs:
        print("  all shards already exist", flush=True)
    else:
        print(f"  starting {len(jobs)} workers, each tokenizing ~{per_worker/1e6:.0f}M tokens", flush=True)
        t0 = time.time()
        with mp.Pool(len(jobs)) as pool:
            results = pool.map(worker, jobs)
        elapsed = time.time() - t0
        total = sum(results)
        print(f"  workers done: {total/1e9:.2f}B tokens in {elapsed/60:.1f}min "
              f"({total/elapsed/1e6:.1f}M tok/s aggregate)", flush=True)

    final_path = out_dir / "tokens.bin"
    print(f"  concatenating shards → {final_path}", flush=True)
    shard_paths = sorted(out_dir.glob("shard_*.i32"))
    with open(final_path, "wb") as fout:
        total = 0
        for sp in shard_paths:
            data = sp.read_bytes()
            fout.write(data)
            total += len(data) // 4
            print(f"    + {sp.name}: +{len(data)/4/1e6:.0f}M tokens (total {total/1e9:.2f}B)",
                  flush=True)
    print(f"  done: {final_path} = {final_path.stat().st_size/1e9:.2f}GB", flush=True)

    meta_path = out_dir / "tokens.meta.json"
    import json
    meta_path.write_text(json.dumps({
        "n_tokens": total, "dtype": "int32", "raw_int32": True,
        "tokenizer": "tiktoken/gpt2", "source": "fineweb-edu/sample-10BT",
        "format": "raw int32 little-endian, no header",
    }, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
