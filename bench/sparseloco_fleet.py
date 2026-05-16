"""SparseLoCo fleet worker — runs across heterogeneous Lium boxes via S3.

Each invocation is ONE worker. Workers self-coordinate through S3 barriers,
no orchestrator process is required. This is the production-style architecture
from the SparseLoCo paper, deployed on Cloudflare R2 (S3-compatible).

Wire layout:
    runs/<id>/sparseloco/config.json                       # cfg + hparams
    runs/<id>/sparseloco/init/W.bin                        # initial weights (rank 0 writes)
    runs/<id>/sparseloco/init/ready.json                   # rank 0 done bootstrap
    runs/<id>/sparseloco/outer={t}/peer={r}.pt             # compressed pseudo-grad
    runs/<id>/sparseloco/outer={t}/peer={r}.json           # telemetry per-outer per-peer

The barrier model:
    - Rank 0 writes init/W.bin then init/ready.json. All ranks wait for ready.json.
    - At each outer step t, all ranks upload outer={t}/peer={r}.pt.
    - All ranks poll until all R uploads exist, then download + aggregate.

Usage on each box:
    python bench/sparseloco_fleet.py worker \
        --rank 0 --world-size 5 \
        --run-id slfleet1 \
        --moe on \
        --tokens-uri s3://bucket/runs/<global>/data/tokens.bin \
        --device cuda:0 \
        --H 15 --T 100 \
        --out /workspace/sl_fleet_rank0.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reuse model + helpers from the local validation script
sys.path.insert(0, "/root/Teuton")
from bench.sparseloco_moe_validation import (  # noqa: E402
    Cfg, GPT, evaluate, shard_iter, snapshot_params, write_params,
    diff_params, apply_outer, sparseloco_compress,
    chunked_topk_quant_encode, chunked_topk_quant_decode,
)

from teuton_legacy_v2.storage import S3Bucket  # noqa: E402
from teuton import tensor_io  # noqa: E402


def _build_bucket():
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("S3_REGION", "us-east-1")
    return S3Bucket(bucket=bucket, region=region)


def _key(run_id, *parts):
    return "runs/" + run_id + "/sparseloco/" + "/".join(parts)


def _wait_for_key(bucket, key, timeout_sec=600.0, poll_sec=2.0,
                    log_prefix=""):
    uri = bucket.uri_for_key(key)
    deadline = time.time() + timeout_sec
    waited_for = 0
    while time.time() < deadline:
        if bucket.exists(uri):
            return True
        time.sleep(poll_sec)
        waited_for += poll_sec
        if waited_for >= 30 and waited_for % 30 < poll_sec:
            print(f"{log_prefix}still waiting for {key} ({waited_for:.0f}s)",
                  flush=True)
    return False


def _wait_for_all_peers(bucket, run_id, outer, world_size, *,
                         timeout_sec=600.0, poll_sec=2.0, my_rank=None):
    deadline = time.time() + timeout_sec
    seen = set()
    if my_rank is not None:
        seen.add(my_rank)
    while time.time() < deadline:
        for r in range(world_size):
            if r in seen:
                continue
            uri = bucket.uri_for_key(_key(run_id, f"outer={outer}", f"peer={r}.pt"))
            if bucket.exists(uri):
                seen.add(r)
        if len(seen) == world_size:
            return True
        time.sleep(poll_sec)
    return False


_CODEC = "torch"      # set by cmd_worker via --codec flag
_AGGREGATOR = "peer"  # "peer" (every worker downloads R-1) or "star" (rank 0 collects + broadcasts)
_PARALLEL_GETS = 8    # max concurrent S3 GETs in peer mode (set via --parallel-gets)


def _serialize_payloads(payloads: dict) -> bytes:
    """Serialize {name: sparse_payload_dict} for upload. Format depends on
    the global _CODEC switch: 'torch' uses torch.save (~3x bigger but
    cross-version-safe); 'packed' uses our compact bit-packed binary
    format (~50x compression vs full fp32)."""
    if _CODEC == "packed":
        from bench.sparseloco_codec import encode_payloads
        return encode_payloads(payloads)
    buf = io.BytesIO()
    torch.save(payloads, buf)
    return buf.getvalue()


def _deserialize_payloads(body: bytes) -> dict:
    if _CODEC == "packed" or body[:4] == b"SLCD":
        from bench.sparseloco_codec import decode_payloads
        return decode_payloads(body)
    return torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)


def _decode_payloads_to_dense(payloads: dict, ref_tensors: dict, device) -> dict:
    """Inverse of compression: per-name sparse payload -> dense tensor on device."""
    out = {}
    for name, p in payloads.items():
        out[name] = chunked_topk_quant_decode(p, device=device)
    return out


def _fetch_and_aggregate_peers(bucket, run_id, outer, R, my_rank,
                                  my_compressed_dense, theta_prev, device):
    """Wait for all R peers' uploads at outer step `outer`, download them
    in parallel, decode, and average together with our own contribution.
    Returns (avg_delta dict, bytes_dn).

    Designed to be safe to call from a background thread.
    """
    from concurrent.futures import ThreadPoolExecutor

    ok = _wait_for_all_peers(bucket, run_id, outer, R,
                                timeout_sec=900.0, my_rank=my_rank,
                                poll_sec=1.0)
    if not ok:
        raise RuntimeError(f"timeout waiting for peers at outer {outer}")

    other_ranks = [r for r in range(R) if r != my_rank]
    uris = [
        bucket.uri_for_key(_key(run_id, f"outer={outer}", f"peer={r}.pt"))
        for r in other_ranks
    ]
    bodies = [None] * len(uris)
    with ThreadPoolExecutor(max_workers=_PARALLEL_GETS) as pool:
        futs = {pool.submit(bucket.get, uri): i for i, uri in enumerate(uris)}
        for f in futs:
            bodies[futs[f]] = f.result()

    avg_delta = {n: torch.zeros_like(theta_prev[n]) for n in theta_prev}
    for n in avg_delta:
        avg_delta[n].add_(my_compressed_dense[n] / R)

    bytes_dn = 0
    for body in bodies:
        bytes_dn += len(body)
        peer_payloads = _deserialize_payloads(body)
        peer_decoded = _decode_payloads_to_dense(peer_payloads, theta_prev, device)
        for n in avg_delta:
            avg_delta[n].add_(peer_decoded[n] / R)
    return avg_delta, bytes_dn


def _save_full_weights(model, path_local):
    state = {n: p.detach().cpu() for n, p in model.named_parameters()}
    torch.save(state, path_local)


def _load_full_weights(model, body, device):
    state = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(state[n].to(device))


def _download_tokens(bucket, tokens_uri, local_path):
    """Download tokens.bin from S3 to local disk if not present."""
    if Path(local_path).exists():
        return
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    body = bucket.get(tokens_uri)
    Path(local_path).write_bytes(body)


def cmd_worker(args):
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed_all(args.seed + args.rank)
    torch.set_float32_matmul_precision("high")

    global _CODEC, _AGGREGATOR, _PARALLEL_GETS
    _CODEC = args.codec
    _AGGREGATOR = args.aggregator
    _PARALLEL_GETS = args.parallel_gets

    bucket = _build_bucket()
    device = torch.device(args.device)
    is_main = (args.rank == 0)
    log_pfx = f"[r{args.rank}] "

    cfg = Cfg(use_moe=(args.moe == "on"))
    cfg.block_size = args.seq_len
    cfg.n_layer = args.n_layer
    cfg.d = args.d_model
    cfg.n_head = args.n_head
    cfg.d_ff = args.d_ff
    cfg.n_loops = args.n_loops
    cfg.anchor_inject = args.anchor_inject

    print(log_pfx + f"starting cfg={cfg.__dict__} device={args.device}", flush=True)

    # ----- bootstrap (rank 0 writes config + initial weights) -----
    cfg_uri = bucket.uri_for_key(_key(args.run_id, "config.json"))
    init_uri = bucket.uri_for_key(_key(args.run_id, "init", "W.bin"))
    ready_uri = bucket.uri_for_key(_key(args.run_id, "init", "ready.json"))

    if is_main:
        if not bucket.exists(ready_uri):
            cfg_payload = {
                "vocab": cfg.vocab, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
                "d": cfg.d, "d_ff": cfg.d_ff, "T": cfg.block_size,
                "n_experts": cfg.n_experts, "moe_top_k": cfg.moe_top_k,
                "use_moe": cfg.use_moe,
                "world_size": args.world_size, "H": args.H, "T_outer": args.T,
                "lr_inner": args.lr_inner, "lr_outer": args.lr_outer,
                "density": args.density, "ef_beta": args.ef_beta,
                "quant_bits": args.quant_bits, "chunk_size": args.chunk_size,
                "bsz": args.bsz, "tokens_uri": args.tokens_uri,
                "started_unix": int(time.time()),
            }
            bucket.put_json(cfg_uri, cfg_payload)
            # Build model + write initial weights
            torch.manual_seed(args.seed)
            init_model = GPT(cfg)
            tmp = "/tmp/init_W.bin"
            _save_full_weights(init_model, tmp)
            bucket.put(init_uri, Path(tmp).read_bytes())
            del init_model
            bucket.put_json(ready_uri, {"ready_unix": int(time.time())})
            print(log_pfx + f"bootstrap done: cfg + init weights written ({Path(tmp).stat().st_size / 1e6:.1f}MB)",
                  flush=True)
    else:
        if not _wait_for_key(bucket, _key(args.run_id, "init", "ready.json"),
                              timeout_sec=600.0, log_prefix=log_pfx):
            print(log_pfx + "TIMEOUT waiting for bootstrap", flush=True)
            return 2

    # ----- load initial state -----
    cfg_loaded = bucket.get_json(cfg_uri)
    print(log_pfx + f"loaded cfg from S3, world_size={cfg_loaded['world_size']}", flush=True)

    model = GPT(cfg).to(device)
    init_body = bucket.get(init_uri)
    _load_full_weights(model, init_body, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(log_pfx + f"model loaded: {n_params / 1e6:.1f}M params", flush=True)

    # ----- download tokens -----
    local_tokens = "/workspace/teuton_data/tokens.bin"
    _download_tokens(bucket, args.tokens_uri, local_tokens)
    body = Path(local_tokens).read_bytes()
    tokens = tensor_io.decode_tensor(body).to(torch.long)
    train_n = int(0.95 * tokens.numel())
    train_tokens = tokens[:train_n]
    val_tokens = tokens[train_n:]
    R = args.world_size
    shard_size = train_tokens.numel() // R
    shard_start = args.rank * shard_size
    shard = train_tokens[shard_start: shard_start + shard_size]
    train_iter = shard_iter(shard, args.bsz, args.seq_len, seed=args.seed + args.rank * 1000)
    val_iter = shard_iter(val_tokens, args.bsz, args.seq_len, seed=args.seed + 999999)
    print(log_pfx + f"tokens loaded: shard={shard.numel() / 1e6:.1f}M, "
          f"val={val_tokens.numel() / 1e6:.1f}M", flush=True)

    # ----- per-worker EF buffer -----
    ef = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    # ----- main loop -----
    metrics = {
        "rank": args.rank, "world_size": R, "device": args.device,
        "n_params": n_params, "use_moe": cfg.use_moe,
        "T_outer": args.T, "H": args.H, "density": args.density,
        "outer_curve": [], "wall_clock_per_outer": [],
        "bytes_uploaded_per_outer": [], "bytes_downloaded_per_outer": [],
        "comm_sec_per_outer": [], "compute_sec_per_outer": [],
        "started_unix": time.time(),
    }
    chunk_size = args.chunk_size
    from concurrent.futures import ThreadPoolExecutor
    bg_pool = ThreadPoolExecutor(max_workers=2)
    pending_agg_future = None         # background fetch of prev outer's aggregate
    pending_agg_outer = -1            # which outer it corresponds to
    pending_theta_prev_for_apply = None  # weights snapshot at start of pending outer

    for outer in range(args.T):
        outer_start = time.time()

        theta_prev = snapshot_params(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_inner,
                                betas=(0.9, 0.95), weight_decay=0.1)

        # ---- async mode: collect previous outer's aggregate (if pending) ----
        # In '1step' mode, we don't wait for peers right after upload — instead
        # we kick off a background fetch and let inner training of THIS outer
        # proceed using current weights. The previous outer's aggregate is
        # applied at the start of this iteration, BEFORE inner training.
        async_apply_sec = 0.0
        if args.async_mode == "1step" and pending_agg_future is not None:
            t_apply = time.time()
            try:
                prev_avg_delta, prev_bytes_dn = pending_agg_future.result()
            except Exception as e:
                print(log_pfx + f"async aggregate fetch failed: {e}", flush=True)
                prev_avg_delta = None
                prev_bytes_dn = 0
            if prev_avg_delta is not None:
                # Apply to model: model has been trained for one outer step
                # since pending_theta_prev_for_apply was snapshotted. We
                # apply the delta as a correction.
                # Standard DiLoCo: theta = theta_prev_outer - alpha * avg_delta.
                # In async, we don't have theta_prev_outer in memory anymore —
                # the model has already moved. Two choices:
                #   (a) apply correction to the CURRENT model: model -= alpha * avg_delta
                #       (treats the merge as a directional correction relative to where we are)
                #   (b) restore pending_theta_prev_for_apply, apply, then re-do inner steps
                # We use (a) — far cheaper, still convergent under bounded staleness.
                apply_outer(model, prev_avg_delta, args.lr_outer)
            async_apply_sec = time.time() - t_apply
            theta_prev = snapshot_params(model)  # refresh snapshot

        # ---- inner loop ----
        # Two modes:
        #   default (--target-compute-sec 0): fixed H inner steps (DiLoCo standard)
        #   time-bounded (--target-compute-sec X): run for ~X seconds. Fast
        #   workers do MORE than H steps in the same wall-time as slow workers.
        #   Combined with async-1step, this means fast workers contribute MORE
        #   inner-step work per outer step (variable-H per worker).
        compute_start = time.time()
        last_loss = float("nan")
        inner_steps_done = 0
        if args.target_compute_sec > 0:
            min_h = max(1, args.H)  # always do at least min_h, then continue while time permits
            while inner_steps_done < min_h or (time.time() - compute_start) < args.target_compute_sec:
                x, y = next(train_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss, _ = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                last_loss = float(loss.item())
                inner_steps_done += 1
        else:
            for h in range(args.H):
                x, y = next(train_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss, _ = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                last_loss = float(loss.item())
                inner_steps_done += 1
        compute_sec = time.time() - compute_start

        # ---- pseudo-grad + SparseLoCo compress ----
        delta = diff_params(theta_prev, model)
        compressed_dense, payloads, total_bytes_est = sparseloco_compress(
            delta, ef, beta=args.ef_beta, density=args.density,
            chunk_size=chunk_size, bits=args.quant_bits,
            return_payloads=True,
        )

        # ---- upload our compressed grad (sparse wire format) ----
        comm_start = time.time()
        my_bytes = _serialize_payloads(payloads)
        my_uri = bucket.uri_for_key(_key(args.run_id, f"outer={outer}",
                                          f"peer={args.rank}.pt"))
        bucket.put(my_uri, my_bytes)
        bytes_up = len(my_bytes)
        print(log_pfx + f"outer {outer:3d}: inner_loss={last_loss:.4f} "
              f"compute={compute_sec:.2f}s ({inner_steps_done} steps) "
              f"upload={bytes_up / 1e6:.2f}MB "
              f"(theoretical={total_bytes_est / 1e6:.2f}MB) "
              f"async_apply={async_apply_sec:.1f}s", flush=True)

        # ---- async '1step' mode: fire-and-forget the peer fetch, then continue ----
        if args.async_mode == "1step":
            # Snapshot what we'd need to apply later: just the dict reference.
            # Submit a background fetch for THIS outer's peer grads. Will be
            # applied at the START of the next outer step.
            pending_agg_future = bg_pool.submit(
                _fetch_and_aggregate_peers,
                bucket, args.run_id, outer, R, args.rank,
                {n: t.clone() for n, t in compressed_dense.items()},  # snapshot
                {n: t.clone() for n, t in theta_prev.items()},
                device,
            )
            pending_agg_outer = outer
            comm_sec = time.time() - comm_start

            outer_wallclock = time.time() - outer_start
            metrics["outer_curve"].append({
                "outer": outer, "inner_loss": last_loss,
                "compute_sec": compute_sec, "comm_sec": comm_sec,
                "wallclock_sec": outer_wallclock,
                "bytes_up": bytes_up, "bytes_dn": 0,
                "async_apply_sec": async_apply_sec,
                "inner_steps_done": inner_steps_done,
            })
            metrics["wall_clock_per_outer"].append(outer_wallclock)
            metrics["compute_sec_per_outer"].append(compute_sec)
            metrics["comm_sec_per_outer"].append(comm_sec)
            metrics["bytes_uploaded_per_outer"].append(bytes_up)
            metrics["bytes_downloaded_per_outer"].append(0)

            if outer % args.val_every == 0 or outer == args.T - 1:
                v, rinfo = evaluate(model, val_iter, args.n_val, device,
                                      collect_routing=cfg.use_moe)
                metrics["outer_curve"][-1]["val_loss"] = v
                metrics["outer_curve"][-1]["routing"] = rinfo
                print(log_pfx + f"outer {outer:3d}: val_loss={v:.4f} "
                      f"comm={comm_sec:.1f}s wall={outer_wallclock:.1f}s "
                      f"util={compute_sec / outer_wallclock:.1%} (async)",
                      flush=True)
            continue   # skip the synchronous aggregation block below

        # ---- aggregation (synchronous mode): 'peer' or 'star' ----
        bytes_dn = 0
        avg_delta = {n: torch.zeros_like(theta_prev[n]) for n in theta_prev}

        if _AGGREGATOR == "peer":
            # ---- wait for all peers ----
            ok = _wait_for_all_peers(bucket, args.run_id, outer, R,
                                       timeout_sec=900.0, my_rank=args.rank,
                                       poll_sec=1.0)
            if not ok:
                print(log_pfx + f"TIMEOUT waiting for peers at outer {outer}", flush=True)
                break

            # Parallel GETs: download all R-1 peer payloads concurrently
            # (boto3 connections are thread-safe; max_concurrency in
            # _PARALLEL_GETS bounds this).
            from concurrent.futures import ThreadPoolExecutor
            other_ranks = [r for r in range(R) if r != args.rank]
            uris = [
                bucket.uri_for_key(_key(args.run_id, f"outer={outer}",
                                         f"peer={r}.pt"))
                for r in other_ranks
            ]
            bodies = [None] * len(uris)
            with ThreadPoolExecutor(max_workers=_PARALLEL_GETS) as pool:
                futs = {pool.submit(bucket.get, uri): i for i, uri in enumerate(uris)}
                for f in futs:
                    i = futs[f]
                    bodies[i] = f.result()

            # Add own contribution
            for n in avg_delta:
                avg_delta[n].add_(compressed_dense[n] / R)
            # Decode + accumulate peers
            for body in bodies:
                bytes_dn += len(body)
                peer_payloads = _deserialize_payloads(body)
                peer_decoded = _decode_payloads_to_dense(peer_payloads, theta_prev, device)
                for n in avg_delta:
                    avg_delta[n].add_(peer_decoded[n] / R)
        else:
            # 'star': rank 0 collects all, computes mean, broadcasts as
            # outer={t}/aggregate.pt. All other ranks just download aggregate.
            agg_uri = bucket.uri_for_key(_key(args.run_id, f"outer={outer}",
                                                 "aggregate.pt"))
            if args.rank == 0:
                ok = _wait_for_all_peers(bucket, args.run_id, outer, R,
                                           timeout_sec=900.0, my_rank=0,
                                           poll_sec=1.0)
                if not ok:
                    print(log_pfx + f"TIMEOUT waiting for peers at outer {outer}", flush=True)
                    break
                for n in avg_delta:
                    avg_delta[n].add_(compressed_dense[n] / R)
                for r in range(1, R):
                    peer_uri = bucket.uri_for_key(_key(args.run_id, f"outer={outer}",
                                                          f"peer={r}.pt"))
                    body = bucket.get(peer_uri)
                    bytes_dn += len(body)
                    peer_payloads = _deserialize_payloads(body)
                    peer_decoded = _decode_payloads_to_dense(peer_payloads, theta_prev, device)
                    for n in avg_delta:
                        avg_delta[n].add_(peer_decoded[n] / R)
                # Encode aggregate as a torch state dict (small enough; no need
                # to use the sparse codec since the mean is dense).
                agg_payload = io.BytesIO()
                torch.save({n: t.detach().cpu().to(torch.float32) for n, t in avg_delta.items()},
                            agg_payload)
                bucket.put(agg_uri, agg_payload.getvalue())
            else:
                # Wait for aggregate.pt
                deadline = time.time() + 900.0
                while time.time() < deadline and not bucket.exists(agg_uri):
                    time.sleep(1.0)
                if not bucket.exists(agg_uri):
                    print(log_pfx + f"TIMEOUT waiting for aggregate at outer {outer}", flush=True)
                    break
                body = bucket.get(agg_uri)
                bytes_dn += len(body)
                avg_state = torch.load(io.BytesIO(body), map_location="cpu",
                                          weights_only=True)
                for n in avg_delta:
                    avg_delta[n] = avg_state[n].to(device, dtype=torch.float32)
        comm_sec = time.time() - comm_start

        # ---- restore prev weights, apply outer ----
        write_params(model, theta_prev)
        apply_outer(model, avg_delta, args.lr_outer)

        outer_wallclock = time.time() - outer_start
        metrics["outer_curve"].append({
            "outer": outer, "inner_loss": last_loss,
            "compute_sec": compute_sec, "comm_sec": comm_sec,
            "wallclock_sec": outer_wallclock,
            "bytes_up": bytes_up, "bytes_dn": bytes_dn,
            "inner_steps_done": inner_steps_done,
        })
        metrics["wall_clock_per_outer"].append(outer_wallclock)
        metrics["compute_sec_per_outer"].append(compute_sec)
        metrics["comm_sec_per_outer"].append(comm_sec)
        metrics["bytes_uploaded_per_outer"].append(bytes_up)
        metrics["bytes_downloaded_per_outer"].append(bytes_dn)

        # ---- val eval (every K outer steps) ----
        if outer % args.val_every == 0 or outer == args.T - 1:
            v, rinfo = evaluate(model, val_iter, args.n_val, device,
                                  collect_routing=cfg.use_moe)
            metrics["outer_curve"][-1]["val_loss"] = v
            metrics["outer_curve"][-1]["routing"] = rinfo
            print(log_pfx + f"outer {outer:3d}: val_loss={v:.4f} "
                  f"comm={comm_sec:.1f}s wall={outer_wallclock:.1f}s "
                  f"util={compute_sec / outer_wallclock:.1%}",
                  flush=True)

    # ---- drain final pending aggregate (async mode) ----
    if args.async_mode == "1step" and pending_agg_future is not None:
        try:
            prev_avg_delta, _ = pending_agg_future.result(timeout=300.0)
            apply_outer(model, prev_avg_delta, args.lr_outer)
        except Exception as e:
            print(log_pfx + f"failed to drain final aggregate: {e}", flush=True)

    metrics["finished_unix"] = time.time()
    metrics["total_elapsed_sec"] = metrics["finished_unix"] - metrics["started_unix"]
    if metrics["wall_clock_per_outer"]:
        metrics["avg_wallclock_per_outer"] = float(np.mean(metrics["wall_clock_per_outer"]))
        metrics["avg_compute_sec"] = float(np.mean(metrics["compute_sec_per_outer"]))
        metrics["avg_comm_sec"] = float(np.mean(metrics["comm_sec_per_outer"]))
        metrics["avg_utilization"] = (
            metrics["avg_compute_sec"] / max(metrics["avg_wallclock_per_outer"], 1e-9)
        )
        metrics["avg_bytes_up_mb"] = float(np.mean(metrics["bytes_uploaded_per_outer"])) / 1e6
        metrics["avg_bytes_dn_mb"] = float(np.mean(metrics["bytes_downloaded_per_outer"])) / 1e6

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2, default=str))

    # Also upload to S3 for centralized analysis
    bucket.put_json(
        bucket.uri_for_key(_key(args.run_id, f"final/peer={args.rank}.json")),
        metrics,
    )
    print(log_pfx + f"DONE total={metrics['total_elapsed_sec']:.0f}s "
          f"util={metrics.get('avg_utilization', 0):.1%} "
          f"avg_bytes_up={metrics.get('avg_bytes_up_mb', 0):.1f}MB", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="sparseloco_fleet")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--rank", type=int, required=True)
    w.add_argument("--world-size", type=int, required=True)
    w.add_argument("--run-id", required=True)
    w.add_argument("--moe", choices=["on", "off"], required=True)
    w.add_argument("--tokens-uri", required=True)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--out", required=True)
    w.add_argument("--H", type=int, default=15)
    w.add_argument("--T", type=int, default=100)
    w.add_argument("--bsz", type=int, default=8)
    w.add_argument("--seq-len", type=int, default=256)
    w.add_argument("--n-layer", type=int, default=8)
    w.add_argument("--d-model", type=int, default=384)
    w.add_argument("--n-head", type=int, default=6)
    w.add_argument("--d-ff", type=int, default=1024)
    w.add_argument("--lr-inner", type=float, default=8e-4)
    w.add_argument("--lr-outer", type=float, default=0.7)
    w.add_argument("--density", type=float, default=0.03)
    w.add_argument("--quant-bits", type=int, default=2)
    w.add_argument("--ef-beta", type=float, default=0.95)
    w.add_argument("--chunk-size", type=int, default=4096)
    w.add_argument("--val-every", type=int, default=10)
    w.add_argument("--n-val", type=int, default=10)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--codec", choices=["torch", "packed"], default="torch",
                   help="Wire serialization codec for SparseLoCo payloads.")
    w.add_argument("--aggregator", choices=["peer", "star"], default="peer",
                   help="'peer' = every worker downloads R-1 peer grads (R*(R-1) total GETs). "
                        "'star' = rank 0 collects all + broadcasts mean (R PUTs + R+1 GETs).")
    w.add_argument("--parallel-gets", type=int, default=8,
                   help="Max concurrent S3 GETs in peer aggregator mode.")
    w.add_argument("--async-mode", choices=["sync", "1step"], default="sync",
                   help="'sync' = wait for peers each outer step (DiLoCo). "
                        "'1step' = launch peer fetch in background, apply at "
                        "next outer step (one-step stale, ~2x util).")
    w.add_argument("--target-compute-sec", type=float, default=0.0,
                   help="If > 0, use time-bounded inner training: run for "
                        "this many seconds (in addition to a minimum of --H "
                        "steps). Lets fast workers do more inner steps per "
                        "outer than slow workers, balancing variable hardware.")
    w.add_argument("--n-loops", type=int, default=1,
                   help="Quasar-style looped transformer: cycle through the "
                        "n_layer unique blocks this many times. Effective depth "
                        "= n_layer * n_loops. Same FLOPs as deep model, "
                        "n_loops× fewer unique parameters.")
    w.add_argument("--anchor-inject", action="store_true",
                   help="Quasar anchor-P injection: add input embedding back "
                        "into hidden state at each loop iteration (1/n_loops "
                        "scale). Helps stabilize gradients in deep loops.")
    w.set_defaults(fn=cmd_worker)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
