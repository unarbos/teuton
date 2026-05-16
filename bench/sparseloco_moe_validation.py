"""SparseLoCo + MoE validation experiment.

Tests the hypothesis from the conversation: does MoE expert specialization
survive averaged pseudo-gradients across workers seeing different data
shards, especially under SparseLoCo's compression?

Compares 6 conditions:
   {DDP-AdamW, DiLoCo-dense, SparseLoCo} × {dense-FFN, MoE-FFN}

Each "worker" is simulated serially on one GPU (R workers each running H
inner steps with their own data shard, then merging pseudo-gradients).
This is mathematically equivalent to true distributed since workers don't
communicate during inner steps.

Designed to be RUNNABLE on a single H200 GPU per condition. Run 6
conditions in parallel across 6 GPUs of an 8xH200 box.

Metrics emitted to JSON:
- final_val_loss
- val_loss_curve (every K outer steps)
- (MoE only) per-expert routing entropy
- (MoE only) per-expert utilization variance across workers
- (MoE only) per-layer expert weight cosine similarity across workers
  (measured BEFORE merge at the last outer step — the question is whether
  experts have specialized differently and how the merge handles it)

Usage:
    python bench/sparseloco_moe_validation.py run \
        --condition sparseloco --moe on \
        --tokens /workspace/teuton_data/tokens.bin \
        --device cuda:0 --out /workspace/sl_sparseloco_moe.json

The "ddp" condition uses 1 model on a (R*B, T) batch — mathematically
identical to R-worker DDP with mean-reduction.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Cfg:
    vocab: int = 50304
    n_layer: int = 8
    n_head: int = 6
    d: int = 384
    d_ff: int = 1024
    block_size: int = 256
    n_experts: int = 8
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.5
    use_moe: bool = False
    # Looped transformer support (Quasar-style parameter sharing across depth):
    # If n_loops > 1, n_layer is interpreted as the number of UNIQUE blocks,
    # and the model loops through them n_loops times for n_layer*n_loops
    # effective depth at the cost of just n_layer unique parameters.
    n_loops: int = 1
    anchor_inject: bool = False     # Quasar anchor-P injection at each loop


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.w


class MHA(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.cfg.n_head
        dh = D // H
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoEFFN(nn.Module):
    """Top-K MoE with SwiGLU experts. Capacity-bound (drops overflow tokens
    per the standard Switch / Mixtral pattern). Returns (out, aux_routing_info)."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(cfg.d, cfg.d_ff) for _ in range(cfg.n_experts)
        ])

    def forward(self, x, collect_routing=False):
        B, T, D = x.shape
        n = B * T
        x_flat = x.reshape(n, D)
        logits = self.gate(x_flat)
        scores = F.softmax(logits, dim=-1)            # (n, E)
        topk_scores, topk_idx = scores.topk(self.cfg.moe_top_k, dim=-1)  # (n, K)
        topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x_flat)
        # Per-expert dispatch
        for e in range(self.cfg.n_experts):
            # mask of token-positions where this expert is selected (any K slot)
            is_e = (topk_idx == e)              # (n, K)
            if not is_e.any():
                continue
            # weights for expert e (sum of scores in slots where idx==e)
            w_e = (topk_scores * is_e.float()).sum(-1)   # (n,)
            mask = w_e > 0
            tokens_e = x_flat[mask]
            if tokens_e.shape[0] == 0:
                continue
            y_e = self.experts[e](tokens_e)
            out[mask] += y_e * w_e[mask].unsqueeze(-1)

        info = None
        if collect_routing:
            # Routing distribution: how often each expert is in top-K
            with torch.no_grad():
                util = torch.zeros(self.cfg.n_experts, device=x.device)
                for e in range(self.cfg.n_experts):
                    util[e] = (topk_idx == e).float().sum() / max(n, 1)
                # Gate entropy: mean over tokens of -sum(p log p)
                ent = -(scores * (scores.clamp_min(1e-9).log())).sum(-1).mean()
                info = {"util": util.cpu(), "entropy": float(ent.item())}
        return out.reshape(B, T, D), info


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d)
        self.attn = MHA(cfg)
        self.norm2 = RMSNorm(cfg.d)
        self.ffn = MoEFFN(cfg) if cfg.use_moe else SwiGLU(cfg.d, cfg.d_ff)
        self.use_moe = cfg.use_moe

    def forward(self, x, collect_routing=False):
        x = x + self.attn(self.norm1(x))
        if self.use_moe:
            ffn_out, info = self.ffn(self.norm2(x), collect_routing=collect_routing)
            x = x + ffn_out
            return x, info
        else:
            x = x + self.ffn(self.norm2(x))
            return x, None


class GPT(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.block_size, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d)
        self.lm_head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.lm_head.weight = self.tok.weight                   # tied
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, targets=None, collect_routing=False):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok(ids) + self.pos(pos)
        anchor = x if self.cfg.anchor_inject and self.cfg.n_loops > 1 else None
        infos = []
        # Looped forward: cycle through blocks `n_loops` times. Each loop reuses
        # the same parameters; effective depth = n_layer * n_loops.
        for loop_idx in range(self.cfg.n_loops):
            for b in self.blocks:
                x, info = b(x, collect_routing=collect_routing)
                if collect_routing and info is not None:
                    infos.append(info)
            if anchor is not None and loop_idx < self.cfg.n_loops - 1:
                # Quasar anchor-P injection (gradient scaled by 1/n_loops).
                x = x + anchor * (1.0 / self.cfg.n_loops)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    targets.reshape(-1).long())
        return logits, loss, infos


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_tokens(path: str) -> torch.Tensor:
    body = Path(path).read_bytes()
    # Try Teuton tensor_io first; else raw int32/uint16 fallback
    try:
        sys.path.insert(0, "/root/Teuton")
        from teuton import tensor_io
        return tensor_io.decode_tensor(body).to(torch.long)
    except Exception:
        try:
            return torch.from_numpy(np.frombuffer(body, dtype=np.int32).copy()).to(torch.long)
        except Exception:
            return torch.from_numpy(np.frombuffer(body, dtype=np.uint16).copy()).to(torch.long)


def shard_iter(tokens: torch.Tensor, B: int, T: int, seed: int):
    """Infinite iterator of (input_ids, target_ids) batches drawn from the
    given sub-tensor (one shard)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = tokens.numel()
    while True:
        starts = torch.randint(0, n - T - 1, (B,), generator=g)
        rows = torch.stack([tokens[s : s + T + 1] for s in starts.tolist()])
        yield rows[:, :T].contiguous(), rows[:, 1:].contiguous()


# ---------------------------------------------------------------------------
# SparseLoCo codec: chunked Top-k with optional 2-bit quantization + EF
# Per Algorithm 1 of the SparseLoCo paper (arxiv 2508.15706).
# ---------------------------------------------------------------------------


def chunked_topk_quant_encode(grad: torch.Tensor, density: float, chunk_size: int,
                                bits: int = 2):
    """Encode a parameter pseudo-grad: chunked Top-k + quantization.

    Returns:
      - decompressed: dense tensor (same shape as grad) for local EF use
      - payload:      sparse compressed dict for transmission
                      {q (int8), scale (fp32), idx (int32), pad (int), shape (tuple)}
      - total_bytes:  estimated wire size of payload
    """
    flat = grad.detach().reshape(-1)
    pad = (chunk_size - flat.numel() % chunk_size) % chunk_size
    if pad:
        flat = F.pad(flat, (0, pad))
    chunks = flat.reshape(-1, chunk_size)
    n_chunks, C = chunks.shape
    k = max(1, int(round(C * density)))

    abs_chunks = chunks.abs()
    _, topk_idx = abs_chunks.topk(k, dim=-1)
    selected = chunks.gather(-1, topk_idx)        # (n_chunks, k)

    if bits == 2:
        scale = selected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 3.0
        q = (selected / scale).round().clamp(-3, 3).to(torch.int8)
        deq = q.float() * scale
    else:
        deq = selected
        q = selected
        scale = torch.ones(n_chunks, 1, device=selected.device)

    dense = torch.zeros_like(chunks).scatter_(-1, topk_idx, deq)
    dense_flat = dense.reshape(-1)
    if pad:
        dense_flat = dense_flat[: -pad]
    decompressed = dense_flat.reshape(grad.shape)

    bytes_q = q.numel() * (bits / 8.0)
    bytes_idx = topk_idx.numel() * (math.ceil(math.log2(C)) / 8.0)
    bytes_scale = scale.numel() * 4
    total_bytes = bytes_q + bytes_idx + bytes_scale

    payload = {
        "q": q.to(torch.int8).cpu(),
        "scale": scale.to(torch.float32).cpu(),
        "idx": topk_idx.to(torch.int32).cpu(),
        "pad": int(pad),
        "shape": list(grad.shape),
        "chunk_size": int(chunk_size),
    }
    return decompressed, payload, total_bytes


def chunked_topk_quant_decode(payload: dict, device=None) -> torch.Tensor:
    """Inverse of chunked_topk_quant_encode (sparse payload -> dense tensor)."""
    q = payload["q"]
    scale = payload["scale"]
    idx = payload["idx"]
    pad = int(payload["pad"])
    shape = list(payload["shape"])
    C = int(payload["chunk_size"])
    if device is not None:
        q = q.to(device)
        scale = scale.to(device)
        idx = idx.to(device)
    deq = q.float() * scale                        # (n_chunks, k)
    n_chunks = idx.shape[0]
    full_chunks = torch.zeros(n_chunks, C, dtype=torch.float32, device=q.device)
    full_chunks.scatter_(-1, idx.long(), deq)
    flat = full_chunks.reshape(-1)
    if pad:
        flat = flat[: -pad]
    return flat.reshape(shape)


def sparseloco_compress(delta: dict, ef: dict, beta: float, density: float,
                          chunk_size: int, bits: int, return_payloads: bool = False):
    """SparseLoCo step on per-param dict of tensors.
    delta: pseudo-gradient {name: tensor}
    ef:    error-feedback buffer {name: tensor} (mutated in place)
    Returns:
        if return_payloads=False: (dense_compressed {name: tensor}, total_bytes_sent)
        if return_payloads=True:  (dense_compressed, payloads {name: dict}, total_bytes_sent)
    """
    out_dense = {}
    out_payload = {}
    total_bytes = 0
    for name, d in delta.items():
        ef[name].mul_(beta).add_(d)
        d_hat, payload, n_bytes = chunked_topk_quant_encode(
            ef[name], density, chunk_size, bits
        )
        ef[name].sub_(d_hat)
        out_dense[name] = d_hat
        if return_payloads:
            out_payload[name] = payload
        total_bytes += n_bytes
    if return_payloads:
        return out_dense, out_payload, total_bytes
    return out_dense, total_bytes


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def snapshot_params(model: nn.Module) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def write_params(model: nn.Module, snap: dict) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(snap[n])


def diff_params(prev: dict, current: nn.Module) -> dict:
    return {n: (prev[n] - p.detach()) for n, p in current.named_parameters()}


def apply_outer(model: nn.Module, avg_delta: dict, alpha: float):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.sub_(alpha * avg_delta[n])


@torch.no_grad()
def evaluate(model: nn.Module, val_iter, n_steps: int, device, collect_routing=False):
    model.eval()
    losses = []
    routing = []
    for _ in range(n_steps):
        x, y = next(val_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss, infos = model(x, y, collect_routing=collect_routing)
        losses.append(loss.item())
        if collect_routing and infos:
            routing.append(infos)
    model.train()
    routing_summary = None
    if collect_routing and routing:
        n_layers = len(routing[0])
        utils = []
        ents = []
        for li in range(n_layers):
            u = torch.stack([torch.tensor(r[li]["util"]) for r in routing]).mean(0)
            e = float(np.mean([r[li]["entropy"] for r in routing]))
            utils.append(u.tolist())
            ents.append(e)
        routing_summary = {"per_layer_util": utils, "per_layer_entropy": ents}
    return float(np.mean(losses)), routing_summary


def expert_weight_drift(workers_params: list[dict], cfg: Cfg) -> dict:
    """Pre-merge: how different are expert weights across workers?
    Returns per-layer mean cosine-similarity of expert e across workers."""
    if not cfg.use_moe:
        return {}
    R = len(workers_params)
    sims = []
    for li in range(cfg.n_layer):
        layer_sims = []
        for e in range(cfg.n_experts):
            for w_name in [f"blocks.{li}.ffn.experts.{e}.w1.weight",
                            f"blocks.{li}.ffn.experts.{e}.w2.weight",
                            f"blocks.{li}.ffn.experts.{e}.w3.weight"]:
                if w_name not in workers_params[0]:
                    continue
                ws = [workers_params[r][w_name].reshape(-1) for r in range(R)]
                # Pairwise cosine sim
                pair_sims = []
                for i in range(R):
                    for j in range(i + 1, R):
                        cs = F.cosine_similarity(ws[i].float(), ws[j].float(), dim=0)
                        pair_sims.append(cs.item())
                if pair_sims:
                    layer_sims.append(float(np.mean(pair_sims)))
        sims.append(float(np.mean(layer_sims)) if layer_sims else 1.0)
    return {"per_layer_pre_merge_expert_cos_sim": sims}


def cmd_run(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    cfg = Cfg(use_moe=(args.moe == "on"))
    cfg.block_size = args.seq_len

    print(f"[{args.condition}/moe={args.moe}] cfg: layers={cfg.n_layer} d={cfg.d} "
          f"h={cfg.n_head} ff={cfg.d_ff} T={cfg.block_size} "
          f"experts={cfg.n_experts if cfg.use_moe else 0}", flush=True)

    model = GPT(cfg).to(device)
    print(f"[{args.condition}/moe={args.moe}] {model.num_params() / 1e6:.1f}M params",
          flush=True)

    # Data
    tokens = load_tokens(args.tokens)
    train_n = int(0.95 * tokens.numel())
    train_tokens = tokens[:train_n]
    val_tokens = tokens[train_n:]
    print(f"[data] {tokens.numel() / 1e6:.1f}M tokens "
          f"(train={train_tokens.numel() / 1e6:.1f}M, val={val_tokens.numel() / 1e6:.1f}M)",
          flush=True)

    # Each worker gets a slice of the training data (non-overlapping)
    R = args.workers
    shard_size = train_tokens.numel() // R
    worker_iters = []
    for r in range(R):
        s = r * shard_size
        e = s + shard_size
        shard = train_tokens[s:e]
        worker_iters.append(shard_iter(shard, args.bsz, args.seq_len,
                                         seed=args.seed + r * 1000))
    val_iter = shard_iter(val_tokens, args.bsz, args.seq_len, seed=args.seed + 999999)

    metrics = {
        "condition": args.condition, "moe": args.moe, "R": R,
        "H_inner": args.H_inner, "T_outer": args.T_outer, "bsz": args.bsz,
        "seq_len": args.seq_len, "lr_inner": args.lr_inner, "lr_outer": args.lr_outer,
        "density": args.density, "ef_beta": args.ef_beta, "quant_bits": args.quant_bits,
        "n_params": model.num_params(),
        "loss_curve": [], "val_curve": [], "bytes_per_outer": [],
        "expert_specialization": None,
        "elapsed_sec": None,
    }

    t0 = time.time()

    if args.condition == "ddp":
        # Mathematically equivalent to R workers DDP-averaging at every step:
        # one optimizer step per "inner iteration", batch is R times bigger.
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_inner,
                                betas=(0.9, 0.95), weight_decay=0.1)
        total_steps = args.T_outer * args.H_inner
        for step in range(total_steps):
            xs, ys = [], []
            for r in range(R):
                x, y = next(worker_iters[r])
                xs.append(x); ys.append(y)
            x = torch.cat(xs, dim=0).to(device, non_blocking=True)
            y = torch.cat(ys, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % max(args.H_inner, 1) == 0:
                metrics["loss_curve"].append({"step": step, "loss": float(loss.item())})
                if step % args.val_every == 0:
                    v, rinfo = evaluate(model, val_iter, args.n_val, device,
                                          collect_routing=cfg.use_moe)
                    metrics["val_curve"].append({"step": step, "val_loss": v,
                                                  "routing": rinfo})
                    print(f"  [ddp/{args.moe}] step {step:5d}  loss={loss.item():.4f}  val={v:.4f}",
                          flush=True)
        # Final eval + (no expert drift to measure since one model)
        v, rinfo = evaluate(model, val_iter, args.n_val, device, collect_routing=cfg.use_moe)
        metrics["val_curve"].append({"step": total_steps, "val_loss": v, "routing": rinfo})
        metrics["final_val_loss"] = v

    else:
        # DiLoCo / SparseLoCo
        # Each worker maintains its own AdamW state and EF buffer (SparseLoCo only).
        worker_states = []
        for r in range(R):
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr_inner,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            ef = ({n: torch.zeros_like(p) for n, p in model.named_parameters()}
                  if args.condition == "sparseloco" else None)
            worker_states.append({"opt_state": copy.deepcopy(opt.state_dict()),
                                   "ef": ef})

        chunk_size = args.chunk_size

        for outer in range(args.T_outer):
            theta_prev = snapshot_params(model)
            worker_deltas = []
            worker_param_snapshots = []           # for expert-drift measurement at end
            losses_this_outer = []

            for r in range(R):
                # Restore base weights
                write_params(model, theta_prev)
                # Restore this worker's optimizer state (or create fresh)
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr_inner,
                                        betas=(0.9, 0.95), weight_decay=0.1)
                opt.load_state_dict(worker_states[r]["opt_state"])

                last_loss = float("nan")
                for h in range(args.H_inner):
                    x, y = next(worker_iters[r])
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, loss, _ = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    last_loss = float(loss.item())
                losses_this_outer.append(last_loss)
                worker_states[r]["opt_state"] = copy.deepcopy(opt.state_dict())

                # Pseudo-gradient
                delta = diff_params(theta_prev, model)
                worker_deltas.append(delta)
                # If we're at the LAST outer step and MoE, snapshot pre-merge weights
                if cfg.use_moe and outer == args.T_outer - 1:
                    worker_param_snapshots.append({
                        n: p.detach().clone() for n, p in model.named_parameters()
                    })

            # Now merge
            avg_delta = {n: torch.zeros_like(theta_prev[n]) for n in theta_prev}
            bytes_sent_this_outer = 0
            if args.condition == "diloco":
                for delta in worker_deltas:
                    for n in avg_delta:
                        avg_delta[n].add_(delta[n] / R)
                # Bytes: full pseudo-gradient per worker
                bytes_sent_this_outer = sum(
                    p.numel() * 4 for p in theta_prev.values()
                ) * R
            else:  # sparseloco
                for r in range(R):
                    compressed, n_bytes = sparseloco_compress(
                        worker_deltas[r], worker_states[r]["ef"],
                        beta=args.ef_beta, density=args.density,
                        chunk_size=chunk_size, bits=args.quant_bits,
                    )
                    bytes_sent_this_outer += n_bytes
                    for n in avg_delta:
                        avg_delta[n].add_(compressed[n] / R)

            # Restore prev weights and apply outer step
            write_params(model, theta_prev)
            apply_outer(model, avg_delta, args.lr_outer)

            avg_loss_this_outer = float(np.mean(losses_this_outer))
            metrics["loss_curve"].append({"outer": outer,
                                           "loss": avg_loss_this_outer,
                                           "n_inner": args.H_inner})
            metrics["bytes_per_outer"].append(bytes_sent_this_outer)
            if outer % args.val_every_outer == 0 or outer == args.T_outer - 1:
                v, rinfo = evaluate(model, val_iter, args.n_val, device,
                                      collect_routing=cfg.use_moe)
                metrics["val_curve"].append({"outer": outer, "val_loss": v,
                                              "routing": rinfo})
                print(f"  [{args.condition}/{args.moe}] outer {outer:4d}  "
                      f"inner_loss={avg_loss_this_outer:.4f}  val={v:.4f}  "
                      f"bytes={bytes_sent_this_outer / 1e6:.2f}MB",
                      flush=True)

            if cfg.use_moe and outer == args.T_outer - 1 and worker_param_snapshots:
                metrics["expert_specialization"] = expert_weight_drift(
                    worker_param_snapshots, cfg
                )

        v, rinfo = evaluate(model, val_iter, args.n_val, device, collect_routing=cfg.use_moe)
        metrics["val_curve"].append({"outer": args.T_outer, "val_loss": v, "routing": rinfo})
        metrics["final_val_loss"] = v

    metrics["elapsed_sec"] = time.time() - t0
    metrics["mean_bytes_per_outer"] = (
        float(np.mean(metrics["bytes_per_outer"]))
        if metrics["bytes_per_outer"] else 0.0
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[{args.condition}/{args.moe}] DONE  final_val={metrics['final_val_loss']:.4f}  "
          f"elapsed={metrics['elapsed_sec']:.0f}s  wrote {args.out}", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="sparseloco_moe_validation")
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--condition", choices=["ddp", "diloco", "sparseloco"], required=True)
    rp.add_argument("--moe", choices=["on", "off"], required=True)
    rp.add_argument("--tokens", required=True)
    rp.add_argument("--device", default="cuda:0")
    rp.add_argument("--out", required=True)
    rp.add_argument("--workers", type=int, default=4)
    rp.add_argument("--H-inner", type=int, default=15)
    rp.add_argument("--T-outer", type=int, default=200)
    rp.add_argument("--bsz", type=int, default=8)
    rp.add_argument("--seq-len", type=int, default=256)
    rp.add_argument("--lr-inner", type=float, default=8e-4)
    rp.add_argument("--lr-outer", type=float, default=0.7)
    rp.add_argument("--density", type=float, default=0.03)
    rp.add_argument("--quant-bits", type=int, default=2)
    rp.add_argument("--ef-beta", type=float, default=0.95)
    rp.add_argument("--chunk-size", type=int, default=4096)
    rp.add_argument("--val-every", type=int, default=200,
                    help="DDP: validate every N inner steps")
    rp.add_argument("--val-every-outer", type=int, default=10,
                    help="DiLoCo/SparseLoCo: validate every N outer steps")
    rp.add_argument("--n-val", type=int, default=20)
    rp.add_argument("--seed", type=int, default=42)
    rp.set_defaults(fn=cmd_run)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
