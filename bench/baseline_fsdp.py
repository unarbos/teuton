"""Centralized FSDP baseline for the streaming/Pluralis comparison.

Runs a configurable GPT (100M / 300M / 1B) on a single multi-GPU box
(intended target: 8xH200), using FSDP for sharded parameters + activation
recomputation. Trains on the FineWeb-edu tokens.bin we built with
`locus.data.fineweb_loader`. Emits a JSON with tokens/sec, val_loss,
$/billion-tokens for the chosen reference price.

Usage:
    # Quick smoke (single GPU, 100M, 100 steps):
    python bench/baseline_fsdp.py run \\
        --size 100M --n-steps 100 \\
        --tokens /tmp/locus_data/tokens.bin \\
        --out /tmp/baseline_100m.json

    # Real 8xH200 run via torchrun:
    torchrun --standalone --nproc_per_node=8 bench/baseline_fsdp.py run \\
        --size 100M --n-steps 5000 \\
        --tokens /workspace/tokens.bin \\
        --hourly-cost 32.0 \\
        --out /workspace/baseline_100m.json

The model architecture matches what the gpt_pipe task will use (RMSNorm +
MHA + SwiGLU FFN), so the comparison is apples-to-apples at the same
parameter count and step count.
"""
from __future__ import annotations

import argparse
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
# Model: pre-norm GPT with MHA + SwiGLU + RMSNorm. ~Same architecture we'll
# pipeline in gpt_pipe so the cost comparison is direct.
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 50304    # GPT-2 BPE rounded up to multiple of 128
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048           # SwiGLU "inner": effective FFN dim is d_ff
    block_size: int = 1024
    dropout: float = 0.0       # not used during training
    bias: bool = False
    rope_theta: float = 10000.0


def _config_for_size(size: str) -> GPTConfig:
    size = size.upper()
    if size == "100M":
        # ~100M params. d=768, 12 heads, 12 layers, ff=2048
        return GPTConfig(n_layer=12, n_head=12, d_model=768, d_ff=2048,
                         block_size=1024)
    if size == "300M":
        # ~300M
        return GPTConfig(n_layer=24, n_head=16, d_model=1024, d_ff=2816,
                         block_size=1024)
    if size == "1B":
        return GPTConfig(n_layer=24, n_head=16, d_model=2048, d_ff=5632,
                         block_size=1024)
    raise ValueError(f"unknown --size {size!r}; pick 100M | 300M | 1B")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def _precompute_rope(d: int, T: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _apply_rope(q, k, cos, sin):
    # q, k: [B, H, T, dh]
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1) + \
               x_rot * sin.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
    return rotate(q), rotate(k)


class MHA(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H = self.cfg.n_head
        dh = D // H
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, cos, sin)
        # Force the flash-attention backend (cuDNN backend has issues on
        # some torch+cu130 / Hopper combos: "No valid execution plans built").
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MHA(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tied embeddings (saves params and helps loss).
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.tok_emb(ids)
        cos, sin = _precompute_rope(self.cfg.d_model // self.cfg.n_head,
                                    T, self.cfg.rope_theta,
                                    device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1).long())
        return logits, loss


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def _load_tokens(path: str) -> torch.Tensor:
    """Load tokens.bin produced by locus.data.fineweb_loader. Format: a
    Locus tensor_io blob (header + payload). Falls back to raw int32/uint16
    np.fromfile for portability."""
    p = Path(path)
    body = p.read_bytes()
    # Try Locus tensor_io decoding first (for files written via cli.prepare).
    try:
        sys.path.insert(0, str(p.parent.parent / "Locus" / "Locus"))  # best-effort
        from locus import tensor_io
        return tensor_io.decode_tensor(body).to(torch.long)
    except Exception:
        pass
    # Fallback: try common raw layouts
    try:
        return torch.from_numpy(np.frombuffer(body, dtype=np.uint16).copy()).to(torch.long)
    except Exception:
        return torch.from_numpy(np.frombuffer(body, dtype=np.int32).copy()).to(torch.long)


def _batch_iter(tokens: torch.Tensor, B: int, T: int, seed: int = 0):
    """Infinite iterator of (input_ids, target_ids) batches."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = tokens.numel()
    while True:
        starts = torch.randint(0, n - T - 1, (B,), generator=g)
        rows = torch.stack([tokens[s : s + T + 1] for s in starts.tolist()])
        yield rows[:, :T].contiguous(), rows[:, 1:].contiguous()


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def _is_dist():
    return os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) > 1


def _setup_dist():
    if _is_dist():
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    return 0, 0, 1


def _wrap_fsdp(model, mp_dtype):
    """Wrap with FSDP (full shard) using activation recomputation per block."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial

    mp_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
    )


def cmd_run(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    local_rank, rank, world_size = _setup_dist()
    is_main = (rank == 0)
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        return 2
    device = torch.device(f"cuda:{local_rank}")
    torch.set_float32_matmul_precision("high")

    # Build model
    cfg = _config_for_size(args.size)
    cfg.block_size = args.seq_len
    if is_main:
        print(f"[{args.size}] cfg: layers={cfg.n_layer} d={cfg.d_model} h={cfg.n_head} "
              f"ff={cfg.d_ff} T={cfg.block_size}", flush=True)

    model = GPT(cfg).to(device)
    n_params = model.num_parameters()
    if is_main:
        print(f"[{args.size}] {n_params / 1e6:.1f}M params", flush=True)

    if world_size > 1:
        model = _wrap_fsdp(model, mp_dtype=torch.bfloat16)
    else:
        model = model.to(torch.bfloat16)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=0.1)

    # Data
    tokens = _load_tokens(args.tokens)
    if is_main:
        print(f"[data] {tokens.numel() / 1e6:.1f}M tokens loaded from {args.tokens}", flush=True)
    train_n = int(0.95 * tokens.numel())
    train_tokens = tokens[:train_n]
    val_tokens = tokens[train_n:]

    # Per-rank batches (data parallel: each rank reads a different slice).
    train_iter = _batch_iter(train_tokens, B=args.micro_bsz, T=args.seq_len,
                              seed=args.seed + rank * 1000)
    val_iter = _batch_iter(val_tokens, B=args.micro_bsz, T=args.seq_len,
                            seed=args.seed + 999999)

    # Train
    metrics = {"size": args.size, "n_params": n_params, "n_steps": args.n_steps,
               "world_size": world_size, "micro_bsz": args.micro_bsz,
               "seq_len": args.seq_len, "lr": args.lr, "seed": args.seed}
    losses = []
    val_losses = []
    t_start = time.time()
    tokens_seen = 0
    for step in range(args.n_steps):
        model.train()
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        tokens_seen += args.micro_bsz * args.seq_len * world_size
        if is_main and (step % args.log_every == 0 or step == args.n_steps - 1):
            losses.append({"step": step, "loss": float(loss.item()),
                           "tokens_seen": tokens_seen,
                           "elapsed_sec": time.time() - t_start})
            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"tokens/sec={tokens_seen / (time.time() - t_start):.0f}",
                  flush=True)
        if args.val_every > 0 and step > 0 and step % args.val_every == 0:
            # All ranks must participate in val (FSDP requires the same
            # forward graph across ranks for grad reduce).
            v = _eval(model, val_iter, args.n_val_steps, device)
            if is_main:
                val_losses.append({"step": step, "val_loss": v})
                print(f"  step {step:5d}  val_loss={v:.4f}", flush=True)

    # Final val (all ranks)
    v = _eval(model, val_iter, args.n_val_steps, device)
    if is_main:
        val_losses.append({"step": args.n_steps, "val_loss": v})

    elapsed = time.time() - t_start
    tokens_per_sec = tokens_seen / max(elapsed, 1.0)

    cost_per_hour = args.hourly_cost
    cost_per_sec = cost_per_hour / 3600.0
    cost = elapsed * cost_per_sec
    cost_per_b_tokens = (cost / max(tokens_seen, 1)) * 1e9

    metrics.update({
        "elapsed_sec": elapsed,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": tokens_per_sec,
        "hourly_cost_usd": cost_per_hour,
        "total_cost_usd": cost,
        "dollars_per_b_tokens": cost_per_b_tokens,
        "final_val_loss": val_losses[-1]["val_loss"] if val_losses else None,
        "loss_curve": losses,
        "val_curve": val_losses,
    })

    if is_main:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"\n[result] tokens/sec={tokens_per_sec:.0f}", flush=True)
        print(f"[result] $/B-tokens (at ${cost_per_hour}/hr) = ${cost_per_b_tokens:.2f}",
              flush=True)
        print(f"[result] wrote {args.out}", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0


@torch.no_grad()
def _eval(model, val_iter, n_steps, device):
    model.eval()
    losses = []
    for _ in range(n_steps):
        x, y = next(val_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="baseline_fsdp")
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--size", type=str, default="100M", help="100M | 300M | 1B")
    rp.add_argument("--tokens", type=str, required=True, help="tokens.bin path")
    rp.add_argument("--n-steps", type=int, default=2000)
    rp.add_argument("--micro-bsz", type=int, default=8)
    rp.add_argument("--seq-len", type=int, default=1024)
    rp.add_argument("--lr", type=float, default=3e-4)
    rp.add_argument("--seed", type=int, default=42)
    rp.add_argument("--log-every", type=int, default=10)
    rp.add_argument("--val-every", type=int, default=200)
    rp.add_argument("--n-val-steps", type=int, default=20)
    rp.add_argument("--hourly-cost", type=float, default=32.0,
                    help="USD/hour for the box (8xH200 ~ $24-32/hr)")
    rp.add_argument("--out", type=str, default="/tmp/baseline_fsdp.json")
    rp.set_defaults(fn=cmd_run)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
