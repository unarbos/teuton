"""Streaming-pipeline GPT training task.

A real GPT broken into N pipeline stages. Each stage has 1 transformer
block (RMSNorm + MHA + RMSNorm + SwiGLU FFN). The head stage adds the
token + positional embeddings; the tail stage adds the final RMSNorm,
lm_head projection, and cross-entropy loss.

All forward, backward, and outer-step graphs are built by hand in Teuton
IR. There is no autograd at runtime — backward is computed analytically
in the per-stage `build_bwd_graph`.

We keep this Phase-1 simple by:
  * Using learned positional embedding (not RoPE) — IR cost lower.
  * Using "naive" attention (Q @ K^T -> mask -> softmax -> @ V), not
    flash-style. Each microbatch is small (B<=8, T=128 or 256) so memory
    is fine, and this lets us write a clean closed-form backward.
  * Storing per-block weights as a single concatenated W_blob per stage
    plus a small SHAPES manifest, so the streaming orch sees one weight
    file per stage. (Avoids changing the streaming protocol to support
    multi-tensor weights per stage.)

Wire layout per microbatch (mb=M) at epoch E for stage K:

    runs/<id>/streaming/epoch=E/stage=K/outputs/mb=M/x.bin       (B, T, D) fp32
    runs/<id>/streaming/epoch=E/stage=K/outputs/mb=M/done.json   tail only
    runs/<id>/streaming/epoch=E/stage=K/bwd/mb=M/dW.bin           per-stage dW concat
    runs/<id>/streaming/epoch=E/stage=K/bwd/mb=M/dL_dx_in.bin    (B, T, D) fp32

Per-epoch state:
    runs/<id>/weights/epoch=E/stage_K_W.bin                       per-stage W concat
    runs/<id>/static/tokens.bin                                   shared corpus
"""
from __future__ import annotations

import math

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..streaming import PipelineStage, StreamingParams
from ..storage import LocalBucket


# --------------------------------------------------------------------------- #
# Hyperparameters (overridable in tests by setting attribute before
# build_streaming_inputs is called)
# --------------------------------------------------------------------------- #

# Model
VOCAB = 50304
D = 384
N_HEAD = 6
D_FF = 1024     # SwiGLU FFN inner dim
T = 128         # seq len per microbatch
B = 4           # microbatch size
N_STAGES = 4    # number of pipeline stages
# Number of transformer blocks PER stage. With N_STAGES=4 and
# N_BLOCKS_PER_STAGE=2, total layers = 8. Fused stages reduce S3 round-trips
# per microbatch by N_BLOCKS_PER_STAGE×: each stage does block_1 -> block_2
# locally with NO wire crossing between blocks 1 and 2.
N_BLOCKS_PER_STAGE = 1
N_MICROBATCHES = 16
MAX_EPOCHS = 5

# Optimizer
LR = 3e-4

# Data corpus URI (set by bootstrap to runs/<id>/static/tokens.bin or shared)
TOKENS_URI = ""

# Init
WEIGHTS_SEED = 42
INIT_SCALE = 0.02
DTYPE = "float32"
EPS = 1e-5

# Pluralis subspace-projection on the activation wire (Phase 2.1).
# When SUBSPACE_K is None: full-rank D-dim activations cross the wire (Phase 1).
# When SUBSPACE_K=k (e.g. 40): forward emits x_proj = x @ U_k (B, T, k), and
# the receiving stage expands via U_k^T (B, T, D). U_k is a static (D, k)
# random-orthonormal matrix shared across all stage boundaries, written once
# at bootstrap to runs/<id>/static/u_k.bin.
# Same scheme on the backward gradient lane: dL_dx is projected to k-dim
# before being sent upstream. Lossy when k < D (we drop everything outside
# span(U_k)) but the constraint preserves the dominant directions.
SUBSPACE_K: int | None = None
# Phase 2.2: smaller subspace for the backward gradient lane (asymmetric).
# Defaults to SUBSPACE_K when None; set independently to e.g. 10 to get
# another ~k_x/k_dy compression on the bwd lane.
SUBSPACE_K_DY: int | None = None
SUBSPACE_SEED = 31415
SUBSPACE_DY_SEED = 41718  # different seed -> independent random U_k matrix

# Phase 2.3: int8 per-channel quantization on the projected wire. When True,
# both the forward x_proj and the backward dL_dx_proj are packed as
# (uint8 blob = scale_fp32 || int8_values). Receiver unpacks/dequantizes.
# Composes with subspace projection: ~4x bytes reduction on top of the
# subspace ratio.
WIRE_INT8: bool = False

# Tied embeddings — when True, tok_emb is stored ONCE as a shared static
# blob (runs/<id>/static/tok_emb.bin) instead of being part of stage 0's W
# blob. The tail stage's lm_head also references this same blob (transposed)
# instead of having its own W_lm. Saves ~2 * V*D fp32 bytes per epoch on
# weight downloads (76 MB at 100M scale). Safe because we currently freeze
# tok_emb gradient anyway (scatter_add not yet in IR).
TIED_EMBED: bool = False

# K-inner-step streaming LocoProp. When K_INNER > 1, each backward job at
# stage K runs K LOCAL SGD steps to minimize ||block(x_in; W) - target||^2
# where target = x_out_initial - LOCO_ETA_OUT * dL_dx_out is the LocoProp
# preferred-output target. The job emits dW = (W_K_local - W_initial), which
# the outer step averages across microbatches as usual. Effect: K updates
# worth of progress per S3 round-trip; wire stays the same.
#
# K_INNER=1 reproduces the current (Phase 2) behavior exactly. K_INNER=4-8
# is the "missing optimization" win.
K_INNER: int = 1
# Local SGD step size inside the K-inner loop. Total magnitude of the local
# update is bounded by K_INNER * LOCO_ETA_INNER, so set ETA_INNER ~ LR / K
# to keep the sum at the same scale as plain SGD.
LOCO_ETA_INNER: float = 0.0
# Step size for forming the LocoProp target from (x_out, dL_dx_out).
LOCO_ETA_OUT: float = 1.0


# --------------------------------------------------------------------------- #
# Per-stage weight blob layout
#
# We pack a stage's parameters into a single 1-D fp32 tensor `W_blob` so
# that the streaming protocol's "one weights tensor per stage" assumption
# holds. The packing order is fixed:
#
#   stage 0 (head):
#     tok_emb     (V, D)
#     pos_emb     (T, D)
#     ... block params (see below) ...
#   any stage:
#     g_norm1     (D,)
#     W_qkv       (D, 3D)
#     W_o         (D, D)
#     g_norm2     (D,)
#     W_ffn1      (D, F)        # FFN expansion (silu branch)
#     W_ffn2      (D, F)        # FFN expansion (gate branch)
#     W_ffn3      (F, D)        # FFN contraction
#   stage S-1 (tail) extra:
#     g_norm_f    (D,)
#     W_lm        (D, V)
#
# Total elements per stage are precomputed below.
# --------------------------------------------------------------------------- #


def _single_block_param_sizes() -> list[tuple[str, list[int]]]:
    return [
        ("g_norm1", [D]),
        ("W_qkv", [D, 3 * D]),
        ("W_o", [D, D]),
        ("g_norm2", [D]),
        ("W_ffn1", [D, D_FF]),
        ("W_ffn2", [D, D_FF]),
        ("W_ffn3", [D_FF, D]),
    ]


def _block_param_sizes() -> list[tuple[str, list[int]]]:
    """Per-stage block params, repeated N_BLOCKS_PER_STAGE times. Names get
    a "_b{i}" suffix so they remain unique within the W blob."""
    out: list[tuple[str, list[int]]] = []
    for b in range(N_BLOCKS_PER_STAGE):
        for name, shape in _single_block_param_sizes():
            out.append((f"{name}_b{b}", shape))
    return out


def _block_names_for_stage() -> list[str]:
    """Just the per-block param names (without head/tail extras), useful
    for the K-inner LocoProp loop."""
    return [f"{n}_b{b}" for b in range(N_BLOCKS_PER_STAGE)
            for n, _ in _single_block_param_sizes()]


def _head_extra_sizes() -> list[tuple[str, list[int]]]:
    # When tied, tok_emb lives in a separate static blob, NOT in stage 0's W blob.
    if TIED_EMBED:
        return [("pos_emb", [T, D])]
    return [("tok_emb", [VOCAB, D]), ("pos_emb", [T, D])]


def _tail_extra_sizes() -> list[tuple[str, list[int]]]:
    # When tied, lm_head is the transpose of tok_emb (read from static blob),
    # so no W_lm in stage_S-1's W blob.
    if TIED_EMBED:
        return [("g_norm_f", [D])]
    return [("g_norm_f", [D]), ("W_lm", [D, VOCAB])]


def _stage_param_sizes(stage: int, n_stages: int) -> list[tuple[str, list[int]]]:
    sizes: list[tuple[str, list[int]]] = []
    if stage == 0:
        sizes.extend(_head_extra_sizes())
    sizes.extend(_block_param_sizes())
    if stage == n_stages - 1:
        sizes.extend(_tail_extra_sizes())
    return sizes


def _bind_tok_emb(gb: GraphBuilder, bucket: LocalBucket, run_id: str):
    """const_blob ref to the shared tok_emb. Used by head (embedding lookup)
    and tail (lm_head matmul) when TIED_EMBED is True."""
    uri = bucket.uri_for_key(f"runs/{run_id}/static/tok_emb.bin")
    return gb.const_blob(uri, shape=[VOCAB, D], dtype=DTYPE)


def _stage_total_numel(stage: int, n_stages: int) -> int:
    return sum(math.prod(shape) for _, shape in _stage_param_sizes(stage, n_stages))


def _make_uk(d: int, k: int, seed: int = SUBSPACE_SEED) -> torch.Tensor:
    """Random orthonormal (d, k) matrix, deterministic given d, k, seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.empty(d, d).normal_(generator=g)
    Q, _ = torch.linalg.qr(m)
    return Q[:, :k].contiguous().to(torch.float32)


def _wire_dim() -> int:
    return SUBSPACE_K if SUBSPACE_K is not None else D


def _project_in_ir(gb: GraphBuilder, x, u_k):
    """x: (B, T, D) -> (B, T, k); just x @ U_k."""
    return gb.matmul(x, u_k)


def _unproject_in_ir(gb: GraphBuilder, x_proj, u_k):
    """x_proj: (B, T, k) -> (B, T, D); x_proj @ U_k.T."""
    return gb.matmul(x_proj, gb.transpose(u_k, dims=[1, 0]))


def _bind_uk(gb: GraphBuilder, bucket: LocalBucket, run_id: str, *, dy: bool = False):
    """Add a const_blob ref to the static U_k matrix.
    When dy=True, refs the (potentially smaller) backward U_k_dy."""
    if dy:
        k = SUBSPACE_K_DY if SUBSPACE_K_DY is not None else SUBSPACE_K
        name = "u_k_dy.bin"
    else:
        k = SUBSPACE_K
        name = "u_k.bin"
    uri = bucket.uri_for_key(f"runs/{run_id}/static/{name}")
    return gb.const_blob(uri, shape=[D, int(k)], dtype=DTYPE)


def _split_blob_in_ir(gb: GraphBuilder, blob, stage: int, n_stages: int) -> dict:
    """Given the 1-D weight blob for this stage, slice + reshape into
    named per-parameter views. Returns a dict {name: ref}."""
    out: dict = {}
    offset = 0
    for name, shape in _stage_param_sizes(stage, n_stages):
        n = math.prod(shape)
        s = gb.slice(blob, dim=0, start=offset, end=offset + n)
        s = gb.reshape(s, shape=shape)
        out[name] = s
        offset += n
    return out


# --------------------------------------------------------------------------- #
# IR sub-routines (forward + backward fragments)
# --------------------------------------------------------------------------- #


def _rmsnorm_fwd(gb: GraphBuilder, x, g, *, eps=EPS):
    """y = x * rsqrt(mean(x^2)+eps) * g, broadcasting g over (B, T)."""
    x2 = gb.mul(x, x)
    ms = gb.mean(x2, dim=-1, keepdim=True)
    eps_c = gb.const(float(eps))
    rstd = gb.div(gb.const(1.0), gb.sqrt(gb.add(ms, eps_c)))
    x_hat = gb.mul(x, rstd)
    g3 = gb.reshape(g, shape=[1, 1, D])
    g_btd = gb.broadcast(g3, shape=[B, T, D])
    y = gb.mul(x_hat, g_btd)
    return y, rstd, x_hat


def _rmsnorm_bwd(gb: GraphBuilder, dy, x, g, rstd, x_hat):
    """Compute dx and dg given upstream dy and saved rstd, x_hat.

    Closed-form for RMSNorm (no recentering): with N = D,
      dx_hat = dy * g
      dx = rstd * (dx_hat - x_hat * mean(dx_hat * x_hat, dim=-1, keepdim=True))
      dg = sum_{B,T} (dy * x_hat) reduced over (B, T)
    """
    g3 = gb.reshape(g, shape=[1, 1, D])
    g_btd = gb.broadcast(g3, shape=[B, T, D])
    dxh = gb.mul(dy, g_btd)
    dotmean = gb.mean(gb.mul(dxh, x_hat), dim=-1, keepdim=True)
    inner = gb.sub(dxh, gb.mul(x_hat, dotmean))
    dx = gb.mul(rstd, inner)
    # dg: sum over B, T of dy * x_hat
    dg_per = gb.mul(dy, x_hat)
    dg_sum = gb.sum(dg_per, dim=0, keepdim=False)   # (T, D)
    dg = gb.sum(dg_sum, dim=0, keepdim=False)       # (D,)
    return dx, dg


def _attn_fwd(gb: GraphBuilder, x, W_qkv, W_o, *, B_, T_, D_, H_):
    """Naive multi-head causal self-attention.

    Returns (y, saved) where saved has the tensors needed for backward.
      y: (B, T, D)
      saved.q, k, v: (B, H, T, dh)
      saved.attn:   (B, H, T, T) softmax weights
    """
    dh = D_ // H_
    scale = 1.0 / math.sqrt(dh)

    qkv = gb.matmul(x, W_qkv)                                # (B, T, 3D)
    qkv = gb.reshape(qkv, shape=[B_, T_, 3, H_, dh])
    qkv = gb.transpose(qkv, dims=[2, 0, 3, 1, 4])            # (3, B, H, T, dh)
    q = gb.emit("slice", [qkv], kwargs={"dim": 0, "start": 0, "end": 1})
    q = gb.reshape(q, shape=[B_, H_, T_, dh])
    k = gb.emit("slice", [qkv], kwargs={"dim": 0, "start": 1, "end": 2})
    k = gb.reshape(k, shape=[B_, H_, T_, dh])
    v = gb.emit("slice", [qkv], kwargs={"dim": 0, "start": 2, "end": 3})
    v = gb.reshape(v, shape=[B_, H_, T_, dh])

    # scores = q @ k^T  -> (B, H, T, T)
    k_t = gb.transpose(k, dims=[0, 1, 3, 2])
    scores = gb.matmul(q, k_t)
    scores = gb.mul(scores, gb.const(scale))
    # Causal mask: subtract a large value where i < j (upper triangle)
    big_neg = gb.full(shape=[T_, T_], value=-1e9, dtype=DTYPE)
    mask = gb.triu(big_neg, diagonal=1)
    mask4 = gb.broadcast(
        gb.reshape(mask, shape=[1, 1, T_, T_]),
        shape=[B_, H_, T_, T_],
    )
    scores = gb.add(scores, mask4)
    attn = gb.softmax(scores, dim=-1)
    # h = attn @ v  -> (B, H, T, dh)
    h = gb.matmul(attn, v)
    h = gb.transpose(h, dims=[0, 2, 1, 3])
    h = gb.reshape(h, shape=[B_, T_, D_])
    y = gb.matmul(h, W_o)
    saved = {"q": q, "k": k, "v": v, "attn": attn, "h": h, "scale": scale,
             "x": x, "W_qkv": W_qkv, "W_o": W_o}
    return y, saved


def _attn_bwd(gb: GraphBuilder, dy, saved, *, B_, T_, D_, H_):
    """Backward through naive multi-head causal self-attention.

    Returns (dx, dW_qkv, dW_o).
    """
    dh = D_ // H_
    q, k, v = saved["q"], saved["k"], saved["v"]
    attn = saved["attn"]
    h = saved["h"]
    x = saved["x"]
    W_qkv = saved["W_qkv"]
    W_o = saved["W_o"]
    scale = saved["scale"]

    # dh' (out of o) = dy @ W_o^T;  dW_o = h^T @ dy
    W_o_t = gb.transpose(W_o, dims=[1, 0])
    dh_flat = gb.matmul(dy, W_o_t)                       # (B, T, D)
    dW_o = gb.einsum(h, dy, equation="btd,bte->de")      # (D, D)

    # Reshape dh_flat back to (B, H, T, dh)
    dh_4 = gb.reshape(dh_flat, shape=[B_, T_, H_, dh])
    dh_4 = gb.transpose(dh_4, dims=[0, 2, 1, 3])         # (B, H, T, dh)

    # h = attn @ v
    # dv = attn^T @ dh_4
    attn_t = gb.transpose(attn, dims=[0, 1, 3, 2])
    dv = gb.matmul(attn_t, dh_4)                          # (B, H, T, dh)
    # dattn = dh_4 @ v^T
    v_t = gb.transpose(v, dims=[0, 1, 3, 2])
    dattn = gb.matmul(dh_4, v_t)                          # (B, H, T, T)

    # softmax bwd: dscores = attn * (dattn - sum(dattn * attn, dim=-1, keep))
    da_a = gb.mul(dattn, attn)
    s = gb.sum(da_a, dim=-1, keepdim=True)
    dscores = gb.mul(attn, gb.sub(dattn, s))             # (B, H, T, T)
    dscores = gb.mul(dscores, gb.const(scale))

    # scores = q @ k^T
    # dq = dscores @ k
    dq = gb.matmul(dscores, k)                            # (B, H, T, dh)
    # dk = dscores^T @ q
    dscores_t = gb.transpose(dscores, dims=[0, 1, 3, 2])
    dk = gb.matmul(dscores_t, q)                          # (B, H, T, dh)

    # Stack dq, dk, dv back into a (3, B, H, T, dh) tensor, then reshape
    # to (B, T, 3D) and combine with x to compute dW_qkv and dx.
    # We do it via concat (since stack ops only concat along an existing or
    # new axis).
    dqkv = gb.stack([dq, dk, dv], dim=0)                  # (3, B, H, T, dh)
    dqkv = gb.transpose(dqkv, dims=[1, 3, 0, 2, 4])       # (B, T, 3, H, dh)
    dqkv_flat = gb.reshape(dqkv, shape=[B_, T_, 3 * D_])

    # dW_qkv = x^T @ dqkv summed over batch/time
    dW_qkv = gb.einsum(x, dqkv_flat, equation="btd,bte->de")   # (D, 3D)
    # dx_attn = dqkv_flat @ W_qkv^T
    W_qkv_t = gb.transpose(W_qkv, dims=[1, 0])
    dx = gb.matmul(dqkv_flat, W_qkv_t)                    # (B, T, D)
    return dx, dW_qkv, dW_o


def _ffn_fwd(gb: GraphBuilder, x, W1, W2, W3):
    """SwiGLU FFN: out = (silu(x W1) * (x W2)) W3."""
    h1 = gb.matmul(x, W1)         # (B, T, F)
    h2 = gb.matmul(x, W2)         # (B, T, F)
    s = gb.silu(h1)               # (B, T, F)
    g = gb.mul(s, h2)             # (B, T, F)
    out = gb.matmul(g, W3)        # (B, T, D)
    saved = {"x": x, "h1": h1, "h2": h2, "s": s, "g": g,
             "W1": W1, "W2": W2, "W3": W3}
    return out, saved


def _ffn_bwd(gb: GraphBuilder, dy, saved):
    """Backward through SwiGLU FFN.

    silu(z) = z * sigmoid(z); dsilu/dz = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    """
    x = saved["x"]; h1 = saved["h1"]; h2 = saved["h2"]
    s = saved["s"]; g = saved["g"]
    W1 = saved["W1"]; W2 = saved["W2"]; W3 = saved["W3"]

    # dg = dy @ W3^T;  dW3 = g^T @ dy
    W3_t = gb.transpose(W3, dims=[1, 0])
    dg = gb.matmul(dy, W3_t)                          # (B, T, F)
    dW3 = gb.einsum(g, dy, equation="btf,btd->fd")    # (F, D)

    # g = s * h2 -> ds = dg * h2; dh2 = dg * s
    ds = gb.mul(dg, h2)
    dh2 = gb.mul(dg, s)

    # silu derivative: z = h1, sig = sigmoid(z); dsilu = sig * (1 + z*(1-sig))
    sig = gb.sigmoid(h1)
    one = gb.const(1.0)
    one_minus_sig = gb.sub(gb.broadcast(gb.reshape(one, shape=[1, 1, 1]),
                                        shape=[B, T, D_FF]), sig)
    inner = gb.add(
        gb.broadcast(gb.reshape(one, shape=[1, 1, 1]), shape=[B, T, D_FF]),
        gb.mul(h1, one_minus_sig),
    )
    dsilu = gb.mul(sig, inner)
    dh1 = gb.mul(ds, dsilu)                           # (B, T, F)

    # dW1 = x^T @ dh1; dW2 = x^T @ dh2
    dW1 = gb.einsum(x, dh1, equation="btd,btf->df")
    dW2 = gb.einsum(x, dh2, equation="btd,btf->df")

    # dx_attn = dh1 @ W1^T + dh2 @ W2^T
    W1_t = gb.transpose(W1, dims=[1, 0])
    W2_t = gb.transpose(W2, dims=[1, 0])
    dx = gb.add(gb.matmul(dh1, W1_t), gb.matmul(dh2, W2_t))
    return dx, dW1, dW2, dW3


# --------------------------------------------------------------------------- #
# Forward graph per stage
# --------------------------------------------------------------------------- #


def build_fwd_graph(*, stage: int, n_stages: int, is_tail: bool,
                     bucket: LocalBucket | None = None,
                     run_id: str | None = None) -> Graph:
    gb = GraphBuilder()
    gb.param("mb_seed", "int")
    blob = gb.input("W", [_stage_total_numel(stage, n_stages)], DTYPE)
    P = _split_blob_in_ir(gb, blob, stage, n_stages)

    use_subspace = SUBSPACE_K is not None and bucket is not None and run_id is not None
    u_k_ref = _bind_uk(gb, bucket, run_id) if use_subspace else None
    use_tied = TIED_EMBED and bucket is not None and run_id is not None
    tok_emb_ref_static = _bind_tok_emb(gb, bucket, run_id) if use_tied else None

    if stage == 0:
        # Head stage: read tokens corpus + indices, embed.
        tokens_in = gb.input("tokens", [-1], "int32")
        ids, target_ids = gb.data_indexer(tokens_in, B=B, T=T, mb_seed=ref_param("mb_seed"))
        # Use the shared static tok_emb if tied, else the per-stage W blob slice.
        tok_emb_for_embed = tok_emb_ref_static if use_tied else P["tok_emb"]
        x = gb.embedding(tok_emb_for_embed, ids)        # (B, T, D)
        # Add positional embedding (T, D) broadcast over B.
        pos_btd = gb.broadcast(gb.reshape(P["pos_emb"], shape=[1, T, D]),
                               shape=[B, T, D])
        x = gb.add(x, pos_btd)
    else:
        if use_subspace:
            if WIRE_INT8:
                # Wire is a 1-D uint8 blob; unpack + reshape back to (B, T, k)
                packed_size = B * T * SUBSPACE_K + SUBSPACE_K * 4
                x_packed = gb.input("x", [packed_size], "uint8")
                x_proj = gb.unpack_dequantize_int8(
                    x_packed, shape=[B, T, SUBSPACE_K], dim=-1,
                )
            else:
                x_proj = gb.input("x", [B, T, SUBSPACE_K], DTYPE)
            x = _unproject_in_ir(gb, x_proj, u_k_ref)   # (B, T, D)
        else:
            x = gb.input("x", [B, T, D], DTYPE)

    # All N_BLOCKS_PER_STAGE blocks at this stage, fused (no wire crossing
    # between blocks within the same stage).
    x_block_out, _ = _block_fwd_full(gb, x, P)

    # Track whether we already declared the "tokens" input — for N_STAGES=1
    # (head and tail are the same stage), the head branch already declared
    # it and we need to re-use the same input ref in the tail branch instead
    # of declaring a duplicate.
    head_tokens_ref = tokens_in if stage == 0 else None
    head_target_ids = target_ids if stage == 0 else None

    # Always emit x (the orchestrator's forward_output_specs requires it).
    # For the tail stage we also emit `loss`.
    if use_subspace and not is_tail:
        # Project to k-dim before sending downstream. Tail doesn't need x for
        # downstream (it's the last stage), so we keep the tail's x at full D.
        x_wire = _project_in_ir(gb, x_block_out, u_k_ref)
        if WIRE_INT8:
            x_wire = gb.quantize_pack_int8(x_wire, dim=-1)
        gb.output("x", x_wire)
    else:
        gb.output("x", x_block_out)
    if is_tail:
        # Final norm + lm_head + cross_entropy.
        # When tied, lm_head is tok_emb.T (shape (D, V)), shared with head.
        nf, _, _ = _rmsnorm_fwd(gb, x_block_out, P["g_norm_f"])
        if use_tied:
            lm_head = gb.transpose(tok_emb_ref_static, dims=[1, 0])  # (D, V)
        else:
            lm_head = P["W_lm"]
        logits = gb.matmul(nf, lm_head)               # (B, T, V)
        # Re-derive target ids from mb_seed. If we're also the head stage,
        # the tokens input + target_ids were already declared above; reuse.
        if head_target_ids is not None:
            target_ids2 = head_target_ids
        else:
            tokens_in = gb.input("tokens", [-1], "int32")
            _, target_ids2 = gb.data_indexer(
                tokens_in, B=B, T=T, mb_seed=ref_param("mb_seed"),
            )
        logits_flat = gb.reshape(logits, shape=[B * T, VOCAB])
        targets_flat = gb.reshape(target_ids2, shape=[B * T])
        loss = gb.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        gb.output("loss", gb.unsqueeze(loss, dim=0))   # (1,) scalar
    return gb.build()


# --------------------------------------------------------------------------- #
# Backward graph per stage
#
# For Phase 1, we re-run the forward locally at each backward stage to
# recover the saved activations (no upstream "save and ship" of saved
# tensors). This costs ~2x compute on the backward pass but avoids
# inflating the wire payload — Pluralis-style activation compression in
# Phase 2 will further amortize.
# --------------------------------------------------------------------------- #


def _pack_grad_blob(gb: GraphBuilder, parts: dict, stage: int, n_stages: int):
    """Concatenate per-parameter gradients into a single 1-D blob in the
    SAME order as `_stage_param_sizes(stage, n_stages)`. Missing parts are
    replaced with zeros (e.g. tok_emb gradient on Phase-1 head where we
    haven't implemented scatter_add yet).
    """
    pieces = []
    for name, shape in _stage_param_sizes(stage, n_stages):
        n = math.prod(shape)
        g = parts.get(name)
        if g is None:
            g = gb.full(shape=[n], value=0.0, dtype=DTYPE)
        else:
            g = gb.reshape(g, shape=[n])
        pieces.append(g)
    if len(pieces) == 1:
        return pieces[0]
    return gb.concat(pieces, dim=0)


# --------------------------------------------------------------------------- #
# Block fwd/bwd as helpers callable in the K-inner LocoProp loop.
# --------------------------------------------------------------------------- #


def _single_block_fwd(gb: GraphBuilder, x_in, P, *, suffix: str = ""):
    """Forward through ONE transformer block (RMSNorm + Attn + RMSNorm + FFN
    + residuals). P is the per-stage param dict; suffix is "_b{i}" so the
    helper finds the right block's params (g_norm1_b0 vs g_norm1_b1 etc).
    Returns (x_block_out, saved-tensors-dict)."""
    s = suffix  # short alias
    n1, rstd1, xhat1 = _rmsnorm_fwd(gb, x_in, P[f"g_norm1{s}"])
    attn_out, attn_saved = _attn_fwd(gb, n1, P[f"W_qkv{s}"], P[f"W_o{s}"],
                                      B_=B, T_=T, D_=D, H_=N_HEAD)
    x_post_attn = gb.add(x_in, attn_out)
    n2, rstd2, xhat2 = _rmsnorm_fwd(gb, x_post_attn, P[f"g_norm2{s}"])
    ffn_out, ffn_saved = _ffn_fwd(gb, n2, P[f"W_ffn1{s}"], P[f"W_ffn2{s}"],
                                    P[f"W_ffn3{s}"])
    x_out = gb.add(x_post_attn, ffn_out)
    saved = {
        "suffix": s,
        "x_in": x_in, "x_post_attn": x_post_attn,
        "rstd1": rstd1, "xhat1": xhat1,
        "rstd2": rstd2, "xhat2": xhat2,
        "attn_saved": attn_saved, "ffn_saved": ffn_saved,
    }
    return x_out, saved


def _single_block_bwd(gb: GraphBuilder, dy, saved, P):
    """Backward through ONE transformer block."""
    s = saved["suffix"]
    dffn = dy
    dx_post_attn = dy
    dn2, dW_ffn1, dW_ffn2, dW_ffn3 = _ffn_bwd(gb, dffn, saved["ffn_saved"])
    dx_post_attn_norm, dg_norm2 = _rmsnorm_bwd(
        gb, dn2, saved["x_post_attn"],
        P[f"g_norm2{s}"], saved["rstd2"], saved["xhat2"],
    )
    dx_post_attn_total = gb.add(dx_post_attn, dx_post_attn_norm)

    dattn = dx_post_attn_total
    dx_residual = dx_post_attn_total
    dn1, dW_qkv, dW_o = _attn_bwd(gb, dattn, saved["attn_saved"],
                                    B_=B, T_=T, D_=D, H_=N_HEAD)
    dx_in_norm, dg_norm1 = _rmsnorm_bwd(
        gb, dn1, saved["x_in"],
        P[f"g_norm1{s}"], saved["rstd1"], saved["xhat1"],
    )
    dx_in_total = gb.add(dx_residual, dx_in_norm)

    dgrads = {
        f"g_norm1{s}": dg_norm1, f"W_qkv{s}": dW_qkv, f"W_o{s}": dW_o,
        f"g_norm2{s}": dg_norm2, f"W_ffn1{s}": dW_ffn1,
        f"W_ffn2{s}": dW_ffn2, f"W_ffn3{s}": dW_ffn3,
    }
    return dgrads, dx_in_total


def _block_fwd_full(gb: GraphBuilder, x_in, P_block: dict):
    """Forward through ALL N_BLOCKS_PER_STAGE blocks at this stage. Returns
    (x_stage_out, list_of_saved). P_block must contain block params with
    "_b0", "_b1", ... suffixes."""
    saved_list = []
    x = x_in
    for b in range(N_BLOCKS_PER_STAGE):
        x, saved = _single_block_fwd(gb, x, P_block, suffix=f"_b{b}")
        saved_list.append(saved)
    return x, saved_list


def _block_bwd_full(gb: GraphBuilder, dy_block_out, saved_list, P_block: dict):
    """Backward through ALL N_BLOCKS_PER_STAGE blocks (reversed order).
    Returns (dgrads_dict_with_all_block_params, dL_dx_into_first_block).
    """
    dgrads_all: dict = {}
    dy = dy_block_out
    for b in range(N_BLOCKS_PER_STAGE - 1, -1, -1):
        dgrads_b, dy = _single_block_bwd(gb, dy, saved_list[b], P_block)
        dgrads_all.update(dgrads_b)
    return dgrads_all, dy


def _apply_local_sgd(gb: GraphBuilder, P_block: dict, dgrads: dict, eta: float) -> dict:
    """Return new per-param refs with W -= eta * dW for each block param
    (across all N_BLOCKS_PER_STAGE blocks)."""
    eta_c = gb.const(float(eta))
    out = dict(P_block)
    for name in dgrads:
        out[name] = gb.sub(P_block[name], gb.mul(eta_c, dgrads[name]))
    return out


def _eta_inner_effective() -> float:
    """If LOCO_ETA_INNER is 0 (auto), default to LR — each inner step is a
    full SGD step on the local quadratic (validated empirically: gives
    K=8 ~0.08 nats better loss/outer-step than K=1 at the same per-mb wire
    cost). Override via env / config when needed."""
    if LOCO_ETA_INNER and LOCO_ETA_INNER > 0:
        return LOCO_ETA_INNER
    return float(LR)


def build_bwd_graph(*, stage: int, n_stages: int, is_tail: bool, is_head: bool,
                     bucket: LocalBucket | None = None,
                     run_id: str | None = None) -> Graph:
    """Backward stage K. Emits a packed 1-D `dW` blob (same numel as W) and,
    if not the head, the upstream gradient `dL_dx_in`.

    We re-run the forward inside the bwd graph so the worker doesn't need
    to receive saved activations. This costs ~2x compute but keeps the
    wire payload small.
    """
    gb = GraphBuilder()
    gb.param("mb_seed", "int")
    blob = gb.input("W", [_stage_total_numel(stage, n_stages)], DTYPE)
    P = _split_blob_in_ir(gb, blob, stage, n_stages)

    use_subspace = SUBSPACE_K is not None and bucket is not None and run_id is not None
    u_k_ref = _bind_uk(gb, bucket, run_id) if use_subspace else None
    # Asymmetric: backward grad lane gets its own (smaller) U_k_dy.
    u_k_dy_ref = _bind_uk(gb, bucket, run_id, dy=True) if use_subspace else None
    k_dy = (SUBSPACE_K_DY if SUBSPACE_K_DY is not None else SUBSPACE_K) if use_subspace else None
    use_tied = TIED_EMBED and bucket is not None and run_id is not None
    tok_emb_ref_static = _bind_tok_emb(gb, bucket, run_id) if use_tied else None

    head_tokens_ref = None
    head_target_ids = None
    if is_head:
        tokens_in = gb.input("tokens", [-1], "int32")
        ids, target_ids = gb.data_indexer(tokens_in, B=B, T=T, mb_seed=ref_param("mb_seed"))
        head_tokens_ref = tokens_in
        head_target_ids = target_ids
        tok_emb_for_embed = tok_emb_ref_static if use_tied else P["tok_emb"]
        x_in = gb.embedding(tok_emb_for_embed, ids)
        pos_btd = gb.broadcast(gb.reshape(P["pos_emb"], shape=[1, T, D]),
                               shape=[B, T, D])
        x_in = gb.add(x_in, pos_btd)
    else:
        if use_subspace:
            # Forward (x_in) stays projected with the FULL k_x (=SUBSPACE_K).
            if WIRE_INT8:
                packed_size = B * T * SUBSPACE_K + SUBSPACE_K * 4
                x_in_packed = gb.input("x_in", [packed_size], "uint8")
                x_in_proj = gb.unpack_dequantize_int8(
                    x_in_packed, shape=[B, T, SUBSPACE_K], dim=-1,
                )
            else:
                x_in_proj = gb.input("x_in", [B, T, SUBSPACE_K], DTYPE)
            x_in = _unproject_in_ir(gb, x_in_proj, u_k_ref)
        else:
            x_in = gb.input("x_in", [B, T, D], DTYPE)

    # Forward through all stage's blocks (fused).
    x_block_out, saved_blocks = _block_fwd_full(gb, x_in, P)

    grads: dict = {}

    if is_tail:
        nf, rstd_f, xhat_f = _rmsnorm_fwd(gb, x_block_out, P["g_norm_f"])
        # Tied: lm_head = tok_emb.T (D, V); else use stage's W_lm.
        if use_tied:
            lm_head = gb.transpose(tok_emb_ref_static, dims=[1, 0])
        else:
            lm_head = P["W_lm"]
        logits = gb.matmul(nf, lm_head)
        # Reuse the head's tokens input + target_ids if we are also the head.
        if head_target_ids is not None:
            target_ids_t = head_target_ids
        else:
            tokens_in_t = gb.input("tokens", [-1], "int32")
            _, target_ids_t = gb.data_indexer(
                tokens_in_t, B=B, T=T, mb_seed=ref_param("mb_seed"),
            )
        probs = gb.softmax(logits, dim=-1)
        zero_logits = gb.full(shape=[B, T, VOCAB], value=0.0, dtype=DTYPE)
        idx = gb.unsqueeze(target_ids_t, dim=-1)
        ones = gb.full(shape=[B, T, 1], value=1.0, dtype=DTYPE)
        one_hot = gb.scatter(zero_logits, idx, ones, dim=-1)
        n_bt = float(B * T)
        dlogits = gb.mul(gb.sub(probs, one_hot), gb.const(1.0 / n_bt))
        # dW_lm = nf^T @ dlogits  (only emitted when NOT tied — when tied,
        # the lm_head IS tok_emb and we freeze it).
        if not use_tied:
            grads["W_lm"] = gb.einsum(nf, dlogits, equation="btd,btv->dv")
        # Backward through lm_head: dnf = dlogits @ lm_head.T
        lm_head_t = gb.transpose(lm_head, dims=[1, 0])  # (V, D) when tied
        dnf = gb.matmul(dlogits, lm_head_t)
        dx_block_out, dg_norm_f = _rmsnorm_bwd(gb, dnf, x_block_out,
                                                P["g_norm_f"], rstd_f, xhat_f)
        grads["g_norm_f"] = dg_norm_f
    else:
        if use_subspace:
            # Receive dL_dx in projected (k_dy-dim) form from downstream.
            if WIRE_INT8:
                packed_size_dy = B * T * k_dy + k_dy * 4
                dL_dx_packed = gb.input("dL_dx_out", [packed_size_dy], "uint8")
                dL_dx_proj = gb.unpack_dequantize_int8(
                    dL_dx_packed, shape=[B, T, k_dy], dim=-1,
                )
            else:
                dL_dx_proj = gb.input("dL_dx_out", [B, T, k_dy], DTYPE)
            dx_block_out = _unproject_in_ir(gb, dL_dx_proj, u_k_dy_ref)
        else:
            dx_block_out = gb.input("dL_dx_out", [B, T, D], DTYPE)

    block_names = _block_names_for_stage()
    if K_INNER > 1:
        # K-inner LocoProp: K local SGD steps minimizing the per-stage local
        # quadratic ||block(x_in; W) - target||^2 where
        # target = x_block_out_initial - LOCO_ETA_OUT * dx_block_out.
        #
        # Effect: K updates worth of progress per S3 round-trip. Wire stays
        # the same per microbatch; compute multiplies by K.
        P_init_block = {n: P[n] for n in block_names}
        P_current = {n: P[n] for n in block_names}
        eta_out_c = gb.const(float(LOCO_ETA_OUT))
        target = gb.sub(x_block_out, gb.mul(eta_out_c, dx_block_out))
        # NOTE: NO (2/N) factor on the local grad — the local "loss" is
        # interpreted as ||x_pred - target||^2 (sum, not mean) so that
        # K=1 LocoProp with LOCO_ETA_OUT=1 reproduces the plain-SGD grad
        # direction exactly. The magnitude is then controlled by the
        # combination of LOCO_ETA_OUT and eta_inner.
        eta_inner = _eta_inner_effective()
        last_dx_in = None
        for _k in range(K_INNER):
            x_pred_k, saved_k = _block_fwd_full(gb, x_in, P_current)
            g_local = gb.sub(x_pred_k, target)
            dgrads_k, dx_in_k = _block_bwd_full(gb, g_local, saved_k, P_current)
            P_current = _apply_local_sgd(gb, P_current, dgrads_k, eta_inner)
            last_dx_in = dx_in_k
        # Convert (W_init - W_current) -> dW_for_outer such that
        # outer's `W_new = W - LR * mean(dW)` reproduces averaging the
        # local W trajectories. So dW = (W_init - W_current) / LR.
        inv_lr_c = gb.const(1.0 / float(LR))
        for name in block_names:
            delta = gb.sub(P_init_block[name], P_current[name])
            grads[name] = gb.mul(delta, inv_lr_c)
        dx_in_total = last_dx_in
    else:
        # K_INNER == 1: regular block bwd through all N_BLOCKS_PER_STAGE blocks.
        dgrads_blocks, dx_in_total = _block_bwd_full(gb, dx_block_out,
                                                       saved_blocks, P)
        for name, val in dgrads_blocks.items():
            grads[name] = val

    if is_head:
        # Pos emb bwd is exact: dpos_emb[t, :] = sum_b dx_in[b, t, :]
        grads["pos_emb"] = gb.sum(dx_in_total, dim=0, keepdim=False)
        # tok_emb bwd needs scatter_add (not yet in IR). For Phase 1 we
        # leave tok_emb gradient as zero; this freezes the embedding
        # initialization. We'll add scatter_add in Phase 2 polish.

    dW_blob = _pack_grad_blob(gb, grads, stage, n_stages)
    gb.output("dW", dW_blob)

    if not is_head:
        if use_subspace:
            # Project dL_dx_in to k_dy-dim using U_k_dy, then send upstream.
            dx_in_proj = _project_in_ir(gb, dx_in_total, u_k_dy_ref)
            if WIRE_INT8:
                dx_in_proj = gb.quantize_pack_int8(dx_in_proj, dim=-1)
            gb.output("dL_dx_in", dx_in_proj)
        else:
            gb.output("dL_dx_in", dx_in_total)
    return gb.build()


# --------------------------------------------------------------------------- #
# Outer (per-stage SGD update of the packed W blob)
# --------------------------------------------------------------------------- #


def build_outer_graph(*, stage: int, n_stages: int, n_microbatches: int) -> Graph:
    """Outer step for stage K. Reads:
      - W (packed blob)
      - dW_<m> for m in 0..M-1 — each is itself a packed blob in the
        same layout as the param blob's *trainable* slots, but only
        the fields the bwd graph emits as outputs.

    For Phase 1 we expect the worker to PACK the dW outputs into the same
    blob layout (this is done in the orch's PipelineStage.outer_graph
    plumbing), giving us a uniform "W += -lr * mean(dW_blob)" update.

    To avoid changing the streaming protocol, we instead build the outer
    graph to take ALL named per-parameter gradients and produce a new
    packed W. The orchestrator will need to know which gradients are
    emitted by the bwd graph.
    """
    gb = GraphBuilder()
    n = _stage_total_numel(stage, n_stages)
    blob = gb.input("W", [n], DTYPE)
    # Also accept M packed gradient blobs and avg-update.
    dblobs = []
    for m in range(n_microbatches):
        dblobs.append(gb.input(f"dW_{m}", [n], DTYPE))
    if n_microbatches == 1:
        dW_mean = dblobs[0]
    else:
        stacked = gb.stack(dblobs, dim=0)            # (M, n)
        dW_mean = gb.mean(stacked, dim=0, keepdim=False)
    lr_c = gb.const(float(LR))
    new_w = gb.sub(blob, gb.mul(lr_c, dW_mean))
    gb.output("W_new", new_w)
    return gb.build()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _initial_blob(stage: int, n_stages: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(WEIGHTS_SEED + stage * 7919)
    chunks: list[torch.Tensor] = []
    for name, shape in _stage_param_sizes(stage, n_stages):
        if name.startswith("g_"):
            chunks.append(torch.ones(shape).reshape(-1))
        elif name in ("tok_emb", "pos_emb"):
            t = torch.empty(shape).normal_(generator=g) * INIT_SCALE
            chunks.append(t.reshape(-1))
        else:
            t = torch.empty(shape).normal_(generator=g) * INIT_SCALE
            chunks.append(t.reshape(-1))
    return torch.cat(chunks).contiguous()


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    import os as _os

    # Optional: point at a pre-tokenized corpus at e.g.
    # s3://<bucket>/runs/<other>/data/tokens.bin
    tokens_uri_override = _os.environ.get("GPT_PIPE_TOKENS_URI", "")

    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    # SUBSPACE_K and SUBSPACE_K_DY can be overridden at bootstrap time
    # via env (so a single python process can launch runs with different k).
    sk_env = _os.environ.get("GPT_PIPE_SUBSPACE_K")
    if sk_env:
        global SUBSPACE_K
        SUBSPACE_K = int(sk_env) if sk_env != "0" else None
    sk_dy_env = _os.environ.get("GPT_PIPE_SUBSPACE_K_DY")
    if sk_dy_env:
        global SUBSPACE_K_DY
        SUBSPACE_K_DY = int(sk_dy_env) if sk_dy_env != "0" else None
    int8_env = _os.environ.get("GPT_PIPE_WIRE_INT8")
    if int8_env:
        global WIRE_INT8
        WIRE_INT8 = int8_env not in ("", "0", "false", "False")
    tied_env = _os.environ.get("GPT_PIPE_TIED_EMBED")
    if tied_env:
        global TIED_EMBED
        TIED_EMBED = tied_env not in ("", "0", "false", "False")
    k_inner_env = _os.environ.get("GPT_PIPE_K_INNER")
    if k_inner_env:
        global K_INNER
        K_INNER = max(int(k_inner_env), 1)
    eta_inner_env = _os.environ.get("GPT_PIPE_LOCO_ETA_INNER")
    if eta_inner_env:
        global LOCO_ETA_INNER
        LOCO_ETA_INNER = float(eta_inner_env)
    eta_out_env = _os.environ.get("GPT_PIPE_LOCO_ETA_OUT")
    if eta_out_env:
        global LOCO_ETA_OUT
        LOCO_ETA_OUT = float(eta_out_env)
    n_blocks_env = _os.environ.get("GPT_PIPE_N_BLOCKS_PER_STAGE")
    if n_blocks_env:
        global N_BLOCKS_PER_STAGE
        N_BLOCKS_PER_STAGE = max(int(n_blocks_env), 1)

    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "gpt_pipe", "n_stages": N_STAGES, "d": D,
         "vocab": VOCAB, "n_head": N_HEAD, "d_ff": D_FF,
         "B": B, "T": T,
         "n_microbatches": N_MICROBATCHES,
         "max_epochs": int(max_rounds), "lr": LR,
         "tokens_uri": tokens_uri_override,
         "subspace_k": SUBSPACE_K,
         "subspace_k_dy": SUBSPACE_K_DY,
         "wire_int8": WIRE_INT8,
         "tied_embed": TIED_EMBED,
         "k_inner": K_INNER,
         "loco_eta_inner": LOCO_ETA_INNER,
         "loco_eta_out": LOCO_ETA_OUT,
         "n_blocks_per_stage": N_BLOCKS_PER_STAGE},
    )

    # If subspace projection is enabled, write the static U_k (forward) and
    # U_k_dy (backward) blobs. U_k_dy uses a different seed so it spans a
    # different subspace from U_k.
    if SUBSPACE_K is not None and SUBSPACE_K > 0:
        u_k = _make_uk(D, SUBSPACE_K, seed=SUBSPACE_SEED)
        u_k_uri = bucket.uri_for_key(f"runs/{run_id}/static/u_k.bin")
        if not bucket.exists(u_k_uri):
            bucket.put(u_k_uri, tensor_io.encode_tensor(u_k))
        k_dy = SUBSPACE_K_DY if SUBSPACE_K_DY is not None else SUBSPACE_K
        u_k_dy = _make_uk(D, k_dy, seed=SUBSPACE_DY_SEED)
        u_k_dy_uri = bucket.uri_for_key(f"runs/{run_id}/static/u_k_dy.bin")
        if not bucket.exists(u_k_dy_uri):
            bucket.put(u_k_dy_uri, tensor_io.encode_tensor(u_k_dy))
    # If tied, write tok_emb ONCE as a shared static blob first; the
    # per-stage W blobs for stage 0 (head) and stage N-1 (tail) then
    # exclude tok_emb / W_lm respectively.
    if TIED_EMBED:
        g_te = torch.Generator(device="cpu").manual_seed(WEIGHTS_SEED)
        tok_emb = torch.empty(VOCAB, D).normal_(generator=g_te) * INIT_SCALE
        te_uri = bucket.uri_for_key(f"runs/{run_id}/static/tok_emb.bin")
        if not bucket.exists(te_uri):
            bucket.put(te_uri, tensor_io.encode_tensor(tok_emb.contiguous()))

    for s in range(N_STAGES):
        blob = _initial_blob(s, N_STAGES)
        uri = bucket.uri_for_key(
            f"runs/{run_id}/weights/epoch=0/stage_{s}_W.bin"
        )
        bucket.put(uri, tensor_io.encode_tensor(blob))

    # Synthesize a dummy uniform-random "corpus" only if no override AND no
    # existing local tokens.bin. (When override is set, skip — workers will
    # read directly from the override URI.)
    tokens_uri = bucket.uri_for_key(f"runs/{run_id}/static/tokens.bin")
    if tokens_uri_override or bucket.exists(tokens_uri):
        return
    n = max(VOCAB * 100, 100_000)
    g = torch.Generator(device="cpu").manual_seed(123)
    toks = torch.randint(0, VOCAB, (n,), generator=g, dtype=torch.int32)
    bucket.put(tokens_uri, tensor_io.encode_tensor(toks))


def build_streaming_inputs(*, bucket: LocalBucket, run_id: str):
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    cfg = bucket.get_json(cfg_uri) if bucket.exists(cfg_uri) else {}
    max_epochs = int(cfg.get("max_epochs", MAX_EPOCHS))
    n_microbatches = int(cfg.get("n_microbatches", N_MICROBATCHES))
    # Read ALL model dims + flags from the manifest config so the orch
    # process matches what bootstrap wrote (they're separate Python procs
    # and module globals don't survive across process boundaries).
    global D, T, B, N_HEAD, D_FF, VOCAB, N_STAGES, LR
    global SUBSPACE_K, SUBSPACE_K_DY, WIRE_INT8, TIED_EMBED
    global K_INNER, LOCO_ETA_INNER, LOCO_ETA_OUT, N_BLOCKS_PER_STAGE
    if "d" in cfg: D = int(cfg["d"])
    if "T" in cfg: T = int(cfg["T"])
    if "B" in cfg: B = int(cfg["B"])
    if "n_head" in cfg: N_HEAD = int(cfg["n_head"])
    if "d_ff" in cfg: D_FF = int(cfg["d_ff"])
    if "vocab" in cfg: VOCAB = int(cfg["vocab"])
    if "n_stages" in cfg: N_STAGES = int(cfg["n_stages"])
    if "lr" in cfg: LR = float(cfg["lr"])
    cfg_subspace_k = cfg.get("subspace_k")
    if cfg_subspace_k is not None:
        SUBSPACE_K = int(cfg_subspace_k) if cfg_subspace_k else None
    cfg_subspace_k_dy = cfg.get("subspace_k_dy")
    if cfg_subspace_k_dy is not None:
        SUBSPACE_K_DY = int(cfg_subspace_k_dy) if cfg_subspace_k_dy else None
    cfg_wire_int8 = cfg.get("wire_int8")
    if cfg_wire_int8 is not None:
        WIRE_INT8 = bool(cfg_wire_int8)
    cfg_tied_embed = cfg.get("tied_embed")
    if cfg_tied_embed is not None:
        TIED_EMBED = bool(cfg_tied_embed)
    cfg_k_inner = cfg.get("k_inner")
    if cfg_k_inner is not None:
        K_INNER = max(int(cfg_k_inner), 1)
    cfg_eta_inner = cfg.get("loco_eta_inner")
    if cfg_eta_inner is not None:
        LOCO_ETA_INNER = float(cfg_eta_inner)
    cfg_eta_out = cfg.get("loco_eta_out")
    if cfg_eta_out is not None:
        LOCO_ETA_OUT = float(cfg_eta_out)
    cfg_n_blocks = cfg.get("n_blocks_per_stage")
    if cfg_n_blocks is not None:
        N_BLOCKS_PER_STAGE = max(int(cfg_n_blocks), 1)

    use_subspace = SUBSPACE_K is not None
    wire_dim = SUBSPACE_K if use_subspace else D
    wire_dtype = "uint8" if (use_subspace and WIRE_INT8) else DTYPE
    if use_subspace and WIRE_INT8:
        wire_shape_x = [B * T * SUBSPACE_K + SUBSPACE_K * 4]
    else:
        wire_shape_x = [B, T, wire_dim]

    stages: list[PipelineStage] = []
    for s in range(N_STAGES):
        is_tail = (s == N_STAGES - 1)
        is_head = (s == 0)
        fwd_g = build_fwd_graph(stage=s, n_stages=N_STAGES, is_tail=is_tail,
                                 bucket=bucket, run_id=run_id)
        bwd_g = build_bwd_graph(stage=s, n_stages=N_STAGES, is_tail=is_tail,
                                 is_head=is_head,
                                 bucket=bucket, run_id=run_id)
        outer_g = build_outer_graph(stage=s, n_stages=N_STAGES,
                                     n_microbatches=n_microbatches)
        in_specs: list[tuple[str, list[int], str]] = []
        if not is_head:
            in_specs.append(("x", wire_shape_x, wire_dtype))
        # Tail's forward x output is full-D (no downstream); other stages
        # emit on the wire (subspace + optional int8).
        if is_tail:
            out_specs = [("x", [B, T, D], DTYPE)]
        else:
            out_specs = [("x", wire_shape_x, wire_dtype)]
        stages.append(PipelineStage(
            stage_id=s,
            forward_graph=fwd_g,
            backward_graph=bwd_g,
            outer_graph=outer_g,
            forward_input_specs=in_specs,
            forward_output_specs=out_specs,
            weights_input_name="W",
            weights_shape=[_stage_total_numel(s, N_STAGES)],
            weights_dtype=DTYPE,
            backward_takes_loss_target=is_tail,
            backward_emits_dx_in=not is_head,
        ))

    tokens_uri = (
        cfg.get("tokens_uri")
        or TOKENS_URI
        or bucket.uri_for_key(f"runs/{run_id}/static/tokens.bin")
    )
    # Cost telemetry: gpu_class -> $/hour for the box. Pulled from the
    # manifest config (set by the bootstrap or pre-populated) plus a
    # built-in default for the Lium fleet we typically run.
    default_prices = {
        "H200": 31.92, "B200": 39.92,
        "RTX3090": 1.04, "RTX4090": 0.70,
        "L40": 1.5, "L40S": 2.0, "A6000": 1.0,
    }
    prices = dict(default_prices)
    prices.update(cfg.get("gpu_class_price_per_hour", {}) or {})

    params = StreamingParams(
        n_stages=N_STAGES,
        n_microbatches=n_microbatches,
        max_epochs=max_epochs,
        training=True,
        target_static_uri=None,   # CE loss is built into the tail stage graph
        lr=LR,
        static_inputs={
            "tokens": (tokens_uri, "head_and_tail"),
        },
        tokens_per_mb=B * T,
        gpu_class_price_per_hour=prices,
    )
    return stages, params
