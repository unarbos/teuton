"""Phase 1 — depth-parallel mini-GPT, ~2M trainable params.

Architecture (4 residual MLP blocks in a frozen embedding sandwich):

    input_ids   : (B, T) int64
    h_0  = embedding(emb_table, input_ids) + pe          # (B, T, D)
    h_1  = h_0 + silu(h_0 @ W_0_a) @ W_0_b               # block 0 (UB-0)
    h_2  = h_1 + silu(h_1 @ W_1_a) @ W_1_b               # block 1 (UB-1)
    h_3  = h_2 + silu(h_2 @ W_2_a) @ W_2_b               # block 2 (UB-2)
    h_4  = h_3 + silu(h_3 @ W_3_a) @ W_3_b               # block 3 (UB-3)
    logits = h_4 @ emb_table.T                            # (B, T, V)
    loss   = cross_entropy(logits, input_ids)            # predict-self

UBs: 4 (one per block). Each UB owns its (W_a, W_b) packed as a single tensor
of shape (2, D, D) so the existing weights_key infrastructure works.

Forward pass (on the master) computes the full forward + manual backward
through every block to produce per-cut (h_i, target_i) where
target_i = h_i - dL/dh_i. This is shipped to the workers for that UB.

Inner step (on each pinned worker) takes (x_in, target) and runs K-step SGD
on the local quadratic ½‖block_i(x_in) - target‖² where block_i means the
residual MLP block parameterized by the worker's owned weights.

Outer step is AdamW-style with persistent (m, v) state per UB.

This phase deliberately skips:
  - Subspace compression on the wire (Phase 3)
  - Weight tying (Phase 3)
  - int8 / DeMo (Phase 3)
  - Pipelining (Phase 2)
  - Multi-head attention (would need full backward through softmax etc.)

The architecture preserves the residual stream, which is the property that
makes per-block LocoProp targets meaningful (gradient flows through the
identity branch, so dL/dh_i is a well-defined, not-collapsed-to-zero signal).
"""
from __future__ import annotations

import math

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #

V = 2048                # vocab size
D = 256                 # residual stream dim
T = 64                  # sequence length
B_TRAIN = 16
B_EVAL = 64
N_BLOCKS = 4
N_UB = N_BLOCKS         # 4 UBs, one per block
# Step 2: with multi-microbatch, each UB only needs 1 replica (microbatches
# provide diversity instead of replicas). Total inner jobs per round =
# n_microbatches * N_UB * INNER_REPLICAS = 8*4*1 = 32, well-distributed
# across 23 workers.
INNER_REPLICAS = 1
N_MICROBATCHES = 8
K_INNER = 8
INNER_LR = 0.001
ADAM_LR = 0.005
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
EMB_SEED = 7777
PE_SEED = 5555
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Static tensors written by bootstrap
# --------------------------------------------------------------------------- #


def _make_emb_table() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(EMB_SEED)
    e = torch.empty(V, D).normal_(generator=g)
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _make_pe() -> torch.Tensor:
    pe = torch.zeros(T, D)
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe * 0.1


# --------------------------------------------------------------------------- #
# Per-block helpers (reused in forward / inner / eval graphs)
# --------------------------------------------------------------------------- #


def _block_forward(gb, x, w_a, w_b):
    """Residual MLP block: out = x + silu(x @ W_a) @ W_b.

    Returns (out, h_pre, h_post) where h_pre = x @ W_a (pre-activation) and
    h_post = silu(h_pre). Saving these lets the backward pass avoid recomputing.
    """
    h_pre = gb.matmul(x, w_a)
    h_post = gb.silu(h_pre)
    y = gb.matmul(h_post, w_b)
    out = gb.add(x, y)
    return out, h_pre, h_post


def _block_backward(gb, dL_dout, h_pre, h_post, x, w_a, w_b):
    """Backward through `out = x + silu(x @ W_a) @ W_b`.

    Given upstream dL/dout, returns dL/dx (input-side gradient).
    The residual identity branch contributes dL/dout directly.
    """
    # dL/dy = dL/dout (since out = x + y)
    dL_dy = dL_dout
    # dL/dW_b = h_post.T @ dL_dy   (not needed at this layer; computed by inner_step)
    # dL/dh_post = dL_dy @ W_b.T
    w_b_t = gb.transpose(w_b, dims=[1, 0])
    dL_dh_post = gb.matmul(dL_dy, w_b_t)
    # silu'(h_pre) = sigmoid(h_pre) * (1 + h_pre * (1 - sigmoid(h_pre)))
    sig = gb.sigmoid(h_pre)
    one = gb.const(1.0)
    one_minus_sig = gb.sub(one, sig)
    silu_grad = gb.mul(sig, gb.add(one, gb.mul(h_pre, one_minus_sig)))
    dL_dh_pre = gb.mul(dL_dh_post, silu_grad)
    # dL/dW_a = x.T @ dL/dh_pre  (used by inner_step internally; not returned)
    # dL/dx_branch = dL/dh_pre @ W_a.T
    w_a_t = gb.transpose(w_a, dims=[1, 0])
    dL_dx_branch = gb.matmul(dL_dh_pre, w_a_t)
    # Residual: dL/dx = dL/dx_branch + dL/dout
    dL_dx = gb.add(dL_dx_branch, dL_dout)
    return dL_dx


def _emb_blob(gb: GraphBuilder, bucket: str, run_id: str):
    return gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'emb_table')}",
        shape=[V, D], dtype=DTYPE,
    )


def _pe_blob(gb: GraphBuilder, bucket: str, run_id: str):
    return gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'pe')}",
        shape=[T, D], dtype=DTYPE,
    )


def _unpack_block(gb, packed):
    """packed: (2, D, D) -> (W_a, W_b) each (D, D)."""
    w_a = gb.squeeze(gb.slice(packed, dim=0, start=0, end=1), dim=0)
    w_b = gb.squeeze(gb.slice(packed, dim=0, start=1, end=2), dim=0)
    return w_a, w_b


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #


def build_forward_graph(*, bucket: str, run_id: str) -> Graph:
    """Full forward + manual backward through 4 residual MLP blocks.

    Emits per-cut packed tensor `target_i` = stack([h_i_input, target_i])
    of shape (2, B, T, D) where h_i_input is the input to block i and
    target_i = h_{i+1} - dL/dh_{i+1} is the LocoProp target at the cut.
    """
    gb = GraphBuilder()
    weights = [
        gb.input(f"weights_{i}", [2, D, D], DTYPE) for i in range(N_UB)
    ]
    gb.param("round_id", "int")

    emb_table = _emb_blob(gb, bucket, run_id)
    pe = _pe_blob(gb, bucket, run_id)

    # Random batch per round
    ids_f = gb.emit("uniform", args=[],
                    kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, T], "dtype": DTYPE})
    input_ids = gb.cast(gb.mul(ids_f, gb.const(float(V) - 1e-3)), dtype="int64")
    target_ids = input_ids

    emb = gb.embedding(emb_table, input_ids)        # (B, T, D)
    h_0 = gb.add(emb, pe)                            # (B, T, D)

    # Forward through all blocks; cache (h_pre, h_post, w_a, w_b) per block.
    block_inputs = [h_0]                             # h_0, h_1, h_2, h_3
    block_caches = []                                # (h_pre_i, h_post_i, w_a_i, w_b_i)
    h = h_0
    for i in range(N_BLOCKS):
        w_a, w_b = _unpack_block(gb, weights[i])
        h_next, h_pre, h_post = _block_forward(gb, h, w_a, w_b)
        block_caches.append((h_pre, h_post, w_a, w_b))
        if i < N_BLOCKS - 1:
            block_inputs.append(h_next)              # input to next block
        h = h_next
    h_final = h                                      # h_4

    # logits + loss-grad chain (tied lm_head)
    emb_t = gb.transpose(emb_table, dims=[1, 0])
    logits = gb.matmul(h_final, emb_t)               # (B, T, V)
    probs = gb.softmax(logits, dim=-1)

    arange_v = gb.arange(start=0, end=V, step=1, dtype="int64")
    arange_btv = gb.broadcast(gb.reshape(arange_v, shape=[1, 1, V]),
                              shape=[B_TRAIN, T, V])
    target_btv = gb.broadcast(gb.unsqueeze(target_ids, dim=-1),
                              shape=[B_TRAIN, T, V])
    one_hot = gb.cast(gb.eq(arange_btv, target_btv), dtype=DTYPE)
    inv_n = gb.const(1.0 / float(B_TRAIN * T))
    dL_dlogits = gb.mul(gb.sub(probs, one_hot), inv_n)   # (B, T, V)

    # dL/dh_final = dL/dlogits @ emb_table  (since logits = h_final @ emb_table.T)
    dL_dh = gb.matmul(dL_dlogits, emb_table)             # (B, T, D)

    # Backward through blocks in reverse to get dL/dh_i for every i.
    # We collect dL_dh for the OUTPUT of each block (= input of next block).
    dL_dh_outputs = [None] * N_BLOCKS    # dL/dh_{i+1}  (output of block i)
    dL_dh_outputs[N_BLOCKS - 1] = dL_dh  # output of last block = h_final
    cur = dL_dh
    for i in range(N_BLOCKS - 1, -1, -1):
        h_pre, h_post, w_a, w_b = block_caches[i]
        x_in = block_inputs[i]
        cur = _block_backward(gb, cur, h_pre, h_post, x_in, w_a, w_b)
        if i > 0:
            dL_dh_outputs[i - 1] = cur                  # dL/d(input of block i) = dL/d(output of block i-1)

    # Per-UB packed (h_input, target = h_output - dL/dh_output)
    for i in range(N_BLOCKS):
        h_in = block_inputs[i]
        # h_out = block forward output. Recompute? Already in block_caches — but
        # we don't directly cache h_out. h_out = h_in + h_post @ w_b. Let's
        # re-emit it (the IR will dedupe via subgraph identity? actually it
        # won't, but it's fine — small re-cost).
        h_pre_i, h_post_i, w_a_i, w_b_i = block_caches[i]
        y_branch = gb.matmul(h_post_i, w_b_i)
        h_out = gb.add(h_in, y_branch)
        target_i = gb.sub(h_out, dL_dh_outputs[i])
        pack = gb.stack([h_in, target_i], dim=0)         # (2, B, T, D)
        gb.output(f"target_{i}", pack)
    return gb.build()


def build_inner_graph() -> Graph:
    """K-step SGD on a single residual MLP block.

    Local objective: ½‖block(x_in) - target‖² over (B, T, D) where
    block(x) = x + silu(x @ W_a) @ W_b.

    The gradient for W_a and W_b is computed by the same backward function
    used in the forward pass — but here we differentiate through THIS block's
    weights only (not back into x_in).
    """
    gb = GraphBuilder()
    packed_w = gb.input("weights", [2, D, D], DTYPE)
    packed_target = gb.input("target", [2, B_TRAIN, T, D], DTYPE)

    w_a_init, w_b_init = _unpack_block(gb, packed_w)
    x_in = gb.squeeze(gb.slice(packed_target, dim=0, start=0, end=1), dim=0)
    target = gb.squeeze(gb.slice(packed_target, dim=0, start=1, end=2), dim=0)

    w_a = w_a_init
    w_b = w_b_init
    lr = gb.const(float(INNER_LR))
    inv_n = gb.const(1.0 / float(B_TRAIN * T))

    for _ in range(K_INNER):
        h_pre = gb.matmul(x_in, w_a)                 # (B, T, D)
        h_post = gb.silu(h_pre)
        y_branch = gb.matmul(h_post, w_b)
        out = gb.add(x_in, y_branch)
        # MSE loss = ½‖out - target‖²; dL/dout = (out - target) / N
        dL_dout = gb.mul(gb.sub(out, target), inv_n)
        # Gradients w.r.t. block weights
        dL_dy = dL_dout                              # (B, T, D)
        # dL/dW_b = sum_b sum_t h_post.T @ dL_dy ; via einsum
        dL_dW_b = gb.einsum(h_post, dL_dy, equation="btd,bte->de")
        # dL/dh_post = dL_dy @ W_b.T
        w_b_t = gb.transpose(w_b, dims=[1, 0])
        dL_dh_post = gb.matmul(dL_dy, w_b_t)
        sig = gb.sigmoid(h_pre)
        one = gb.const(1.0)
        silu_grad = gb.mul(sig, gb.add(one, gb.mul(h_pre, gb.sub(one, sig))))
        dL_dh_pre = gb.mul(dL_dh_post, silu_grad)
        dL_dW_a = gb.einsum(x_in, dL_dh_pre, equation="btd,bte->de")
        # SGD step
        w_a = gb.sub(w_a, gb.mul(lr, dL_dW_a))
        w_b = gb.sub(w_b, gb.mul(lr, dL_dW_b))

    delta = gb.sub(gb.stack([w_a, w_b], dim=0), packed_w)
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    """AdamW outer with persistent (m, v) state per UB."""
    gb = GraphBuilder()
    w = gb.input("weights", [2, D, D], DTYPE)
    rd = gb.input("reduced_delta", [2, D, D], DTYPE)
    m = gb.input("m", [2, D, D], DTYPE)
    v = gb.input("v", [2, D, D], DTYPE)
    gb.param("round_id", "int")

    # Treat -delta as the "gradient" direction (delta = -lr*g, so g ~ -delta).
    g = gb.neg(rd)

    b1 = gb.const(float(ADAM_BETA1))
    b2 = gb.const(float(ADAM_BETA2))
    one_b1 = gb.const(1.0 - float(ADAM_BETA1))
    one_b2 = gb.const(1.0 - float(ADAM_BETA2))
    new_m = gb.add(gb.mul(b1, m), gb.mul(one_b1, g))
    new_v = gb.add(gb.mul(b2, v), gb.mul(one_b2, gb.mul(g, g)))

    eps = gb.const(float(ADAM_EPS))
    lr = gb.const(float(ADAM_LR))
    update = gb.div(new_m, gb.add(gb.sqrt(new_v), eps))
    new_w = gb.sub(w, gb.mul(lr, update))

    gb.output("new_weights", new_w)
    gb.output("new_m", new_m)
    gb.output("new_v", new_v)
    return gb.build()


def build_eval_graph(*, bucket: str, run_id: str) -> Graph:
    gb = GraphBuilder()
    weights = [gb.input(f"weights_{i}", [2, D, D], DTYPE) for i in range(N_UB)]
    emb_table = _emb_blob(gb, bucket, run_id)
    pe = _pe_blob(gb, bucket, run_id)
    ids_f = gb.emit("uniform", args=[],
                    kwargs={"seed": EVAL_SEED, "shape": [B_EVAL, T], "dtype": DTYPE})
    input_ids = gb.cast(gb.mul(ids_f, gb.const(float(V) - 1e-3)), dtype="int64")
    h = gb.add(gb.embedding(emb_table, input_ids), pe)
    for i in range(N_BLOCKS):
        w_a, w_b = _unpack_block(gb, weights[i])
        h, _, _ = _block_forward(gb, h, w_a, w_b)
    emb_t = gb.transpose(emb_table, dims=[1, 0])
    logits = gb.matmul(h, emb_t)
    loss = gb.cross_entropy(logits, input_ids, ignore_index=-100)
    gb.output("metrics", gb.unsqueeze(loss, dim=0))
    return gb.build()


def build_reduce_graph(n_inputs: int) -> Graph:
    gb = GraphBuilder()
    refs = []
    for i in range(n_inputs):
        ri = gb.input(f"d_{i}", [2, D, D], DTYPE)
        refs.append(gb.unsqueeze(ri, dim=0))
    stack = refs[0] if n_inputs == 1 else gb.concat(refs, dim=0)
    avg = gb.mean(stack, dim=0)
    gb.output("reduced", avg)
    return gb.build()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def initial_weights() -> list[torch.Tensor]:
    """Per-UB packed weights (2, D, D) where [0] = W_a, [1] = W_b.

    Init uses small Gaussian (∼0.02 std) so that residual-stream magnitude
    is preserved at init (block(x) ≈ x for small W_a, W_b).
    """
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    out = []
    scale = (1.0 / D) ** 0.5
    for _ in range(N_UB):
        w_a = torch.empty(D, D).normal_(generator=g) * scale
        w_b = torch.empty(D, D).normal_(generator=g) * (scale * 0.1)
        out.append(torch.stack([w_a, w_b], dim=0))
    return out


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pluralis_gpt_10M", "n_unique_blocks": N_UB, "n_blocks": N_BLOCKS,
         "d": D, "V": V, "T": T, "B_train": B_TRAIN, "B_eval": B_EVAL,
         "K_inner": K_INNER, "inner_lr": INNER_LR, "adam_lr": ADAM_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
    )
    weights = initial_weights()
    for i, w in enumerate(weights):
        bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, i)),
                   tensor_io.encode_tensor(w))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "emb_table")),
               tensor_io.encode_tensor(_make_emb_table()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "pe")),
               tensor_io.encode_tensor(_make_pe()))
    # Adam state zeros at round 0
    zero = torch.zeros(2, D, D, dtype=torch.float32)
    for ub in range(N_UB):
        bucket.put(bucket.uri_for_key(paths.optim_state_key(run_id, 0, ub, "m")),
                   tensor_io.encode_tensor(zero))
        bucket.put(bucket.uri_for_key(paths.optim_state_key(run_id, 0, ub, "v")),
                   tensor_io.encode_tensor(zero))


def build_orchestrator_inputs(
    *, bucket: LocalBucket, run_id: str
) -> tuple[TaskGraphs, OrchestratorParams]:
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    cfg = bucket.get_json(cfg_uri) if bucket.exists(cfg_uri) else {}
    max_rounds = int(cfg.get("max_rounds", 5))
    inner_replicas = int(cfg.get("inner_replicas_per_ub", INNER_REPLICAS))

    graphs = TaskGraphs(
        forward=build_forward_graph(bucket=bucket.bucket, run_id=run_id),
        inner=build_inner_graph(),
        outer=build_outer_graph(),
        eval=build_eval_graph(bucket=bucket.bucket, run_id=run_id),
        reduce_for_n=build_reduce_graph,
    )
    params = OrchestratorParams(
        n_unique_blocks=N_UB,
        inner_replicas_per_ub=inner_replicas,
        common_params={},
        inner_params={},
        outer_params={},
        eval_params={},
        reduce_params={},
        max_rounds=max_rounds,
        m_target=inner_replicas,           # per-(mb) target; total m_target
                                           # = m_target * n_microbatches
        m_min=max(1, inner_replicas // 2),
        t_max_sec=5.0,
        n_microbatches=N_MICROBATCHES,     # Step 2: 8 microbatches per round
        outer_extra_state=["m", "v"],
    )
    return graphs, params
