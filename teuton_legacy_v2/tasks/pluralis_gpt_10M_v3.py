"""Phase 3 — `pluralis_gpt_10M_v3`: same model as `pluralis_gpt_10M`, with the
v3 wire stack layered in.

Wire compression on the activation lane (the dominant payload):

    forward emits, per cut i:
        x_in_proj   = x_in @ U[:, :k_x]            # (B, T, k_x)
        target_proj = target @ U[:, :k_dy]         # (B, T, k_dy)
    inner_step receives (x_in_proj, target_proj) and reconstructs:
        x_in    ≈ x_in_proj   @ U[:, :k_x].T       # rank-k_x reconstruction
        target  ≈ target_proj @ U[:, :k_dy].T      # rank-k_dy reconstruction

This is asymmetric Pluralis (k_x > k_dy) per the v3 100B plan: x_in needs more
rank because it carries the residual stream signal forward; the gradient
direction d_y is sparser so a smaller subspace suffices.

For DTYPE=float32 and (D=256, k_x=40, k_dy=10), wire bytes per cut:
    raw       = 2 × B × T × D × 4 = 2,097,152 B  (2.0 MiB)  — what Phase 1 ships
    projected = (B × T × k_x + B × T × k_dy) × 4 = ~205 KiB  — 10× reduction

int8 quantization on the projected blobs (Stage-10 ops) gives another ~4×.

Inner step compute is unchanged from `pluralis_gpt_10M` after reconstruction —
the per-block residual MLP runs in full d-dim. Only the WIRE is compressed.
This sacrifices some quality (rank-k_dy reconstruction is lossy) but is the
right Phase-3 tradeoff: prove compression saves bytes without breaking
training; lossless v3 (constraining W to S) is a Phase-3.5 follow-up.

Outer step is identical to the parent task (AdamW with persistent m, v).
"""
from __future__ import annotations

import math

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket
from . import pluralis_gpt_10M as base


D = base.D
V = base.V
T = base.T
B_TRAIN = base.B_TRAIN
B_EVAL = base.B_EVAL
N_BLOCKS = base.N_BLOCKS
N_UB = base.N_UB
INNER_REPLICAS = base.INNER_REPLICAS
K_INNER = base.K_INNER
INNER_LR = base.INNER_LR
ADAM_LR = base.ADAM_LR
ADAM_BETA1 = base.ADAM_BETA1
ADAM_BETA2 = base.ADAM_BETA2
ADAM_EPS = base.ADAM_EPS
EMB_SEED = base.EMB_SEED
PE_SEED = base.PE_SEED
EVAL_SEED = base.EVAL_SEED
STUDENT_INIT_SEED = base.STUDENT_INIT_SEED
DTYPE = base.DTYPE

# Pluralis subspace params (per v3 100B plan)
K_X = 40           # input-side projection rank
K_DY = 10          # gradient-side projection rank
SUBSPACE_SEED = 3001


def _make_uk() -> torch.Tensor:
    """Random orthonormal basis of D, take first max(K_X, K_DY) columns."""
    g = torch.Generator(device="cpu").manual_seed(SUBSPACE_SEED)
    m = torch.empty(D, D).normal_(generator=g)
    Q, _ = torch.linalg.qr(m)
    return Q.contiguous()                        # (D, D), orthonormal


def _uk_blob(gb: GraphBuilder, bucket: str, run_id: str):
    return gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'U_basis')}",
        shape=[D, D], dtype=DTYPE,
    )


def build_forward_graph(*, bucket: str, run_id: str) -> Graph:
    """Same forward + manual-backward chain as the parent task, but each per-cut
    output is now (x_in_proj, target_proj) with asymmetric ranks.
    """
    gb = GraphBuilder()
    weights = [
        gb.input(f"weights_{i}", [2, D, D], DTYPE) for i in range(N_UB)
    ]
    gb.param("round_id", "int")

    emb_table = base._emb_blob(gb, bucket, run_id)
    pe = base._pe_blob(gb, bucket, run_id)
    u_basis = _uk_blob(gb, bucket, run_id)
    # Slice the orthonormal basis into U_x and U_dy (column subsets).
    u_x = gb.slice(u_basis, dim=1, start=0, end=K_X)        # (D, K_X)
    u_dy = gb.slice(u_basis, dim=1, start=0, end=K_DY)      # (D, K_DY)

    ids_f = gb.emit("uniform", args=[],
                    kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, T], "dtype": DTYPE})
    input_ids = gb.cast(gb.mul(ids_f, gb.const(float(V) - 1e-3)), dtype="int64")
    target_ids = input_ids

    emb = gb.embedding(emb_table, input_ids)
    h_0 = gb.add(emb, pe)

    block_inputs = [h_0]
    block_caches = []
    h = h_0
    for i in range(N_BLOCKS):
        w_a, w_b = base._unpack_block(gb, weights[i])
        h_next, h_pre, h_post = base._block_forward(gb, h, w_a, w_b)
        block_caches.append((h_pre, h_post, w_a, w_b))
        if i < N_BLOCKS - 1:
            block_inputs.append(h_next)
        h = h_next
    h_final = h

    emb_t = gb.transpose(emb_table, dims=[1, 0])
    logits = gb.matmul(h_final, emb_t)
    probs = gb.softmax(logits, dim=-1)

    arange_v = gb.arange(start=0, end=V, step=1, dtype="int64")
    arange_btv = gb.broadcast(gb.reshape(arange_v, shape=[1, 1, V]),
                              shape=[B_TRAIN, T, V])
    target_btv = gb.broadcast(gb.unsqueeze(target_ids, dim=-1),
                              shape=[B_TRAIN, T, V])
    one_hot = gb.cast(gb.eq(arange_btv, target_btv), dtype=DTYPE)
    inv_n = gb.const(1.0 / float(B_TRAIN * T))
    dL_dlogits = gb.mul(gb.sub(probs, one_hot), inv_n)

    dL_dh = gb.matmul(dL_dlogits, emb_table)
    dL_dh_outputs = [None] * N_BLOCKS
    dL_dh_outputs[N_BLOCKS - 1] = dL_dh
    cur = dL_dh
    for i in range(N_BLOCKS - 1, -1, -1):
        h_pre, h_post, w_a, w_b = block_caches[i]
        x_in = block_inputs[i]
        cur = base._block_backward(gb, cur, h_pre, h_post, x_in, w_a, w_b)
        if i > 0:
            dL_dh_outputs[i - 1] = cur

    # Per-UB packed (x_in_proj, target_proj) with asymmetric ranks.
    for i in range(N_BLOCKS):
        h_in = block_inputs[i]
        h_pre_i, h_post_i, w_a_i, w_b_i = block_caches[i]
        y_branch = gb.matmul(h_post_i, w_b_i)
        h_out = gb.add(h_in, y_branch)
        target_i = gb.sub(h_out, dL_dh_outputs[i])

        x_in_proj = gb.matmul(h_in, u_x)              # (B, T, K_X)
        target_proj = gb.matmul(target_i, u_dy)       # (B, T, K_DY)
        # Pad the K_DY tensor to K_X by zero-extending so we can stack.
        pad = gb.full(shape=[B_TRAIN, T, K_X - K_DY], value=0.0, dtype=DTYPE)
        target_proj_padded = gb.concat([target_proj, pad], dim=-1)   # (B, T, K_X)
        pack = gb.stack([x_in_proj, target_proj_padded], dim=0)       # (2, B, T, K_X)
        gb.output(f"target_{i}", pack)
    return gb.build()


def build_inner_graph(*, bucket: str, run_id: str) -> Graph:
    """Reconstruct (x_in, target) from projections, then run the parent
    inner step (K-step SGD on local quadratic).
    """
    gb = GraphBuilder()
    packed_w = gb.input("weights", [2, D, D], DTYPE)
    packed_target = gb.input("target", [2, B_TRAIN, T, K_X], DTYPE)
    u_basis = _uk_blob(gb, bucket, run_id)
    u_x = gb.slice(u_basis, dim=1, start=0, end=K_X)        # (D, K_X)
    u_dy = gb.slice(u_basis, dim=1, start=0, end=K_DY)      # (D, K_DY)
    u_x_t = gb.transpose(u_x, dims=[1, 0])                  # (K_X, D)
    u_dy_t = gb.transpose(u_dy, dims=[1, 0])                # (K_DY, D)

    w_a_init, w_b_init = base._unpack_block(gb, packed_w)
    x_in_proj = gb.squeeze(gb.slice(packed_target, dim=0, start=0, end=1), dim=0)  # (B,T,K_X)
    target_padded = gb.squeeze(gb.slice(packed_target, dim=0, start=1, end=2), dim=0)
    target_proj = gb.slice(target_padded, dim=-1, start=0, end=K_DY)
    # Reconstruct
    x_in = gb.matmul(x_in_proj, u_x_t)                       # (B,T,D)
    target = gb.matmul(target_proj, u_dy_t)                  # (B,T,D)

    w_a = w_a_init
    w_b = w_b_init
    lr = gb.const(float(INNER_LR))
    inv_n = gb.const(1.0 / float(B_TRAIN * T))

    for _ in range(K_INNER):
        h_pre = gb.matmul(x_in, w_a)
        h_post = gb.silu(h_pre)
        y_branch = gb.matmul(h_post, w_b)
        out = gb.add(x_in, y_branch)
        dL_dout = gb.mul(gb.sub(out, target), inv_n)
        dL_dy = dL_dout
        dL_dW_b = gb.einsum(h_post, dL_dy, equation="btd,bte->de")
        w_b_t = gb.transpose(w_b, dims=[1, 0])
        dL_dh_post = gb.matmul(dL_dy, w_b_t)
        sig = gb.sigmoid(h_pre)
        one = gb.const(1.0)
        silu_grad = gb.mul(sig, gb.add(one, gb.mul(h_pre, gb.sub(one, sig))))
        dL_dh_pre = gb.mul(dL_dh_post, silu_grad)
        dL_dW_a = gb.einsum(x_in, dL_dh_pre, equation="btd,bte->de")
        w_a = gb.sub(w_a, gb.mul(lr, dL_dW_a))
        w_b = gb.sub(w_b, gb.mul(lr, dL_dW_b))

    delta = gb.sub(gb.stack([w_a, w_b], dim=0), packed_w)
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    return base.build_outer_graph()


def build_eval_graph(*, bucket: str, run_id: str) -> Graph:
    return base.build_eval_graph(bucket=bucket, run_id=run_id)


def build_reduce_graph(n_inputs: int) -> Graph:
    return base.build_reduce_graph(n_inputs)


def initial_weights() -> list[torch.Tensor]:
    return base.initial_weights()


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pluralis_gpt_10M_v3", "n_unique_blocks": N_UB, "n_blocks": N_BLOCKS,
         "d": D, "V": V, "T": T, "B_train": B_TRAIN, "B_eval": B_EVAL,
         "k_x": K_X, "k_dy": K_DY,
         "K_inner": K_INNER, "inner_lr": INNER_LR, "adam_lr": ADAM_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
    )
    weights = initial_weights()
    for i, w in enumerate(weights):
        bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, i)),
                   tensor_io.encode_tensor(w))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "emb_table")),
               tensor_io.encode_tensor(base._make_emb_table()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "pe")),
               tensor_io.encode_tensor(base._make_pe()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "U_basis")),
               tensor_io.encode_tensor(_make_uk()))
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
        inner=build_inner_graph(bucket=bucket.bucket, run_id=run_id),
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
        m_target=inner_replicas,
        m_min=max(1, inner_replicas // 2),
        t_max_sec=60.0,
        outer_extra_state=["m", "v"],
    )
    return graphs, params
