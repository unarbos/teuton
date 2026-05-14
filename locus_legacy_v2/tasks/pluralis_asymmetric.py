"""Stage 9 — asymmetric Pluralis (k_x ≠ k_dy).

Like Stage 8, but UB-1's wire-format uses two different ranks:
  - x_in is projected with U_k[:, :k_x]   (k_x = SUBSPACE_K, lossless)
  - target is projected with U_k[:, :k_dy]  (k_dy < k_x, sub-rank lossy)

The two projected tensors are concatenated along the last axis on the wire:
shape (B, T, k_x + k_dy). Inner_step splits and unprojects each.

Stage-9 deliverable: wire bytes for UB-1 are smaller than Stage 8's even
though we keep x lossless. Convergence may be slower because d_y is lossy.
"""
from __future__ import annotations

import math

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket


V = 32
D = 16
K_X = 8
K_DY = 6
T = 8
B_TRAIN = 16
B_EVAL = 64
N_UB = 2
INNER_REPLICAS = 2
K_INNER = 4
INNER_LR = 0.005
OUTER_LR = 1.0
EMB_SEED = 7777
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
SUBSPACE_SEED = 3001
DTYPE = "float32"


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


def _make_uk() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(SUBSPACE_SEED)
    m = torch.empty(D, D).normal_(generator=g)
    Q, _ = torch.linalg.qr(m)
    return Q[:, :K_X].contiguous()        # full architectural rank k_x


def build_forward_graph(*, bucket: str, run_id: str) -> Graph:
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")

    emb_table = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'emb_table')}",
        shape=[V, D], dtype=DTYPE,
    )
    pe = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'pe')}",
        shape=[T, D], dtype=DTYPE,
    )
    u_k_x = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'U_k_x')}",
        shape=[D, K_X], dtype=DTYPE,
    )
    u_k_dy = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'U_k_dy')}",
        shape=[D, K_DY], dtype=DTYPE,
    )

    ids_f = gb.emit("uniform", args=[],
                    kwargs={"seed": ref_param("round_id"),
                            "shape": [B_TRAIN, T], "dtype": DTYPE})
    input_ids = gb.cast(gb.mul(ids_f, gb.const(float(V) - 1e-3)), dtype="int64")
    target_ids = input_ids

    emb = gb.embedding(emb_table, input_ids)
    h_0 = gb.add(emb, pe)
    h_1 = gb.matmul(h_0, w0)
    h_2 = gb.matmul(h_1, w1)
    emb_t = gb.transpose(emb_table, dims=[1, 0])
    logits = gb.matmul(h_2, emb_t)

    probs = gb.softmax(logits, dim=-1)
    arange_v = gb.arange(start=0, end=V, step=1, dtype="int64")
    arange_btv = gb.broadcast(gb.reshape(arange_v, shape=[1, 1, V]),
                               shape=[B_TRAIN, T, V])
    target_btv = gb.broadcast(gb.unsqueeze(target_ids, dim=-1),
                               shape=[B_TRAIN, T, V])
    one_hot = gb.cast(gb.eq(arange_btv, target_btv), dtype=DTYPE)
    inv_n = gb.const(1.0 / float(B_TRAIN * T))
    dL_dlogits = gb.mul(gb.sub(probs, one_hot), inv_n)
    dL_dh_2 = gb.matmul(dL_dlogits, emb_table)
    w1_t = gb.transpose(w1, dims=[1, 0])
    dL_dh_1 = gb.matmul(dL_dh_2, w1_t)

    target_0 = gb.sub(h_1, dL_dh_1)
    target_1 = gb.sub(h_2, dL_dh_2)

    pack_0 = gb.stack([h_0, target_0], dim=0)        # (2, B, T, d)

    h_1_proj = gb.matmul(h_1, u_k_x)                 # (B, T, K_X)
    target_1_proj = gb.matmul(target_1, u_k_dy)      # (B, T, K_DY)
    pack_1 = gb.concat([h_1_proj, target_1_proj], dim=-1)   # (B, T, K_X+K_DY)

    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    return gb.build()


def build_inner_graph_ub0() -> Graph:
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed = gb.input("target", [2, B_TRAIN, T, D], DTYPE)
    x_in = gb.squeeze(gb.slice(packed, dim=0, start=0, end=1), dim=0)
    target = gb.squeeze(gb.slice(packed, dim=0, start=1, end=2), dim=0)
    w_curr = weights
    lr = gb.const(float(INNER_LR))
    for _ in range(K_INNER):
        y_pred = gb.matmul(x_in, w_curr)
        resid = gb.sub(y_pred, target)
        grad = gb.einsum(x_in, resid, equation="btd,bte->de")
        w_curr = gb.sub(w_curr, gb.mul(lr, grad))
    delta = gb.sub(w_curr, weights)
    gb.output("delta", delta)
    return gb.build()


def build_inner_graph_ub1(*, bucket: str, run_id: str) -> Graph:
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed_proj = gb.input("target", [B_TRAIN, T, K_X + K_DY], DTYPE)
    u_k_x = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'U_k_x')}",
        shape=[D, K_X], dtype=DTYPE,
    )
    u_k_dy = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'U_k_dy')}",
        shape=[D, K_DY], dtype=DTYPE,
    )

    # Split last axis: [:K_X] and [K_X:K_X+K_DY]
    h_1_proj = gb.slice(packed_proj, dim=-1, start=0, end=K_X)             # (B, T, K_X)
    target_proj = gb.slice(packed_proj, dim=-1, start=K_X, end=K_X + K_DY)  # (B, T, K_DY)

    u_k_x_t = gb.transpose(u_k_x, dims=[1, 0])
    u_k_dy_t = gb.transpose(u_k_dy, dims=[1, 0])

    x_in = gb.matmul(h_1_proj, u_k_x_t)         # (B, T, d) — lossless
    target = gb.matmul(target_proj, u_k_dy_t)   # (B, T, d) — sub-rank lossy

    w_curr = weights
    lr = gb.const(float(INNER_LR))
    for _ in range(K_INNER):
        y_pred = gb.matmul(x_in, w_curr)
        resid = gb.sub(y_pred, target)
        grad = gb.einsum(x_in, resid, equation="btd,bte->de")
        w_curr = gb.sub(w_curr, gb.mul(lr, grad))

    p_full = gb.matmul(u_k_x, u_k_x_t)
    w_proj = gb.matmul(w_curr, p_full)
    delta = gb.sub(w_proj, weights)
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    gb = GraphBuilder()
    w = gb.input("weights", [D, D], DTYPE)
    rd = gb.input("reduced_delta", [D, D], DTYPE)
    nw = gb.add(w, gb.mul(rd, gb.const(float(OUTER_LR))))
    gb.output("new_weights", nw)
    return gb.build()


def build_eval_graph(*, bucket: str, run_id: str) -> Graph:
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    emb_table = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'emb_table')}",
        shape=[V, D], dtype=DTYPE,
    )
    pe = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'pe')}",
        shape=[T, D], dtype=DTYPE,
    )
    ids_f = gb.emit("uniform", args=[],
                    kwargs={"seed": EVAL_SEED, "shape": [B_EVAL, T], "dtype": DTYPE})
    input_ids = gb.cast(gb.mul(ids_f, gb.const(float(V) - 1e-3)), dtype="int64")
    emb = gb.embedding(emb_table, input_ids)
    h_0 = gb.add(emb, pe)
    h_1 = gb.matmul(h_0, w0)
    h_2 = gb.matmul(h_1, w1)
    emb_t = gb.transpose(emb_table, dims=[1, 0])
    logits = gb.matmul(h_2, emb_t)
    loss = gb.cross_entropy(logits, input_ids)
    gb.output("metrics", gb.unsqueeze(loss, dim=0))
    return gb.build()


def build_reduce_graph(n_inputs: int) -> Graph:
    gb = GraphBuilder()
    refs = []
    for i in range(n_inputs):
        ri = gb.input(f"d_{i}", [D, D], DTYPE)
        refs.append(gb.unsqueeze(ri, dim=0))
    stack = refs[0] if n_inputs == 1 else gb.concat(refs, dim=0)
    avg = gb.mean(stack, dim=0)
    gb.output("reduced", avg)
    return gb.build()


def initial_weights() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    eye = torch.eye(D, dtype=torch.float32)
    w0 = eye + torch.empty(D, D).normal_(generator=g) * 0.05
    u_k = _make_uk()
    p = u_k @ u_k.T
    w1_raw = eye + torch.empty(D, D).normal_(generator=g) * 0.05
    w1 = w1_raw @ p
    return w0, w1


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pluralis_asymmetric", "n_unique_blocks": N_UB, "d": D,
         "k_x": K_X, "k_dy": K_DY, "V": V, "T": T,
         "B_train": B_TRAIN, "B_eval": B_EVAL,
         "K_inner": K_INNER, "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
    )
    w0, w1 = initial_weights()
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 0)),
               tensor_io.encode_tensor(w0))
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 1)),
               tensor_io.encode_tensor(w1))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "emb_table")),
               tensor_io.encode_tensor(_make_emb_table()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "pe")),
               tensor_io.encode_tensor(_make_pe()))
    u_k = _make_uk()
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "U_k_x")),
               tensor_io.encode_tensor(u_k[:, :K_X].contiguous()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "U_k_dy")),
               tensor_io.encode_tensor(u_k[:, :K_DY].contiguous()))


def build_orchestrator_inputs(
    *, bucket: LocalBucket, run_id: str
) -> tuple[TaskGraphs, OrchestratorParams]:
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    cfg = bucket.get_json(cfg_uri) if bucket.exists(cfg_uri) else {}
    max_rounds = int(cfg.get("max_rounds", 5))
    inner_replicas = int(cfg.get("inner_replicas_per_ub", INNER_REPLICAS))

    inner_ub0 = build_inner_graph_ub0()
    inner_ub1 = build_inner_graph_ub1(bucket=bucket.bucket, run_id=run_id)

    graphs = TaskGraphs(
        forward=build_forward_graph(bucket=bucket.bucket, run_id=run_id),
        inner=inner_ub0,
        outer=build_outer_graph(),
        eval=build_eval_graph(bucket=bucket.bucket, run_id=run_id),
        reduce_for_n=build_reduce_graph,
        inner_per_ub=[inner_ub0, inner_ub1],
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
        m_min=1,
        t_max_sec=60.0,
    )
    return graphs, params
