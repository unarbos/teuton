"""Stage 7 — Pluralis subspace constraint (architectural).

Tiny GPT with the second linear layer's output constrained to lie in a fixed
k-dimensional subspace S ⊂ R^d. The orthonormal basis U_k ∈ R^{d×k} is a
const_blob written by bootstrap once and read by the inner_step graph.

The constraint mechanism: after each K-step inner SGD, project the candidate
W1's columns onto S via P = U_k @ U_k^T, i.e. W1 ← W1 @ P. (Equivalently, W1
is constrained to satisfy W1 = W1 @ U_k @ U_k^T after every update.)

Outer step uses modified AdamW: for the constrained UB-1, the V-tensor is
made row-constant (mean over the last axis kept) so the per-element rescaling
preserves W1's column-subspace membership. UB-0 is unconstrained and uses
standard AdamW.

This stage doesn't change the wire format yet — that's Stage 8.

Stage-7 deliverable: subspace-constrained training converges (loss drops).
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
# Hyperparameters (mostly inherited from tiny_gpt, plus subspace_k)
# --------------------------------------------------------------------------- #

V = 32
D = 16
SUBSPACE_K = 8
T = 8
B_TRAIN = 16
B_EVAL = 64
N_UB = 2
INNER_REPLICAS = 2
K_INNER = 4
INNER_LR = 0.005
OUTER_LR = 1.0
EMB_SEED = 7777
PE_SEED = 5555
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
SUBSPACE_SEED = 3001
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Static-blob construction
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


def _make_uk() -> torch.Tensor:
    """Random orthonormal basis of R^d, kept to first SUBSPACE_K columns."""
    g = torch.Generator(device="cpu").manual_seed(SUBSPACE_SEED)
    m = torch.empty(D, D).normal_(generator=g)
    Q, _ = torch.linalg.qr(m)
    return Q[:, :SUBSPACE_K].contiguous()      # (D, K)


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #


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

    pack_0 = gb.stack([h_0, target_0], dim=0)
    pack_1 = gb.stack([h_1, target_1], dim=0)
    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    return gb.build()


def build_inner_graph(*, bucket: str, run_id: str) -> Graph:
    """K-step SGD on local quadratic, with subspace projection of W1.

    Uses `ub` param to switch behavior: ub=0 is unconstrained, ub=1 is
    subspace-constrained via post-step `W <- W @ U_k @ U_k^T`. Since the
    IR has no conditionals on params, we instead always compute `W @ P`
    and use `where(constrain_flag, projected, raw)` to select. We pass
    `constrain_flag` as a constant tensor read from a static blob —
    bootstrap writes one (1,)-shaped boolean per UB.
    """
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed = gb.input("target", [2, B_TRAIN, T, D], DTYPE)

    u_k = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'U_k')}",
        shape=[D, SUBSPACE_K], dtype=DTYPE,
    )
    gb.param("ub", "int")
    # constrain_flag = float(ub == 1) — only UB-1 is constrained.
    cf_scalar = gb.cast(gb.eq(ref_param("ub"), gb.const(1)), dtype=DTYPE)

    x_in = gb.squeeze(gb.slice(packed, dim=0, start=0, end=1), dim=0)
    target = gb.squeeze(gb.slice(packed, dim=0, start=1, end=2), dim=0)

    w_curr = weights
    lr = gb.const(float(INNER_LR))
    for _ in range(K_INNER):
        y_pred = gb.matmul(x_in, w_curr)
        resid = gb.sub(y_pred, target)
        grad = gb.einsum(x_in, resid, equation="btd,bte->de")
        w_curr = gb.sub(w_curr, gb.mul(lr, grad))

    # Subspace projection P = U_k U_k^T  (D, D)
    u_k_t = gb.transpose(u_k, dims=[1, 0])
    p_full = gb.matmul(u_k, u_k_t)
    w_proj = gb.matmul(w_curr, p_full)

    # Blend by cf_scalar (1.0 = projected, 0.0 = raw). Implements the
    # per-UB switch without IR-level conditionals.
    one_minus_cf = gb.sub(gb.const(1.0), cf_scalar)
    w_final = gb.add(gb.mul(cf_scalar, w_proj), gb.mul(one_minus_cf, w_curr))

    delta = gb.sub(w_final, weights)
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    """Vanilla SGD-style outer (Stage 7's contribution is the constraint, not
    a fancy optimizer). Stage 14 will combine subspace + AdamW + sign-Loco."""
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


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def initial_weights() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    eye = torch.eye(D, dtype=torch.float32)
    w0 = eye + torch.empty(D, D).normal_(generator=g) * 0.05
    # W1 starts in the subspace S
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
        {"task": "pluralis_subspace", "n_unique_blocks": N_UB, "d": D, "k": SUBSPACE_K,
         "V": V, "T": T, "B_train": B_TRAIN, "B_eval": B_EVAL,
         "K_inner": K_INNER, "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
    )
    w0, w1 = initial_weights()
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 0)),
               tensor_io.encode_tensor(w0))
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 1)),
               tensor_io.encode_tensor(w1))
    # Static blobs
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "emb_table")),
               tensor_io.encode_tensor(_make_emb_table()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "pe")),
               tensor_io.encode_tensor(_make_pe()))
    bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, "U_k")),
               tensor_io.encode_tensor(_make_uk()))
    # Per-UB constrain_flag tensors (so the inner_step graph can branch
    # between constrained / unconstrained behavior via a `where` blend).
    flag_ub = {0: 0.0, 1: 1.0}
    for ub, val in flag_ub.items():
        flag = torch.tensor([val], dtype=torch.float32)
        bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, f"constrain_flag_ub{ub}")),
                   tensor_io.encode_tensor(flag))


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
        m_min=1,
        t_max_sec=60.0,
    )
    return graphs, params
