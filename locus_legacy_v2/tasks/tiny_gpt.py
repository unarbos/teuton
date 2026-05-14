"""Stage 6 — tiny GPT (a real LM) trained through the bucket.

Architecture (intentionally minimal so the IR can express the full backprop
chain manually):

    input_ids: (B, T) int64
    emb       = embedding(emb_table, input_ids)            # (B, T, d)
    h_0       = emb + pe                                    # (B, T, d)
    h_1       = h_0 @ W0                                    # (B, T, d)
    h_2       = h_1 @ W1                                    # (B, T, d)
    logits    = h_2 @ emb_table.T                           # (B, T, V)  (tied head)
    targets   = input_ids   (predict-self task)
    loss      = cross_entropy(logits, targets)

UBs: 2  (UB-0 = W0 in d×d, UB-1 = W1 in d×d).

Static const_blobs (written by bootstrap, referenced via `const_blob` ref):
  - emb_table  shape (V, d)  — random init, then frozen
  - pe         shape (T, d)  — sinusoidal

This task exercises:
  - softmax / log_softmax / cross_entropy IR ops
  - embedding IR op
  - const_blob ref kind (emb_table + pe)

The "predict-self" target is intentionally trivial: with W0 = W1 = I the
model learns to output the input token's logits dominated by the diagonal
of emb_table @ emb_table^T — so cross-entropy can decrease meaningfully in
just a few rounds even with K=4 LocoProp inner SGD.
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

V = 32
D = 16
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
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Static-blob construction (written by bootstrap, referenced by graphs)
# --------------------------------------------------------------------------- #


def _make_emb_table() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(EMB_SEED)
    e = torch.empty(V, D).normal_(generator=g)
    # Normalize each row so emb @ emb.T has unit diagonal -> "predict self"
    # is well-conditioned.
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _make_pe() -> torch.Tensor:
    pe = torch.zeros(T, D)
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe * 0.1   # small magnitude so emb dominates


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #


def _emb_blob_ref(gb: GraphBuilder, run_id: str):
    return gb.const_blob(
        f"s3://{{BUCKET}}/{paths.static_blob_key(run_id, 'emb_table')}",
        shape=[V, D], dtype=DTYPE,
    )


def _pe_blob_ref(gb: GraphBuilder, run_id: str):
    return gb.const_blob(
        f"s3://{{BUCKET}}/{paths.static_blob_key(run_id, 'pe')}",
        shape=[T, D], dtype=DTYPE,
    )


def build_forward_graph(*, bucket: str, run_id: str) -> Graph:
    """Forward + manual gradient computation. Emits per-UB packed (x_in, target)
    of shape (2, B, T, d)."""
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")

    # Static blobs (URIs concretized at task-build time)
    emb_table = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'emb_table')}",
        shape=[V, D], dtype=DTYPE,
    )
    pe = gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, 'pe')}",
        shape=[T, D], dtype=DTYPE,
    )

    # Generate random ids per round: cast(uniform([B, T]) * V, int64)
    ids_f = gb.emit(
        "uniform", args=[],
        kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, T], "dtype": DTYPE},
    )
    ids_scaled = gb.mul(ids_f, gb.const(float(V) - 1e-3))   # avoid V edge case
    input_ids = gb.cast(ids_scaled, dtype="int64")
    target_ids = input_ids   # predict-self

    # Forward
    emb = gb.embedding(emb_table, input_ids)            # (B, T, d)
    h_0 = gb.add(emb, pe)                                # (B, T, d) (pe broadcasts)
    h_1 = gb.matmul(h_0, w0)
    h_2 = gb.matmul(h_1, w1)

    # logits = h_2 @ emb_table^T  (tied lm_head)
    emb_t = gb.transpose(emb_table, dims=[1, 0])         # (d, V)
    logits = gb.matmul(h_2, emb_t)                       # (B, T, V)

    # Compute dL/dlogits = (probs - one_hot) / (B*T)
    probs = gb.softmax(logits, dim=-1)                   # (B, T, V)

    # one_hot via eq + cast
    arange_v = gb.arange(start=0, end=V, step=1, dtype="int64")           # (V,)
    arange_btv = gb.broadcast(
        gb.reshape(arange_v, shape=[1, 1, V]),
        shape=[B_TRAIN, T, V],
    )
    target_btv = gb.broadcast(
        gb.unsqueeze(target_ids, dim=-1),
        shape=[B_TRAIN, T, V],
    )
    one_hot_bool = gb.eq(arange_btv, target_btv)
    one_hot = gb.cast(one_hot_bool, dtype=DTYPE)

    inv_n = gb.const(1.0 / float(B_TRAIN * T))
    dL_dlogits = gb.mul(gb.sub(probs, one_hot), inv_n)   # (B, T, V)

    # dL/dh_2 = dL/dlogits @ emb_table   (since logits = h_2 @ emb_t)
    dL_dh_2 = gb.matmul(dL_dlogits, emb_table)           # (B, T, d)

    # dL/dh_1 = dL/dh_2 @ W1.T
    w1_t = gb.transpose(w1, dims=[1, 0])
    dL_dh_1 = gb.matmul(dL_dh_2, w1_t)                   # (B, T, d)

    # LocoProp targets
    target_0 = gb.sub(h_1, dL_dh_1)
    target_1 = gb.sub(h_2, dL_dh_2)

    # Pack (x_in, target) -> (2, B, T, d)
    pack_0 = gb.stack([h_0, target_0], dim=0)
    pack_1 = gb.stack([h_1, target_1], dim=0)

    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """K-step SGD on local quadratic ½‖x_in @ W - target‖². Same shape for
    both UBs."""
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed = gb.input("target", [2, B_TRAIN, T, D], DTYPE)

    x_in = gb.squeeze(gb.slice(packed, dim=0, start=0, end=1), dim=0)  # (B, T, d)
    target = gb.squeeze(gb.slice(packed, dim=0, start=1, end=2), dim=0)

    w_curr = weights
    lr = gb.const(float(INNER_LR))
    for _ in range(K_INNER):
        y_pred = gb.matmul(x_in, w_curr)                 # (B, T, d)
        resid = gb.sub(y_pred, target)
        # grad shape (d, d); contract over (B, T)
        grad = gb.einsum(x_in, resid, equation="btd,bte->de")
        w_curr = gb.sub(w_curr, gb.mul(lr, grad))
    delta = gb.sub(w_curr, weights)
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
    """Validation cross-entropy on a fixed eval batch."""
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

    loss = gb.cross_entropy(logits, input_ids, ignore_index=-100)
    out = gb.unsqueeze(loss, dim=0)
    gb.output("metrics", out)
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
    # Initialize close to identity so the loss starts at a sensible value.
    eye = torch.eye(D, dtype=torch.float32)
    noise = torch.empty(D, D).normal_(generator=g) * 0.05
    w0 = eye + noise
    noise2 = torch.empty(D, D).normal_(generator=g) * 0.05
    w1 = eye + noise2
    return w0, w1


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "tiny_gpt", "n_unique_blocks": N_UB, "d": D, "V": V, "T": T,
         "B_train": B_TRAIN, "B_eval": B_EVAL, "K_inner": K_INNER,
         "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
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
        m_target=inner_replicas,
        m_min=1,
        t_max_sec=60.0,
    )
    return graphs, params
