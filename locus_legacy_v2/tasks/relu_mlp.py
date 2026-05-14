"""Stage 1 — ReLU MLP with full backprop in IR.

Same shape as the linear `tasks/mlp.py` but with a ReLU non-linearity in the
hidden layer. The forward graph emits the *negative gradient* of MSE w.r.t.
each weight matrix as the per-UB target; inner_step scales by inner_lr to
produce the SGD delta; reduce averages; outer_step adds.

The point of this task is to exercise the IR's ability to express the
backprop chain through a non-linearity using comparison ops:

    h_pre = x @ W1
    h     = relu(h_pre)
    y     = h @ W2
    err   = y - y_true
    dL_dy   = err * (2/B)
    dL_dW2  = h.T @ dL_dy
    dL_dh   = dL_dy @ W2.T
    relu_mask = cast(gt(h_pre, 0.0), float32)
    dL_dh_pre = dL_dh * relu_mask
    dL_dW1    = x.T @ dL_dh_pre
"""
from __future__ import annotations

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #

D = 16
B_TRAIN = 64
B_EVAL = 256
N_UB = 2
INNER_REPLICAS = 2
INNER_LR = 0.01
OUTER_LR = 1.0
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #


def build_forward_graph() -> Graph:
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")

    x = gb.emit(
        "normal", args=[],
        kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, D], "dtype": DTYPE},
    )
    # Teacher uses ReLU too (so the student can actually fit it well)
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    teacher_h_pre = gb.matmul(x, t1)
    teacher_h = gb.relu(teacher_h_pre)
    y_true = gb.matmul(teacher_h, t2)

    # Student forward
    h_pre = gb.matmul(x, w0)
    h = gb.relu(h_pre)
    y = gb.matmul(h, w1)

    err = gb.sub(y, y_true)
    dL_dy = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))

    # dL/dW2 = h^T @ dL_dy
    h_t = gb.transpose(h, dims=[1, 0])
    dL_dW2 = gb.matmul(h_t, dL_dy)

    # dL/dh = dL_dy @ W2^T
    w1_t = gb.transpose(w1, dims=[1, 0])
    dL_dh = gb.matmul(dL_dy, w1_t)

    # ReLU derivative mask: 1 where h_pre > 0, else 0 (cast bool -> float32)
    mask_bool = gb.gt(h_pre, gb.const(0.0))
    mask_f32 = gb.cast(mask_bool, dtype=DTYPE)
    dL_dh_pre = gb.mul(dL_dh, mask_f32)

    # dL/dW1 = x^T @ dL_dh_pre
    x_t = gb.transpose(x, dims=[1, 0])
    dL_dW1 = gb.matmul(x_t, dL_dh_pre)

    target_0 = gb.neg(dL_dW1)
    target_1 = gb.neg(dL_dW2)

    gb.output("target_0", target_0)
    gb.output("target_1", target_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """Scale the negative-gradient target by inner_lr."""
    gb = GraphBuilder()
    gb.input("weights", [D, D], DTYPE)
    target = gb.input("target", [D, D], DTYPE)
    delta = gb.mul(target, gb.const(float(INNER_LR)))
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    gb = GraphBuilder()
    w = gb.input("weights", [D, D], DTYPE)
    rd = gb.input("reduced_delta", [D, D], DTYPE)
    scaled = gb.mul(rd, gb.const(float(OUTER_LR)))
    nw = gb.add(w, scaled)
    gb.output("new_weights", nw)
    return gb.build()


def build_eval_graph() -> Graph:
    """MSE on a fixed eval batch passed through ReLU MLP."""
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)

    x = gb.emit("normal", args=[],
                kwargs={"seed": EVAL_SEED, "shape": [B_EVAL, D], "dtype": DTYPE})
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    teacher_h = gb.relu(gb.matmul(x, t1))
    y_true = gb.matmul(teacher_h, t2)

    h = gb.relu(gb.matmul(x, w0))
    y = gb.matmul(h, w1)
    err = gb.sub(y, y_true)
    sq = gb.mul(err, err)
    mse = gb.mean(sq)
    out = gb.unsqueeze(mse, dim=0)
    gb.output("metrics", out)
    return gb.build()


def build_reduce_graph(n_inputs: int) -> Graph:
    gb = GraphBuilder()
    refs = []
    for i in range(n_inputs):
        ri = gb.input(f"d_{i}", [D, D], DTYPE)
        refs.append(gb.unsqueeze(ri, dim=0))
    if n_inputs == 1:
        stack = refs[0]
    else:
        stack = gb.concat(refs, dim=0)
    avg = gb.mean(stack, dim=0)
    gb.output("reduced", avg)
    return gb.build()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def initial_weights() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    scale = (1.0 / D) ** 0.5
    w0 = torch.empty(D, D).normal_(generator=g) * scale
    w1 = torch.empty(D, D).normal_(generator=g) * scale
    return w0, w1


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {
            "run_id": run_id,
            "current_round": 0,
            "max_rounds": int(max_rounds),
            "completed_rounds": [],
            "failed_rounds": [],
        },
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {
            "task": "relu_mlp",
            "n_unique_blocks": N_UB,
            "d": D,
            "B_train": B_TRAIN,
            "B_eval": B_EVAL,
            "inner_lr": INNER_LR,
            "outer_lr": OUTER_LR,
            "inner_replicas_per_ub": INNER_REPLICAS,
            "max_rounds": int(max_rounds),
        },
    )
    w0, w1 = initial_weights()
    bucket.put(
        bucket.uri_for_key(paths.weights_key(run_id, 0, 0)),
        tensor_io.encode_tensor(w0),
    )
    bucket.put(
        bucket.uri_for_key(paths.weights_key(run_id, 0, 1)),
        tensor_io.encode_tensor(w1),
    )


def build_orchestrator_inputs(
    *, bucket: LocalBucket, run_id: str
) -> tuple[TaskGraphs, OrchestratorParams]:
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    cfg = bucket.get_json(cfg_uri) if bucket.exists(cfg_uri) else {}
    max_rounds = int(cfg.get("max_rounds", 5))
    inner_replicas = int(cfg.get("inner_replicas_per_ub", INNER_REPLICAS))

    graphs = TaskGraphs(
        forward=build_forward_graph(),
        inner=build_inner_graph(),
        outer=build_outer_graph(),
        eval=build_eval_graph(),
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
        t_max_sec=10.0,
    )
    return graphs, params
