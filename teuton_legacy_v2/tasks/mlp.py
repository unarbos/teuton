"""Toy MLP task.

A 2-layer linear "MLP" (no activations) y = (x @ W1) @ W2 trained against a
fixed teacher MLP via gradient descent, expressed as Teuton IR graphs.

Why no activation? It keeps the IR tiny and avoids the need for a comparison
op for the relu mask. The system-level demonstration is identical: forward
pass produces a target (= negative gradient of MSE), inner_step scales it by
inner_lr, reduce averages across replicas, outer_step applies it.

Both unique blocks have identical shape `[d, d]` so a single inner_step graph
and a single outer_step graph cover both.

Per round R, replica r reads the same target — the inner step produces the
same delta deterministically. Replication exists to exercise the reduce path;
the run still converges because each round is a real gradient step on a
fresh batch.
"""
from __future__ import annotations

from typing import Any

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ParamSpec, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #

D = 16              # square model: d_in = d_hid = d_out
B_TRAIN = 64
B_EVAL = 256
N_UB = 2            # UB-0 = W1, UB-1 = W2
INNER_REPLICAS = 2
INNER_LR = 0.01
OUTER_LR = 1.0      # inner already produced lr-scaled delta
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #


def build_forward_graph() -> Graph:
    """Compute negative gradients of MSE loss w.r.t. W1 and W2.

    Inputs:
      weights_0: (D, D)  — student W1
      weights_1: (D, D)  — student W2

    Param:
      round_id: int      — used as the seed for the per-round training batch

    Outputs:
      target_0: (D, D)   — -dL/dW1
      target_1: (D, D)   — -dL/dW2
    """
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")

    # x ~ N(0, 1) using round_id as the seed (param-ref kwarg)
    x = gb.emit(
        "normal",
        args=[],
        kwargs={
            "seed": ref_param("round_id"),
            "shape": [B_TRAIN, D],
            "dtype": DTYPE,
        },
    )
    # teacher generates y_true (fixed seeds)
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)

    # student forward
    h = gb.matmul(x, w0)             # (B, D)
    y = gb.matmul(h, w1)             # (B, D)

    err = gb.sub(y, y_true)          # (B, D)
    dL_dy = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))

    # dL/dW2 = h^T @ dL/dy
    h_t = gb.transpose(h, dims=[1, 0])
    dL_dW2 = gb.matmul(h_t, dL_dy)

    # dL/dh = dL/dy @ W2^T
    w1_t = gb.transpose(w1, dims=[1, 0])
    dL_dh = gb.matmul(dL_dy, w1_t)

    # dL/dW1 = x^T @ dL/dh
    x_t = gb.transpose(x, dims=[1, 0])
    dL_dW1 = gb.matmul(x_t, dL_dh)

    target_0 = gb.neg(dL_dW1)
    target_1 = gb.neg(dL_dW2)

    gb.output("target_0", target_0)
    gb.output("target_1", target_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """Scale the negative-gradient target by inner_lr to produce a delta.

    Inputs:
      weights: (D, D)
      target:  (D, D)

    Outputs:
      delta:   (D, D) = inner_lr * target
    """
    gb = GraphBuilder()
    gb.input("weights", [D, D], DTYPE)
    target = gb.input("target", [D, D], DTYPE)
    delta = gb.mul(target, gb.const(float(INNER_LR)))
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    """Apply the reduced delta to the weights.

    Inputs:
      weights:        (D, D)
      reduced_delta:  (D, D)

    Outputs:
      new_weights: (D, D) = weights + outer_lr * reduced_delta
    """
    gb = GraphBuilder()
    w = gb.input("weights", [D, D], DTYPE)
    rd = gb.input("reduced_delta", [D, D], DTYPE)
    scaled = gb.mul(rd, gb.const(float(OUTER_LR)))
    nw = gb.add(w, scaled)
    gb.output("new_weights", nw)
    return gb.build()


def build_eval_graph() -> Graph:
    """Compute MSE on a fixed eval batch.

    Inputs:
      weights_0: (D, D)
      weights_1: (D, D)

    Outputs:
      metrics: (1,) tensor — val MSE.
    """
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)

    x = gb.emit("normal", args=[],
                kwargs={"seed": EVAL_SEED, "shape": [B_EVAL, D], "dtype": DTYPE})
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)

    h = gb.matmul(x, w0)
    y = gb.matmul(h, w1)
    err = gb.sub(y, y_true)
    sq = gb.mul(err, err)
    mse = gb.mean(sq)

    out = gb.unsqueeze(mse, dim=0)
    gb.output("metrics", out)
    return gb.build()


def build_reduce_graph(n_inputs: int) -> Graph:
    """Reduce N delta tensors of shape (D, D) by mean.

    Stacks the N inputs into one tensor of shape (N, D, D) via
    unsqueeze + concat, then takes the mean along axis 0.
    """
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
# Bootstrap (writes initial weights + state.json)
# --------------------------------------------------------------------------- #


def initial_weights() -> tuple[torch.Tensor, torch.Tensor]:
    """Xavier-ish init for student W1, W2."""
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    scale = (1.0 / D) ** 0.5
    w0 = torch.empty(D, D).normal_(generator=g) * scale
    w1 = torch.empty(D, D).normal_(generator=g) * scale
    return w0, w1


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    """Initialize a run: write state.json, config.json, and round-0 weights."""
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
            "task": "mlp",
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


# --------------------------------------------------------------------------- #
# Orchestrator-side glue: build TaskGraphs + OrchestratorParams
# --------------------------------------------------------------------------- #


def build_orchestrator_inputs(
    *, bucket: LocalBucket, run_id: str
) -> tuple[TaskGraphs, OrchestratorParams]:
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    if bucket.exists(cfg_uri):
        cfg = bucket.get_json(cfg_uri)
    else:
        cfg = {}
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
