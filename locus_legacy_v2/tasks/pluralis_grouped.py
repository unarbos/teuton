"""Stage 13 — group_layers > 1 (worker holds N consecutive layers).

L=2 layers, group_layers=2 → 1 worker-group holding both layers locally.
This is the extreme case: the entire 2-layer chain runs on one worker, and
the only wire transfer is (group input, group target) — no per-layer
intermediate activations cross the wire.

Encoded in v2 by collapsing N_UB to the number of GROUPS (not layers):
    N_UB = 1, n_layers = 2 (held internally by the inner_step graph),
    weight shape per UB = (2, D, D) (packed [W0, W1] for the 2-layer group).

Compared to the un-grouped Stage 0 baseline (N_UB=2, two cuts shipped),
the group_layers=2 variant ships ~half the activation-wire bytes per round
because there's only one cut.
"""
from __future__ import annotations

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket


D = 16
B_TRAIN = 64
B_EVAL = 256
N_UB = 1                # 1 group
N_LAYERS_IN_GROUP = 2   # the group holds 2 consecutive layers
INNER_REPLICAS = 2
K_INNER = 4
INNER_LR = 0.001
OUTER_LR = 1.0
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


def build_forward_graph() -> Graph:
    """Forward computes the full 2-layer chain and emits ONE packed target
    of shape (2, B, D): input x and the LocoProp target for the group's
    output (which is the full network output)."""
    gb = GraphBuilder()
    weights = gb.input("weights_0", [N_LAYERS_IN_GROUP, D, D], DTYPE)
    gb.param("round_id", "int")

    w0 = gb.squeeze(gb.slice(weights, dim=0, start=0, end=1), dim=0)
    w1 = gb.squeeze(gb.slice(weights, dim=0, start=1, end=2), dim=0)

    x = gb.emit("normal", args=[],
                kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, D], "dtype": DTYPE})
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)

    h_1 = gb.matmul(x, w0)
    h_2 = gb.matmul(h_1, w1)

    err = gb.sub(h_2, y_true)
    dL_dh2 = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))

    target = gb.sub(h_2, dL_dh2)         # group-output target (B, D)

    pack = gb.stack([x, target], dim=0)  # (2, B, D)
    gb.output("target_0", pack)
    return gb.build()


def build_inner_graph() -> Graph:
    """Worker holds the full 2-layer group. Computes joint forward+backward
    on the held weights, runs K-step joint SGD, returns delta for both layers
    packed as (2, D, D)."""
    gb = GraphBuilder()
    weights = gb.input("weights", [N_LAYERS_IN_GROUP, D, D], DTYPE)
    packed = gb.input("target", [2, B_TRAIN, D], DTYPE)

    x_in = gb.squeeze(gb.slice(packed, dim=0, start=0, end=1), dim=0)
    target = gb.squeeze(gb.slice(packed, dim=0, start=1, end=2), dim=0)

    w0_curr = gb.squeeze(gb.slice(weights, dim=0, start=0, end=1), dim=0)
    w1_curr = gb.squeeze(gb.slice(weights, dim=0, start=1, end=2), dim=0)

    lr = gb.const(float(INNER_LR))
    inv_b = gb.const(1.0 / float(B_TRAIN))
    for _ in range(K_INNER):
        # Forward
        h1 = gb.matmul(x_in, w0_curr)
        h2 = gb.matmul(h1, w1_curr)
        # Loss = MSE(h2, target). dL/dh2 = (2/B)(h2 - target)
        resid = gb.sub(h2, target)
        dL_dh2 = gb.mul(resid, gb.const(2.0 / float(B_TRAIN)))
        # Gradients (no /B since we already factored it in dL_dh2)
        # dL/dW1 = h1.T @ dL/dh2
        h1_t = gb.transpose(h1, dims=[1, 0])
        gW1 = gb.matmul(h1_t, dL_dh2)
        # dL/dh1 = dL/dh2 @ W1.T
        w1_t = gb.transpose(w1_curr, dims=[1, 0])
        dL_dh1 = gb.matmul(dL_dh2, w1_t)
        # dL/dW0 = x.T @ dL/dh1
        x_t = gb.transpose(x_in, dims=[1, 0])
        gW0 = gb.matmul(x_t, dL_dh1)
        # SGD
        w0_curr = gb.sub(w0_curr, gb.mul(lr, gW0))
        w1_curr = gb.sub(w1_curr, gb.mul(lr, gW1))

    # Pack new weights and compute delta
    new_pack = gb.stack([w0_curr, w1_curr], dim=0)
    delta = gb.sub(new_pack, weights)
    gb.output("delta", delta)
    return gb.build()


def build_outer_graph() -> Graph:
    gb = GraphBuilder()
    w = gb.input("weights", [N_LAYERS_IN_GROUP, D, D], DTYPE)
    rd = gb.input("reduced_delta", [N_LAYERS_IN_GROUP, D, D], DTYPE)
    nw = gb.add(w, gb.mul(rd, gb.const(float(OUTER_LR))))
    gb.output("new_weights", nw)
    return gb.build()


def build_eval_graph() -> Graph:
    gb = GraphBuilder()
    weights = gb.input("weights_0", [N_LAYERS_IN_GROUP, D, D], DTYPE)
    w0 = gb.squeeze(gb.slice(weights, dim=0, start=0, end=1), dim=0)
    w1 = gb.squeeze(gb.slice(weights, dim=0, start=1, end=2), dim=0)
    x = gb.emit("normal", args=[],
                kwargs={"seed": EVAL_SEED, "shape": [B_EVAL, D], "dtype": DTYPE})
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)
    h = gb.matmul(gb.matmul(x, w0), w1)
    err = gb.sub(h, y_true)
    sq = gb.mul(err, err)
    out = gb.unsqueeze(gb.mean(sq), dim=0)
    gb.output("metrics", out)
    return gb.build()


def build_reduce_graph(n_inputs: int) -> Graph:
    gb = GraphBuilder()
    refs = []
    for i in range(n_inputs):
        ri = gb.input(f"d_{i}", [N_LAYERS_IN_GROUP, D, D], DTYPE)
        refs.append(gb.unsqueeze(ri, dim=0))
    stack = refs[0] if n_inputs == 1 else gb.concat(refs, dim=0)
    avg = gb.mean(stack, dim=0)
    gb.output("reduced", avg)
    return gb.build()


def initial_weights() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    scale = (1.0 / D) ** 0.5
    w0 = torch.empty(D, D).normal_(generator=g) * scale
    w1 = torch.empty(D, D).normal_(generator=g) * scale
    return torch.stack([w0, w1], dim=0)


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pluralis_grouped", "n_unique_blocks": N_UB,
         "n_layers_in_group": N_LAYERS_IN_GROUP, "d": D,
         "B_train": B_TRAIN, "B_eval": B_EVAL, "K_inner": K_INNER,
         "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
    )
    weights = initial_weights()
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 0)),
               tensor_io.encode_tensor(weights))


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
        t_max_sec=60.0,
    )
    return graphs, params
