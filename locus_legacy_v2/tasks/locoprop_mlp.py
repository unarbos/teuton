"""Stage 2 — LocoProp MLP with K-step inner loop.

Linear 2-layer MLP, MSE loss against a linear teacher. The forward graph
emits per-UB LocoProp local targets `target_l = y_l - dL/dy_l` along with
the per-UB block input `x_in_l`, packed together as a (2, B, D) tensor:

    packed[0] = x_in
    packed[1] = target

The inner_step graph slices these out, then runs K=K_INNER SGD steps on
the local loss `½‖x_in @ W - target‖²` (no /B, so the per-step gradient
has natural scale and inner_lr has its conventional meaning), returning
the displacement W_K - W_0. Outer applies this directly.

Why packing? Schedule.py emits one URI per UB output from the forward step.
Packing keeps the schedule's per-cut shape uniform without forcing a
schedule change to support multiple tensors per cut.
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
K_INNER = 4
INNER_LR = 0.001     # without /B; small step keeps inter-layer interaction stable
OUTER_LR = 1.0
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #


def build_forward_graph() -> Graph:
    """Compute LocoProp local targets + block inputs, packed as (2, B, D).

    Forward grads use 2/B normalization (standard MSE).
    """
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")

    x = gb.emit(
        "normal", args=[],
        kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, D], "dtype": DTYPE},
    )
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)

    h = gb.matmul(x, w0)
    y = gb.matmul(h, w1)

    err = gb.sub(y, y_true)
    dL_dy = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))

    w1_t = gb.transpose(w1, dims=[1, 0])
    dL_dh = gb.matmul(dL_dy, w1_t)

    target_0 = gb.sub(h, dL_dh)
    target_1 = gb.sub(y, dL_dy)

    x_in_0_u = gb.unsqueeze(x, dim=0)
    target_0_u = gb.unsqueeze(target_0, dim=0)
    pack_0 = gb.concat([x_in_0_u, target_0_u], dim=0)

    x_in_1_u = gb.unsqueeze(h, dim=0)
    target_1_u = gb.unsqueeze(target_1, dim=0)
    pack_1 = gb.concat([x_in_1_u, target_1_u], dim=0)

    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """K=K_INNER SGD steps on the local loss ½‖x_in @ W - target‖²
    (no /B in the gradient — gives inner_lr its conventional scale)."""
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed = gb.input("target", [2, B_TRAIN, D], DTYPE)

    x_in = gb.squeeze(gb.slice(packed, dim=0, start=0, end=1), dim=0)
    target = gb.squeeze(gb.slice(packed, dim=0, start=1, end=2), dim=0)

    x_t = gb.transpose(x_in, dims=[1, 0])

    w_curr = weights
    lr = gb.const(float(INNER_LR))
    for _ in range(K_INNER):
        y_pred = gb.matmul(x_in, w_curr)
        resid = gb.sub(y_pred, target)
        grad = gb.matmul(x_t, resid)
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


def build_eval_graph() -> Graph:
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
            "task": "locoprop_mlp",
            "n_unique_blocks": N_UB,
            "d": D, "B_train": B_TRAIN, "B_eval": B_EVAL,
            "K_inner": K_INNER, "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
            "inner_replicas_per_ub": INNER_REPLICAS,
            "max_rounds": int(max_rounds),
        },
    )
    w0, w1 = initial_weights()
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 0)),
               tensor_io.encode_tensor(w0))
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 1)),
               tensor_io.encode_tensor(w1))


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
