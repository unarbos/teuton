"""Stage 4 — AdamWLocoSign outer optimizer.

Builds on Stage 3 (AdamW outer with persistent state). The forward graph
now emits a *second* per-UB tensor — the actual backprop gradient `bp_grad`
— alongside the LocoProp packed (x_in, target). The orchestrator threads
this as an extra forward output via `forward_extra_outputs = ["bp_grad"]`,
making it accessible to the outer step.

Sign-Loco update rule:
    magnitude_per_elem = AdamW(bp_grad).abs   (Adam's per-element step size)
    direction_per_elem = -sign(reduced_delta) (LocoProp displacement direction,
                                                negated since reduced_delta is
                                                the descent direction)
    update = magnitude_per_elem * direction_per_elem
    new_weights = weights - update            (note: minus, since direction
                                                already encodes "go negative
                                                gradient")
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
N_UB = 2
INNER_REPLICAS = 2
K_INNER = 4
INNER_LR = 0.001
ADAM_LR = 0.05
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


def build_forward_graph() -> Graph:
    """Forward emits two outputs per UB:
      - target_<ub>: packed (2, B, D) of (x_in, locoprop_target)
      - bp_grad_<ub>: actual BP gradient w.r.t. that UB's weights.
    """
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")

    x = gb.emit("normal", args=[],
                kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, D], "dtype": DTYPE})
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

    pack_0 = gb.concat([gb.unsqueeze(x, dim=0), gb.unsqueeze(target_0, dim=0)], dim=0)
    pack_1 = gb.concat([gb.unsqueeze(h, dim=0), gb.unsqueeze(target_1, dim=0)], dim=0)

    # Actual BP gradients — used as the "magnitude" signal in sign-Loco.
    h_t = gb.transpose(h, dims=[1, 0])
    bp_grad_1 = gb.matmul(h_t, dL_dy)
    x_t = gb.transpose(x, dims=[1, 0])
    bp_grad_0 = gb.matmul(x_t, dL_dh)

    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    gb.output("bp_grad_0", bp_grad_0)
    gb.output("bp_grad_1", bp_grad_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """Standard locoprop inner; ignores bp_grad — it's only used by outer."""
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed = gb.input("target", [2, B_TRAIN, D], DTYPE)
    # bp_grad is plumbed through schedule as an extra input; bind it but don't
    # use it in the inner step.
    _ = gb.input("bp_grad", [D, D], DTYPE)
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
    """Sign-Loco outer:
        m, v <- AdamW EMAs over bp_grad (not reduced_delta!)
        magnitude = m_hat / (sqrt(v_hat) + eps)   (per-element)
        direction = -sign(reduced_delta)          (per-element)
        new_w = w - lr * |magnitude| * direction
    """
    gb = GraphBuilder()
    w = gb.input("weights", [D, D], DTYPE)
    m = gb.input("m", [D, D], DTYPE)
    v = gb.input("v", [D, D], DTYPE)
    rd = gb.input("reduced_delta", [D, D], DTYPE)
    bp = gb.input("bp_grad", [D, D], DTYPE)
    gb.param("round_id", "int")

    beta1 = gb.const(float(ADAM_BETA1))
    beta2 = gb.const(float(ADAM_BETA2))
    one_minus_b1 = gb.const(1.0 - float(ADAM_BETA1))
    one_minus_b2 = gb.const(1.0 - float(ADAM_BETA2))

    new_m = gb.add(gb.mul(beta1, m), gb.mul(one_minus_b1, bp))
    new_v = gb.add(gb.mul(beta2, v), gb.mul(one_minus_b2, gb.mul(bp, bp)))

    eps = gb.const(float(ADAM_EPS))
    lr = gb.const(float(ADAM_LR))

    denom = gb.add(gb.sqrt(new_v), eps)
    magnitude = gb.abs_(gb.div(new_m, denom))     # per-element
    direction = gb.neg(gb.sign(rd))               # rd is descent direction
    update = gb.mul(magnitude, direction)
    new_w = gb.sub(w, gb.mul(lr, update))

    gb.output("new_weights", new_w)
    gb.output("new_m", new_m)
    gb.output("new_v", new_v)
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
    out = gb.unsqueeze(gb.mean(sq), dim=0)
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


def initial_weights() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    scale = (1.0 / D) ** 0.5
    w0 = torch.empty(D, D).normal_(generator=g) * scale
    w1 = torch.empty(D, D).normal_(generator=g) * scale
    return w0, w1


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "sign_loco_mlp", "n_unique_blocks": N_UB, "d": D,
         "B_train": B_TRAIN, "B_eval": B_EVAL, "K_inner": K_INNER,
         "inner_lr": INNER_LR, "adam_lr": ADAM_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
    )
    w0, w1 = initial_weights()
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 0)),
               tensor_io.encode_tensor(w0))
    bucket.put(bucket.uri_for_key(paths.weights_key(run_id, 0, 1)),
               tensor_io.encode_tensor(w1))
    zero = torch.zeros(D, D, dtype=torch.float32)
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
        outer_extra_state=["m", "v"],
        forward_extra_outputs=["bp_grad"],
    )
    return graphs, params
