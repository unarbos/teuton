"""Stage 12 — weight tying (G < L).

L = 4 layers, G = 2 unique blocks. The layer→UB mapping is:
    layer 0,1 → UB-0  (W0 applied twice in sequence)
    layer 2,3 → UB-1  (W1 applied twice in sequence)

Forward chains 4 layers with the tied weights:
    h_0 = x                                # input
    h_1 = h_0 @ W0                         # layer 0
    h_2 = h_1 @ W0                         # layer 1
    h_3 = h_2 @ W1                         # layer 2
    h_4 = h_3 @ W1                         # layer 3
    loss = MSE(h_4, y_true)

For each UB, the forward emits a stacked (M_layers_per_ub=2, 2, B, D) tensor:
    pack_ub[m, 0] = x_in_m  (layer m's input)
    pack_ub[m, 1] = target_m (layer m's locoprop target)

The inner step does *joint* K-step SGD: at each step it averages gradients
across the M tied layers' loss terms before updating W. This is the
"federated-averaging across tied positions" pattern.

Schedule is unchanged: n_unique_blocks=2 (each UB is one tied weight).
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
N_LAYERS = 4              # L
M_PER_UB = N_LAYERS // N_UB  # 2 — layers per UB (tying group size)
INNER_REPLICAS = 2
K_INNER = 4
INNER_LR = 0.0005
OUTER_LR = 1.0
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


def build_forward_graph() -> Graph:
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

    # 4-layer tied chain
    h_0 = x
    h_1 = gb.matmul(h_0, w0)         # layer 0 (UB-0)
    h_2 = gb.matmul(h_1, w0)         # layer 1 (UB-0)
    h_3 = gb.matmul(h_2, w1)         # layer 2 (UB-1)
    h_4 = gb.matmul(h_3, w1)         # layer 3 (UB-1)

    err = gb.sub(h_4, y_true)
    dL_dh4 = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))

    w1_t = gb.transpose(w1, dims=[1, 0])
    w0_t = gb.transpose(w0, dims=[1, 0])
    dL_dh3 = gb.matmul(dL_dh4, w1_t)
    dL_dh2 = gb.matmul(dL_dh3, w1_t)
    dL_dh1 = gb.matmul(dL_dh2, w0_t)
    dL_dh0 = gb.matmul(dL_dh1, w0_t)

    target_l0 = gb.sub(h_1, dL_dh1)
    target_l1 = gb.sub(h_2, dL_dh2)
    target_l2 = gb.sub(h_3, dL_dh3)
    target_l3 = gb.sub(h_4, dL_dh4)

    # Pack per UB. Each pack has shape (M=2, 2, B, D).
    def _pack(x_in_a, target_a, x_in_b, target_b):
        # one layer: stack [x_in, target] along axis 0 -> (2, B, D)
        layer_a = gb.stack([x_in_a, target_a], dim=0)
        layer_b = gb.stack([x_in_b, target_b], dim=0)
        # both layers: stack -> (2, 2, B, D)
        return gb.stack([layer_a, layer_b], dim=0)

    pack_0 = _pack(h_0, target_l0, h_1, target_l1)   # UB-0 (layers 0, 1)
    pack_1 = _pack(h_2, target_l2, h_3, target_l3)   # UB-1 (layers 2, 3)

    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """Joint K-step SGD across M tied layers' local losses, with grad-averaging."""
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed = gb.input("target", [M_PER_UB, 2, B_TRAIN, D], DTYPE)

    # Slice each layer's (x_in, target)
    layers = []
    for m in range(M_PER_UB):
        layer_pkd = gb.squeeze(gb.slice(packed, dim=0, start=m, end=m + 1), dim=0)   # (2, B, D)
        x_in = gb.squeeze(gb.slice(layer_pkd, dim=0, start=0, end=1), dim=0)
        target = gb.squeeze(gb.slice(layer_pkd, dim=0, start=1, end=2), dim=0)
        x_t = gb.transpose(x_in, dims=[1, 0])
        layers.append((x_in, x_t, target))

    w_curr = weights
    lr = gb.const(float(INNER_LR))
    inv_m = gb.const(1.0 / float(M_PER_UB))
    for _ in range(K_INNER):
        # Per-layer grad, then average
        grads = []
        for x_in, x_t, target in layers:
            y_pred = gb.matmul(x_in, w_curr)
            resid = gb.sub(y_pred, target)
            grads.append(gb.matmul(x_t, resid))
        if M_PER_UB == 1:
            avg_grad = grads[0]
        else:
            sum_grad = grads[0]
            for g in grads[1:]:
                sum_grad = gb.add(sum_grad, g)
            avg_grad = gb.mul(sum_grad, inv_m)
        w_curr = gb.sub(w_curr, gb.mul(lr, avg_grad))

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
    h = gb.matmul(gb.matmul(gb.matmul(gb.matmul(x, w0), w0), w1), w1)
    err = gb.sub(h, y_true)
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
    eye = torch.eye(D, dtype=torch.float32)
    w0 = eye + torch.empty(D, D).normal_(generator=g) * 0.02
    w1 = eye + torch.empty(D, D).normal_(generator=g) * 0.02
    return w0, w1


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pluralis_tied", "n_unique_blocks": N_UB, "n_layers": N_LAYERS,
         "m_per_ub": M_PER_UB, "d": D, "B_train": B_TRAIN, "B_eval": B_EVAL,
         "K_inner": K_INNER, "inner_lr": INNER_LR,
         "inner_replicas_per_ub": INNER_REPLICAS, "max_rounds": int(max_rounds)},
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
        t_max_sec=60.0,
    )
    return graphs, params
