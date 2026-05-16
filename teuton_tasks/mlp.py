"""Small deterministic MLP task for Teuton v3 smoke and subnet testing."""
from __future__ import annotations

import torch

from teuton_core.ir import Graph, GraphBuilder, ref_param

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


def build_forward_graph() -> Graph:
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    gb.param("round_id", "int")
    x = gb.emit("normal", args=[], kwargs={"seed": ref_param("round_id"), "shape": [B_TRAIN, D], "dtype": DTYPE})
    t1 = gb.emit("normal", args=[], kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[], kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)
    h = gb.matmul(x, w0)
    y = gb.matmul(h, w1)
    err = gb.sub(y, y_true)
    dL_dy = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))
    dL_dW2 = gb.matmul(gb.transpose(h, dims=[1, 0]), dL_dy)
    dL_dh = gb.matmul(dL_dy, gb.transpose(w1, dims=[1, 0]))
    dL_dW1 = gb.matmul(gb.transpose(x, dims=[1, 0]), dL_dh)
    gb.output("target_0", gb.neg(dL_dW1))
    gb.output("target_1", gb.neg(dL_dW2))
    return gb.build()


def build_inner_graph() -> Graph:
    gb = GraphBuilder()
    gb.input("weights", [D, D], DTYPE)
    target = gb.input("target", [D, D], DTYPE)
    gb.output("delta", gb.mul(target, gb.const(float(INNER_LR))))
    return gb.build()


def build_reduce_graph(n_inputs: int) -> Graph:
    gb = GraphBuilder()
    refs = []
    for i in range(n_inputs):
        refs.append(gb.unsqueeze(gb.input(f"d_{i}", [D, D], DTYPE), dim=0))
    stack = refs[0] if len(refs) == 1 else gb.concat(refs, dim=0)
    gb.output("reduced", gb.mean(stack, dim=0))
    return gb.build()


def build_outer_graph() -> Graph:
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    reduced = gb.input("reduced_delta", [D, D], DTYPE)
    gb.output("new_weights", gb.add(weights, gb.mul(reduced, gb.const(float(OUTER_LR)))))
    return gb.build()


def build_eval_graph() -> Graph:
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    x = gb.emit("normal", args=[], kwargs={"seed": EVAL_SEED, "shape": [B_EVAL, D], "dtype": DTYPE})
    t1 = gb.emit("normal", args=[], kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[], kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})
    y_true = gb.matmul(gb.matmul(x, t1), t2)
    err = gb.sub(gb.matmul(gb.matmul(x, w0), w1), y_true)
    gb.output("metrics", gb.unsqueeze(gb.mean(gb.mul(err, err)), dim=0))
    return gb.build()


def initial_weights() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(STUDENT_INIT_SEED)
    scale = (1.0 / D) ** 0.5
    return (
        torch.empty(D, D).normal_(generator=g) * scale,
        torch.empty(D, D).normal_(generator=g) * scale,
    )


def graph_bundle() -> dict[str, Graph]:
    return {
        "forward": build_forward_graph(),
        "inner": build_inner_graph(),
        "outer": build_outer_graph(),
        "eval": build_eval_graph(),
    }
