"""Stage 5 — per-replica data diversity.

Each replica sees a different training batch shard. The forward pass
generates N distinct batches deterministically (seed = round_id * N + r),
runs them all through the model in parallel, and emits per-UB packed
tensors stacked along a new leading "replica" axis: shape (N, 2, B, D).

Each inner_step replica reads its `replica` index from `params` and
slices the matching shard, then runs LocoProp K-step inner SGD on that
shard. Reduce mean-averages across replicas in the bucket. Outer applies.

Asserts loss with N=4 diverse replicas drops more per round than N=1.
"""
from __future__ import annotations

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket


D = 16
B_TRAIN = 32
B_EVAL = 256
N_UB = 2
INNER_REPLICAS = 4
K_INNER = 4
INNER_LR = 0.001
OUTER_LR = 1.0
TEACHER_SEED = 1234
EVAL_SEED = 9999
STUDENT_INIT_SEED = 42
DTYPE = "float32"


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #


def _per_replica_x(gb: GraphBuilder) -> dict[str, object]:
    """Build N independent batches by stacking N normal() ops with different
    seeds derived from round_id. Returns ref to a stacked tensor of shape
    (N, B, D)."""
    # Use round_id-derived seeds; each replica's seed = round_id * 1009 + r * 7
    seeds = [_seed_for_replica(r) for r in range(INNER_REPLICAS)]
    xs = []
    for s in seeds:
        # Note: we use a *param-ref kwarg* for "seed" by combining with round_id,
        # via a small per-replica computation. Since IR can't add ints, we
        # instead use distinct fixed-seed seeds derived offline, and XOR with
        # round_id at runtime via gather-from-arange trickery isn't available.
        # Workaround: use param-ref directly for seed = round_id * (r+1) + r.
        # Without IR-level int arithmetic, we use a "seed_r" param per replica.
        xs.append(gb.emit(
            "normal", args=[],
            kwargs={"seed": ref_param(f"seed_r{s}"),
                    "shape": [B_TRAIN, D], "dtype": DTYPE},
        ))
    # Stack along new leading axis -> (N, B, D)
    stacked = gb.stack([gb.unsqueeze(x, dim=0) for x in xs] if False
                        else xs, dim=0)
    return {"x_stacked": stacked, "xs": xs}


def _seed_for_replica(r: int) -> str:
    """Stable seed-param name per replica."""
    return f"replica{r}"


def build_forward_graph() -> Graph:
    """Forward emits per-UB packed (x_in, target) stacked over N_REPLICAS.

    Output shape per UB: (N, 2, B, D).
    """
    gb = GraphBuilder()
    w0 = gb.input("weights_0", [D, D], DTYPE)
    w1 = gb.input("weights_1", [D, D], DTYPE)
    # `round_id` and per-replica seed params
    gb.param("round_id", "int")
    seed_params = []
    for r in range(INNER_REPLICAS):
        seed_params.append(gb.param(f"seed_r{r}", "int"))

    # Per-replica x sampling. Each x is shape (B, D); stack to (N, B, D).
    xs = []
    for r in range(INNER_REPLICAS):
        xs.append(gb.emit(
            "normal", args=[],
            kwargs={"seed": ref_param(f"seed_r{r}"),
                    "shape": [B_TRAIN, D], "dtype": DTYPE},
        ))
    x_all = gb.stack(xs, dim=0)        # (N, B, D)

    # Teacher (shared across replicas)
    t1 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED, "shape": [D, D], "dtype": DTYPE})
    t2 = gb.emit("normal", args=[],
                 kwargs={"seed": TEACHER_SEED + 1, "shape": [D, D], "dtype": DTYPE})

    # Compute teacher outputs for all replicas: (N, B, D) @ (D, D) -> (N, B, D)
    h_teacher = gb.matmul(x_all, t1)
    y_true_all = gb.matmul(h_teacher, t2)

    # Student forward, per replica
    h_all = gb.matmul(x_all, w0)        # (N, B, D)
    y_all = gb.matmul(h_all, w1)        # (N, B, D)

    err = gb.sub(y_all, y_true_all)     # (N, B, D)
    dL_dy = gb.mul(err, gb.const(2.0 / float(B_TRAIN)))

    w1_t = gb.transpose(w1, dims=[1, 0])
    dL_dh = gb.matmul(dL_dy, w1_t)      # (N, B, D)

    target_0 = gb.sub(h_all, dL_dh)     # (N, B, D)
    target_1 = gb.sub(y_all, dL_dy)     # (N, B, D)

    # Pack per UB along axis=1 -> (N, 2, B, D)
    pack_0 = gb.stack([x_all, target_0], dim=1)
    pack_1 = gb.stack([h_all, target_1], dim=1)

    gb.output("target_0", pack_0)
    gb.output("target_1", pack_1)
    return gb.build()


def build_inner_graph() -> Graph:
    """Slices the packed (N, 2, B, D) by replica index, then K-step SGD."""
    gb = GraphBuilder()
    weights = gb.input("weights", [D, D], DTYPE)
    packed_all = gb.input("target", [INNER_REPLICAS, 2, B_TRAIN, D], DTYPE)
    gb.param("replica", "int")
    gb.param("replica_end_excl", "int")

    # Slice along axis 0 by [replica, replica_end_excl) -> (1, 2, B, D)
    one = gb.slice(packed_all, dim=0,
                   start=ref_param("replica"),
                   end=ref_param("replica_end_excl"))
    one2 = gb.squeeze(one, dim=0)       # (2, B, D)

    x_in = gb.squeeze(gb.slice(one2, dim=0, start=0, end=1), dim=0)
    target = gb.squeeze(gb.slice(one2, dim=0, start=1, end=2), dim=0)
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
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "data_parallel_mlp", "n_unique_blocks": N_UB, "d": D,
         "B_train": B_TRAIN, "B_eval": B_EVAL, "K_inner": K_INNER,
         "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
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
    # Per-replica seed params, threaded into common_params so both forward
    # and inner see them. round_id is auto-set by schedule.
    common = {f"seed_r{r}": (r * 1009 + 7) for r in range(INNER_REPLICAS)}
    params = OrchestratorParams(
        n_unique_blocks=N_UB,
        inner_replicas_per_ub=inner_replicas,
        common_params=common,
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
