"""Streaming-pipeline training task — forward + backward + outer step.

Architecture (linear chain of S matmul stages, no nonlinearity):

    x_0 ---W_0---> x_1 ---W_1---> ... ---W_{S-1}---> x_S
                                                       |
                                                    loss = MSE(x_S, target)

Each stage K has weights W_k (D, D). The pipeline streams M microbatches
through the forward chain, then back through the backward chain. After all
microbatches' backward complete, a per-stage outer step aggregates all M
dW_k gradients and updates W_k. Then the next epoch starts with the new
weights.

Why linear (no nonlinearity)? Keeps the backward graph tiny (~10 ops per
stage instead of ~30 with silu derivative) so the IR stays simple. We
still learn — initial random W_k and a fixed target make this a real
optimization problem with a clear loss curve.

Static state per epoch:
    runs/<id>/weights/epoch=E/stage_K_W.bin    # (D, D) fp32
    runs/<id>/static/target.bin                 # (D,) fp32, fixed across run

Per microbatch m (in epoch E):
    runs/<id>/streaming/epoch=E/stage=K/outputs/mb=M/x.bin       # (B, T, D)
    runs/<id>/streaming/epoch=E/stage=K/outputs/mb=M/done.json   # tail only
    runs/<id>/streaming/epoch=E/stage=K/bwd/mb=M/dL_dx.bin       # (B, T, D)
    runs/<id>/streaming/epoch=E/stage=K/bwd/mb=M/dW.bin          # (D, D)
    runs/<id>/streaming/epoch=E/stage=K/loss/mb=M/loss.json      # tail loss

After backward fully drains:
    runs/<id>/streaming/epoch=E/outer/stage=K/applied.json       # marker
    runs/<id>/weights/epoch=E+1/stage_K_W.bin                    # new weights
"""
from __future__ import annotations

import math

import torch

from locus_legacy_v2 import paths, tensor_io
from locus_core.ir import Graph, GraphBuilder, ref_param
from locus_legacy_v2.streaming import PipelineStage, StreamingParams
from locus_runtime.storage import LocalBucket


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #

D = 512
B = 4
T = 8
N_STAGES = 4
N_MICROBATCHES = 32        # 32 in flight, ~8 per stage with 39 workers
MAX_EPOCHS = 5
LR = 0.1                   # aggressive — we want visible loss decrease in 5 epochs
DTYPE = "float32"
WEIGHTS_SEED = 4242
TARGET_SEED = 9999
INIT_SCALE = 0.05          # close-to-identity start


# --------------------------------------------------------------------------- #
# Per-stage graphs
# --------------------------------------------------------------------------- #


def build_fwd_graph(*, stage: int, is_tail: bool) -> Graph:
    """Forward stage K: x_out = x_in @ W_k.

    Stage 0 synthesizes x_in from `mb_seed`. Tail stage also writes
    `loss` (a 1-element tensor = MSE(x_out, target)).
    """
    gb = GraphBuilder()
    gb.param("mb_seed", "int")
    w = gb.input("W", [D, D], DTYPE)

    if stage == 0:
        x = gb.emit("normal", args=[],
                     kwargs={"seed": ref_param("mb_seed"),
                             "shape": [B, T, D], "dtype": DTYPE})
    else:
        x = gb.input("x", [B, T, D], DTYPE)
    x_out = gb.matmul(x, w)
    gb.output("x", x_out)

    if is_tail:
        target = gb.input("target", [D], DTYPE)
        # broadcast target to (B, T, D)
        target_btd = gb.broadcast(
            gb.reshape(target, shape=[1, 1, D]),
            shape=[B, T, D],
        )
        diff = gb.sub(x_out, target_btd)
        sq = gb.mul(diff, diff)
        loss_scalar = gb.mean(sq)
        gb.output("loss", gb.unsqueeze(loss_scalar, dim=0))
    return gb.build()


def build_bwd_graph(*, stage: int, is_tail: bool, is_head: bool) -> Graph:
    """Backward stage K.

    Inputs:
      - W (D, D)
      - x_in (B, T, D) for non-head stages; head re-synthesizes from `mb_seed`.
      - tail: also reads `target` (D,)
      - non-tail: also reads `dL_dx_out` (B, T, D) from upstream bwd
    Outputs:
      - dW (D, D)
      - non-head: dL_dx_in (B, T, D)
    """
    gb = GraphBuilder()
    gb.param("mb_seed", "int")
    w = gb.input("W", [D, D], DTYPE)

    if is_head:
        # Re-synthesize x_in from the same seed used in fwd.
        x_in = gb.emit("normal", args=[],
                        kwargs={"seed": ref_param("mb_seed"),
                                "shape": [B, T, D], "dtype": DTYPE})
    else:
        x_in = gb.input("x_in", [B, T, D], DTYPE)

    if is_tail:
        target = gb.input("target", [D], DTYPE)
        x_out = gb.matmul(x_in, w)
        target_btd = gb.broadcast(
            gb.reshape(target, shape=[1, 1, D]),
            shape=[B, T, D],
        )
        inv_n = gb.const(2.0 / float(B * T * D))
        dL_dx_out = gb.mul(gb.sub(x_out, target_btd), inv_n)
    else:
        dL_dx_out = gb.input("dL_dx_out", [B, T, D], DTYPE)

    dW = gb.einsum(x_in, dL_dx_out, equation="btd,bte->de")
    gb.output("dW", dW)

    if not is_head:
        w_t = gb.transpose(w, dims=[1, 0])
        dL_dx_in = gb.matmul(dL_dx_out, w_t)
        gb.output("dL_dx_in", dL_dx_in)
    return gb.build()


def build_outer_graph(n_microbatches: int) -> Graph:
    """Outer SGD: W_new = W - lr * mean(all dW_m for m in 0..M-1).

    Inputs: W (D, D), and dW_0..dW_{M-1} each (D, D).
    Output: W_new (D, D).
    """
    gb = GraphBuilder()
    w = gb.input("W", [D, D], DTYPE)
    dW_inputs = [
        gb.input(f"dW_{m}", [D, D], DTYPE) for m in range(n_microbatches)
    ]
    # Mean of dWs
    if n_microbatches == 1:
        dW_mean = dW_inputs[0]
    else:
        stacked = gb.stack(
            [gb.unsqueeze(d, dim=0) for d in dW_inputs], dim=0,
        )
        dW_mean = gb.mean(gb.squeeze(stacked, dim=1), dim=0)
    lr = gb.const(float(LR))
    new_w = gb.sub(w, gb.mul(lr, dW_mean))
    gb.output("W_new", new_w)
    return gb.build()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _initial_weights() -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(WEIGHTS_SEED)
    eye = torch.eye(D, dtype=torch.float32)
    out = []
    for _ in range(N_STAGES):
        noise = torch.empty(D, D).normal_(generator=g) * INIT_SCALE
        out.append(eye + noise)
    return out


def _make_target() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(TARGET_SEED)
    return torch.empty(D).normal_(generator=g) * 0.5


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pipe_train", "n_stages": N_STAGES, "d": D, "B": B, "T": T,
         "n_microbatches": N_MICROBATCHES, "max_epochs": int(max_rounds),
         "lr": LR},
    )
    weights = _initial_weights()
    for s, w in enumerate(weights):
        # Epoch 0 weights live under streaming/weights/epoch=0/
        uri = bucket.uri_for_key(
            f"runs/{run_id}/weights/epoch=0/stage_{s}_W.bin"
        )
        bucket.put(uri, tensor_io.encode_tensor(w))
    # Fixed target
    bucket.put(bucket.uri_for_key(f"runs/{run_id}/static/target.bin"),
               tensor_io.encode_tensor(_make_target()))


def build_streaming_inputs(*, bucket: LocalBucket, run_id: str):
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    cfg = bucket.get_json(cfg_uri) if bucket.exists(cfg_uri) else {}
    max_epochs = int(cfg.get("max_epochs", MAX_EPOCHS))
    n_microbatches = int(cfg.get("n_microbatches", N_MICROBATCHES))

    stages: list[PipelineStage] = []
    outer_g = build_outer_graph(n_microbatches)
    for s in range(N_STAGES):
        is_tail = (s == N_STAGES - 1)
        is_head = (s == 0)
        fwd_g = build_fwd_graph(stage=s, is_tail=is_tail)
        bwd_g = build_bwd_graph(stage=s, is_tail=is_tail, is_head=is_head)
        in_specs: list[tuple[str, list[int], str]] = []
        if not is_head:
            in_specs.append(("x", [B, T, D], DTYPE))
        out_specs = [("x", [B, T, D], DTYPE)]
        stages.append(PipelineStage(
            stage_id=s,
            forward_graph=fwd_g,
            backward_graph=bwd_g,
            outer_graph=outer_g,
            forward_input_specs=in_specs,
            forward_output_specs=out_specs,
            weights_input_name="W",
            weights_shape=[D, D],
            weights_dtype=DTYPE,
            backward_takes_loss_target=is_tail,
            backward_emits_dx_in=not is_head,
        ))

    target_uri = bucket.uri_for_key(f"runs/{run_id}/static/target.bin")
    params = StreamingParams(
        n_stages=N_STAGES,
        n_microbatches=n_microbatches,
        max_epochs=max_epochs,
        training=True,
        target_static_uri=target_uri,
        lr=LR,
    )
    return stages, params
