"""Streaming-pipeline demo task — minimum viable for utilization measurement.

4 stages, each does a (D, D) matmul + nonlinearity. Microbatches of shape
(B, T, D) flow through the pipeline. No backward, no training — purely a
forward "ping" pipeline whose ONLY purpose is to demonstrate that the
streaming orchestrator achieves >> 5% utilization.

Stage 0: synthesize input from `mb_seed` param + initial weights matmul
Stage 1, 2: matmul + silu
Stage 3 (tail): matmul + write final activation + done marker

Each stage's compute is intentionally non-trivial (D=512 matmul, K_INNER=4
sequential matmuls) so utilization measurements aren't dominated by Python
overhead.
"""
from __future__ import annotations

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..streaming import PipelineStage, StreamingParams
from ..storage import LocalBucket


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #

D = 4096               # bigger D so GPU compute (matmul) is meaningful
B = 8
T = 64
N_STAGES = 6           # 6-deep pipeline
# 61 GPUs / 6 stages ≈ 10 workers per stage. With 60 microbatches in flight,
# every stage worker processes ~6 microbatches per epoch (saturated pipeline).
N_MICROBATCHES = 60
MAX_EPOCHS = 3
# Per stage: 80 matmuls of (B*T=512, D=4096) × (D, D) = 80 × 6.4 GFLOP = 0.5 TFLOP.
# H200 (~1000 TFLOPS bf16): ~0.5 ms per stage — IO dominates.
# 3090 (~35 TFLOPS bf16): ~15 ms per stage — IO still dominates.
# Wire per stage transition: B*T*D*4 = 8 MB (fp32).
INNER_MATMULS = 80
DTYPE = "float32"      # keep fp32 for now (IR stricter on dtypes)
WEIGHTS_SEED = 4242


# --------------------------------------------------------------------------- #
# Per-stage graphs
# --------------------------------------------------------------------------- #


def _stage_weights_blob(gb, run_id: str, bucket: str, stage: int):
    return gb.const_blob(
        f"s3://{bucket}/{paths.static_blob_key(run_id, f'stage_{stage}_W')}",
        shape=[D, D], dtype=DTYPE,
    )


def build_stage_graph(*, bucket: str, run_id: str, stage: int) -> Graph:
    """Stage-K forward graph.

    Stage 0 takes no input (synthesizes from `mb_seed` param).
    Stages 1..N take `x` (B,T,D) input from previous stage.
    All stages emit `x` (B,T,D) output for next stage.
    """
    gb = GraphBuilder()
    gb.param("mb_seed", "int")
    w = _stage_weights_blob(gb, run_id, bucket, stage)

    if stage == 0:
        x = gb.emit("normal", args=[],
                     kwargs={"seed": ref_param("mb_seed"),
                             "shape": [B, T, D], "dtype": DTYPE})
    else:
        x = gb.input("x", [B, T, D], DTYPE)

    # Synthetic compute load: K sequential matmul + silu
    h = x
    for _ in range(INNER_MATMULS):
        h = gb.matmul(h, w)
        h = gb.silu(h)
    gb.output("x", h)
    return gb.build()


def build_done_graph(*, bucket: str, run_id: str) -> Graph:
    """Tail-stage adds a `done` marker output (just a small JSON payload).

    Built as a separate graph that takes x and emits both x and a tiny
    constant tensor to mark completion. We use the metric-style JSON output
    convention by checking ref.uri.endswith('.json') in the worker.
    """
    gb = GraphBuilder()
    gb.param("mb_seed", "int")
    w = _stage_weights_blob(gb, run_id, bucket, N_STAGES - 1)
    x = gb.input("x", [B, T, D], DTYPE)
    h = x
    for _ in range(INNER_MATMULS):
        h = gb.matmul(h, w)
        h = gb.silu(h)
    gb.output("x", h)
    # `done` is a 1-element tensor written to a .json URI by the worker.
    done = gb.unsqueeze(gb.mean(h), dim=0)
    gb.output("done", done)
    return gb.build()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _initial_weights() -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(WEIGHTS_SEED)
    out = []
    scale = (1.0 / D) ** 0.5
    for _ in range(N_STAGES):
        w = torch.empty(D, D).normal_(generator=g) * scale
        out.append(w)
    return out


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    bucket.put_json(
        bucket.uri_for_key(paths.state_key(run_id)),
        {"run_id": run_id, "current_round": 0, "max_rounds": int(max_rounds),
         "completed_rounds": [], "failed_rounds": []},
    )
    bucket.put_json(
        bucket.uri_for_key(paths.manifest_config_key(run_id)),
        {"task": "pipe_demo", "n_stages": N_STAGES, "d": D, "B": B, "T": T,
         "n_microbatches": N_MICROBATCHES, "max_epochs": int(max_rounds)},
    )
    weights = _initial_weights()
    for s, w in enumerate(weights):
        bucket.put(bucket.uri_for_key(paths.static_blob_key(run_id, f"stage_{s}_W")),
                   tensor_io.encode_tensor(w))


def build_streaming_inputs(*, bucket: LocalBucket, run_id: str):
    cfg_uri = bucket.uri_for_key(paths.manifest_config_key(run_id))
    cfg = bucket.get_json(cfg_uri) if bucket.exists(cfg_uri) else {}
    max_epochs = int(cfg.get("max_epochs", MAX_EPOCHS))
    n_microbatches = int(cfg.get("n_microbatches", N_MICROBATCHES))

    stages: list[PipelineStage] = []
    for s in range(N_STAGES):
        if s == N_STAGES - 1:
            g = build_done_graph(bucket=bucket.bucket, run_id=run_id)
        else:
            g = build_stage_graph(bucket=bucket.bucket, run_id=run_id, stage=s)
        # Specs: stage 0 has no input, stages 1+ take x; all emit x
        if s == 0:
            in_specs: list[tuple[str, list[int], str]] = []
        else:
            in_specs = [("x", [B, T, D], DTYPE)]
        out_specs = [("x", [B, T, D], DTYPE)]
        stages.append(PipelineStage(
            stage_id=s,
            forward_graph=g,
            forward_input_specs=in_specs,
            forward_output_specs=out_specs,
        ))

    params = StreamingParams(
        n_stages=N_STAGES,
        n_microbatches=n_microbatches,
        max_epochs=max_epochs,
    )
    return stages, params
