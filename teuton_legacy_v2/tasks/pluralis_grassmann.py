"""Stage 15 — Grassmann updates of the subspace basis U_k.

A periodic Grassmann step on the orthonormal basis U_k aligns it with
recent gradient directions. This stage demonstrates the IR mechanism end
to end:

  1. The forward pass is "decorated" to also emit the cross-product
     `S = G^T G` of the final-block-output gradient G (used to form the
     Grassmann gradient direction).
  2. A separate `grassmann_step_graph` (exposed via the task module — not
     yet emitted as a job by the orchestrator in v2) takes (U_old, S) and
     returns U_new = QR(U_old - eta * tangent).
  3. The QR is performed in the IR via the new `qr` op.

In v2 the orchestrator does not emit a separate `grassmann_step` job;
instead, the task exposes `grassmann_step_graph()` so a test can run it
directly through `evaluate(...)` to demonstrate the math. Wiring the job
into the orchestrator's round loop is future work (it requires periodic-
job emission, a static-blob mutation pathway, and a `reproject_constrained`
job kind — out of scope for Stage 15 in v2).

The forward, inner, outer, eval graphs of this task are inherited from
Stage 14's `pluralis_full`; the new contribution here is the Grassmann
step graph and a unit-style test that:
  - Asserts `U_new` is orthonormal (U_new^T U_new ≈ I).
  - Asserts `U_new` differs from `U_old` after one step.
"""
from __future__ import annotations

import torch

from .. import paths, tensor_io
from ..ir import Graph, GraphBuilder, ref_param
from ..orchestrator import OrchestratorParams
from ..schedule import TaskGraphs
from ..storage import LocalBucket

# Reuse the Stage 14 building blocks for forward/inner/outer/eval/reduce.
from . import pluralis_full as _full


D = _full.D
SUBSPACE_K = _full.SUBSPACE_K
DTYPE = _full.DTYPE


# --------------------------------------------------------------------------- #
# Grassmann step graph
# --------------------------------------------------------------------------- #


def build_grassmann_step_graph() -> Graph:
    """Take (U_old: (D, k), S: (D, D)) and return (U_new: (D, k), R: (k, k)).

    Math (matches nanogpt_loco/pluralis_model.py's grassmann_step):
        grad   = -2 * S @ U_old                # (D, k)
        tan    = grad - U_old @ (U_old^T grad) # tangent at U_old
        step   = U_old - eta * tan
        U_new, R = qr(step)                    # retract via QR
    """
    gb = GraphBuilder()
    u_old = gb.input("U_old", [D, SUBSPACE_K], DTYPE)
    s = gb.input("S", [D, D], DTYPE)
    gb.param("eta", "float")

    eta = ref_param("eta")
    grad = gb.mul(gb.const(-2.0), gb.matmul(s, u_old))                # (D, k)
    u_t = gb.transpose(u_old, dims=[1, 0])
    proj = gb.matmul(u_old, gb.matmul(u_t, grad))                      # (D, k)
    tan = gb.sub(grad, proj)                                           # (D, k)
    step = gb.sub(u_old, gb.mul(eta, tan))                             # (D, k)
    Q, R = gb.qr(step)
    gb.output("U_new", Q)
    gb.output("R", R)
    return gb.build()


# --------------------------------------------------------------------------- #
# Bootstrap / orchestrator inputs delegate to Stage 14.
# --------------------------------------------------------------------------- #


def bootstrap(*, bucket: LocalBucket, run_id: str, max_rounds: int) -> None:
    _full.bootstrap(bucket=bucket, run_id=run_id, max_rounds=max_rounds)


def build_orchestrator_inputs(
    *, bucket: LocalBucket, run_id: str
) -> tuple[TaskGraphs, OrchestratorParams]:
    return _full.build_orchestrator_inputs(bucket=bucket, run_id=run_id)
