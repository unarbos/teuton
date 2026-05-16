"""Round DAG builder.

Pure-MapReduce: every job is `assigned`. The orchestrator passes in the set
of graphs it wants to run (built by the task module) and the worker pin
table.

The round shape (per round R):

    forward_pass            (1 job, assigned to a designated worker)
       └─ for each UB i:
            inner_step replicas r=0..N-1  (N jobs, assigned to pinned workers)
                ↓
            reduce_i        (1 job, assigned; emitted by the orchestrator
                             AFTER quorum, with concrete delta input URIs)
                ↓
            outer_step_i    (1 job, assigned; writes weights/round=R+1/UB-i.bin
                             plus any extra optim_state outputs)
       └─ eval              (1 job, assigned)

`build_initial_round_jobs` emits the *non-reduce* jobs at round-start; the
reduce job for each UB is emitted later by the orchestrator once it has
observed enough deltas.

Outer-step extra state (e.g., Adam m & v, DeMo error-feedback residuals) is
threaded through `outer_extra_state` — a list of state-tensor names that
become both extra inputs (read from round R) and extra outputs (write to
round R+1) of the outer_step manifest. Bootstrap is responsible for writing
the initial round-0 values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .ir import Graph
from .paths import (
    job_manifest_key,
    jobs_index_key,
    metrics_key,
    optim_state_key,
    output_delta_key,
    reduced_delta_key,
    target_key,
    weights_key,
)
from .storage import join_uri
from .types import GraphRef, IORef, JobManifest


# --------------------------------------------------------------------------- #
# Per-task graph bundle
# --------------------------------------------------------------------------- #


@dataclass
class TaskGraphs:
    forward: Graph
    inner: Graph
    outer: Graph
    eval: Graph
    reduce_for_n: "callable"  # type: ignore[name-defined]
    # Optional: per-UB overrides. If provided (length must equal n_unique_blocks),
    # each UB gets its own graph; the corresponding default field is ignored.
    inner_per_ub: "list[Graph] | None" = None
    outer_per_ub: "list[Graph] | None" = None


# --------------------------------------------------------------------------- #
# Job-id helpers
# --------------------------------------------------------------------------- #


def jid_forward(round_id: int) -> str:
    return f"j-r{round_id}-fwd"


def jid_inner(round_id: int, ub: int, replica: int) -> str:
    return f"j-r{round_id}-ub{ub}-inner-r{replica}"


def jid_reduce(round_id: int, ub: int) -> str:
    return f"j-r{round_id}-ub{ub}-reduce"


def jid_outer(round_id: int, ub: int) -> str:
    return f"j-r{round_id}-ub{ub}-outer"


def jid_eval(round_id: int) -> str:
    return f"j-r{round_id}-eval"


# --------------------------------------------------------------------------- #
# Initial DAG
# --------------------------------------------------------------------------- #


def _round_robin(items: Sequence[str], i: int) -> str:
    if not items:
        raise ValueError("no workers available for assignment")
    return items[i % len(items)]


def jid_forward_mb(round_id: int, mb: int) -> str:
    return f"j-r{round_id}-mb{mb}-fwd"


def jid_inner_mb(round_id: int, mb: int, ub: int, replica: int) -> str:
    return f"j-r{round_id}-mb{mb}-ub{ub}-inner-r{replica}"


def build_initial_round_jobs(
    *,
    bucket: str,
    run_id: str,
    round_id: int,
    n_unique_blocks: int,
    inner_replicas_per_ub: int,
    workers: Sequence[str],
    pin_table: dict[int, list[str]],
    forward_worker: str,
    eval_worker: str,
    graphs: TaskGraphs,
    common_params: dict[str, Any] | None = None,
    inner_params: dict[str, Any] | None = None,
    outer_params: dict[str, Any] | None = None,
    eval_params: dict[str, Any] | None = None,
    forward_extra_outputs: Sequence[str] = (),
    inner_extra_outputs: Sequence[str] = (),
    outer_extra_state: Sequence[str] = (),
    deadline_seconds: int = 600,
    now_unix: int | None = None,
    n_microbatches: int = 1,
    forward_workers: Sequence[str] | None = None,
) -> tuple[list[JobManifest], dict[str, Graph]]:
    """Build the round's initial job manifests + the set of graphs they reference.

    `outer_extra_state`: list of state-tensor names threaded into outer_step.
        Each name `s` becomes an additional input URI
        `optim_state/round=R/UB-i/{s}.bin` and an additional output URI
        `optim_state/round=R+1/UB-i/{s}.bin`. Bootstrap must write round-0 values.

    `forward_extra_outputs`: list of additional per-UB tensor outputs from the
        forward pass (e.g., per-UB BP gradients for sign-Loco). Each name `s`
        becomes a per-UB URI `targets/round=R/cut-{ub}_{s}.bin`. Inner_step
        manifests automatically pull in the matching name.

    `inner_extra_outputs`: list of additional per-replica tensor outputs from
        inner_step (e.g., DeMo error-feedback residuals). Each name `s`
        produces a URI under `outputs/...`. The orchestrator's reduce step
        is unchanged; consumers that need these read them directly.
    """
    if not workers:
        raise ValueError("workers list is empty")
    common = dict(common_params or {})
    inner_p = dict(inner_params or {})
    outer_p = dict(outer_params or {})
    eval_p = dict(eval_params or {})
    fwd_extras = list(forward_extra_outputs)
    inner_extras = list(inner_extra_outputs)
    outer_state = list(outer_extra_state)
    now = int(now_unix if now_unix is not None else time.time())
    deadline = now + int(deadline_seconds)

    fwd_sha = graphs.forward.graph_id()
    eval_sha = graphs.eval.graph_id()

    # Per-UB inner / outer graphs (with default fallback)
    def _inner_for(ub: int) -> Graph:
        if graphs.inner_per_ub is not None:
            return graphs.inner_per_ub[ub]
        return graphs.inner

    def _outer_for(ub: int) -> Graph:
        if graphs.outer_per_ub is not None:
            return graphs.outer_per_ub[ub]
        return graphs.outer

    out_jobs: list[JobManifest] = []
    out_graphs: dict[str, Graph] = {
        fwd_sha: graphs.forward,
        eval_sha: graphs.eval,
    }
    # Add inner / outer graphs per UB (deduped by sha)
    inner_refs_per_ub: list[GraphRef] = []
    for ub in range(n_unique_blocks):
        g = _inner_for(ub)
        sha = g.graph_id()
        out_graphs[sha] = g
        inner_refs_per_ub.append(
            GraphRef(sha256=sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{sha}.json"))
        )
    outer_refs_per_ub: list[GraphRef] = []
    for ub in range(n_unique_blocks):
        g = _outer_for(ub)
        sha = g.graph_id()
        out_graphs[sha] = g
        outer_refs_per_ub.append(
            GraphRef(sha256=sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{sha}.json"))
        )

    fwd_graph_ref = GraphRef(sha256=fwd_sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{fwd_sha}.json"))
    eval_graph_ref = GraphRef(sha256=eval_sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{eval_sha}.json"))

    # ----- forward_pass (one per microbatch) -----
    # The forward graph is parameterized by `round_id` (used as the input-data
    # seed). For multi-microbatch we pass `round_id*1000 + mb` so each
    # microbatch sees a different random batch — yielding diverse gradient
    # signal that gets averaged in reduce.
    if forward_workers and n_microbatches > 1:
        fwd_pool = list(forward_workers)
    else:
        fwd_pool = [forward_worker]
    for mb in range(n_microbatches):
        assigned = fwd_pool[mb % len(fwd_pool)]
        fwd_inputs = [
            IORef(name=f"weights_{ub}", uri=join_uri(bucket, weights_key(run_id, round_id, ub)))
            for ub in range(n_unique_blocks)
        ]
        fwd_outputs = [
            IORef(name=f"target_{cut}",
                  uri=join_uri(bucket, target_key(run_id, round_id, cut, mb=mb)))
            for cut in range(n_unique_blocks)
        ]
        for s in fwd_extras:
            for ub in range(n_unique_blocks):
                extra_uri = join_uri(
                    bucket,
                    f"runs/{run_id}/targets/round={round_id}/mb={mb}/cut-{ub}_{s}.bin",
                )
                fwd_outputs.append(IORef(name=f"{s}_{ub}", uri=extra_uri))
        out_jobs.append(
            JobManifest(
                job_id=jid_forward_mb(round_id, mb) if n_microbatches > 1 else jid_forward(round_id),
                run_id=run_id,
                round_id=round_id,
                kind="forward_pass",
                graph_ref=fwd_graph_ref,
                # Seed = round_id * 1000 + mb for distinct random batches per mb
                params={**common, "round_id": round_id * 1000 + mb, "mb": mb},
                inputs=fwd_inputs,
                outputs=fwd_outputs,
                assigned_to=assigned,
                deadline_unix=deadline,
                created_unix=now,
            )
        )

    # ----- inner_step jobs (M microbatches × N_UB UBs × R replicas) -----
    for ub in range(n_unique_blocks):
        pinned = pin_table.get(ub) or list(workers)
        for mb in range(n_microbatches):
            for r in range(inner_replicas_per_ub):
                # Round-robin within (mb, replica) so different mbs land on
                # different workers as much as possible.
                slot = mb * inner_replicas_per_ub + r
                assignee = _round_robin(pinned, slot)
                jid = (jid_inner_mb(round_id, mb, ub, r)
                       if n_microbatches > 1 else jid_inner(round_id, ub, r))
                inputs = [
                    IORef(name="weights",
                          uri=join_uri(bucket, weights_key(run_id, round_id, ub))),
                    IORef(name="target",
                          uri=join_uri(bucket, target_key(run_id, round_id, ub, mb=mb))),
                ]
                for s in fwd_extras:
                    inputs.append(IORef(
                        name=s,
                        uri=join_uri(bucket, f"runs/{run_id}/targets/round={round_id}/mb={mb}/cut-{ub}_{s}.bin"),
                    ))
                outputs = [
                    IORef(name="delta",
                          uri=join_uri(bucket, output_delta_key(run_id, round_id, ub, assignee, mb=mb))),
                ]
                for s in inner_extras:
                    outputs.append(IORef(
                        name=s,
                        uri=join_uri(
                            bucket,
                            f"runs/{run_id}/outputs/round={round_id}/ub={ub}/mb={mb}/worker={assignee}/{s}.bin",
                        ),
                    ))
                out_jobs.append(
                    JobManifest(
                        job_id=jid,
                        run_id=run_id,
                        round_id=round_id,
                        kind="inner_step",
                        graph_ref=inner_refs_per_ub[ub],
                        params={**common, **inner_p, "ub": ub, "mb": mb,
                                "replica": r, "replica_end_excl": r + 1,
                                "n_replicas": inner_replicas_per_ub,
                                "n_microbatches": n_microbatches},
                        inputs=inputs,
                        outputs=outputs,
                        assigned_to=assignee,
                        deadline_unix=deadline,
                        created_unix=now,
                    )
                )

    # NB: outer_step + eval are NOT emitted here. They are emitted late by
    # the orchestrator (`build_outer_job` / `build_eval_job`) once the
    # required upstream outputs are observed. This lets us co-locate outer
    # with the worker that just produced the reduced delta (saves one GET
    # per UB per round at scale).
    return out_jobs, out_graphs


def build_outer_job(
    *,
    bucket: str,
    run_id: str,
    round_id: int,
    ub: int,
    outer_graph: Graph,
    assigned_to: str,
    common_params: dict[str, Any] | None = None,
    outer_params: dict[str, Any] | None = None,
    forward_extra_outputs: Sequence[str] = (),
    outer_extra_state: Sequence[str] = (),
    deadline_seconds: int = 600,
    now_unix: int | None = None,
) -> tuple[JobManifest, dict[str, Graph]]:
    """Late-emitted outer_step for one UB. Assigned to the worker that just
    produced the reduced delta — that worker has the tensor in its local
    output cache, so the GET on `reduced_delta` short-circuits."""
    now = int(now_unix if now_unix is not None else time.time())
    common = dict(common_params or {})
    outer_p = dict(outer_params or {})
    sha = outer_graph.graph_id()
    graph_ref = GraphRef(sha256=sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{sha}.json"))
    inputs = [
        IORef(name="weights",
              uri=join_uri(bucket, weights_key(run_id, round_id, ub))),
        IORef(name="reduced_delta",
              uri=join_uri(bucket, reduced_delta_key(run_id, round_id, ub))),
    ]
    for s in outer_extra_state:
        inputs.append(IORef(
            name=s,
            uri=join_uri(bucket, optim_state_key(run_id, round_id, ub, s)),
        ))
    for s in forward_extra_outputs:
        # Forward writes per-mb extras; outer reads from mb=0 (the canonical
        # target tensor for outer-step purposes — forward_extras are aux
        # signals that shouldn't differ across microbatches).
        inputs.append(IORef(
            name=s,
            uri=join_uri(bucket, f"runs/{run_id}/targets/round={round_id}/mb=0/cut-{ub}_{s}.bin"),
        ))
    outputs = [
        IORef(name="new_weights",
              uri=join_uri(bucket, weights_key(run_id, round_id + 1, ub))),
    ]
    for s in outer_extra_state:
        outputs.append(IORef(
            name=f"new_{s}",
            uri=join_uri(bucket, optim_state_key(run_id, round_id + 1, ub, s)),
        ))
    return JobManifest(
        job_id=jid_outer(round_id, ub),
        run_id=run_id,
        round_id=round_id,
        kind="outer_step",
        graph_ref=graph_ref,
        params={**common, **outer_p, "ub": ub, "round_id": round_id},
        inputs=inputs,
        outputs=outputs,
        assigned_to=assigned_to,
        deadline_unix=now + int(deadline_seconds),
        created_unix=now,
    ), {sha: outer_graph}


def build_eval_job(
    *,
    bucket: str,
    run_id: str,
    round_id: int,
    n_unique_blocks: int,
    eval_graph: Graph,
    assigned_to: str,
    common_params: dict[str, Any] | None = None,
    eval_params: dict[str, Any] | None = None,
    deadline_seconds: int = 600,
    now_unix: int | None = None,
) -> tuple[JobManifest, dict[str, Graph]]:
    """Late-emitted eval. All outers must be done first."""
    now = int(now_unix if now_unix is not None else time.time())
    common = dict(common_params or {})
    eval_p = dict(eval_params or {})
    sha = eval_graph.graph_id()
    graph_ref = GraphRef(sha256=sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{sha}.json"))
    inputs = [
        IORef(name=f"weights_{ub}",
              uri=join_uri(bucket, weights_key(run_id, round_id + 1, ub)))
        for ub in range(n_unique_blocks)
    ]
    outputs = [
        IORef(name="metrics", uri=join_uri(bucket, metrics_key(run_id, round_id))),
    ]
    return JobManifest(
        job_id=jid_eval(round_id),
        run_id=run_id,
        round_id=round_id,
        kind="eval",
        graph_ref=graph_ref,
        params={**common, **eval_p, "round_id": round_id + 1},
        inputs=inputs,
        outputs=outputs,
        assigned_to=assigned_to,
        deadline_unix=now + int(deadline_seconds),
        created_unix=now,
    ), {sha: eval_graph}


# --------------------------------------------------------------------------- #
# Late-emitted reduce job
# --------------------------------------------------------------------------- #


def build_reduce_job(
    *,
    bucket: str,
    run_id: str,
    round_id: int,
    ub: int,
    delta_uris: Sequence[str],
    reduce_graph: Graph,
    assigned_to: str,
    common_params: dict[str, Any] | None = None,
    reduce_params: dict[str, Any] | None = None,
    deadline_seconds: int = 600,
    now_unix: int | None = None,
) -> tuple[JobManifest, dict[str, Graph]]:
    now = int(now_unix if now_unix is not None else time.time())
    sha = reduce_graph.graph_id()
    graph_ref = GraphRef(sha256=sha, uri=join_uri(bucket, f"runs/{run_id}/graphs/{sha}.json"))
    common = dict(common_params or {})
    rp = dict(reduce_params or {})
    inputs = [
        IORef(name=f"d_{i}", uri=u)
        for i, u in enumerate(sorted(delta_uris))
    ]
    outputs = [
        IORef(name="reduced", uri=join_uri(bucket, reduced_delta_key(run_id, round_id, ub))),
    ]
    job = JobManifest(
        job_id=jid_reduce(round_id, ub),
        run_id=run_id,
        round_id=round_id,
        kind="reduce",
        graph_ref=graph_ref,
        params={**common, **rp, "ub": ub, "n_inputs": len(inputs)},
        inputs=inputs,
        outputs=outputs,
        assigned_to=assigned_to,
        deadline_unix=now + int(deadline_seconds),
        created_unix=now,
    )
    return job, {sha: reduce_graph}


# --------------------------------------------------------------------------- #
# Round-completion predicate
# --------------------------------------------------------------------------- #


def round_completion_uris(*, bucket: str, run_id: str, round_id: int, n_unique_blocks: int) -> list[str]:
    return [
        join_uri(bucket, weights_key(run_id, round_id + 1, ub))
        for ub in range(n_unique_blocks)
    ]
