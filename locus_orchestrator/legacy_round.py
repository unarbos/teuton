"""V3-owned access to the preserved v2 round scheduler.

The subnet-native v3 run manager uses signed manifests and grants. This module
keeps the wider v2 round DAG implementation available for migration,
benchmark replay, and task validation without importing the old checkout.
"""
from __future__ import annotations

from locus_legacy_v2.orchestrator import Orchestrator, OrchestratorParams
from locus_legacy_v2.schedule import (  # noqa: F401
    TaskGraphs,
    build_eval_job,
    build_initial_round_jobs,
    build_outer_job,
    build_reduce_job,
    jid_eval,
    jid_forward,
    jid_forward_mb,
    jid_inner,
    jid_inner_mb,
    jid_outer,
    jid_reduce,
    round_completion_uris,
)

__all__ = [
    "Orchestrator",
    "OrchestratorParams",
    "TaskGraphs",
    "build_eval_job",
    "build_initial_round_jobs",
    "build_outer_job",
    "build_reduce_job",
    "jid_eval",
    "jid_forward",
    "jid_forward_mb",
    "jid_inner",
    "jid_inner_mb",
    "jid_outer",
    "jid_reduce",
    "round_completion_uris",
]
