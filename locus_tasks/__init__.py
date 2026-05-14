"""Task plugin registry for Locus v3."""
from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable


NATIVE_ROUND_TASKS = {"mlp"}
LEGACY_ROUND_TASKS = {
    "adam_mlp",
    "data_parallel_mlp",
    "legacy_mlp",
    "locoprop_mlp",
    "pluralis_asymmetric",
    "pluralis_demo",
    "pluralis_full",
    "pluralis_gpt_10M",
    "pluralis_gpt_10M_v3",
    "pluralis_grassmann",
    "pluralis_grouped",
    "pluralis_int8",
    "pluralis_lossless_wire",
    "pluralis_subspace",
    "pluralis_tied",
    "relu_mlp",
    "sign_loco_mlp",
    "tiny_gpt",
}
STREAMING_TASKS = {"gpt_pipe", "pipe_demo", "pipe_train"}
KNOWN_TASKS = NATIVE_ROUND_TASKS | LEGACY_ROUND_TASKS


@runtime_checkable
class RoundTask(Protocol):
    N_UB: int
    INNER_REPLICAS: int

    def graph_bundle(self): ...
    def initial_weights(self): ...
    def build_reduce_graph(self, n_inputs: int): ...


@runtime_checkable
class StreamingTask(Protocol):
    def bootstrap(self, *, bucket, run_id: str, max_rounds: int) -> None: ...
    def build_streaming_inputs(self, *, bucket, run_id: str): ...


def load_task(name: str):
    if name in STREAMING_TASKS:
        raise ValueError(
            f"task {name!r} is a streaming task. Its IR builders are available "
            "under locus_tasks and should be run through StreamingRunManager."
        )
    if name in LEGACY_ROUND_TASKS:
        raise ValueError(
            f"task {name!r} is preserved from v2 under locus_tasks.{name}, but "
            "the v3 RunManager supports only native round tasks. Use the "
            "`locus-v2-legacy` runner for v2-style round orchestration."
        )
    if name not in KNOWN_TASKS:
        known = sorted(KNOWN_TASKS | STREAMING_TASKS)
        raise ValueError(f"unknown v3 task {name!r}; known={known}")
    return importlib.import_module(f"locus_tasks.{name}")


def load_streaming_task(name: str):
    if name not in STREAMING_TASKS:
        raise ValueError(f"unknown v3 streaming task {name!r}; known={sorted(STREAMING_TASKS)}")
    return importlib.import_module(f"locus_tasks.{name}")
