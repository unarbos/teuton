from __future__ import annotations

import json

import torch

from locus_core.ir import Graph, GraphBuilder, ref_param
from locus_runtime import tensor_io
from locus_runtime.eval import evaluate


def test_tensor_io_v2_dtype_round_trips() -> None:
    tensors = [
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        torch.tensor([True, False, True, True], dtype=torch.bool),
        torch.tensor([1.5, -0.25, 3.0], dtype=torch.bfloat16),
    ]
    for tensor in tensors:
        decoded = tensor_io.decode_tensor(tensor_io.encode_tensor(tensor))
        assert decoded.dtype == tensor.dtype
        assert list(decoded.shape) == list(tensor.shape)
        assert torch.equal(decoded, tensor)


def test_v2_ir_hash_and_param_seed_semantics() -> None:
    gb = GraphBuilder()
    gb.param("seed", "int")
    out = gb.emit("normal", args=[], kwargs={"seed": ref_param("seed"), "shape": [4], "dtype": "float32"})
    gb.output("y", out)
    graph = gb.build()

    graph2 = Graph.from_dict(json.loads(graph.to_canonical_json()))
    assert graph2.graph_id() == graph.graph_id()
    a = evaluate(graph2, {}, {"seed": 1234})["y"]
    b = evaluate(graph2, {}, {"seed": 1234})["y"]
    c = evaluate(graph2, {}, {"seed": 9999})["y"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_legacy_v2_task_catalog_imports() -> None:
    import locus_legacy_v2.tasks.adam_mlp as adam_mlp
    import locus_legacy_v2.tasks.gpt_pipe as gpt_pipe
    import locus_tasks.adam_mlp as public_adam_mlp
    import locus_tasks.gpt_pipe as public_gpt_pipe

    assert adam_mlp.build_forward_graph().graph_id()
    assert public_adam_mlp.build_forward_graph().graph_id() == adam_mlp.build_forward_graph().graph_id()
    assert hasattr(gpt_pipe, "build_streaming_inputs")
    assert hasattr(public_gpt_pipe, "build_streaming_inputs")
