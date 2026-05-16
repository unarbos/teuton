"""Tensor IR — the language Teuton ships with each job.

A `Graph` is a typed DAG of `Op`s over named tensor `Input`s and scalar
`Param`s, producing one or more named `Output`s. Graphs are pure; their
identity is the sha256 of a canonical JSON encoding.

`ir.py` defines the data structures + JSON codec + the registry of op
signatures (name, kwarg names). Op semantics live in `eval.py`.

Refs encode where a tensor comes from:
  - {"kind": "input",     "name": str}                              — graph input
  - {"kind": "op",        "id": int, "idx": int}                    — output of another op
  - {"kind": "param",     "name": str}                              — scalar param
  - {"kind": "const",     "value": JsonScalar | list}               — inline literal
  - {"kind": "const_blob","uri": str, "shape": [...], "dtype": str} — bucket-resident tensor

This file does not import torch, so the orchestrator-side graph-construction
path stays light.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------- #
# Type-system primitives (shape/dtype are protocol concepts, used for declared
# input/output specs; the evaluator double-checks at runtime).
# --------------------------------------------------------------------------- #


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TensorSpec":
        return TensorSpec(name=d["name"], shape=[int(x) for x in d["shape"]], dtype=d["dtype"])


@dataclass
class ParamSpec:
    name: str
    type: str  # "int" | "float" | "bool" | "str" | "list[int]" | "list[float]" | "list[str]"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ParamSpec":
        return ParamSpec(name=d["name"], type=d["type"])


# --------------------------------------------------------------------------- #
# Refs (where an operand comes from)
# --------------------------------------------------------------------------- #


def ref_input(name: str) -> dict[str, Any]:
    return {"kind": "input", "name": name}


def ref_op(op_id: int, idx: int = 0) -> dict[str, Any]:
    return {"kind": "op", "id": int(op_id), "idx": int(idx)}


def ref_param(name: str) -> dict[str, Any]:
    return {"kind": "param", "name": name}


def ref_const(value: Any) -> dict[str, Any]:
    return {"kind": "const", "value": value}


def ref_const_blob(uri: str, shape: Sequence[int], dtype: str) -> dict[str, Any]:
    """A pointer to a tensor stored in the bucket. Resolved at evaluator time
    by reading the URI and decoding via tensor_io. Used for fixed teachers,
    PE buffers, U_k bases, DCT bases, and any other static tensor that's too
    big to inline as a JSON literal but doesn't change between rounds."""
    return {
        "kind": "const_blob",
        "uri": uri,
        "shape": [int(s) for s in shape],
        "dtype": dtype,
    }


# --------------------------------------------------------------------------- #
# Op + Graph
# --------------------------------------------------------------------------- #


@dataclass
class Op:
    id: int
    op: str
    args: list[dict[str, Any]] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    out: list[TensorSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "op": self.op,
            "args": [dict(a) for a in self.args],
            "kwargs": dict(self.kwargs),
            "out": [t.to_dict() for t in self.out],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Op":
        return Op(
            id=int(d["id"]),
            op=d["op"],
            args=[dict(a) for a in d.get("args", [])],
            kwargs=dict(d.get("kwargs") or {}),
            out=[TensorSpec.from_dict(x) for x in d.get("out", [])],
        )


@dataclass
class Graph:
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    ir_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [{"name": o["name"], "ref": dict(o["ref"])} for o in self.outputs],
            "params": [p.to_dict() for p in self.params],
            "ops": [o.to_dict() for o in self.ops],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Graph":
        return Graph(
            ir_version=int(d.get("ir_version", 1)),
            inputs=[TensorSpec.from_dict(x) for x in d.get("inputs", [])],
            outputs=[{"name": o["name"], "ref": dict(o["ref"])} for o in d.get("outputs", [])],
            params=[ParamSpec.from_dict(x) for x in d.get("params", [])],
            ops=[Op.from_dict(x) for x in d.get("ops", [])],
        )

    def to_canonical_json(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def graph_id(self) -> str:
        return hashlib.sha256(self.to_canonical_json()).hexdigest()


# --------------------------------------------------------------------------- #
# Op registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OpSig:
    name: str
    n_args: int               # exact number of tensor args; -1 means variadic
    kwargs: tuple[str, ...]


_REGISTRY: dict[str, OpSig] = {}


def _register(name: str, n_args: int, kwargs: Iterable[str] = ()) -> None:
    _REGISTRY[name] = OpSig(name=name, n_args=n_args, kwargs=tuple(kwargs))


# Elementwise (binary)
for _n in ("add", "sub", "mul", "div", "pow"):
    _register(_n, 2)
# Elementwise (unary)
for _n in ("neg", "exp", "log", "sqrt", "abs", "round",
           "relu", "gelu", "silu", "sin", "cos",
           "sigmoid", "tanh", "identity"):
    _register(_n, 1)
# Comparisons (binary -> bool tensor)
for _n in ("gt", "lt", "ge", "le", "eq"):
    _register(_n, 2)
_register("clamp", 1, kwargs=("min", "max"))
_register("where", 3)
_register("cast", 1, kwargs=("dtype",))
_register("sign", 1)

# Linalg
_register("matmul", 2)
_register("transpose", 1, kwargs=("dims",))
_register("reshape", 1, kwargs=("shape",))
_register("einsum", 2, kwargs=("equation",))

# Reductions
for _n in ("sum", "mean", "max", "min"):
    _register(_n, 1, kwargs=("dim", "keepdim"))

# Shape
_register("concat", -1, kwargs=("dim",))
_register("split", 1, kwargs=("sizes", "dim"))
_register("broadcast", 1, kwargs=("shape",))
_register("squeeze", 1, kwargs=("dim",))
_register("unsqueeze", 1, kwargs=("dim",))
_register("slice", 1, kwargs=("dim", "start", "end"))
_register("stack", -1, kwargs=("dim",))

# Indexing
_register("gather", 2, kwargs=("dim",))
_register("scatter", 3, kwargs=("dim",))
_register("arange", 0, kwargs=("start", "end", "step", "dtype"))

# Random
_register("normal", 0, kwargs=("seed", "shape", "dtype"))
_register("uniform", 0, kwargs=("seed", "shape", "dtype"))

# Sort / top-k
_register("sort", 1, kwargs=("dim", "descending"))
_register("topk", 1, kwargs=("k", "dim"))   # multi-output: (values, indices)

# NN compositions (single ops for clarity)
_register("softmax", 1, kwargs=("dim",))
_register("log_softmax", 1, kwargs=("dim",))
_register("layer_norm", 3, kwargs=("eps",))   # (x, weight, bias)
_register("rmsnorm", 2, kwargs=("eps",))      # (x, weight)
_register("cross_entropy", 2, kwargs=("ignore_index",))   # (logits[..., V], targets[...])
_register("embedding", 2)                                 # (weight[V, d], ids[...])

# Masks / constants
_register("tril", 1, kwargs=("diagonal",))
_register("triu", 1, kwargs=("diagonal",))
_register("full", 0, kwargs=("shape", "value", "dtype"))

# Quantization (multi-output)
_register("quantize_int8_per_channel", 1, kwargs=("dim",))    # -> (q_int8, scale_fp32)
_register("dequantize_int8_per_channel", 2)                   # (q, scale) -> fp32

# Single-output packed quantization for the wire: int8 values + per-channel
# fp32 scales packed into a single uint8 blob. Layout is
#   [scale_bytes (4*k) | int8_values_bytes (B*T*k)]
# where k is the size along `dim` (default -1).
_register("quantize_pack_int8", 1, kwargs=("dim",))           # x -> uint8 blob
_register("unpack_dequantize_int8", 1, kwargs=("dim", "shape", "scale_dim"))  # blob -> fp32

# Linalg ops added later (Grassmann)
_register("qr", 1)                                            # multi-output (Q, R)

# Data indexing: pull (input_ids, target_ids) of shape [B, T] from a
# uint16/int32 token stream tensor. Deterministically seeded by mb_seed.
_register("data_indexer", 1, kwargs=("B", "T", "mb_seed"))


def known_ops() -> list[str]:
    return sorted(_REGISTRY.keys())


def op_signature(name: str) -> OpSig:
    sig = _REGISTRY.get(name)
    if sig is None:
        raise KeyError(f"unknown op: {name!r}")
    return sig


# --------------------------------------------------------------------------- #
# GraphBuilder — convenience constructor used by task modules and tests.
# --------------------------------------------------------------------------- #


class GraphBuilder:
    """Mutable helper that produces a `Graph`."""

    def __init__(self) -> None:
        self._inputs: list[TensorSpec] = []
        self._outputs: list[dict[str, Any]] = []
        self._params: list[ParamSpec] = []
        self._ops: list[Op] = []

    def input(self, name: str, shape: Sequence[int], dtype: str) -> dict[str, Any]:
        if any(t.name == name for t in self._inputs):
            raise ValueError(f"duplicate input name: {name!r}")
        self._inputs.append(TensorSpec(name=name, shape=[int(s) for s in shape], dtype=dtype))
        return ref_input(name)

    def param(self, name: str, type_: str) -> dict[str, Any]:
        if any(p.name == name for p in self._params):
            raise ValueError(f"duplicate param name: {name!r}")
        self._params.append(ParamSpec(name=name, type=type_))
        return ref_param(name)

    def const(self, value: Any) -> dict[str, Any]:
        return ref_const(value)

    def const_blob(self, uri: str, shape: Sequence[int], dtype: str) -> dict[str, Any]:
        return ref_const_blob(uri, shape, dtype)

    def output(self, name: str, ref: dict[str, Any]) -> None:
        self._outputs.append({"name": name, "ref": dict(ref)})

    def emit(
        self,
        op: str,
        args: Sequence[dict[str, Any]] = (),
        kwargs: dict[str, Any] | None = None,
        out: Sequence[TensorSpec] = (),
    ) -> dict[str, Any]:
        sig = op_signature(op)
        if sig.n_args >= 0 and len(args) != sig.n_args:
            raise ValueError(
                f"op {op!r} expects {sig.n_args} args, got {len(args)}"
            )
        kw = dict(kwargs or {})
        for k in kw.keys():
            if k not in sig.kwargs:
                raise ValueError(f"op {op!r} got unexpected kwarg: {k!r}")
        op_id = len(self._ops)
        self._ops.append(
            Op(
                id=op_id,
                op=op,
                args=[dict(a) for a in args],
                kwargs=kw,
                out=list(out),
            )
        )
        return ref_op(op_id, idx=0)

    def emit_multi(
        self,
        op: str,
        args: Sequence[dict[str, Any]] = (),
        kwargs: dict[str, Any] | None = None,
        n_outputs: int = 2,
    ) -> tuple[dict[str, Any], ...]:
        """Emit an op and return refs to all `n_outputs` outputs."""
        sig = op_signature(op)
        if sig.n_args >= 0 and len(args) != sig.n_args:
            raise ValueError(
                f"op {op!r} expects {sig.n_args} args, got {len(args)}"
            )
        kw = dict(kwargs or {})
        for k in kw.keys():
            if k not in sig.kwargs:
                raise ValueError(f"op {op!r} got unexpected kwarg: {k!r}")
        op_id = len(self._ops)
        self._ops.append(
            Op(id=op_id, op=op, args=[dict(a) for a in args], kwargs=kw, out=[])
        )
        return tuple(ref_op(op_id, idx=i) for i in range(n_outputs))

    # --- ergonomic shortcuts ---

    def add(self, a, b): return self.emit("add", [a, b])
    def sub(self, a, b): return self.emit("sub", [a, b])
    def mul(self, a, b): return self.emit("mul", [a, b])
    def div(self, a, b): return self.emit("div", [a, b])
    def pow_(self, a, b): return self.emit("pow", [a, b])
    def neg(self, a): return self.emit("neg", [a])
    def exp(self, a): return self.emit("exp", [a])
    def log(self, a): return self.emit("log", [a])
    def sqrt(self, a): return self.emit("sqrt", [a])
    def abs_(self, a): return self.emit("abs", [a])
    def round_(self, a): return self.emit("round", [a])
    def relu(self, a): return self.emit("relu", [a])
    def gelu(self, a): return self.emit("gelu", [a])
    def silu(self, a): return self.emit("silu", [a])
    def sin(self, a): return self.emit("sin", [a])
    def cos(self, a): return self.emit("cos", [a])
    def sigmoid(self, a): return self.emit("sigmoid", [a])
    def tanh(self, a): return self.emit("tanh", [a])
    def identity(self, a): return self.emit("identity", [a])
    def sign(self, a): return self.emit("sign", [a])

    def gt(self, a, b): return self.emit("gt", [a, b])
    def lt(self, a, b): return self.emit("lt", [a, b])
    def ge(self, a, b): return self.emit("ge", [a, b])
    def le(self, a, b): return self.emit("le", [a, b])
    def eq(self, a, b): return self.emit("eq", [a, b])

    def clamp(self, a, *, min=None, max=None):
        kw = {}
        if min is not None: kw["min"] = min
        if max is not None: kw["max"] = max
        return self.emit("clamp", [a], kwargs=kw)
    def where(self, cond, a, b): return self.emit("where", [cond, a, b])
    def cast(self, a, *, dtype): return self.emit("cast", [a], kwargs={"dtype": dtype})

    def matmul(self, a, b): return self.emit("matmul", [a, b])
    def transpose(self, a, *, dims): return self.emit("transpose", [a], kwargs={"dims": list(dims)})
    def reshape(self, a, *, shape): return self.emit("reshape", [a], kwargs={"shape": list(shape)})
    def einsum(self, a, b, *, equation): return self.emit("einsum", [a, b], kwargs={"equation": equation})

    def sum(self, a, *, dim=None, keepdim=False):
        return self.emit("sum", [a], kwargs={"dim": dim, "keepdim": bool(keepdim)})
    def mean(self, a, *, dim=None, keepdim=False):
        return self.emit("mean", [a], kwargs={"dim": dim, "keepdim": bool(keepdim)})
    def max_(self, a, *, dim=None, keepdim=False):
        return self.emit("max", [a], kwargs={"dim": dim, "keepdim": bool(keepdim)})
    def min_(self, a, *, dim=None, keepdim=False):
        return self.emit("min", [a], kwargs={"dim": dim, "keepdim": bool(keepdim)})

    def concat(self, tensors, *, dim):
        return self.emit("concat", list(tensors), kwargs={"dim": int(dim)})
    def stack(self, tensors, *, dim):
        return self.emit("stack", list(tensors), kwargs={"dim": int(dim)})
    def split(self, a, *, sizes, dim):
        return self.emit("split", [a], kwargs={"sizes": list(sizes), "dim": int(dim)})
    def broadcast(self, a, *, shape):
        return self.emit("broadcast", [a], kwargs={"shape": list(shape)})
    def squeeze(self, a, *, dim):
        return self.emit("squeeze", [a], kwargs={"dim": int(dim)})
    def unsqueeze(self, a, *, dim):
        return self.emit("unsqueeze", [a], kwargs={"dim": int(dim)})
    def slice(self, a, *, dim, start, end):
        # Allow start/end to be ref dicts (typically param refs); only cast ints.
        ks = start if isinstance(start, dict) else int(start)
        ke = end if isinstance(end, dict) else int(end)
        return self.emit(
            "slice", [a], kwargs={"dim": int(dim), "start": ks, "end": ke}
        )

    def gather(self, a, index, *, dim):
        return self.emit("gather", [a, index], kwargs={"dim": int(dim)})
    def scatter(self, a, index, src, *, dim):
        return self.emit("scatter", [a, index, src], kwargs={"dim": int(dim)})
    def arange(self, *, start, end, step=1, dtype="int64"):
        return self.emit(
            "arange", [], kwargs={"start": start, "end": end, "step": step, "dtype": dtype}
        )

    def normal(self, *, seed, shape, dtype="float32"):
        return self.emit(
            "normal", [], kwargs={"seed": int(seed), "shape": list(shape), "dtype": dtype}
        )
    def uniform(self, *, seed, shape, dtype="float32"):
        return self.emit(
            "uniform", [], kwargs={"seed": int(seed), "shape": list(shape), "dtype": dtype}
        )

    def sort(self, a, *, dim=-1, descending=False):
        return self.emit("sort", [a], kwargs={"dim": int(dim), "descending": bool(descending)})

    def topk(self, a, *, k, dim=-1):
        return self.emit_multi("topk", [a], kwargs={"k": int(k), "dim": int(dim)}, n_outputs=2)

    def softmax(self, a, *, dim=-1):
        return self.emit("softmax", [a], kwargs={"dim": int(dim)})
    def log_softmax(self, a, *, dim=-1):
        return self.emit("log_softmax", [a], kwargs={"dim": int(dim)})
    def layer_norm(self, a, weight, bias, *, eps=1e-5):
        return self.emit("layer_norm", [a, weight, bias], kwargs={"eps": float(eps)})
    def rmsnorm(self, a, weight, *, eps=1e-5):
        return self.emit("rmsnorm", [a, weight], kwargs={"eps": float(eps)})
    def tril(self, a, *, diagonal=0):
        return self.emit("tril", [a], kwargs={"diagonal": int(diagonal)})
    def triu(self, a, *, diagonal=0):
        return self.emit("triu", [a], kwargs={"diagonal": int(diagonal)})
    def full(self, *, shape, value, dtype="float32"):
        return self.emit("full", [], kwargs={"shape": list(shape), "value": value, "dtype": dtype})
    def cross_entropy(self, logits, targets, *, ignore_index=-100):
        return self.emit(
            "cross_entropy", [logits, targets], kwargs={"ignore_index": int(ignore_index)}
        )
    def embedding(self, weight, ids):
        return self.emit("embedding", [weight, ids])

    def quantize_int8_per_channel(self, x, *, dim=-1):
        return self.emit_multi(
            "quantize_int8_per_channel", [x],
            kwargs={"dim": int(dim)}, n_outputs=2,
        )
    def dequantize_int8_per_channel(self, q, scale):
        return self.emit("dequantize_int8_per_channel", [q, scale])

    def quantize_pack_int8(self, x, *, dim=-1):
        """Quantize x to int8 per-channel along `dim`, then pack the scale
        and quantized values into a single 1-D uint8 blob suitable for the
        wire. Returns the packed blob ref."""
        return self.emit("quantize_pack_int8", [x], kwargs={"dim": int(dim)})
    def unpack_dequantize_int8(self, blob, *, shape, dim=-1):
        """Reverse of `quantize_pack_int8`. `shape` is the original tensor's
        shape, `dim` is the per-channel dim (the size along that dim
        determines how much of the blob is the scale)."""
        return self.emit(
            "unpack_dequantize_int8", [blob],
            kwargs={"dim": int(dim), "shape": list(shape), "scale_dim": int(dim)},
        )

    def qr(self, a):
        return self.emit_multi("qr", [a], n_outputs=2)

    def data_indexer(self, tokens, *, B, T, mb_seed):
        """Pull (input_ids, target_ids) of shape [B, T] from a flat token
        stream. Deterministic given mb_seed: each B-batch picks B random
        starting offsets in the token stream, then takes T+1 consecutive
        tokens, splits into (input, target).
        mb_seed may be an int OR a ref-dict (e.g. ref_param("mb_seed")).
        Returns 2 outputs: input_ids and target_ids.
        """
        seed = mb_seed if isinstance(mb_seed, dict) else int(mb_seed)
        return self.emit_multi(
            "data_indexer",
            [tokens],
            kwargs={"B": int(B), "T": int(T), "mb_seed": seed},
            n_outputs=2,
        )

    def build(self) -> Graph:
        return Graph(
            inputs=list(self._inputs),
            outputs=list(self._outputs),
            params=list(self._params),
            ops=list(self._ops),
            ir_version=1,
        )
