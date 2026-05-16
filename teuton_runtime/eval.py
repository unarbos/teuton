"""PyTorch evaluator for Teuton v3 IR graphs."""
from __future__ import annotations

import os
from typing import Any

import torch

from teuton_core.ir import Graph
from . import tensor_io

_DETERMINISM_SET = False
_ACTIVE_DEVICE: torch.device | None = None


def _ensure_determinism() -> None:
    global _DETERMINISM_SET
    if _DETERMINISM_SET:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    _DETERMINISM_SET = True


def _is_ref_dict(value: Any) -> bool:
    return isinstance(value, dict) and value.get("kind") in {"input", "op", "param", "const", "const_blob"}


def _resolve_ref(ref: dict[str, Any], inputs: dict[str, torch.Tensor], params: dict[str, Any], op_results: list[Any], *, bucket=None, blob_cache: dict[str, torch.Tensor] | None = None) -> Any:
    kind = ref["kind"]
    if kind == "input":
        name = ref["name"]
        if name not in inputs:
            raise KeyError(f"missing graph input: {name!r}")
        return inputs[name]
    if kind == "op":
        op_id = int(ref["id"])
        idx = int(ref.get("idx", 0))
        result = op_results[op_id]
        if isinstance(result, (list, tuple)):
            return result[idx]
        if idx != 0:
            raise IndexError(f"op {op_id} produced single output but idx={idx}")
        return result
    if kind == "param":
        name = ref["name"]
        if name not in params:
            raise KeyError(f"missing param: {name!r}")
        return params[name]
    if kind == "const":
        return ref["value"]
    if kind == "const_blob":
        uri = ref["uri"]
        if blob_cache is not None and uri in blob_cache:
            return blob_cache[uri]
        if bucket is None:
            raise RuntimeError(f"const_blob ref {uri!r} but no bucket provided to evaluate()")
        value = tensor_io.decode_tensor(bucket.get(uri))
        declared_shape = list(ref.get("shape") or [])
        if declared_shape and list(value.shape) != declared_shape:
            raise ValueError(f"const_blob {uri!r}: shape {list(value.shape)} != declared {declared_shape}")
        declared_dtype = ref.get("dtype")
        if declared_dtype and tensor_io.wire_dtype(value) != declared_dtype:
            raise TypeError(f"const_blob {uri!r}: dtype {tensor_io.wire_dtype(value)!r} != declared {declared_dtype!r}")
        if _ACTIVE_DEVICE is not None and value.device != _ACTIVE_DEVICE:
            value = value.to(_ACTIVE_DEVICE)
        if blob_cache is not None:
            blob_cache[uri] = value
        return value
    raise ValueError(f"unknown ref kind: {kind!r}")


def _resolve_kwargs(kwargs: dict[str, Any], inputs: dict[str, torch.Tensor], params: dict[str, Any], op_results: list[Any], *, bucket=None, blob_cache: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    return {
        key: _resolve_ref(value, inputs, params, op_results, bucket=bucket, blob_cache=blob_cache) if _is_ref_dict(value) else value
        for key, value in kwargs.items()
    }


def _dtype(name: str) -> torch.dtype:
    return tensor_io.torch_dtype(name)


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _reduce_kwargs(kwargs: dict[str, Any]) -> tuple[Any, bool]:
    return kwargs.get("dim"), bool(kwargs.get("keepdim", False))


def _op_add(args, kwargs): return args[0] + args[1]
def _op_sub(args, kwargs): return args[0] - args[1]
def _op_mul(args, kwargs): return args[0] * args[1]
def _op_div(args, kwargs): return args[0] / args[1]
def _op_pow(args, kwargs): return args[0] ** args[1]
def _op_neg(args, kwargs): return -args[0]
def _op_exp(args, kwargs): return torch.exp(args[0])
def _op_log(args, kwargs): return torch.log(args[0])
def _op_sqrt(args, kwargs): return torch.sqrt(args[0])
def _op_abs(args, kwargs): return torch.abs(args[0])
def _op_round(args, kwargs): return torch.round(args[0])
def _op_relu(args, kwargs): return torch.relu(args[0])
def _op_gelu(args, kwargs): return torch.nn.functional.gelu(args[0], approximate="tanh")
def _op_silu(args, kwargs): return torch.nn.functional.silu(args[0])
def _op_sin(args, kwargs): return torch.sin(args[0])
def _op_cos(args, kwargs): return torch.cos(args[0])
def _op_sigmoid(args, kwargs): return torch.sigmoid(args[0])
def _op_tanh(args, kwargs): return torch.tanh(args[0])
def _op_identity(args, kwargs): return args[0]
def _op_sign(args, kwargs): return torch.sign(args[0])
def _op_clamp(args, kwargs): return torch.clamp(args[0], min=kwargs.get("min"), max=kwargs.get("max"))
def _op_where(args, kwargs): return torch.where(args[0].to(torch.bool), args[1], args[2])
def _op_cast(args, kwargs): return args[0].to(_dtype(kwargs["dtype"]))
def _op_gt(args, kwargs): return args[0] > args[1]
def _op_lt(args, kwargs): return args[0] < args[1]
def _op_ge(args, kwargs): return args[0] >= args[1]
def _op_le(args, kwargs): return args[0] <= args[1]
def _op_eq(args, kwargs): return args[0] == args[1]
def _op_matmul(args, kwargs): return torch.matmul(args[0], args[1])
def _op_transpose(args, kwargs): return args[0].permute(*kwargs["dims"]).contiguous()
def _op_reshape(args, kwargs): return args[0].reshape(kwargs["shape"])
def _op_einsum(args, kwargs): return torch.einsum(kwargs["equation"], args[0], args[1])


def _op_sum(args, kwargs):
    dim, keepdim = _reduce_kwargs(kwargs)
    return torch.sum(args[0]) if dim is None else torch.sum(args[0], dim=dim, keepdim=keepdim)


def _op_mean(args, kwargs):
    dim, keepdim = _reduce_kwargs(kwargs)
    return torch.mean(args[0]) if dim is None else torch.mean(args[0], dim=dim, keepdim=keepdim)


def _op_max(args, kwargs):
    dim, keepdim = _reduce_kwargs(kwargs)
    return torch.amax(args[0]) if dim is None else torch.amax(args[0], dim=dim, keepdim=keepdim)


def _op_min(args, kwargs):
    dim, keepdim = _reduce_kwargs(kwargs)
    return torch.amin(args[0]) if dim is None else torch.amin(args[0], dim=dim, keepdim=keepdim)


def _op_concat(args, kwargs): return torch.cat(list(args), dim=int(kwargs["dim"]))
def _op_stack(args, kwargs): return torch.stack(list(args), dim=int(kwargs["dim"]))
def _op_split(args, kwargs): return tuple(t.contiguous() for t in torch.split(args[0], list(kwargs["sizes"]), dim=int(kwargs["dim"])))
def _op_broadcast(args, kwargs): return args[0].broadcast_to(kwargs["shape"]).contiguous()
def _op_squeeze(args, kwargs): return args[0].squeeze(int(kwargs["dim"]))
def _op_unsqueeze(args, kwargs): return args[0].unsqueeze(int(kwargs["dim"]))
def _op_slice(args, kwargs): return args[0].narrow(int(kwargs["dim"]), int(kwargs["start"]), int(kwargs["end"]) - int(kwargs["start"])).contiguous()
def _op_gather(args, kwargs): return torch.gather(args[0], int(kwargs["dim"]), args[1].to(torch.int64))
def _op_scatter(args, kwargs): return args[0].clone().scatter_(int(kwargs["dim"]), args[1].to(torch.int64), args[2])
def _op_arange(args, kwargs): return torch.arange(kwargs["start"], kwargs["end"], kwargs.get("step", 1), dtype=_dtype(kwargs.get("dtype", "int64")))


def _op_normal(args, kwargs):
    dtype = _dtype(kwargs.get("dtype", "float32"))
    generator = torch.Generator(device="cpu").manual_seed(int(kwargs["seed"]))
    out = torch.empty(list(kwargs["shape"]), dtype=dtype)
    if dtype.is_floating_point:
        out.normal_(generator=generator)
    else:
        out = torch.empty(list(kwargs["shape"]), dtype=torch.float32).normal_(generator=generator).to(dtype)
    return out.to(_ACTIVE_DEVICE) if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu" else out


def _op_uniform(args, kwargs):
    dtype = _dtype(kwargs.get("dtype", "float32"))
    generator = torch.Generator(device="cpu").manual_seed(int(kwargs["seed"]))
    out = torch.empty(list(kwargs["shape"]), dtype=dtype)
    if dtype.is_floating_point:
        out.uniform_(generator=generator)
    else:
        out = torch.empty(list(kwargs["shape"]), dtype=torch.float32).uniform_(generator=generator).to(dtype)
    return out.to(_ACTIVE_DEVICE) if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu" else out


def _op_sort(args, kwargs):
    values, _ = torch.sort(args[0], dim=int(kwargs.get("dim", -1)), descending=bool(kwargs.get("descending", False)))
    return values.contiguous()


def _op_topk(args, kwargs):
    values, indices = torch.topk(args[0], k=int(kwargs["k"]), dim=int(kwargs.get("dim", -1)), largest=True, sorted=True)
    return values.contiguous(), indices.to(torch.int64).contiguous()


def _op_softmax(args, kwargs): return torch.nn.functional.softmax(args[0], dim=int(kwargs.get("dim", -1)))
def _op_log_softmax(args, kwargs): return torch.nn.functional.log_softmax(args[0], dim=int(kwargs.get("dim", -1)))
def _op_layer_norm(args, kwargs): return torch.nn.functional.layer_norm(args[0], list(args[1].shape), weight=args[1], bias=args[2], eps=float(kwargs.get("eps", 1e-5)))


def _op_rmsnorm(args, kwargs):
    x = args[0]
    x32 = x.to(torch.float32)
    out = x32 * x32.pow(2).mean(dim=-1, keepdim=True).add(float(kwargs.get("eps", 1e-5))).rsqrt() * args[1].to(torch.float32)
    return out.to(x.dtype)


def _op_cross_entropy(args, kwargs):
    return torch.nn.functional.cross_entropy(args[0].reshape(-1, args[0].shape[-1]), args[1].to(torch.int64).reshape(-1), ignore_index=int(kwargs.get("ignore_index", -100)), reduction="mean")


def _op_embedding(args, kwargs): return torch.nn.functional.embedding(args[1].to(torch.int64), args[0])
def _op_tril(args, kwargs): return torch.tril(args[0], diagonal=int(kwargs.get("diagonal", 0)))
def _op_triu(args, kwargs): return torch.triu(args[0], diagonal=int(kwargs.get("diagonal", 0)))


def _op_full(args, kwargs):
    out = torch.full(list(kwargs["shape"]), kwargs.get("value", 0.0), dtype=_dtype(kwargs.get("dtype", "float32")))
    return out.to(_ACTIVE_DEVICE) if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu" else out


def _op_data_indexer(args, kwargs):
    tokens = args[0].reshape(-1).detach().to("cpu", torch.int64)
    b = int(kwargs["B"])
    t = int(kwargs["T"])
    if tokens.numel() < t + 1:
        raise ValueError(f"data_indexer: token stream too short ({tokens.numel()}) for T+1={t + 1}")
    generator = torch.Generator(device="cpu").manual_seed(int(kwargs["mb_seed"]))
    starts = torch.randint(0, tokens.numel() - t - 1, (b,), generator=generator)
    rows = torch.stack([tokens[int(start) : int(start) + t + 1] for start in starts])
    if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu":
        rows = rows.to(_ACTIVE_DEVICE)
    return rows[:, :t].contiguous(), rows[:, 1:].contiguous()


def _op_quantize_int8_per_channel(args, kwargs):
    x = args[0]
    dim = int(kwargs.get("dim", -1))
    if dim < 0:
        dim += x.dim()
    scale = x.abs()
    for axis in [axis for axis in range(x.dim()) if axis != dim]:
        scale = scale.amax(dim=axis, keepdim=True)
    scale = (scale / 127.0).clamp_min(1e-8).to(torch.float32)
    return torch.round(x.to(torch.float32) / scale).clamp(-128, 127).to(torch.int8).contiguous(), scale.contiguous()


def _op_dequantize_int8_per_channel(args, kwargs): return args[0].to(torch.float32) * args[1].to(torch.float32)


def _op_quantize_pack_int8(args, kwargs):
    q, scale = _op_quantize_int8_per_channel(args, kwargs)
    return torch.cat([scale.reshape(-1).contiguous().view(torch.uint8), q.reshape(-1).contiguous().view(torch.uint8)], dim=0).contiguous()


def _op_unpack_dequantize_int8(args, kwargs):
    packed = args[0].to(torch.uint8)
    shape = list(kwargs["shape"])
    dim = int(kwargs.get("dim", -1))
    if dim < 0:
        dim += len(shape)
    k = int(shape[dim])
    scale = packed[: k * 4].contiguous().view(torch.float32)
    scale_shape = [1] * len(shape)
    scale_shape[dim] = k
    return packed[k * 4 :].contiguous().view(torch.int8).reshape(shape).to(torch.float32) * scale.reshape(scale_shape)


def _op_qr(args, kwargs):
    q, r = torch.linalg.qr(args[0], mode="reduced")
    return q.contiguous(), r.contiguous()


_DISPATCH = {
    "add": _op_add, "sub": _op_sub, "mul": _op_mul, "div": _op_div, "pow": _op_pow,
    "neg": _op_neg, "exp": _op_exp, "log": _op_log, "sqrt": _op_sqrt, "abs": _op_abs, "round": _op_round,
    "relu": _op_relu, "gelu": _op_gelu, "silu": _op_silu, "sin": _op_sin, "cos": _op_cos,
    "sigmoid": _op_sigmoid, "tanh": _op_tanh, "identity": _op_identity, "sign": _op_sign,
    "clamp": _op_clamp, "where": _op_where, "cast": _op_cast,
    "gt": _op_gt, "lt": _op_lt, "ge": _op_ge, "le": _op_le, "eq": _op_eq,
    "matmul": _op_matmul, "transpose": _op_transpose, "reshape": _op_reshape, "einsum": _op_einsum,
    "sum": _op_sum, "mean": _op_mean, "max": _op_max, "min": _op_min,
    "concat": _op_concat, "stack": _op_stack, "split": _op_split, "broadcast": _op_broadcast,
    "squeeze": _op_squeeze, "unsqueeze": _op_unsqueeze, "slice": _op_slice,
    "gather": _op_gather, "scatter": _op_scatter, "arange": _op_arange,
    "normal": _op_normal, "uniform": _op_uniform, "sort": _op_sort, "topk": _op_topk,
    "softmax": _op_softmax, "log_softmax": _op_log_softmax, "layer_norm": _op_layer_norm,
    "rmsnorm": _op_rmsnorm, "cross_entropy": _op_cross_entropy, "embedding": _op_embedding,
    "tril": _op_tril, "triu": _op_triu, "full": _op_full, "data_indexer": _op_data_indexer,
    "quantize_int8_per_channel": _op_quantize_int8_per_channel,
    "dequantize_int8_per_channel": _op_dequantize_int8_per_channel,
    "quantize_pack_int8": _op_quantize_pack_int8,
    "unpack_dequantize_int8": _op_unpack_dequantize_int8,
    "qr": _op_qr,
}


def evaluate(graph: Graph, inputs: dict[str, torch.Tensor], params: dict[str, Any] | None = None, *, bucket=None, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    _ensure_determinism()
    global _ACTIVE_DEVICE
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda" and not torch.cuda.is_available():
        dev = None
    previous_device = _ACTIVE_DEVICE
    _ACTIVE_DEVICE = dev
    try:
        params = dict(params or {})
        blob_cache: dict[str, torch.Tensor] = {}
        for spec in graph.inputs:
            if spec.name not in inputs:
                raise KeyError(f"missing input tensor: {spec.name!r}")
            value = inputs[spec.name]
            if tensor_io.wire_dtype(value) != spec.dtype:
                raise TypeError(f"input {spec.name!r}: dtype {tensor_io.wire_dtype(value)!r} != declared {spec.dtype!r}")
            declared = list(spec.shape)
            actual = list(value.shape)
            if len(declared) != len(actual) or any(d != -1 and d != a for d, a in zip(declared, actual)):
                raise ValueError(f"input {spec.name!r}: shape {actual} != declared {declared}")
            if dev is not None and value.device != dev:
                inputs[spec.name] = value.to(dev)
        for spec in graph.params:
            if spec.name not in params:
                raise KeyError(f"missing param: {spec.name!r}")

        op_results: list[Any] = [None] * len(graph.ops)
        for op in graph.ops:
            fn = _DISPATCH.get(op.op)
            if fn is None:
                raise KeyError(f"evaluator missing op: {op.op!r}")
            args = [_resolve_ref(arg, inputs, params, op_results, bucket=bucket, blob_cache=blob_cache) for arg in op.args]
            args = [_as_tensor(arg) if not isinstance(arg, torch.Tensor) else arg for arg in args]
            if dev is not None:
                args = [arg.to(dev) if isinstance(arg, torch.Tensor) and arg.device != dev else arg for arg in args]
            kwargs = _resolve_kwargs(op.kwargs, inputs, params, op_results, bucket=bucket, blob_cache=blob_cache)
            op_results[op.id] = fn(args, kwargs)

        outputs: dict[str, torch.Tensor] = {}
        for output in graph.outputs:
            value = _resolve_ref(output["ref"], inputs, params, op_results, bucket=bucket, blob_cache=blob_cache)
            tensor = _as_tensor(value)
            if dev is not None and tensor.device.type != "cpu":
                tensor = tensor.cpu()
            outputs[output["name"]] = tensor.contiguous()
        return outputs
    finally:
        _ACTIVE_DEVICE = previous_device
