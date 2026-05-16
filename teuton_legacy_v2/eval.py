"""IR evaluator (PyTorch CPU).

`evaluate(graph, inputs, params, *, bucket=None) -> dict[str, torch.Tensor]`
walks the ops in order and dispatches each to a torch counterpart.

Refs:
  - input  : pulled from `inputs` dict
  - op     : pulled from previously-computed op output (idx selects which output)
  - param  : pulled from `params` dict (scalar / list)
  - const  : inline literal value
  - const_blob : tensor stored in the bucket; resolved by reading the URI

Op kwargs may themselves be refs (commonly `param`-refs); these are resolved
per op-call.

Multi-output ops (split, topk, quantize_int8_per_channel, qr) return a tuple
in the per-op slot; refs to them use `idx`.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch

from . import tensor_io
from .ir import Graph, Op


# --------------------------------------------------------------------------- #
# Determinism (best-effort CPU)
# --------------------------------------------------------------------------- #


_DETERMINISM_SET = False
_ACTIVE_DEVICE: torch.device | None = None


def _ensure_determinism() -> None:
    global _DETERMINISM_SET
    if _DETERMINISM_SET:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        # warn_only=True is critical: many fused CUDA ops are non-deterministic;
        # we want them but with a warning, not a hard fail.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    _DETERMINISM_SET = True


# --------------------------------------------------------------------------- #
# Reference resolution
# --------------------------------------------------------------------------- #


def _resolve_ref(
    ref: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    params: dict[str, Any],
    op_results: list[Any],
    *,
    bucket=None,
    blob_cache: dict[str, torch.Tensor] | None = None,
) -> Any:
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
            raise RuntimeError(
                f"const_blob ref {uri!r} but no bucket provided to evaluate()"
            )
        body = bucket.get(uri)
        t = tensor_io.decode_tensor(body)
        # Validate shape/dtype against declared spec
        declared_shape = list(ref.get("shape") or [])
        declared_dtype = ref.get("dtype")
        if declared_shape and list(t.shape) != declared_shape:
            raise ValueError(
                f"const_blob {uri!r}: shape {list(t.shape)} != declared {declared_shape}"
            )
        if declared_dtype:
            wd = tensor_io.wire_dtype(t)
            if wd != declared_dtype:
                raise TypeError(
                    f"const_blob {uri!r}: dtype {wd!r} != declared {declared_dtype!r}"
                )
        if _ACTIVE_DEVICE is not None and t.device != _ACTIVE_DEVICE:
            t = t.to(_ACTIVE_DEVICE)
        if blob_cache is not None:
            blob_cache[uri] = t
        return t
    raise ValueError(f"unknown ref kind: {kind!r}")


def _is_ref_dict(v: Any) -> bool:
    return (
        isinstance(v, dict)
        and "kind" in v
        and v.get("kind") in ("input", "op", "param", "const", "const_blob")
    )


def _resolve_kwargs(
    kwargs: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    params: dict[str, Any],
    op_results: list[Any],
    *,
    bucket=None,
    blob_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if _is_ref_dict(v):
            out[k] = _resolve_ref(v, inputs, params, op_results,
                                   bucket=bucket, blob_cache=blob_cache)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Op dispatch
# --------------------------------------------------------------------------- #


def _dtype(s: str) -> torch.dtype:
    return tensor_io.torch_dtype(s)


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


# --- elementwise / unary ---
def _op_add(args, kwargs):       return args[0] + args[1]
def _op_sub(args, kwargs):       return args[0] - args[1]
def _op_mul(args, kwargs):       return args[0] * args[1]
def _op_div(args, kwargs):       return args[0] / args[1]
def _op_pow(args, kwargs):       return args[0] ** args[1]
def _op_neg(args, kwargs):       return -args[0]
def _op_exp(args, kwargs):       return torch.exp(args[0])
def _op_log(args, kwargs):       return torch.log(args[0])
def _op_sqrt(args, kwargs):      return torch.sqrt(args[0])
def _op_abs(args, kwargs):       return torch.abs(args[0])
def _op_round(args, kwargs):     return torch.round(args[0])
def _op_relu(args, kwargs):      return torch.relu(args[0])
def _op_gelu(args, kwargs):      return torch.nn.functional.gelu(args[0], approximate="tanh")
def _op_silu(args, kwargs):      return torch.nn.functional.silu(args[0])
def _op_sin(args, kwargs):       return torch.sin(args[0])
def _op_cos(args, kwargs):       return torch.cos(args[0])
def _op_sigmoid(args, kwargs):   return torch.sigmoid(args[0])
def _op_tanh(args, kwargs):      return torch.tanh(args[0])
def _op_identity(args, kwargs):  return args[0]
def _op_sign(args, kwargs):      return torch.sign(args[0])
def _op_clamp(args, kwargs):     return torch.clamp(args[0], min=kwargs.get("min"), max=kwargs.get("max"))
def _op_where(args, kwargs):     return torch.where(args[0].to(torch.bool), args[1], args[2])
def _op_cast(args, kwargs):      return args[0].to(_dtype(kwargs["dtype"]))

# --- comparisons (return bool) ---
def _op_gt(args, kwargs): return args[0] > args[1]
def _op_lt(args, kwargs): return args[0] < args[1]
def _op_ge(args, kwargs): return args[0] >= args[1]
def _op_le(args, kwargs): return args[0] <= args[1]
def _op_eq(args, kwargs): return args[0] == args[1]


# --- linalg ---
def _op_matmul(args, kwargs):     return torch.matmul(args[0], args[1])
def _op_transpose(args, kwargs):  return args[0].permute(*kwargs["dims"]).contiguous()
def _op_reshape(args, kwargs):    return args[0].reshape(kwargs["shape"])
def _op_einsum(args, kwargs):     return torch.einsum(kwargs["equation"], args[0], args[1])


# --- reductions ---
def _reduce_kw(kwargs):
    dim = kwargs.get("dim")
    keepdim = bool(kwargs.get("keepdim", False))
    return dim, keepdim


def _op_sum(args, kwargs):
    dim, keepdim = _reduce_kw(kwargs)
    if dim is None: return torch.sum(args[0])
    return torch.sum(args[0], dim=dim, keepdim=keepdim)


def _op_mean(args, kwargs):
    dim, keepdim = _reduce_kw(kwargs)
    if dim is None: return torch.mean(args[0])
    return torch.mean(args[0], dim=dim, keepdim=keepdim)


def _op_max(args, kwargs):
    dim, keepdim = _reduce_kw(kwargs)
    if dim is None: return torch.amax(args[0])
    return torch.amax(args[0], dim=dim, keepdim=keepdim)


def _op_min(args, kwargs):
    dim, keepdim = _reduce_kw(kwargs)
    if dim is None: return torch.amin(args[0])
    return torch.amin(args[0], dim=dim, keepdim=keepdim)


# --- shape ---
def _op_concat(args, kwargs):    return torch.cat(list(args), dim=int(kwargs["dim"]))
def _op_stack(args, kwargs):     return torch.stack(list(args), dim=int(kwargs["dim"]))


def _op_split(args, kwargs):
    sizes = list(kwargs["sizes"])
    dim = int(kwargs["dim"])
    out = torch.split(args[0], sizes, dim=dim)
    return tuple(t.contiguous() for t in out)


def _op_broadcast(args, kwargs): return args[0].broadcast_to(kwargs["shape"]).contiguous()
def _op_squeeze(args, kwargs):   return args[0].squeeze(int(kwargs["dim"]))
def _op_unsqueeze(args, kwargs): return args[0].unsqueeze(int(kwargs["dim"]))


def _op_slice(args, kwargs):
    dim = int(kwargs["dim"])
    start = int(kwargs["start"])
    end = int(kwargs["end"])
    return args[0].narrow(dim, start, end - start).contiguous()


# --- indexing ---
def _op_gather(args, kwargs):
    dim = int(kwargs["dim"])
    return torch.gather(args[0], dim, args[1].to(torch.int64))


def _op_scatter(args, kwargs):
    dim = int(kwargs["dim"])
    return args[0].clone().scatter_(dim, args[1].to(torch.int64), args[2])


def _op_arange(args, kwargs):
    start = kwargs["start"]; end = kwargs["end"]
    step = kwargs.get("step", 1)
    dtype = _dtype(kwargs.get("dtype", "int64"))
    return torch.arange(start, end, step, dtype=dtype)


# --- random ---
def _op_normal(args, kwargs):
    seed = int(kwargs["seed"])
    shape = list(kwargs["shape"])
    dtype = _dtype(kwargs.get("dtype", "float32"))
    # Generate on CPU for determinism, then move to active device.
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty(shape, dtype=dtype)
    if dtype.is_floating_point:
        out.normal_(generator=g)
    else:
        f = torch.empty(shape, dtype=torch.float32).normal_(generator=g)
        out = f.to(dtype)
    if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu":
        out = out.to(_ACTIVE_DEVICE)
    return out


def _op_uniform(args, kwargs):
    seed = int(kwargs["seed"])
    shape = list(kwargs["shape"])
    dtype = _dtype(kwargs.get("dtype", "float32"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty(shape, dtype=dtype)
    if dtype.is_floating_point:
        out.uniform_(generator=g)
    else:
        f = torch.empty(shape, dtype=torch.float32).uniform_(generator=g)
        out = f.to(dtype)
    if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu":
        out = out.to(_ACTIVE_DEVICE)
    return out


# --- sort / topk ---
def _op_sort(args, kwargs):
    dim = int(kwargs.get("dim", -1))
    descending = bool(kwargs.get("descending", False))
    sorted_t, _ = torch.sort(args[0], dim=dim, descending=descending)
    return sorted_t.contiguous()


def _op_topk(args, kwargs):
    k = int(kwargs["k"])
    dim = int(kwargs.get("dim", -1))
    vals, idxs = torch.topk(args[0], k=k, dim=dim, largest=True, sorted=True)
    return (vals.contiguous(), idxs.to(torch.int64).contiguous())


# --- NN compositions ---
def _op_softmax(args, kwargs):
    return torch.nn.functional.softmax(args[0], dim=int(kwargs.get("dim", -1)))


def _op_log_softmax(args, kwargs):
    return torch.nn.functional.log_softmax(args[0], dim=int(kwargs.get("dim", -1)))


def _op_layer_norm(args, kwargs):
    x = args[0]
    weight = args[1]
    bias = args[2]
    eps = float(kwargs.get("eps", 1e-5))
    # normalized_shape derived from weight: trailing dim(s) of x matching weight shape
    normalized_shape = list(weight.shape)
    return torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)


def _op_rmsnorm(args, kwargs):
    """RMSNorm over the trailing axis. args = (x, weight). weight shape = (d,)."""
    x = args[0]
    weight = args[1]
    eps = float(kwargs.get("eps", 1e-5))
    # Compute in fp32 for numerical stability when training in fp16/bf16.
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    rms = xf.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (xf * rms * weight.to(torch.float32)).to(orig_dtype)


def _op_tril(args, kwargs):
    """Lower-triangular mask (or copy) of input. diagonal=0 keeps main diag."""
    diag = int(kwargs.get("diagonal", 0))
    return torch.tril(args[0], diagonal=diag)


def _op_triu(args, kwargs):
    diag = int(kwargs.get("diagonal", 0))
    return torch.triu(args[0], diagonal=diag)


def _op_full(args, kwargs):
    """Constant tensor of given shape filled with `value`."""
    shape = list(kwargs["shape"])
    value = kwargs.get("value", 0.0)
    dtype = _dtype(kwargs.get("dtype", "float32"))
    out = torch.full(shape, value, dtype=dtype)
    if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu":
        out = out.to(_ACTIVE_DEVICE)
    return out


def _op_cross_entropy(args, kwargs):
    logits = args[0]
    targets = args[1].to(torch.int64)
    ignore_index = int(kwargs.get("ignore_index", -100))
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    return torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, ignore_index=ignore_index, reduction="mean"
    )


def _op_embedding(args, kwargs):
    weight = args[0]
    ids = args[1].to(torch.int64)
    return torch.nn.functional.embedding(ids, weight)


def _op_data_indexer(args, kwargs):
    """Pull a (B, T) input batch and (B, T) target batch from a flat token
    stream.

    args[0]: tokens, 1-D int tensor (uint16 packed as int32 over the wire,
             or int64). Treat as a contiguous corpus.
    kwargs:  B, T, mb_seed.

    For each row in the batch, deterministically (seeded by mb_seed) pick a
    starting offset into the stream, take T+1 consecutive tokens, split into
    input_ids[T] and target_ids[T]. Returns (input_ids, target_ids) of shape
    (B, T) each, dtype int64.
    """
    tokens = args[0]
    B = int(kwargs["B"])
    T = int(kwargs["T"])
    seed_v = kwargs["mb_seed"]
    if isinstance(seed_v, torch.Tensor):
        seed_v = int(seed_v.item())
    mb_seed = int(seed_v)
    if tokens.dim() != 1:
        tokens = tokens.reshape(-1)
    n = tokens.numel()
    if n < T + 1:
        raise ValueError(
            f"data_indexer: token stream too short ({n}) for T+1={T + 1}"
        )
    g = torch.Generator(device="cpu").manual_seed(mb_seed)
    starts = torch.randint(0, n - T - 1, (B,), generator=g)
    tok_cpu = tokens.detach().to("cpu", torch.int64)
    rows = torch.empty((B, T + 1), dtype=torch.int64)
    for i in range(B):
        s = int(starts[i])
        rows[i] = tok_cpu[s : s + T + 1]
    if _ACTIVE_DEVICE is not None and _ACTIVE_DEVICE.type != "cpu":
        rows = rows.to(_ACTIVE_DEVICE)
    input_ids = rows[:, :T].contiguous()
    target_ids = rows[:, 1:].contiguous()
    return (input_ids, target_ids)


# --- quantization ---
def _op_quantize_int8_per_channel(args, kwargs):
    """Quantize per "channel" (the dim specified by kwargs['dim'], default -1).
    Returns (q_int8, scale_fp32) where scale's shape has 1s in all non-dim
    positions so it broadcast-divides correctly on dequant."""
    x = args[0]
    dim = int(kwargs.get("dim", -1))
    if dim < 0:
        dim = x.dim() + dim
    # amax over all dims except `dim`, keepdim=True per non-dim axis
    keep_axes = [a for a in range(x.dim()) if a != dim]
    if keep_axes:
        amax = x.abs()
        for a in keep_axes:
            amax = amax.amax(dim=a, keepdim=True)
    else:
        amax = x.abs().amax(keepdim=True)
    scale = (amax / 127.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x.to(torch.float32) / scale).clamp(-128, 127).to(torch.int8)
    # store as int32 on the wire (we don't have int8 in the wire dtype list)
    return (q.contiguous(), scale.contiguous())


def _op_dequantize_int8_per_channel(args, kwargs):
    q = args[0].to(torch.float32)
    scale = args[1].to(torch.float32)
    return (q * scale).to(torch.float32)


def _op_quantize_pack_int8(args, kwargs):
    """Quantize x to int8 per-channel along `dim`, pack [scale_fp32 | q_int8]
    into a single 1-D uint8 blob.

    Layout:
      packed[: 4*K]: scale tensor (K floats, fp32 -> 4*K uint8 bytes)
      packed[4*K :]: int8 values (numel of x bytes)
    where K = size of x along `dim`.
    """
    x = args[0]
    dim = int(kwargs.get("dim", -1))
    q, scale = _op_quantize_int8_per_channel([x], {"dim": dim})
    # scale has shape with 1s except at dim; flatten to (K,)
    scale_flat = scale.reshape(-1).contiguous().to(torch.float32)
    scale_bytes = scale_flat.view(torch.uint8)
    q_bytes = q.contiguous().reshape(-1).view(torch.uint8)
    packed = torch.cat([scale_bytes, q_bytes], dim=0).contiguous()
    return packed


def _op_unpack_dequantize_int8(args, kwargs):
    """Reverse of pack_int8. kwargs:
      shape: original x shape (used to reshape int8 values)
      dim:   per-channel dim
    """
    packed = args[0].to(torch.uint8) if args[0].dtype != torch.uint8 else args[0]
    target_shape = list(kwargs["shape"])
    dim = int(kwargs.get("dim", -1))
    if dim < 0:
        dim = len(target_shape) + dim
    K = int(target_shape[dim])
    n_scale_bytes = K * 4
    scale_bytes = packed[:n_scale_bytes].contiguous()
    q_bytes = packed[n_scale_bytes:].contiguous()
    scale = scale_bytes.view(torch.float32)
    # Build broadcast scale shape: 1s except at dim
    scale_shape = [1] * len(target_shape)
    scale_shape[dim] = K
    scale = scale.reshape(scale_shape)
    q = q_bytes.view(torch.int8).reshape(target_shape).to(torch.float32)
    out = q * scale
    if _ACTIVE_DEVICE is not None and out.device != _ACTIVE_DEVICE:
        out = out.to(_ACTIVE_DEVICE)
    return out.to(torch.float32)


# --- linalg (extra) ---
def _op_qr(args, kwargs):
    Q, R = torch.linalg.qr(args[0], mode="reduced")
    return (Q.contiguous(), R.contiguous())


_DISPATCH = {
    # elementwise
    "add": _op_add, "sub": _op_sub, "mul": _op_mul, "div": _op_div, "pow": _op_pow,
    "neg": _op_neg, "exp": _op_exp, "log": _op_log, "sqrt": _op_sqrt,
    "abs": _op_abs, "round": _op_round, "sign": _op_sign,
    "relu": _op_relu, "gelu": _op_gelu, "silu": _op_silu,
    "sin": _op_sin, "cos": _op_cos,
    "sigmoid": _op_sigmoid, "tanh": _op_tanh,
    "identity": _op_identity,
    "clamp": _op_clamp, "where": _op_where, "cast": _op_cast,
    # comparisons
    "gt": _op_gt, "lt": _op_lt, "ge": _op_ge, "le": _op_le, "eq": _op_eq,
    # linalg
    "matmul": _op_matmul, "transpose": _op_transpose, "reshape": _op_reshape,
    "einsum": _op_einsum,
    # reductions
    "sum": _op_sum, "mean": _op_mean, "max": _op_max, "min": _op_min,
    # shape
    "concat": _op_concat, "stack": _op_stack, "split": _op_split,
    "broadcast": _op_broadcast, "squeeze": _op_squeeze, "unsqueeze": _op_unsqueeze,
    "slice": _op_slice,
    # indexing
    "gather": _op_gather, "scatter": _op_scatter, "arange": _op_arange,
    # random
    "normal": _op_normal, "uniform": _op_uniform,
    # sort / topk
    "sort": _op_sort, "topk": _op_topk,
    # NN compositions
    "softmax": _op_softmax, "log_softmax": _op_log_softmax,
    "layer_norm": _op_layer_norm, "rmsnorm": _op_rmsnorm,
    "cross_entropy": _op_cross_entropy,
    "embedding": _op_embedding,
    "data_indexer": _op_data_indexer,
    # masks / constants
    "tril": _op_tril, "triu": _op_triu, "full": _op_full,
    # quantization
    "quantize_int8_per_channel": _op_quantize_int8_per_channel,
    "dequantize_int8_per_channel": _op_dequantize_int8_per_channel,
    "quantize_pack_int8": _op_quantize_pack_int8,
    "unpack_dequantize_int8": _op_unpack_dequantize_int8,
    # extra linalg
    "qr": _op_qr,
}


# --------------------------------------------------------------------------- #
# Top-level evaluator
# --------------------------------------------------------------------------- #


def evaluate(
    graph: Graph,
    inputs: dict[str, torch.Tensor],
    params: dict[str, Any] | None = None,
    *,
    bucket=None,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate `graph` on the given input tensors and scalar params.

    `bucket` (optional): a `LocalBucket` (or compatible) instance used to
    resolve `const_blob` refs. Required if the graph contains any const_blob
    refs; ignored otherwise.

    `device` (optional): if set to "cuda" / "cuda:0" / a torch.device, all
    inputs and intermediate tensors are moved to that device for compute.
    Outputs are returned on CPU (always contiguous) so downstream codecs and
    tensor_io don't need device awareness.
    """
    _ensure_determinism()
    global _ACTIVE_DEVICE
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda" and not torch.cuda.is_available():
        dev = None
    prev_device = _ACTIVE_DEVICE
    _ACTIVE_DEVICE = dev

    try:
        params = dict(params or {})
        blob_cache: dict[str, torch.Tensor] = {}

        # validate inputs
        for spec in graph.inputs:
            if spec.name not in inputs:
                raise KeyError(f"missing input tensor: {spec.name!r}")
            t = inputs[spec.name]
            wd = tensor_io.wire_dtype(t)
            if wd != spec.dtype:
                raise TypeError(
                    f"input {spec.name!r}: dtype {wd!r} != declared {spec.dtype!r}"
                )
            decl = list(spec.shape)
            actual = list(t.shape)
            shape_ok = (len(decl) == len(actual)) and all(
                d == -1 or d == a for d, a in zip(decl, actual)
            )
            if not shape_ok:
                raise ValueError(
                    f"input {spec.name!r}: shape {actual} != declared {decl}"
                )
            if dev is not None and t.device != dev:
                inputs[spec.name] = t.to(dev)

        # validate params
        for pspec in graph.params:
            if pspec.name not in params:
                raise KeyError(f"missing param: {pspec.name!r}")

        op_results: list[Any] = [None] * len(graph.ops)
        for op in graph.ops:
            fn = _DISPATCH.get(op.op)
            if fn is None:
                raise KeyError(f"evaluator missing op: {op.op!r}")
            resolved = [
                _resolve_ref(a, inputs, params, op_results,
                             bucket=bucket, blob_cache=blob_cache)
                for a in op.args
            ]
            promoted = [_as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in resolved]
            if dev is not None:
                promoted = [t.to(dev) if isinstance(t, torch.Tensor) and t.device != dev else t for t in promoted]
            kwargs_resolved = _resolve_kwargs(op.kwargs, inputs, params, op_results,
                                              bucket=bucket, blob_cache=blob_cache)
            result = fn(promoted, kwargs_resolved)
            op_results[op.id] = result

        out: dict[str, torch.Tensor] = {}
        for o in graph.outputs:
            v = _resolve_ref(o["ref"], inputs, params, op_results,
                             bucket=bucket, blob_cache=blob_cache)
            if not isinstance(v, torch.Tensor):
                v = _as_tensor(v)
            # Return on CPU so wire codec and downstream code don't need device awareness.
            if dev is not None and v.device.type != "cpu":
                v = v.cpu()
            out[o["name"]] = v.contiguous()
        return out
    finally:
        _ACTIVE_DEVICE = prev_device
