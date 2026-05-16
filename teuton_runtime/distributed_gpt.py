"""Tensor-parallel GPT execution primitives.

This module is intentionally narrow: it gives the distributed executor a real
NCCL-backed hot path for GPT-style stage work without forcing the generic IR to
learn placement annotations before the parallel strategy settles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TensorParallelContext:
    rank: int
    world_size: int
    device: torch.device
    process_group: Any = None


def _split_for_rank(tensor: torch.Tensor, *, rank: int, world_size: int, dim: int) -> torch.Tensor:
    return torch.tensor_split(tensor, world_size, dim=dim)[rank].contiguous()


def _maybe_all_reduce(tensor: torch.Tensor, ctx: TensorParallelContext) -> torch.Tensor:
    if ctx.world_size <= 1:
        return tensor
    import torch.distributed as dist

    dist.all_reduce(tensor, group=ctx.process_group)
    return tensor


def _maybe_all_gather(tensor: torch.Tensor, ctx: TensorParallelContext, *, dim: int) -> torch.Tensor:
    if ctx.world_size <= 1:
        return tensor
    import torch.distributed as dist

    parts = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
    dist.all_gather(parts, tensor.contiguous(), group=ctx.process_group)
    return torch.cat(parts, dim=dim).contiguous()


class TensorParallelLinear:
    """Megatron-style linear primitive for local tensor parallel ranks."""

    def __init__(self, weight: torch.Tensor, *, ctx: TensorParallelContext, bias: torch.Tensor | None = None) -> None:
        self.weight = weight.to(ctx.device)
        self.bias = bias.to(ctx.device) if bias is not None else None
        self.ctx = ctx

    def column_parallel(self, x: torch.Tensor, *, gather_output: bool = True) -> torch.Tensor:
        w = _split_for_rank(self.weight, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=1)
        b = _split_for_rank(self.bias, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=0) if self.bias is not None else None
        y = x.to(self.ctx.device).matmul(w)
        if b is not None:
            y = y + b
        return _maybe_all_gather(y, self.ctx, dim=-1) if gather_output else y

    def row_parallel(self, x: torch.Tensor, *, input_is_parallel: bool = False) -> torch.Tensor:
        w = _split_for_rank(self.weight, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=0)
        x_part = x.to(self.ctx.device) if input_is_parallel else _split_for_rank(x.to(self.ctx.device), rank=self.ctx.rank, world_size=self.ctx.world_size, dim=-1)
        y = x_part.matmul(w)
        y = _maybe_all_reduce(y, self.ctx)
        if self.bias is not None:
            y = y + self.bias.to(self.ctx.device)
        return y


class GPTTensorParallelRunner:
    """Small dispatch surface for GPT stage jobs inside DistributedJobExecutor.

    The first supported operation is a tensor-parallel MLP/linear stage. Full
    transformer stages can build on the same column/row primitives without
    changing receipts or sharded artifacts again.
    """

    def __init__(self, ctx: TensorParallelContext) -> None:
        self.ctx = ctx

    def run(self, inputs: dict[str, torch.Tensor], params: dict[str, Any]) -> dict[str, torch.Tensor]:
        op = params.get("tp_op", "linear")
        if op == "linear":
            return self.run_linear(inputs, params)
        if op == "linear_backward":
            return self.run_linear_backward(inputs, params)
        if op == "mlp":
            return self.run_mlp(inputs, params)
        if op == "mlp_backward":
            return self.run_mlp_backward(inputs, params)
        raise ValueError(f"unsupported gpt tensor-parallel op: {op!r}")

    def run_linear(self, inputs: dict[str, torch.Tensor], params: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = inputs[params.get("x", "x")]
        w = inputs[params.get("weight", "W")]
        bias_name = params.get("bias")
        bias = inputs[bias_name] if bias_name else None
        mode = params.get("parallel_mode", "column")
        linear = TensorParallelLinear(w, ctx=self.ctx, bias=bias)
        if mode == "row":
            y = linear.row_parallel(x, input_is_parallel=bool(params.get("input_is_parallel", False)))
        elif mode == "column":
            y = linear.column_parallel(x, gather_output=bool(params.get("gather_output", True)))
        else:
            raise ValueError(f"unsupported linear parallel_mode: {mode!r}")
        return {params.get("output", "y"): y.detach().cpu()}

    def run_mlp(self, inputs: dict[str, torch.Tensor], params: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = inputs[params.get("x", "x")]
        w1 = inputs[params.get("w1", "W1")]
        w2 = inputs[params.get("w2", "W2")]
        hidden = TensorParallelLinear(w1, ctx=self.ctx).column_parallel(x, gather_output=False)
        activation = params.get("activation", "gelu")
        if activation == "gelu":
            hidden = torch.nn.functional.gelu(hidden)
        elif activation == "silu":
            hidden = torch.nn.functional.silu(hidden)
        elif activation != "identity":
            raise ValueError(f"unsupported activation: {activation!r}")
        y = TensorParallelLinear(w2, ctx=self.ctx).row_parallel(hidden, input_is_parallel=True)
        return {params.get("output", "y"): y.detach().cpu()}

    def run_linear_backward(self, inputs: dict[str, torch.Tensor], params: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = inputs[params.get("x", "x")].to(self.ctx.device)
        w = inputs[params.get("weight", "W")].to(self.ctx.device)
        dy = inputs[params.get("dy", "dY")].to(self.ctx.device)
        mode = params.get("parallel_mode", "column")
        if mode == "column":
            w_part = _split_for_rank(w, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=1)
            dy_part = _split_for_rank(dy, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=-1)
            dx = dy_part.matmul(w_part.transpose(-2, -1))
            dx = _maybe_all_reduce(dx, self.ctx)
            dw = self._flat_outer(x, dy_part)
            out = {
                params.get("dx", "dX"): dx.detach().cpu(),
                params.get("dw", "dW"): dw.detach().cpu(),
            }
        elif mode == "row":
            x_part = _split_for_rank(x, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=-1)
            w_part = _split_for_rank(w, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=0)
            dx_part = dy.matmul(w_part.transpose(-2, -1))
            dx = _maybe_all_gather(dx_part, self.ctx, dim=-1) if bool(params.get("gather_dx", True)) else dx_part
            dw = self._flat_outer(x_part, dy)
            out = {
                params.get("dx", "dX"): dx.detach().cpu(),
                params.get("dw", "dW"): dw.detach().cpu(),
            }
        else:
            raise ValueError(f"unsupported linear backward parallel_mode: {mode!r}")
        if params.get("db"):
            out[str(params["db"])] = dy.reshape(-1, dy.shape[-1]).sum(dim=0).detach().cpu()
        return out

    def run_mlp_backward(self, inputs: dict[str, torch.Tensor], params: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = inputs[params.get("x", "x")].to(self.ctx.device)
        w1 = inputs[params.get("w1", "W1")].to(self.ctx.device)
        w2 = inputs[params.get("w2", "W2")].to(self.ctx.device)
        dy = inputs[params.get("dy", "dY")].to(self.ctx.device)
        w1_part = _split_for_rank(w1, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=1)
        w2_part = _split_for_rank(w2, rank=self.ctx.rank, world_size=self.ctx.world_size, dim=0)
        pre = x.matmul(w1_part)
        activation = params.get("activation", "gelu")
        if activation == "gelu":
            hidden = torch.nn.functional.gelu(pre)
            d_hidden = dy.matmul(w2_part.transpose(-2, -1))
            d_pre = d_hidden * self._gelu_grad(pre)
        elif activation == "silu":
            hidden = torch.nn.functional.silu(pre)
            sig = torch.sigmoid(pre)
            d_hidden = dy.matmul(w2_part.transpose(-2, -1))
            d_pre = d_hidden * (sig * (1 + pre * (1 - sig)))
        elif activation == "identity":
            hidden = pre
            d_pre = dy.matmul(w2_part.transpose(-2, -1))
        else:
            raise ValueError(f"unsupported activation: {activation!r}")
        dw2 = self._flat_outer(hidden, dy)
        dw1 = self._flat_outer(x, d_pre)
        dx = d_pre.matmul(w1_part.transpose(-2, -1))
        dx = _maybe_all_reduce(dx, self.ctx)
        return {
            params.get("dx", "dX"): dx.detach().cpu(),
            params.get("dw1", "dW1"): dw1.detach().cpu(),
            params.get("dw2", "dW2"): dw2.detach().cpu(),
        }

    @staticmethod
    def _flat_outer(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1]).transpose(0, 1)
        dy2 = dy.reshape(-1, dy.shape[-1])
        return x2.matmul(dy2)

    @staticmethod
    def _gelu_grad(x: torch.Tensor) -> torch.Tensor:
        # Exact derivative of PyTorch's erf-based GELU approximation.
        c = (2.0 / torch.pi) ** 0.5
        return 0.5 * (1.0 + torch.erf(x / 2.0**0.5)) + 0.5 * x * c * torch.exp(-0.5 * x * x)
