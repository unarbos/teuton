"""Typed tensor wire codec for Locus v3 artifacts."""
from __future__ import annotations

import json
import struct
from collections.abc import Sequence

import numpy as np
import torch

WIRE_VERSION = 1

_TORCH_TO_WIRE: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_WIRE_TO_TORCH: dict[str, torch.dtype] = {v: k for k, v in _TORCH_TO_WIRE.items()}
_DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "int64": 8,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def wire_dtype(t: torch.Tensor) -> str:
    if t.dtype not in _TORCH_TO_WIRE:
        raise ValueError(f"unsupported torch dtype for wire: {t.dtype}")
    return _TORCH_TO_WIRE[t.dtype]


def torch_dtype(wire: str) -> torch.dtype:
    if wire not in _WIRE_TO_TORCH:
        raise ValueError(f"unsupported wire dtype: {wire!r}")
    return _WIRE_TO_TORCH[wire]


def dtype_bytes(wire: str) -> int:
    if wire not in _DTYPE_BYTES:
        raise ValueError(f"unsupported wire dtype: {wire!r}")
    return _DTYPE_BYTES[wire]


def expected_payload_bytes(shape: Sequence[int], dtype: str) -> int:
    n = 1
    for size in shape:
        n *= int(size)
    return n * dtype_bytes(dtype)


def encode_tensor(t: torch.Tensor) -> bytes:
    t_cpu = t.detach().to("cpu").contiguous()
    wd = wire_dtype(t_cpu)
    header = {"shape": list(t_cpu.shape), "dtype": wd, "layout": "row_major", "version": WIRE_VERSION}
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")

    if t_cpu.dtype is torch.bfloat16:
        payload = bytes(t_cpu.view(torch.uint16).contiguous().numpy().tobytes())
    elif t_cpu.dtype is torch.bool:
        payload = bytes(t_cpu.to(torch.uint8).numpy().tobytes())
    else:
        payload = bytes(t_cpu.numpy().tobytes())

    expected = expected_payload_bytes(header["shape"], wd)
    if len(payload) != expected:
        raise AssertionError(f"payload size mismatch: got {len(payload)}, expected {expected} for shape={header['shape']} dtype={wd}")
    return struct.pack("<I", len(header_json)) + header_json + payload


def decode_tensor(blob: bytes) -> torch.Tensor:
    if len(blob) < 4:
        raise ValueError("blob too short for header_len prefix")
    (header_len,) = struct.unpack("<I", blob[:4])
    if 4 + header_len > len(blob):
        raise ValueError("blob truncated in header")
    header = json.loads(blob[4 : 4 + header_len].decode("utf-8"))
    if header.get("version") != WIRE_VERSION:
        raise ValueError(f"unsupported wire version: {header.get('version')}")
    if header.get("layout", "row_major") != "row_major":
        raise ValueError(f"unsupported layout: {header.get('layout')}")

    shape = tuple(int(x) for x in header["shape"])
    wd = header["dtype"]
    payload = blob[4 + header_len :]
    expected = expected_payload_bytes(shape, wd)
    if len(payload) != expected:
        raise ValueError(f"payload truncated: got {len(payload)} bytes, expected {expected} for shape={shape} dtype={wd}")

    if wd == "bfloat16":
        arr = np.frombuffer(payload, dtype=np.uint16).copy()
        arr = arr.reshape(shape) if shape else arr.reshape(())
        return torch.from_numpy(arr).view(torch.bfloat16)
    if wd == "bool":
        arr = np.frombuffer(payload, dtype=np.uint8).copy()
        arr = arr.reshape(shape) if shape else arr.reshape(())
        return torch.from_numpy(arr).to(torch.bool)
    np_dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    arr = np.frombuffer(payload, dtype=np_dtype_map[wd]).copy()
    arr = arr.reshape(shape) if shape else arr.reshape(())
    return torch.from_numpy(arr)
