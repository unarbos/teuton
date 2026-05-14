"""Compact binary codec for SparseLoCo sparse payloads.

Replaces torch.save/load (which adds ~3x overhead due to PyTorch metadata)
with a tight custom binary format. For 100M params at density=0.03 with
2-bit quantization, this drops payload from ~15 MB to ~5-6 MB (matching
the theoretical-minimum estimate).

Format:
    Container: {name: payload_dict} from sparseloco_compress(..., return_payloads=True)

    DICT_HEADER:
        magic   : 4 bytes 'SLCD'   (SparseLoCo Codec)
        version : uint32 = 1
        n_entries : uint32

    Per entry:
        name_len    : uint32
        name        : utf-8 bytes
        payload_len : uint32
        payload_blob: variable

    Per-tensor payload_blob:
        bits_q       : uint32   (2 or 8)
        bits_idx     : uint32   (12, 16, or 32)
        n_chunks     : uint32
        chunk_size   : uint32
        k_per_chunk  : uint32
        pad          : uint32
        shape_dim    : uint32
        shape        : shape_dim x uint32
        scale        : n_chunks x float32
        q_packed     : variable (depends on bits_q)
        idx_packed   : variable (depends on bits_idx)
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np
import torch


_MAGIC = b"SLCD"
_VERSION = 1


# --------------------------------------------------------------------------- #
# Bit packing primitives
# --------------------------------------------------------------------------- #


def _pack_bits_2(arr_int: np.ndarray) -> bytes:
    """Pack int8 array with values in {-3, -1, 1, 3} into 2 bits each.
    Mapping: -3->0, -1->1, 1->2, 3->3   (i.e. (v+3)//2)
    Output: ceil(n/4) bytes, low-bits first.
    """
    n = arr_int.size
    encoded = ((arr_int.astype(np.int8) + 3) // 2).astype(np.uint8)
    pad = (-n) % 4
    if pad:
        encoded = np.concatenate([encoded, np.zeros(pad, dtype=np.uint8)])
    enc4 = encoded.reshape(-1, 4)
    out = (enc4[:, 0]
           | (enc4[:, 1] << 2)
           | (enc4[:, 2] << 4)
           | (enc4[:, 3] << 6))
    return out.astype(np.uint8).tobytes()


def _unpack_bits_2(data: bytes, n: int) -> np.ndarray:
    """Inverse of _pack_bits_2. Returns int32 array of length n."""
    packed = np.frombuffer(data, dtype=np.uint8)
    decoded = np.empty(packed.size * 4, dtype=np.uint8)
    decoded[0::4] = packed & 0x3
    decoded[1::4] = (packed >> 2) & 0x3
    decoded[2::4] = (packed >> 4) & 0x3
    decoded[3::4] = (packed >> 6) & 0x3
    decoded = decoded[:n]
    # Inverse: 0->-3, 1->-1, 2->1, 3->3
    return decoded.astype(np.int32) * 2 - 3


def _pack_bits_idx(arr_int: np.ndarray, bits: int) -> bytes:
    """Pack non-negative int indices using `bits` bits per value (12, 16, or 32)."""
    if bits == 32:
        return arr_int.astype(np.int32).tobytes()
    if bits == 16:
        if arr_int.max() >= 65536:
            raise ValueError("idx values exceed 16-bit range")
        return arr_int.astype(np.uint16).tobytes()
    if bits == 12:
        # Pack pairs of 12-bit values into 3 bytes each.
        if arr_int.max() >= 4096:
            raise ValueError("idx values exceed 12-bit range")
        encoded = arr_int.astype(np.uint16) & 0xFFF
        n = encoded.size
        n_pairs = (n + 1) // 2
        pad = n_pairs * 2 - n
        if pad:
            encoded = np.concatenate([encoded, np.zeros(pad, dtype=np.uint16)])
        v0 = encoded[0::2]
        v1 = encoded[1::2]
        out = np.zeros(n_pairs * 3, dtype=np.uint8)
        out[0::3] = (v0 & 0xFF).astype(np.uint8)
        out[1::3] = (((v0 >> 8) & 0xF) | ((v1 & 0xF) << 4)).astype(np.uint8)
        out[2::3] = ((v1 >> 4) & 0xFF).astype(np.uint8)
        return out.tobytes()
    raise NotImplementedError(f"bits_idx={bits}")


def _unpack_bits_idx(data: bytes, bits: int, n: int) -> np.ndarray:
    if bits == 32:
        return np.frombuffer(data, dtype=np.int32)[:n].copy()
    if bits == 16:
        return np.frombuffer(data, dtype=np.uint16)[:n].astype(np.int32)
    if bits == 12:
        n_pairs = (n + 1) // 2
        packed = np.frombuffer(data, dtype=np.uint8)[: n_pairs * 3]
        b0 = packed[0::3].astype(np.uint16)
        b1 = packed[1::3].astype(np.uint16)
        b2 = packed[2::3].astype(np.uint16)
        v0 = b0 | ((b1 & 0xF) << 8)
        v1 = ((b1 >> 4) & 0xF) | (b2 << 4)
        out = np.empty(n_pairs * 2, dtype=np.uint16)
        out[0::2] = v0
        out[1::2] = v1
        return out[:n].astype(np.int32)
    raise NotImplementedError(f"bits_idx={bits}")


# --------------------------------------------------------------------------- #
# Per-tensor payload encode/decode
# --------------------------------------------------------------------------- #


def _choose_bits(chunk_size: int, q_bits: int) -> tuple[int, int]:
    """Pick optimal bit widths for q and idx given chunk_size and q quant level."""
    bits_q = 2 if q_bits == 2 else 8
    if chunk_size <= 4096:
        bits_idx = 12
    elif chunk_size <= 65536:
        bits_idx = 16
    else:
        bits_idx = 32
    return bits_q, bits_idx


def _encode_one(payload: dict) -> bytes:
    """Encode a single per-tensor payload (output of chunked_topk_quant_encode)."""
    q = payload["q"]                   # int8 tensor
    scale = payload["scale"]           # fp32 tensor
    idx = payload["idx"]               # int32 tensor
    pad = int(payload["pad"])
    shape = list(payload["shape"])
    chunk_size = int(payload["chunk_size"])

    # to numpy
    q_np = q.detach().cpu().numpy().reshape(-1)
    scale_np = scale.detach().cpu().to(torch.float32).numpy().reshape(-1)
    idx_np = idx.detach().cpu().to(torch.int32).numpy().reshape(-1)
    n_chunks, k = idx.shape

    # Determine if q is quantized (values in {-3,-1,1,3}) or raw int8
    if q_np.size > 0:
        q_unique = np.unique(q_np)
        is_2bit = set(q_unique.tolist()).issubset({-3, -1, 1, 3, 0})
    else:
        is_2bit = True
    bits_q, bits_idx = _choose_bits(chunk_size, 2 if is_2bit else 8)

    q_packed = _pack_bits_2(q_np) if bits_q == 2 else q_np.astype(np.int8).tobytes()
    idx_packed = _pack_bits_idx(idx_np, bits_idx)

    # Header
    shape_dim = len(shape)
    header = struct.pack(
        f"<7I{shape_dim}I",
        bits_q, bits_idx, n_chunks, chunk_size, k, pad, shape_dim,
        *shape,
    )
    body = scale_np.astype(np.float32).tobytes() + q_packed + idx_packed
    return header + body


def _decode_one(buf: bytes) -> dict:
    """Decode a single per-tensor payload to a dict equivalent to
    chunked_topk_quant_encode's payload (q int8 tensor, scale fp32, idx int32)."""
    pos = 0
    bits_q, bits_idx, n_chunks, chunk_size, k, pad, shape_dim = struct.unpack_from(
        "<7I", buf, pos
    )
    pos += 4 * 7
    shape = list(struct.unpack_from(f"<{shape_dim}I", buf, pos))
    pos += 4 * shape_dim

    scale_n = n_chunks
    scale_bytes = scale_n * 4
    scale_np = np.frombuffer(buf, dtype=np.float32, count=scale_n, offset=pos).copy()
    pos += scale_bytes

    n_q = n_chunks * k
    if bits_q == 2:
        q_packed_len = (n_q + 3) // 4
    else:
        q_packed_len = n_q
    q_bytes = buf[pos: pos + q_packed_len]
    pos += q_packed_len
    if bits_q == 2:
        q_np = _unpack_bits_2(q_bytes, n_q).astype(np.int8)
    else:
        q_np = np.frombuffer(q_bytes, dtype=np.int8).copy()

    if bits_idx == 12:
        idx_packed_len = ((n_q + 1) // 2) * 3
    elif bits_idx == 16:
        idx_packed_len = n_q * 2
    else:
        idx_packed_len = n_q * 4
    idx_bytes = buf[pos: pos + idx_packed_len]
    pos += idx_packed_len
    idx_np = _unpack_bits_idx(idx_bytes, bits_idx, n_q)

    return {
        "q": torch.from_numpy(q_np).reshape(n_chunks, k),
        "scale": torch.from_numpy(scale_np).reshape(n_chunks, 1),
        "idx": torch.from_numpy(idx_np).reshape(n_chunks, k),
        "pad": pad,
        "shape": shape,
        "chunk_size": chunk_size,
        "_consumed": pos,   # internal: bytes used, useful for streaming decode
    }


# --------------------------------------------------------------------------- #
# Container encode/decode (dict[str, payload])
# --------------------------------------------------------------------------- #


def encode_payloads(payloads: dict) -> bytes:
    """Encode a {name: payload_dict} container into compact binary bytes."""
    parts = [_MAGIC, struct.pack("<2I", _VERSION, len(payloads))]
    for name, p in payloads.items():
        name_b = name.encode("utf-8")
        body = _encode_one(p)
        parts.append(struct.pack("<I", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<I", len(body)))
        parts.append(body)
    return b"".join(parts)


def decode_payloads(data: bytes) -> dict:
    """Inverse of encode_payloads."""
    if data[:4] != _MAGIC:
        raise ValueError(f"bad magic: {data[:4]!r}, expected {_MAGIC!r}")
    pos = 4
    version, n_entries = struct.unpack_from("<2I", data, pos)
    if version != _VERSION:
        raise ValueError(f"unsupported codec version: {version}")
    pos += 8
    out = {}
    for _ in range(n_entries):
        (name_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos: pos + name_len].decode("utf-8")
        pos += name_len
        (body_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        body = data[pos: pos + body_len]
        pos += body_len
        decoded = _decode_one(body)
        decoded.pop("_consumed", None)
        out[name] = decoded
    return out


__all__ = ["encode_payloads", "decode_payloads"]
