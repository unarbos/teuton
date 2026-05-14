"""Worker capability probes carried forward from the v2 fleet runtime."""
from __future__ import annotations

import socket
import time
from typing import Any

import torch


def gpu_index(device: str) -> int | None:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    if device == "cuda":
        return 0
    return None


def probe_torch_device(device: str) -> None:
    """Fail fast if a CUDA worker cannot do basic compute."""
    if device == "cpu":
        return
    with torch.no_grad():
        a = torch.randn(8, 8, device=device)
        b = torch.randn(8, 8, device=device)
        _ = (a @ b).sum().item()


def detect_capabilities(bucket, *, run_id: str, worker_id: str, device: str) -> dict[str, Any]:
    caps: dict[str, Any] = {
        "device": device,
        "worker_id": worker_id,
        "gpu_available": torch.cuda.is_available(),
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        caps["hostname"] = socket.gethostname()
    except Exception:
        caps["hostname"] = "unknown"

    if torch.cuda.is_available():
        try:
            idx = gpu_index(device) or 0
            props = torch.cuda.get_device_properties(idx)
            caps["gpu_class"] = torch.cuda.get_device_name(idx).replace("NVIDIA ", "").replace("GeForce ", "").strip()
            caps["vram_mb"] = int(props.total_memory / (1024 * 1024))
        except Exception:
            caps.setdefault("gpu_class", "cuda")
    else:
        caps["gpu_class"] = "cpu"

    try:
        t0 = time.time()
        probe = bucket.uri_for_key(f"runs/{run_id}/manifest/_rtt_probe_{worker_id}.txt")
        bucket.put(probe, b"x")
        bucket.exists(probe)
        bucket.delete(probe)
        caps["rtt_to_bucket_ms"] = round((time.time() - t0) * 1000.0, 1)
    except Exception:
        caps["rtt_to_bucket_ms"] = 1000.0
    return caps
