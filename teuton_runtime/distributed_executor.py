"""Multi-GPU job execution for single-host worker groups."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import torch.multiprocessing as mp

from teuton_core.protocol import ArtifactDigest, ArtifactRef, JobManifestV3, JobReceiptV3, WorkerIdentity
from teuton_core.signatures import HmacSigner
from . import tensor_io
from .crypto import decode_envelope, encode_envelope, DrandTimelockProvider
from .distributed_gpt import GPTTensorParallelRunner, TensorParallelContext
from .executor import JobExecutor
from .sharded_tensor import encode_manifest as encode_sharded_manifest
from .sharded_tensor import get_sharded_tensor
from .sharded_tensor import put_sharded_tensor
from .storage import ObjectStore
from .transport import ArtifactTransport, DirectArtifactTransport


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rank_entry(
    rank: int,
    world_size: int,
    device_group: list[str],
    master_port: int,
    graph_body: bytes,
    manifest_body: bytes,
    inputs: dict[str, torch.Tensor],
    params: dict[str, Any],
    queue,
) -> None:
    import torch.distributed as dist

    device = torch.device(device_group[rank])
    torch.cuda.set_device(device)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        ctx = TensorParallelContext(rank=rank, world_size=world_size, device=device)
        outputs = GPTTensorParallelRunner(ctx).run(inputs, params)
        if rank == 0:
            queue.put({"ok": True, "outputs": outputs})
    except Exception as e:
        if rank == 0:
            queue.put({"ok": False, "error": repr(e)})
        raise
    finally:
        dist.destroy_process_group()


class DistributedJobExecutor:
    """Executor that treats a local GPU set as one schedulable worker."""

    def __init__(
        self,
        *,
        bucket: ObjectStore,
        devices: list[str],
        encryption_secret: str = "teuton-dev-encryption",
        timelock_provider: DrandTimelockProvider | None = None,
        transport: ArtifactTransport | None = None,
    ) -> None:
        if not devices:
            raise ValueError("DistributedJobExecutor requires at least one device")
        self.bucket = bucket
        self.devices = list(devices)
        self.encryption_secret = encryption_secret
        self.timelock_provider = timelock_provider
        self.transport = transport or DirectArtifactTransport(bucket)
        self.single = JobExecutor(
            bucket=bucket,
            device=self.devices[0],
            encryption_secret=encryption_secret,
            timelock_provider=timelock_provider,
            transport=self.transport,
        )

    @property
    def world_size(self) -> int:
        return len(self.devices)

    def execute(
        self,
        manifest: JobManifestV3,
        *,
        worker: WorkerIdentity,
        miner_secret: str,
        fault_mode: str = "",
        fault_rate: float = 1.0,
        grants: dict[str, Any] | None = None,
    ) -> JobReceiptV3:
        runner = manifest.params.get("distributed_runner") or manifest.params.get("runner")
        if self.world_size <= 1 or runner != "gpt_tensor_parallel_v1":
            receipt = self.single.execute(
                manifest,
                worker=worker,
                miner_secret=miner_secret,
                fault_mode=fault_mode,
                fault_rate=fault_rate,
                grants=grants,
            )
            if self.world_size > 1:
                receipt.execution.update(
                    {
                        "mode": "distributed_fallback_single_rank",
                        "world_size": self.world_size,
                        "device_group": list(self.devices),
                        "reason": "manifest did not request gpt_tensor_parallel_v1",
                    }
                )
                receipt.sign(miner_secret)
            return receipt

        return self._execute_gpt_tensor_parallel(manifest, worker=worker, miner_secret=miner_secret, grants=grants)

    def _execute_gpt_tensor_parallel(
        self,
        manifest: JobManifestV3,
        *,
        worker: WorkerIdentity,
        miner_secret: str,
        grants: dict[str, Any] | None,
    ) -> JobReceiptV3:
        if not all(str(d).startswith("cuda") for d in self.devices):
            raise ValueError("gpt_tensor_parallel_v1 requires CUDA devices")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        t0 = time.time()
        graph_body = self.bucket.get(manifest.graph_ref.uri)
        inputs = self._load_inputs(manifest.inputs, grants=grants)
        t_compute = time.time()
        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        mp.spawn(
            _rank_entry,
            args=(
                self.world_size,
                self.devices,
                _free_port(),
                graph_body,
                json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
                inputs,
                dict(manifest.params),
                queue,
            ),
            nprocs=self.world_size,
            join=True,
        )
        result = queue.get()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "distributed rank failed"))
        outputs: dict[str, torch.Tensor] = result["outputs"]
        t_done = time.time()
        put_jobs: list[tuple[str, bytes]] = []
        output_digests: list[ArtifactDigest] = []
        signer = HmacSigner(miner_secret, identity=worker.hotkey_ss58)
        output_sharding = dict(manifest.params.get("output_sharding") or {})
        for ref in manifest.outputs:
            if ref.name not in outputs:
                raise KeyError(f"distributed runner did not produce output {ref.name!r}")
            shard_cfg = output_sharding.get(ref.name)
            if shard_cfg:
                if ref.crypto is not None:
                    raise ValueError("sharded output artifacts do not yet support crypto envelopes")
                sharded = put_sharded_tensor(
                    self.transport,
                    ref.uri,
                    outputs[ref.name],
                    name=ref.name,
                    world_size=int(shard_cfg.get("world_size", self.world_size)),
                    partition_dim=int(shard_cfg.get("partition_dim", 0)),
                    grants=grants,
                )
                manifest_body = encode_sharded_manifest(sharded)
                output_digests.append(
                    ArtifactDigest(
                        name=ref.name,
                        uri=ref.uri,
                        sha256=hashlib.sha256(manifest_body).hexdigest(),
                        size_bytes=len(manifest_body) + sum(int(s.size_bytes or 0) for s in sharded.shards),
                        sharded=sharded,
                    )
                )
                continue
            body = JobExecutor.encode_output(outputs[ref.name], json_uri=ref.uri.endswith(".json"))
            body = encode_envelope(
                body,
                ref.crypto,
                signer=signer,
                encryption_secret=self.encryption_secret,
                timelock_provider=self.timelock_provider,
            )
            put_jobs.append((ref.uri, body))
            output_digests.append(JobExecutor.digest_body(ref.name, ref.uri, body, ref.crypto))
        with ThreadPoolExecutor(max_workers=min(len(put_jobs), 8) or 1) as ex:
            list(ex.map(lambda item: self.transport.put(item[0], item[1], (grants or {}).get(item[0])), put_jobs))
        t_final = time.time()
        input_digests = [
            JobExecutor.digest_body(ref.name, ref.uri, self.transport.get(ref.uri, (grants or {}).get(ref.uri)), ref.crypto)
            for ref in manifest.inputs
        ]
        receipt = JobReceiptV3(
            receipt_id=f"{manifest.run_id}:{manifest.job_id}:{worker.hotkey_ss58}:{worker.worker_id}:{manifest.attempt}",
            manifest_hash=manifest.manifest_hash(),
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            step_id=manifest.step_id,
            kind=manifest.kind,
            worker=worker,
            input_digests=input_digests,
            output_digests=output_digests,
            started_unix=t0,
            finished_unix=t_final,
            compute_sec=t_done - t_compute,
            claimed_bytes_read=sum(d.size_bytes for d in input_digests),
            claimed_bytes_written=sum(d.size_bytes for d in output_digests),
            execution={
                "mode": "gpt_tensor_parallel_v1",
                "world_size": self.world_size,
                "device_group": list(self.devices),
                "rank_backend": "nccl",
                "sharding_plan_hash": hashlib.sha256(json.dumps(manifest.params.get("parallelism", {}), sort_keys=True).encode("utf-8")).hexdigest(),
            },
        )
        return receipt.sign(miner_secret)

    def _load_inputs(self, refs: list[ArtifactRef], *, grants: dict[str, Any] | None) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = {}
        for ref in refs:
            body = self.transport.get(ref.uri, (grants or {}).get(ref.uri))
            if ref.uri.endswith(".json"):
                try:
                    header = json.loads(body.decode("utf-8"))
                    if "shards" in header and "world_size" in header:
                        inputs[ref.name] = get_sharded_tensor(self.transport, ref.uri, grants=grants).detach().cpu()
                        continue
                except Exception:
                    pass
            body = decode_envelope(
                body,
                ref.crypto,
                verifier=HmacSigner("miner-dev-secret"),
                encryption_secret=self.encryption_secret,
                timelock_provider=self.timelock_provider,
            )
            inputs[ref.name] = tensor_io.decode_tensor(body).detach().cpu()
        return inputs
