"""Bittensor-aware miner facade.

The worker loop is bucket-native and can run without a chain. This facade adds
wallet/commitment seams for real subnet operation while preserving local mode.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from teuton_runtime.storage import ObjectStore
from .worker import MinerWorker, WorkerConfig


@dataclass
class MinerNeuronConfig:
    netuid: int
    run_id: str
    hotkey_ss58: str
    devices: list[str]
    device_group: list[str] | None = None
    miner_secret: str = "miner-dev-secret"
    poll_interval: float = 0.1
    fault_mode: str = ""
    fault_rate: float = 1.0
    encryption_secret: str = "teuton-dev-encryption"
    grant_mode: str = "direct"
    assignment_secret: str = "teuton-dev-assignment"
    assignment_crypto: str = "dev"
    wallet_path: str = "~/.bittensor/wallets"
    wallet_name: str = ""
    hotkey_name: str = ""
    discovery_backend: str = "bucket"
    audit_eligible_hotkeys: list[str] = field(default_factory=list)
    owner_secret: str = "owner-dev-secret"


class MinerNeuron:
    def __init__(self, *, bucket: ObjectStore, config: MinerNeuronConfig) -> None:
        self.bucket = bucket
        self.config = config
        if config.device_group:
            self.workers = [
                MinerWorker(
                    bucket=bucket,
                    config=WorkerConfig(
                        netuid=config.netuid,
                        run_id=config.run_id,
                        hotkey_ss58=config.hotkey_ss58,
                        worker_id=f"{config.hotkey_ss58}-group0",
                        device=config.device_group[0],
                        device_group=config.device_group,
                        miner_secret=config.miner_secret,
                        poll_interval=config.poll_interval,
                        fault_mode=config.fault_mode,
                        fault_rate=config.fault_rate,
                        encryption_secret=config.encryption_secret,
                        grant_mode=config.grant_mode,
                        assignment_secret=config.assignment_secret,
                        assignment_crypto=config.assignment_crypto,
                        wallet_path=config.wallet_path,
                        wallet_name=config.wallet_name,
                        hotkey_name=config.hotkey_name,
                        discovery_backend=config.discovery_backend,
                        audit_eligible_hotkeys=list(config.audit_eligible_hotkeys),
                        owner_secret=config.owner_secret,
                    ),
                )
            ]
        else:
            self.workers = [
                MinerWorker(
                    bucket=bucket,
                    config=WorkerConfig(
                        netuid=config.netuid,
                        run_id=config.run_id,
                        hotkey_ss58=config.hotkey_ss58,
                        worker_id=f"{config.hotkey_ss58}-gpu{i}",
                        device=device,
                        miner_secret=config.miner_secret,
                        poll_interval=config.poll_interval,
                        fault_mode=config.fault_mode,
                        fault_rate=config.fault_rate,
                        encryption_secret=config.encryption_secret,
                        grant_mode=config.grant_mode,
                        assignment_secret=config.assignment_secret,
                        assignment_crypto=config.assignment_crypto,
                        wallet_path=config.wallet_path,
                        wallet_name=config.wallet_name,
                        hotkey_name=config.hotkey_name,
                        discovery_backend=config.discovery_backend,
                        audit_eligible_hotkeys=list(config.audit_eligible_hotkeys),
                        owner_secret=config.owner_secret,
                    ),
                )
                for i, device in enumerate(config.devices)
            ]

    def loop(self) -> None:
        threads = [threading.Thread(target=w.loop, daemon=True) for w in self.workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def stop(self) -> None:
        for worker in self.workers:
            worker.stop()
