"""Quota-aware scheduling primitives for Teuton v3."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from teuton_core.protocol import MinerIdentity, ResourceRequirements, WorkerIdentity


# `StreamingRunManager` now releases quota in `wait_epoch` once an emitted
# job's terminal output exists on the bucket. Default back to the protocol
# value (5.0). Operators can still widen via TEUTON_BASE_QUOTA if needed.
#
# With the queue-based design, per-hotkey queue depth is the authoritative
# backpressure signal; ``QuotaBook`` is retained for ``pick_worker``
# priority math (load-balance across accounts by inflight count). The
# default base quota is intentionally low so ``available_quota`` does not
# pre-emptively exclude accounts before queue-depth has a chance to.
_DEFAULT_BASE_QUOTA = float(os.environ.get("TEUTON_BASE_QUOTA", "5.0"))


@dataclass
class MinerAccount:
    identity: MinerIdentity
    workers: dict[str, WorkerIdentity] = field(default_factory=dict)
    inflight_cu: float = 0.0
    verified_good_cu: float = 0.0
    trust_multiplier: float = 1.0
    base_quota: float = _DEFAULT_BASE_QUOTA

    @property
    def max_inflight_cu(self) -> float:
        return self.base_quota + self.verified_good_cu * 0.5

    @property
    def available_quota(self) -> float:
        return max(0.0, self.max_inflight_cu - self.inflight_cu)


class QuotaBook:
    def __init__(self) -> None:
        self.accounts: dict[str, MinerAccount] = {}

    def update_workers(self, identities: list[MinerIdentity], workers: list[WorkerIdentity]) -> None:
        for identity in identities:
            self.accounts.setdefault(identity.hotkey_ss58, MinerAccount(identity=identity))
        for worker in workers:
            account = self.accounts.setdefault(
                worker.hotkey_ss58,
                MinerAccount(identity=MinerIdentity(netuid=0, hotkey_ss58=worker.hotkey_ss58)),
            )
            account.workers[worker.worker_id] = worker

    def pick_worker(
        self,
        *,
        estimated_cu: float = 1.0,
        requirements: ResourceRequirements | None = None,
        hotkey_filter: Callable[[str], bool] | None = None,
    ) -> WorkerIdentity:
        """Pick the highest-priority eligible worker.

        ``hotkey_filter`` is the queue-depth backpressure hook: the
        orchestrator passes ``lambda hk: queue.depth(hk) < max_inflight``
        so accounts whose miner already has a full queue are skipped before
        priority math runs. Without a filter, the legacy QuotaBook quota
        controls eligibility (used by tests and the non-streaming run
        manager).
        """
        requirements = requirements or ResourceRequirements()
        candidates: list[tuple[float, WorkerIdentity, MinerAccount]] = []
        for account in self.accounts.values():
            if hotkey_filter is not None and not hotkey_filter(account.identity.hotkey_ss58):
                continue
            if account.available_quota < estimated_cu or not account.workers:
                continue
            priority = account.trust_multiplier * (1.0 + len(account.workers)) / max(1.0, account.inflight_cu)
            for worker in account.workers.values():
                if not self.worker_satisfies(worker, requirements):
                    continue
                candidates.append((priority, worker, account))
        if not candidates:
            raise RuntimeError("no miner workers have quota for assignment")
        candidates.sort(key=lambda x: (-x[0], x[1].hotkey_ss58, x[1].worker_id))
        _priority, worker, account = candidates[0]
        account.inflight_cu += estimated_cu
        return worker

    @staticmethod
    def worker_satisfies(worker: WorkerIdentity, requirements: ResourceRequirements) -> bool:
        world_size = int(worker.capabilities.get("world_size") or len(worker.device_group) or (1 if worker.gpu_index is not None else 0))
        if requirements.min_gpus > world_size:
            return False
        if requirements.placement == "single_host" and requirements.min_gpus > 1 and not worker.worker_group_id:
            return False
        return True

    def release(self, hotkey: str, estimated_cu: float = 1.0) -> None:
        account = self.accounts.get(hotkey)
        if account is not None:
            account.inflight_cu = max(0.0, account.inflight_cu - estimated_cu)


class CriticalGate:
    """V1 optimistic gate: all jobs can advance once artifacts exist.

    The class exists so v3 has an explicit policy seam for later verified
    barriers around reduce/outer/checkpoint jobs.
    """

    def requires_validation(self, kind: str) -> bool:
        return kind in {"outer_step", "checkpoint"}

    def can_advance_optimistically(self, kind: str) -> bool:
        return not self.requires_validation(kind)
