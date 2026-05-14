from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator

import pytest

from locus_miner.neuron import MinerNeuron, MinerNeuronConfig
from locus_runtime.storage import LocalBucket


_RUN_COUNTER = itertools.count()


@pytest.fixture
def local_bucket(tmp_path) -> LocalBucket:
    return LocalBucket(root=str(tmp_path), bucket="test-bucket")


@pytest.fixture
def run_id() -> str:
    return f"test-run-{next(_RUN_COUNTER)}"


@pytest.fixture
def start_miners(local_bucket):
    miners: list[MinerNeuron] = []
    threads: list[threading.Thread] = []

    def _start(
        *,
        run_id: str,
        count: int = 2,
        hotkey_prefix: str = "miner",
        fault_index: int | None = None,
        fault_mode: str = "partial_corrupt",
        poll_interval: float = 0.02,
    ) -> list[MinerNeuron]:
        for i in range(count):
            miner = MinerNeuron(
                bucket=local_bucket,
                config=MinerNeuronConfig(
                    netuid=0,
                    run_id=run_id,
                    hotkey_ss58=f"{hotkey_prefix}{i}",
                    devices=["cpu"],
                    poll_interval=poll_interval,
                    fault_mode=fault_mode if fault_index == i else "",
                    fault_rate=1.0,
                ),
            )
            thread = threading.Thread(target=miner.loop, daemon=True)
            thread.start()
            miners.append(miner)
            threads.append(thread)
        return miners

    yield _start

    for miner in miners:
        miner.stop()
    for thread in threads:
        thread.join(timeout=2.0)


@pytest.fixture
def tiny_gpt_pipe(monkeypatch) -> Iterator[object]:
    import locus_tasks.gpt_pipe as gp

    values = {
        "VOCAB": 128,
        "D": 32,
        "N_HEAD": 4,
        "D_FF": 64,
        "T": 8,
        "B": 2,
        "N_STAGES": 2,
        "N_BLOCKS_PER_STAGE": 1,
        "N_MICROBATCHES": 2,
        "MAX_EPOCHS": 1,
        "TIED_EMBED": False,
        "SUBSPACE_K": None,
        "SUBSPACE_K_DY": None,
        "WIRE_INT8": False,
        "K_INNER": 1,
    }
    for name, value in values.items():
        monkeypatch.setattr(gp, name, value)
    yield gp
