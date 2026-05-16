"""Validator neuron facade for Teuton v3."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field

from teuton_core.telemetry import TelemetryWriter
from teuton_runtime.storage import ObjectStore
from .subnet import BittensorAdapter, WeightUpdate
from .verifier import ReplayVerifier, ValidatorConfig, summarize_scores


LOG = logging.getLogger(__name__)


@dataclass
class ValidatorNeuronConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    validator_secret: str = "validator-dev-secret"
    owner_secret: str = "owner-dev-secret"
    miner_secret: str = "miner-dev-secret"
    device: str = "cpu"
    sample_rate: float = 1.0
    encryption_secret: str = "teuton-dev-encryption"
    timelock_provider: object | None = None
    dry_run_weights: bool = True
    wallet_name: str | None = None
    hotkey_name: str | None = None
    network: str | None = None
    audit_mode: str = "local"
    audit_eligible_hotkeys: list[str] = field(default_factory=list)


class ValidatorNeuron:
    def __init__(self, *, bucket: ObjectStore, config: ValidatorNeuronConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.verifier = ReplayVerifier(
            bucket=bucket,
            config=ValidatorConfig(
                netuid=config.netuid,
                run_id=config.run_id,
                validator_hotkey=config.validator_hotkey,
                validator_secret=config.validator_secret,
                owner_secret=config.owner_secret,
                miner_secret=config.miner_secret,
                device=config.device,
                sample_rate=config.sample_rate,
                encryption_secret=config.encryption_secret,
                timelock_provider=config.timelock_provider,
                audit_eligible_hotkeys=list(config.audit_eligible_hotkeys),
            ),
        )
        self.subnet = BittensorAdapter(
            netuid=config.netuid,
            wallet_name=config.wallet_name,
            hotkey_name=config.hotkey_name,
            network=config.network,
            dry_run=config.dry_run_weights,
        )
        # Telemetry writer (best-effort; tolerates missing prefix on bucket).
        self.telemetry = TelemetryWriter(
            bucket=bucket,
            netuid=config.netuid,
            run_id=config.run_id,
            component="validator",
        )

    def run_once(self, *, max_receipts: int | None = None, publish_weights: bool = False) -> dict:
        t0 = time.time()
        if self.config.audit_mode == "consume":
            checked = self.verifier.consume_audit_results(max_receipts=max_receipts)
        else:
            checked = self.verifier.run_once(max_receipts=max_receipts)
        verify_seconds = time.time() - t0

        t1 = time.time()
        windows = summarize_scores(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            window_id=f"run={self.config.run_id}",
            validator_secret=self.config.validator_secret,
        )
        scores = {hotkey: window.score for hotkey, window in windows.items()}
        score_seconds = time.time() - t1

        # Emit the per-pass score snapshot to telemetry (rolling time series).
        try:
            self.telemetry.scores(
                {
                    "validator_hotkey": self.config.validator_hotkey,
                    "audit_mode": self.config.audit_mode,
                    "verify_seconds": round(verify_seconds, 3),
                    "score_seconds": round(score_seconds, 3),
                    "checked": int(checked),
                    "score_windows": {k: v.to_dict() for k, v in windows.items()},
                    "n_hotkeys": len(windows),
                }
            )
        except Exception as e:
            LOG.debug("telemetry.scores emit failed: %r", e)

        update: WeightUpdate | None = None
        if publish_weights:
            try:
                update = self.subnet.publish_weights(scores)
            except Exception as e:
                # Defensive: previous versions could raise; new BittensorAdapter
                # returns a WeightUpdate(submitted=False) instead.
                update = WeightUpdate(
                    uids=[],
                    weights=[],
                    submitted=False,
                    reason="publish_exception",
                    message=repr(e),
                )

            try:
                payload = {
                    "validator_hotkey": self.config.validator_hotkey,
                    "wallet_name": self.config.wallet_name,
                    "hotkey_name": self.config.hotkey_name,
                    "network": self.config.network,
                    "dry_run": self.config.dry_run_weights,
                    "publish_seconds": round(time.time() - t1, 3),
                    "update": asdict(update),
                    "n_score_hotkeys": len(scores),
                }
                self.telemetry.chain(payload)
            except Exception as e:
                LOG.debug("telemetry.chain emit failed: %r", e)

        return {
            "checked": checked,
            "scores": {k: v.to_dict() for k, v in windows.items()},
            "weight_update": asdict(update) if update is not None else None,
        }
