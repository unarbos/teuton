"""Bittensor adapter for Teuton v3 validators.

The adapter is deliberately isolated from the runtime so no-chain fleet tests
can use the same validator with `dry_run=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightUpdate:
    """Outcome of a single set_weights attempt.

    On dry-run, `submitted` is True with synthetic UIDs.
    On chain failure (empty mapping, rate limit, network error), `submitted`
    is False, `reason` is the short failure tag, and the caller can decide to
    sleep / retry. The validator loop treats this as `chain.skipped` /
    `chain.ratelimited` and never crashes.
    """

    uids: list[int]
    weights: list[float]
    submitted: bool = False
    reason: str = ""
    message: str = ""
    extrinsic_hash: str | None = None
    block_hash: str | None = None
    dropped_hotkeys: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BittensorAdapter:
    def __init__(
        self,
        *,
        netuid: int,
        wallet_name: str | None = None,
        hotkey_name: str | None = None,
        network: str | None = None,
        dry_run: bool = True,
    ) -> None:
        self.netuid = int(netuid)
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.network = network
        self.dry_run = dry_run
        self._bt = None
        self._wallet = None
        self._subtensor = None
        if not dry_run:
            import bittensor as bt
            self._bt = bt
            self._wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
            self._subtensor = bt.Subtensor(network=network) if network else bt.Subtensor()

    def hotkey_to_uid(self) -> dict[str, int]:
        if self.dry_run:
            return {}
        metagraph = self._subtensor.metagraph(self.netuid)
        return {hotkey: int(uid) for uid, hotkey in enumerate(metagraph.hotkeys)}

    def publish_weights(self, scores: dict[str, float]) -> WeightUpdate:
        hotkey_to_uid = self.hotkey_to_uid()
        normalized = self.normalize_scores(scores)
        if self.dry_run:
            ordered = sorted(normalized.items())
            update = WeightUpdate(
                uids=list(range(len(ordered))),
                weights=[v for _h, v in ordered],
                submitted=True,
                reason="dry_run",
            )
            print({"dry_run_set_weights": {"uids": update.uids, "weights": update.weights},
                   "hotkeys": [h for h, _v in ordered]})
            return update
        if not normalized:
            return WeightUpdate(uids=[], weights=[], submitted=False, reason="no_scores")
        missing = sorted(set(normalized) - set(hotkey_to_uid))
        if missing:
            print({"dropped_unknown_hotkeys": missing})
        pairs = [(hotkey_to_uid[h], score) for h, score in normalized.items() if h in hotkey_to_uid]
        pairs.sort()
        uids = [uid for uid, _score in pairs]
        weights = [float(score) for _uid, score in pairs]
        if not uids:
            # No score hotkey maps to a current metagraph UID. Used to raise;
            # the validator loop now treats this as a soft skip.
            return WeightUpdate(
                uids=[],
                weights=[],
                submitted=False,
                reason="no_mapped_hotkeys",
                dropped_hotkeys=missing,
            )
        try:
            result = self._subtensor.set_weights(
                wallet=self._wallet,
                netuid=self.netuid,
                uids=uids,
                weights=weights,
            )
        except Exception as e:
            return WeightUpdate(
                uids=uids,
                weights=weights,
                submitted=False,
                reason="set_weights_exception",
                message=repr(e),
                dropped_hotkeys=missing,
            )
        print({"set_weights": result, "uids": uids, "weights": weights})
        # bittensor's set_weights returns (bool, str_or_None) historically;
        # newer versions may return an object with .success / .block_hash.
        submitted = False
        reason = ""
        msg = ""
        ext_hash = None
        blk_hash = None
        extra: dict[str, Any] = {}
        try:
            if isinstance(result, tuple) and len(result) >= 1:
                submitted = bool(result[0])
                if len(result) >= 2:
                    msg = "" if result[1] is None else str(result[1])
                if not submitted and msg:
                    # Heuristic: bittensor returns messages like "WeightsSetRateLimit"
                    low = msg.lower()
                    if "ratelimit" in low or "rate_limit" in low or "too soon" in low:
                        reason = "ratelimited"
                    else:
                        reason = "chain_rejected"
                elif submitted:
                    reason = "success"
            else:
                # Object-style return
                submitted = bool(getattr(result, "success", False) or getattr(result, "is_success", False))
                msg = str(getattr(result, "error_message", "") or getattr(result, "error", "") or "")
                ext_hash = getattr(result, "extrinsic_hash", None)
                blk_hash = getattr(result, "block_hash", None)
                reason = "success" if submitted else ("chain_rejected" if msg else "unknown")
                extra["raw_repr"] = repr(result)[:300]
        except Exception as e:
            extra["parse_error"] = repr(e)
            reason = reason or "parse_error"
        return WeightUpdate(
            uids=uids,
            weights=weights,
            submitted=submitted,
            reason=reason,
            message=msg,
            extrinsic_hash=ext_hash,
            block_hash=blk_hash,
            dropped_hotkeys=missing,
            extra=extra,
        )

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        clean = {k: max(0.0, float(v)) for k, v in scores.items()}
        if not clean:
            return {}
        total = sum(clean.values())
        if total <= 0.0:
            equal = 1.0 / len(clean)
            return {k: equal for k in sorted(clean)}
        return {k: v / total for k, v in clean.items()}
