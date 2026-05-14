"""Validator ledger summaries preserved from the v2 runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from locus_core.protocol import MinerScoreWindow
from .verifier import summarize_scores


@dataclass
class LedgerSummaryV3:
    receipts: int = 0
    verdicts: int = 0
    passed: int = 0
    failed: int = 0
    inconclusive: int = 0
    estimated_cu: float = 0.0
    payable_cu: float = 0.0
    by_hotkey: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipts": int(self.receipts),
            "verdicts": int(self.verdicts),
            "passed": int(self.passed),
            "failed": int(self.failed),
            "inconclusive": int(self.inconclusive),
            "estimated_cu": float(self.estimated_cu),
            "payable_cu": float(self.payable_cu),
            "by_hotkey": self.by_hotkey,
        }


def summarize_ledger(
    bucket,
    *,
    netuid: int,
    run_id: str,
    window_id: str | None = None,
    validator_secret: str = "validator-dev-secret",
) -> LedgerSummaryV3:
    windows: dict[str, MinerScoreWindow] = summarize_scores(
        bucket,
        netuid=netuid,
        run_id=run_id,
        window_id=window_id or f"run={run_id}",
        validator_secret=validator_secret,
    )
    out = LedgerSummaryV3()
    for hotkey, window in windows.items():
        out.receipts += window.receipts
        out.verdicts += window.verdicts
        out.passed += 1 if window.pass_cu > 0 else 0
        out.failed += 1 if window.fail_cu > 0 else 0
        out.inconclusive += 1 if window.unsampled_cu > 0 else 0
        out.estimated_cu += window.pass_cu + window.fail_cu + window.unsampled_cu
        out.payable_cu += window.score
        out.by_hotkey[hotkey] = window.to_dict()
    return out
