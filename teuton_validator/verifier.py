"""Replay verification and score windows for Teuton v3."""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any

from teuton_core import paths
from teuton_core.protocol import AuditResultV3, JobManifestV3, JobReceiptV3, MinerScoreWindow, VerificationVerdictV3
from teuton_core.signatures import verify_dict
from teuton_runtime.crypto import DrandTimelockProvider
from teuton_runtime.storage import ObjectStore
from .audit import AuditReplayConfig, AuditReplayRunner


@dataclass
class ValidatorConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    validator_secret: str = "validator-dev-secret"
    owner_secret: str = "owner-dev-secret"
    miner_secret: str = "miner-dev-secret"
    device: str = "cpu"
    sample_rate: float = 1.0
    max_sample_elements: int = 4096
    encryption_secret: str = "teuton-dev-encryption"
    timelock_provider: DrandTimelockProvider | None = None
    # Operator-controlled allowlist. When non-empty the verifier will only
    # accept AuditResultV3 entries whose ``auditor_hotkey`` is in this set;
    # results from non-allowlisted hotkeys are silently dropped so a
    # compromised / unknown auditor can't influence verdicts.
    audit_eligible_hotkeys: list[str] = field(default_factory=list)


class ReplayVerifier:
    def __init__(self, *, bucket: ObjectStore, config: ValidatorConfig) -> None:
        self.bucket = bucket
        self.config = config

    def run_once(self, *, max_receipts: int | None = None) -> int:
        checked = 0
        for uri, receipt in self.sample_receipts():
            if max_receipts is not None and checked >= max_receipts:
                break
            if self.has_verdict(receipt):
                continue
            verdict = self.verify(uri, receipt)
            self.bucket.put_json(
                self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, receipt.run_id, self.config.validator_hotkey, receipt.receipt_id)),
                verdict.to_dict(),
            )
            checked += 1
        return checked

    def consume_audit_results(self, *, max_receipts: int | None = None) -> int:
        checked = 0
        for _uri, receipt in self.sample_receipts():
            if max_receipts is not None and checked >= max_receipts:
                break
            if self.has_verdict(receipt):
                continue
            audit = self.find_audit_result(receipt)
            if audit is None:
                continue
            if not self.verify_audit_result(audit, receipt):
                continue
            verdict = self.verdict_from_audit(receipt, audit)
            self.bucket.put_json(
                self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, receipt.run_id, self.config.validator_hotkey, receipt.receipt_id)),
                verdict.to_dict(),
            )
            checked += 1
        return checked

    def sample_receipts(self) -> list[tuple[str, JobReceiptV3]]:
        prefix = self.bucket.uri_for_key(paths.receipts_prefix(self.config.netuid, self.config.run_id))
        out: list[tuple[str, JobReceiptV3]] = []
        for uri in self.bucket.list(prefix):
            if not uri.endswith(".json"):
                continue
            try:
                receipt = JobReceiptV3.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if self.config.sample_rate < 1.0:
                h = hashlib.sha256(f"{self.config.validator_hotkey}:{receipt.receipt_id}".encode()).digest()
                if int.from_bytes(h[:8], "big") / float(2**64) >= self.config.sample_rate:
                    continue
            out.append((uri, receipt))
        random.Random(17).shuffle(out)
        return out

    def has_verdict(self, receipt: JobReceiptV3) -> bool:
        return self.bucket.exists(
            self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, receipt.run_id, self.config.validator_hotkey, receipt.receipt_id))
        )

    def verify(self, receipt_uri: str, receipt: JobReceiptV3) -> VerificationVerdictV3:
        audit = AuditReplayRunner(bucket=self.bucket, config=self.audit_config()).run(
            receipt_uri=receipt_uri,
            manifest=self.find_manifest(receipt),
            receipt=receipt,
            auditor_hotkey=self.config.validator_hotkey,
        )
        return self.verdict_from_audit(receipt, audit)

    def find_manifest(self, receipt: JobReceiptV3) -> JobManifestV3:
        uri = self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, receipt.run_id, receipt.job_id))
        return JobManifestV3.from_dict(self.bucket.get_json(uri))

    def audit_config(self) -> AuditReplayConfig:
        return AuditReplayConfig(
            owner_secret=self.config.owner_secret,
            miner_secret=self.config.miner_secret,
            device=self.config.device,
            max_sample_elements=self.config.max_sample_elements,
            encryption_secret=self.config.encryption_secret,
            timelock_provider=self.config.timelock_provider,
        )

    def find_audit_result(self, receipt: JobReceiptV3) -> AuditResultV3 | None:
        allow = set(self.config.audit_eligible_hotkeys)
        prefix = self.bucket.uri_for_key(paths.audit_results_prefix(self.config.netuid, receipt.run_id))
        for uri in self.bucket.list(prefix):
            if not uri.endswith(".json"):
                continue
            try:
                audit = AuditResultV3.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if audit.receipt_id != receipt.receipt_id:
                continue
            if allow and audit.auditor_hotkey not in allow:
                continue
            return audit
        return None

    def verify_audit_result(self, audit: AuditResultV3, receipt: JobReceiptV3) -> bool:
        if not audit.auditor_signature:
            return False
        if audit.receipt_id != receipt.receipt_id or audit.manifest_hash != receipt.manifest_hash:
            return False
        allow = self.config.audit_eligible_hotkeys
        if allow and audit.auditor_hotkey not in set(allow):
            return False
        return verify_dict(audit.unsigned_dict(), audit.auditor_hotkey, audit.auditor_signature)

    def verdict_from_audit(self, receipt: JobReceiptV3, audit: AuditResultV3) -> VerificationVerdictV3:
        estimated_cu = estimate_cu(receipt, audit.replay_compute_sec)
        verdict = VerificationVerdictV3(
            verdict_id=f"{self.config.validator_hotkey}:{receipt.receipt_id}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=receipt.run_id,
            miner_hotkey=receipt.worker.hotkey_ss58,
            validator_hotkey=self.config.validator_hotkey,
            status=audit.status,
            reason=audit.reason,
            estimated_cu=estimated_cu,
            replay_compute_sec=audit.replay_compute_sec,
            checked_unix=time.time(),
            comparison={"audit": audit.to_dict()},
        )
        return verdict.sign(self.config.validator_secret)

    def verdict(
        self,
        receipt: JobReceiptV3,
        status: str,
        reason: str,
        replay_compute_sec: float,
        comparison: dict[str, Any],
        t0: float,
    ) -> VerificationVerdictV3:
        estimated_cu = estimate_cu(receipt, replay_compute_sec)
        verdict = VerificationVerdictV3(
            verdict_id=f"{self.config.validator_hotkey}:{receipt.receipt_id}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=receipt.run_id,
            miner_hotkey=receipt.worker.hotkey_ss58,
            validator_hotkey=self.config.validator_hotkey,
            status=status,
            reason=reason,
            estimated_cu=estimated_cu,
            replay_compute_sec=replay_compute_sec,
            checked_unix=time.time(),
            comparison=comparison,
        )
        return verdict.sign(self.config.validator_secret)


def estimate_cu(receipt: JobReceiptV3, replay_compute_sec: float = 0.0) -> float:
    compute = replay_compute_sec if replay_compute_sec > 0 else receipt.compute_sec
    return compute + (receipt.claimed_bytes_read + receipt.claimed_bytes_written) / 1_000_000_000.0


def summarize_scores(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    window_id: str | None = None,
    validator_secret: str = "validator-dev-secret",
) -> dict[str, MinerScoreWindow]:
    window_id = window_id or f"run={run_id}"
    receipts: dict[str, JobReceiptV3] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))):
        if uri.endswith(".json"):
            r = JobReceiptV3.from_dict(bucket.get_json(uri))
            receipts[r.receipt_id] = r
    verdicts: dict[str, VerificationVerdictV3] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.verdicts_prefix(netuid, run_id))):
        if uri.endswith(".json"):
            v = VerificationVerdictV3.from_dict(bucket.get_json(uri))
            if not v.validator_signature or not verify_dict(v.unsigned_dict(), validator_secret, v.validator_signature):
                continue
            verdicts[v.receipt_id] = v
    windows: dict[str, MinerScoreWindow] = {}
    for receipt in receipts.values():
        hotkey = receipt.worker.hotkey_ss58
        w = windows.setdefault(hotkey, MinerScoreWindow(netuid=netuid, window_id=window_id, hotkey_ss58=hotkey))
        w.receipts += 1
        verdict = verdicts.get(receipt.receipt_id)
        cu = estimate_cu(receipt, verdict.replay_compute_sec if verdict else 0.0)
        if verdict is None:
            w.unsampled_cu += cu
        else:
            w.verdicts += 1
            if verdict.status == "pass":
                w.pass_cu += cu
            elif verdict.status == "fail":
                w.fail_cu += cu
            else:
                w.unsampled_cu += cu * 0.5
    for w in windows.values():
        checked = w.pass_cu + w.fail_cu
        if checked == 0:
            w.trust_multiplier = 0.25
        elif w.fail_cu > 0:
            w.trust_multiplier = max(0.0, (w.pass_cu - 2.0 * w.fail_cu) / checked)
        else:
            w.trust_multiplier = 1.0
        w.score = w.pass_cu + w.unsampled_cu * w.trust_multiplier
    bucket.put_json(
        bucket.uri_for_key(paths.scores_key(netuid, window_id)),
        {hotkey: window.to_dict() for hotkey, window in windows.items()},
    )
    return windows
