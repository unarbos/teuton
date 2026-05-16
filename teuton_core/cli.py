"""Command-line entrypoints for Teuton v3."""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import os

from teuton_core.protocol import ArtifactCryptoPolicy, CryptoMode
from teuton_miner.neuron import MinerNeuron, MinerNeuronConfig
from teuton_orchestrator.run_manager import RunConfig, RunManager
from teuton_orchestrator.streaming import StreamingRunConfig, StreamingRunManager
from teuton_runtime.lifecycle import wipe_run
from teuton_runtime.storage import S3Bucket, open_local_bucket
from teuton_validator.audit_jobs import AuditJobConfig, AuditJobManager
from teuton_validator.ledger import summarize_ledger
from teuton_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


def build_bucket(args):
    s3_bucket = getattr(args, "s3_bucket", "") or os.environ.get("S3_BUCKET", "")
    if s3_bucket:
        return S3Bucket(
            bucket=s3_bucket,
            region=getattr(args, "s3_region", "") or os.environ.get("S3_REGION", "us-east-1"),
            access_key=getattr(args, "aws_access_key_id", "") or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=getattr(args, "aws_secret_access_key", "") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=(getattr(args, "s3_endpoint_url", "") or os.environ.get("S3_ENDPOINT_URL") or None),
        )
    return open_local_bucket(args.local_root, args.bucket)


def build_crypto_policy(args) -> ArtifactCryptoPolicy | None:
    mode = getattr(args, "crypto", "none")
    if mode in ("none", "", None):
        return None
    if mode == "drand-timelock":
        mode = CryptoMode.DRAND_TIMELOCK.value
    return ArtifactCryptoPolicy(
        mode=mode,
        required_signer=getattr(args, "required_signer", None) or None,
        key_id=getattr(args, "crypto_key_id", None) or None,
        drand_round=getattr(args, "drand_round", None),
    )


def discovery_heartbeat_ttl(args) -> float | None:
    ttl = getattr(args, "discovery_heartbeat_ttl_sec", 30.0)
    return float(ttl) if ttl and ttl > 0 else None


def parse_audit_eligible_hotkeys(args) -> list[str]:
    """Normalise --audit-eligible-hotkeys (CSV) into a list of SS58 strings.

    Defaults flow from the ``TEUTON_AUDIT_ELIGIBLE_HOTKEYS`` env var (CSV).
    Empty inputs return an empty list so the legacy behaviour is preserved.
    """
    raw = getattr(args, "audit_eligible_hotkeys", "") or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def require_run_id(args: argparse.Namespace) -> str:
    """Resolve the effective run id for long-running role CLIs.

    The argparse default already pulls from $TEUTON_RUN_ID (set by the
    container entrypoint, which itself prefers compose overrides then the
    image-baked value). We still error here if nothing was supplied so that
    operators get a clear message instead of silently mining run_id="".
    """
    run_id = (getattr(args, "run_id", "") or "").strip()
    if not run_id:
        raise SystemExit(
            "error: --run-id is empty. Provide --run-id, set TEUTON_RUN_ID/RUN_ID "
            "in the environment, or rebuild the image with --build-arg TEUTON_RUN_ID=..."
        )
    args.run_id = run_id
    return run_id


def cmd_local_smoke(args: argparse.Namespace) -> int:
    root = args.local_root or tempfile.mkdtemp(prefix="teuton-v3-")
    bucket = open_local_bucket(root, args.bucket)
    run_id = args.run_id
    miners = []
    for i in range(args.miners):
        fault = args.fault_mode if i == args.bad_miner_index else ""
        miners.append(
            MinerNeuron(
                bucket=bucket,
                config=MinerNeuronConfig(
                    netuid=args.netuid,
                    run_id=run_id,
                    hotkey_ss58=f"miner{i}",
                    devices=["cpu"],
                    fault_mode=fault,
                    fault_rate=args.fault_rate,
                    miner_secret=args.miner_secret,
                    encryption_secret=args.encryption_secret,
                    assignment_crypto=args.assignment_crypto,
                    discovery_backend=args.discovery_backend,
                ),
            )
        )
    threads = []
    for miner in miners:
        t = threading.Thread(target=miner.loop, daemon=True)
        t.start()
        threads.append(t)
    orch = RunManager(
        bucket=bucket,
        config=RunConfig(
            netuid=args.netuid,
            run_id=run_id,
            task=args.task,
            max_steps=args.steps,
            owner_secret=args.owner_secret,
            crypto_policy=build_crypto_policy(args),
            grant_mode=args.grant_mode,
            grant_ttl_sec=args.grant_ttl_sec,
            assignment_secret=args.assignment_secret,
            assignment_crypto=args.assignment_crypto,
            network=args.network,
            discovery_backend=args.discovery_backend,
            discovery_heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
        ),
    )
    orch.run_loop(timeout_sec=args.timeout_sec)
    for miner in miners:
        miner.stop()
    validator = ValidatorNeuron(
        bucket=bucket,
        config=ValidatorNeuronConfig(
            netuid=args.netuid,
            run_id=run_id,
            validator_hotkey="validator0",
            sample_rate=args.sample_rate,
            dry_run_weights=True,
            owner_secret=args.owner_secret,
            miner_secret=args.miner_secret,
            validator_secret=args.validator_secret,
            encryption_secret=args.encryption_secret,
            audit_mode="local",
        ),
    )
    result = validator.run_once(max_receipts=10_000, publish_weights=True)
    print(json.dumps({"root": root, "run_id": run_id, **result}, indent=2, sort_keys=True))
    return 0


def cmd_orchestrator(args: argparse.Namespace) -> int:
    require_run_id(args)
    bucket = build_bucket(args)
    if args.task in {"gpt_pipe"}:
        manager = StreamingRunManager(
            bucket=bucket,
            config=StreamingRunConfig(
                netuid=args.netuid,
                run_id=args.run_id,
                task=args.task,
                max_epochs=args.steps,
                owner_secret=args.owner_secret,
                crypto_policy=build_crypto_policy(args),
                grant_mode=args.grant_mode,
                grant_ttl_sec=args.grant_ttl_sec,
                assignment_secret=args.assignment_secret,
                assignment_crypto=args.assignment_crypto,
                network=args.network,
                discovery_backend=args.discovery_backend,
                discovery_heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
            ),
        )
        manager.run_loop(poll_interval=args.poll_interval, timeout_sec=args.timeout_sec)
        return 0
    manager = RunManager(
        bucket=bucket,
        config=RunConfig(
            netuid=args.netuid,
            run_id=args.run_id,
            task=args.task,
            max_steps=args.steps,
            owner_secret=args.owner_secret,
            crypto_policy=build_crypto_policy(args),
            grant_mode=args.grant_mode,
            grant_ttl_sec=args.grant_ttl_sec,
            assignment_secret=args.assignment_secret,
            assignment_crypto=args.assignment_crypto,
            network=args.network,
            discovery_backend=args.discovery_backend,
            discovery_heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
        ),
    )
    manager.run_loop(poll_interval=args.poll_interval, timeout_sec=args.timeout_sec)
    return 0


def cmd_miner(args: argparse.Namespace) -> int:
    require_run_id(args)
    bucket = build_bucket(args)
    devices = args.devices.split(",") if args.devices else ["cpu"]
    device_group = args.device_group.split(",") if args.device_group else None
    miner = MinerNeuron(
        bucket=bucket,
        config=MinerNeuronConfig(
            netuid=args.netuid,
            run_id=args.run_id,
            hotkey_ss58=args.hotkey,
            devices=devices,
            device_group=device_group,
            poll_interval=args.poll_interval,
            fault_mode=args.fault_mode,
            fault_rate=args.fault_rate,
            miner_secret=args.miner_secret,
            encryption_secret=args.encryption_secret,
            grant_mode=args.grant_mode,
            assignment_secret=args.assignment_secret,
            assignment_crypto=args.assignment_crypto,
            wallet_path=args.wallet_path,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            discovery_backend=args.discovery_backend,
            audit_eligible_hotkeys=parse_audit_eligible_hotkeys(args),
            owner_secret=args.owner_secret,
        ),
    )
    try:
        miner.loop()
    except KeyboardInterrupt:
        miner.stop()
    return 0


def cmd_validator(args: argparse.Namespace) -> int:
    require_run_id(args)
    bucket = build_bucket(args)
    validator = ValidatorNeuron(
        bucket=bucket,
        config=ValidatorNeuronConfig(
            netuid=args.netuid,
            run_id=args.run_id,
            validator_hotkey=args.validator_hotkey,
            device=args.device,
            sample_rate=args.sample_rate,
            dry_run_weights=not args.set_weights,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            network=args.network,
            owner_secret=args.owner_secret,
            miner_secret=args.miner_secret,
            validator_secret=args.validator_secret,
            encryption_secret=args.encryption_secret,
            audit_mode=args.audit_mode,
            audit_eligible_hotkeys=parse_audit_eligible_hotkeys(args),
        ),
    )
    result = validator.run_once(max_receipts=args.max_receipts, publish_weights=args.publish_weights)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_audit_jobs(args: argparse.Namespace) -> int:
    require_run_id(args)
    bucket = build_bucket(args)
    manager = AuditJobManager(
        bucket=bucket,
        config=AuditJobConfig(
            netuid=args.netuid,
            run_id=args.run_id,
            validator_hotkey=args.validator_hotkey,
            owner_secret=args.owner_secret,
            assignment_secret=args.assignment_secret,
            assignment_crypto=args.assignment_crypto,
            network=args.network,
            grant_mode=args.grant_mode,
            grant_ttl_sec=args.grant_ttl_sec,
            sample_rate=args.sample_rate,
            discovery_backend=args.discovery_backend,
            discovery_heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
            audit_eligible_hotkeys=parse_audit_eligible_hotkeys(args),
        ),
    )
    emitted = manager.run_once(max_jobs=args.max_jobs)
    print(json.dumps({"emitted": emitted, "netuid": args.netuid, "run_id": args.run_id}, sort_keys=True))
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    n = wipe_run(bucket, netuid=args.netuid, run_id=args.run_id)
    print(json.dumps({"deleted": n, "netuid": args.netuid, "run_id": args.run_id}, sort_keys=True))
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    summary = summarize_ledger(
        bucket,
        netuid=args.netuid,
        run_id=args.run_id,
        window_id=args.window_id,
        validator_secret=args.validator_secret,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_discovery_ui(args: argparse.Namespace) -> int:
    from teuton_core.discovery_ui import serve_discovery_ui

    bucket = build_bucket(args)
    serve_discovery_ui(
        bucket=bucket,
        netuid=args.netuid,
        run_id=args.run_id or None,
        heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
        refresh_sec=args.refresh_sec,
        snapshot_cache_sec=args.snapshot_cache_sec,
        max_jobs=args.max_jobs,
        max_artifacts=args.max_artifacts,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


def add_bucket_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local-root", default="/tmp/teuton-v3")
    p.add_argument("--bucket", default="teuton-v3")
    p.add_argument("--s3-bucket", default="")
    p.add_argument("--s3-region", default="us-east-1")
    p.add_argument("--s3-endpoint-url", default="")
    p.add_argument("--aws-access-key-id", default="")
    p.add_argument("--aws-secret-access-key", default="")


def add_crypto_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--crypto", choices=["none", "signed", "encrypted", "drand-timelock"], default="none")
    p.add_argument("--crypto-key-id", default=None)
    p.add_argument("--required-signer", default=None)
    p.add_argument("--drand-round", type=int, default=None)


def add_grant_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grant-mode", choices=["direct", "local", "presigned"], default="direct")
    p.add_argument("--grant-ttl-sec", type=int, default=600)
    p.add_argument("--assignment-secret", default=os.environ.get("TEUTON_ASSIGNMENT_SECRET", "teuton-dev-assignment"))
    p.add_argument("--assignment-crypto", choices=["dev", "ed25519"], default=os.environ.get("TEUTON_ASSIGNMENT_CRYPTO", "dev"))


def add_discovery_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--discovery-backend", choices=["bucket"], default=os.environ.get("TEUTON_DISCOVERY_BACKEND", "bucket"))
    p.add_argument(
        "--discovery-heartbeat-ttl-sec",
        type=float,
        default=float(os.environ.get("TEUTON_DISCOVERY_HEARTBEAT_TTL_SEC") or "30"),
        help="Seconds before a discovery heartbeat is treated as stale. Use 0 to disable.",
    )


def add_audit_eligible_args(p: argparse.ArgumentParser) -> None:
    """Comma-separated SS58 allowlist for who may emit / consume audits.

    Used by the validator, the audit-jobs emitter, and audit-eligible
    miners. Empty (the default) means "no allowlist" and preserves the
    legacy off-chain auditor behaviour for back-compat.
    """
    p.add_argument(
        "--audit-eligible-hotkeys",
        default=os.environ.get("TEUTON_AUDIT_ELIGIBLE_HOTKEYS", ""),
        help=(
            "Comma-separated SS58 allowlist of on-chain miner hotkeys that "
            "may receive audit_replay jobs and whose AuditResultV3 the "
            "validator will trust. Default: $TEUTON_AUDIT_ELIGIBLE_HOTKEYS."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="teuton-v3")
    sub = p.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("local-smoke")
    smoke.add_argument("--netuid", type=int, default=0)
    smoke.add_argument("--run-id", default=f"local-{int(time.time())}")
    smoke.add_argument("--local-root", default="")
    smoke.add_argument("--bucket", default="teuton-v3-smoke")
    smoke.add_argument("--task", default="mlp", choices=["mlp"])
    smoke.add_argument("--steps", type=int, default=1)
    smoke.add_argument("--miners", type=int, default=4)
    smoke.add_argument("--bad-miner-index", type=int, default=-1)
    smoke.add_argument("--fault-mode", default="partial_corrupt")
    smoke.add_argument("--fault-rate", type=float, default=1.0)
    smoke.add_argument("--sample-rate", type=float, default=1.0)
    smoke.add_argument("--timeout-sec", type=float, default=60.0)
    smoke.add_argument("--network", default="finney")
    smoke.add_argument("--owner-secret", default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"))
    smoke.add_argument("--miner-secret", default=os.environ.get("TEUTON_MINER_SECRET", "miner-dev-secret"))
    smoke.add_argument("--validator-secret", default=os.environ.get("TEUTON_VALIDATOR_SECRET", "validator-dev-secret"))
    smoke.add_argument("--encryption-secret", default=os.environ.get("TEUTON_ENCRYPTION_SECRET", "teuton-dev-encryption"))
    add_crypto_args(smoke)
    add_grant_args(smoke)
    add_discovery_args(smoke)
    smoke.set_defaults(fn=cmd_local_smoke)

    orch = sub.add_parser("orchestrator")
    add_bucket_args(orch)
    orch.add_argument("--netuid", type=int, default=0)
    orch.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Defaults to $TEUTON_RUN_ID (set by entrypoint from compose override or image-baked value).",
    )
    orch.add_argument("--task", default="mlp")
    orch.add_argument("--steps", type=int, default=1)
    orch.add_argument("--poll-interval", type=float, default=0.1)
    orch.add_argument("--timeout-sec", type=float, default=600.0)
    orch.add_argument("--network", default="finney")
    orch.add_argument("--owner-secret", default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"))
    add_crypto_args(orch)
    add_grant_args(orch)
    add_discovery_args(orch)
    orch.set_defaults(fn=cmd_orchestrator)

    miner = sub.add_parser("miner")
    add_bucket_args(miner)
    miner.add_argument("--netuid", type=int, default=0)
    miner.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Defaults to $TEUTON_RUN_ID (set by entrypoint from compose override or image-baked value).",
    )
    miner.add_argument("--hotkey", required=True)
    miner.add_argument("--devices", default="cpu")
    miner.add_argument("--device-group", default="", help="Comma-separated GPUs that should execute as one multi-GPU worker group.")
    miner.add_argument("--poll-interval", type=float, default=0.1)
    miner.add_argument("--fault-mode", default="")
    miner.add_argument("--fault-rate", type=float, default=1.0)
    miner.add_argument("--miner-secret", default=os.environ.get("TEUTON_MINER_SECRET", "miner-dev-secret"))
    miner.add_argument("--owner-secret", default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"))
    miner.add_argument("--encryption-secret", default=os.environ.get("TEUTON_ENCRYPTION_SECRET", "teuton-dev-encryption"))
    miner.add_argument("--wallet-path", default=os.environ.get("BT_WALLET_PATH", "~/.bittensor/wallets"))
    miner.add_argument("--wallet-name", default=os.environ.get("BT_WALLET_NAME", ""))
    miner.add_argument("--hotkey-name", default=os.environ.get("BT_HOTKEY_NAME", ""))
    add_grant_args(miner)
    add_discovery_args(miner)
    add_audit_eligible_args(miner)
    miner.set_defaults(fn=cmd_miner)

    val = sub.add_parser("validator")
    add_bucket_args(val)
    val.add_argument("--netuid", type=int, default=0)
    val.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Defaults to $TEUTON_RUN_ID (set by entrypoint from compose override or image-baked value).",
    )
    val.add_argument("--validator-hotkey", default="validator0")
    val.add_argument("--device", default="cpu")
    val.add_argument("--sample-rate", type=float, default=1.0)
    val.add_argument("--max-receipts", type=int, default=None)
    val.add_argument("--publish-weights", action="store_true")
    val.add_argument("--set-weights", action="store_true")
    val.add_argument("--wallet-name", default=None)
    val.add_argument("--hotkey-name", default=None)
    val.add_argument("--network", default=None)
    val.add_argument("--owner-secret", default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"))
    val.add_argument("--miner-secret", default=os.environ.get("TEUTON_MINER_SECRET", "miner-dev-secret"))
    val.add_argument("--validator-secret", default=os.environ.get("TEUTON_VALIDATOR_SECRET", "validator-dev-secret"))
    val.add_argument("--encryption-secret", default=os.environ.get("TEUTON_ENCRYPTION_SECRET", "teuton-dev-encryption"))
    val.add_argument("--audit-mode", choices=["local", "consume"], default="local")
    add_audit_eligible_args(val)
    val.set_defaults(fn=cmd_validator)

    audit = sub.add_parser("audit-jobs")
    add_bucket_args(audit)
    audit.add_argument("--netuid", type=int, default=0)
    audit.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Defaults to $TEUTON_RUN_ID (set by entrypoint from compose override or image-baked value).",
    )
    audit.add_argument("--validator-hotkey", default="validator0")
    audit.add_argument("--sample-rate", type=float, default=1.0)
    audit.add_argument("--max-jobs", type=int, default=None)
    audit.add_argument("--network", default="finney")
    audit.add_argument("--owner-secret", default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"))
    add_grant_args(audit)
    add_discovery_args(audit)
    add_audit_eligible_args(audit)
    audit.set_defaults(fn=cmd_audit_jobs)

    wipe = sub.add_parser("wipe-run")
    add_bucket_args(wipe)
    wipe.add_argument("--netuid", type=int, default=0)
    wipe.add_argument("--run-id", required=True)
    wipe.set_defaults(fn=cmd_wipe)

    ledger = sub.add_parser("ledger")
    add_bucket_args(ledger)
    ledger.add_argument("--netuid", type=int, default=0)
    ledger.add_argument("--run-id", required=True)
    ledger.add_argument("--window-id", default=None)
    ledger.add_argument("--validator-secret", default=os.environ.get("TEUTON_VALIDATOR_SECRET", "validator-dev-secret"))
    ledger.set_defaults(fn=cmd_ledger)

    discovery = sub.add_parser("discovery-ui")
    add_bucket_args(discovery)
    discovery.add_argument("--netuid", type=int, default=0)
    discovery.add_argument("--run-id", default="", help="Optional run filter. Omit to show all active runs for the netuid.")
    discovery.add_argument("--host", default="127.0.0.1")
    discovery.add_argument("--port", type=int, default=8765)
    discovery.add_argument("--open-browser", action="store_true")
    discovery.add_argument("--refresh-sec", type=float, default=3.0)
    discovery.add_argument("--snapshot-cache-sec", type=float, default=1.5)
    discovery.add_argument("--max-jobs", type=int, default=500)
    discovery.add_argument("--max-artifacts", type=int, default=300)
    add_discovery_args(discovery)
    discovery.set_defaults(fn=cmd_discovery_ui)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
