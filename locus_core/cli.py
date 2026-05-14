"""Command-line entrypoints for Locus v3."""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import os

from locus_core.protocol import ArtifactCryptoPolicy, CryptoMode
from locus_miner.neuron import MinerNeuron, MinerNeuronConfig
from locus_orchestrator.run_manager import RunConfig, RunManager
from locus_orchestrator.streaming import StreamingRunConfig, StreamingRunManager
from locus_runtime.lifecycle import wipe_run
from locus_runtime.storage import S3Bucket, open_local_bucket
from locus_validator.ledger import summarize_ledger
from locus_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig


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


def cmd_local_smoke(args: argparse.Namespace) -> int:
    root = args.local_root or tempfile.mkdtemp(prefix="locus-v3-")
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
        ),
    )
    result = validator.run_once(max_receipts=10_000, publish_weights=True)
    print(json.dumps({"root": root, "run_id": run_id, **result}, indent=2, sort_keys=True))
    return 0


def cmd_orchestrator(args: argparse.Namespace) -> int:
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
        ),
    )
    manager.run_loop(poll_interval=args.poll_interval, timeout_sec=args.timeout_sec)
    return 0


def cmd_miner(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    devices = args.devices.split(",") if args.devices else ["cpu"]
    miner = MinerNeuron(
        bucket=bucket,
        config=MinerNeuronConfig(
            netuid=args.netuid,
            run_id=args.run_id,
            hotkey_ss58=args.hotkey,
            devices=devices,
            poll_interval=args.poll_interval,
            fault_mode=args.fault_mode,
            fault_rate=args.fault_rate,
            miner_secret=args.miner_secret,
            encryption_secret=args.encryption_secret,
            grant_mode=args.grant_mode,
            assignment_secret=args.assignment_secret,
        ),
    )
    try:
        miner.loop()
    except KeyboardInterrupt:
        miner.stop()
    return 0


def cmd_validator(args: argparse.Namespace) -> int:
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
        ),
    )
    result = validator.run_once(max_receipts=args.max_receipts, publish_weights=args.publish_weights)
    print(json.dumps(result, indent=2, sort_keys=True))
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


def add_bucket_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local-root", default="/tmp/locus-v3")
    p.add_argument("--bucket", default="locus-v3")
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
    p.add_argument("--assignment-secret", default=os.environ.get("LOCUS_ASSIGNMENT_SECRET", "locus-dev-assignment"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locus-v3")
    sub = p.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("local-smoke")
    smoke.add_argument("--netuid", type=int, default=0)
    smoke.add_argument("--run-id", default=f"local-{int(time.time())}")
    smoke.add_argument("--local-root", default="")
    smoke.add_argument("--bucket", default="locus-v3-smoke")
    smoke.add_argument("--task", default="mlp", choices=["mlp"])
    smoke.add_argument("--steps", type=int, default=1)
    smoke.add_argument("--miners", type=int, default=4)
    smoke.add_argument("--bad-miner-index", type=int, default=-1)
    smoke.add_argument("--fault-mode", default="partial_corrupt")
    smoke.add_argument("--fault-rate", type=float, default=1.0)
    smoke.add_argument("--sample-rate", type=float, default=1.0)
    smoke.add_argument("--timeout-sec", type=float, default=60.0)
    smoke.add_argument("--owner-secret", default=os.environ.get("LOCUS_OWNER_SECRET", "owner-dev-secret"))
    smoke.add_argument("--miner-secret", default=os.environ.get("LOCUS_MINER_SECRET", "miner-dev-secret"))
    smoke.add_argument("--validator-secret", default=os.environ.get("LOCUS_VALIDATOR_SECRET", "validator-dev-secret"))
    smoke.add_argument("--encryption-secret", default=os.environ.get("LOCUS_ENCRYPTION_SECRET", "locus-dev-encryption"))
    add_crypto_args(smoke)
    add_grant_args(smoke)
    smoke.set_defaults(fn=cmd_local_smoke)

    orch = sub.add_parser("orchestrator")
    add_bucket_args(orch)
    orch.add_argument("--netuid", type=int, default=0)
    orch.add_argument("--run-id", required=True)
    orch.add_argument("--task", default="mlp")
    orch.add_argument("--steps", type=int, default=1)
    orch.add_argument("--poll-interval", type=float, default=0.1)
    orch.add_argument("--timeout-sec", type=float, default=600.0)
    orch.add_argument("--owner-secret", default=os.environ.get("LOCUS_OWNER_SECRET", "owner-dev-secret"))
    add_crypto_args(orch)
    add_grant_args(orch)
    orch.set_defaults(fn=cmd_orchestrator)

    miner = sub.add_parser("miner")
    add_bucket_args(miner)
    miner.add_argument("--netuid", type=int, default=0)
    miner.add_argument("--run-id", required=True)
    miner.add_argument("--hotkey", required=True)
    miner.add_argument("--devices", default="cpu")
    miner.add_argument("--poll-interval", type=float, default=0.1)
    miner.add_argument("--fault-mode", default="")
    miner.add_argument("--fault-rate", type=float, default=1.0)
    miner.add_argument("--miner-secret", default=os.environ.get("LOCUS_MINER_SECRET", "miner-dev-secret"))
    miner.add_argument("--encryption-secret", default=os.environ.get("LOCUS_ENCRYPTION_SECRET", "locus-dev-encryption"))
    add_grant_args(miner)
    miner.set_defaults(fn=cmd_miner)

    val = sub.add_parser("validator")
    add_bucket_args(val)
    val.add_argument("--netuid", type=int, default=0)
    val.add_argument("--run-id", required=True)
    val.add_argument("--validator-hotkey", default="validator0")
    val.add_argument("--device", default="cpu")
    val.add_argument("--sample-rate", type=float, default=1.0)
    val.add_argument("--max-receipts", type=int, default=None)
    val.add_argument("--publish-weights", action="store_true")
    val.add_argument("--set-weights", action="store_true")
    val.add_argument("--wallet-name", default=None)
    val.add_argument("--hotkey-name", default=None)
    val.add_argument("--network", default=None)
    val.add_argument("--owner-secret", default=os.environ.get("LOCUS_OWNER_SECRET", "owner-dev-secret"))
    val.add_argument("--miner-secret", default=os.environ.get("LOCUS_MINER_SECRET", "miner-dev-secret"))
    val.add_argument("--validator-secret", default=os.environ.get("LOCUS_VALIDATOR_SECRET", "validator-dev-secret"))
    val.add_argument("--encryption-secret", default=os.environ.get("LOCUS_ENCRYPTION_SECRET", "locus-dev-encryption"))
    val.set_defaults(fn=cmd_validator)

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
    ledger.add_argument("--validator-secret", default=os.environ.get("LOCUS_VALIDATOR_SECRET", "validator-dev-secret"))
    ledger.set_defaults(fn=cmd_ledger)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
