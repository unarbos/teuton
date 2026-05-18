"""Command-line entrypoints for Teuton v3."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from typing import Any

from teuton_core import cli_jobs, cli_views
from teuton_core.protocol import ArtifactCryptoPolicy, CryptoMode
from teuton_core.signatures import Signer, load_wallet_hotkey_signer
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


def load_hotkey_signer(args) -> Signer | None:
    wallet_name = getattr(args, "wallet_name", None)
    hotkey_name = getattr(args, "hotkey_name", None)
    wallet_path = getattr(args, "wallet_path", None) or os.environ.get("BT_WALLET_PATH", "~/.bittensor/wallets")
    if not wallet_name or not hotkey_name:
        return None
    return load_wallet_hotkey_signer(
        wallet_path=wallet_path,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
    )


def resolve_hotkey_arg(args, *, attr: str = "hotkey") -> str:
    value = (getattr(args, attr, "") or "").strip()
    if value:
        return value
    signer = load_hotkey_signer(args)
    if signer is None:
        raise SystemExit(f"error: --{attr.replace('_', '-')} is required unless wallet-name and hotkey-name are set")
    setattr(args, attr, signer.identity)
    return signer.identity


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
    owner_signer = load_hotkey_signer(args)
    if args.task in {"gpt_pipe"}:
        manager = StreamingRunManager(
            bucket=bucket,
            config=StreamingRunConfig(
                netuid=args.netuid,
                run_id=args.run_id,
                task=args.task,
                max_epochs=args.steps,
                owner_secret=args.owner_secret,
                owner_signer=owner_signer,
                crypto_policy=build_crypto_policy(args),
                grant_mode=args.grant_mode,
                grant_ttl_sec=args.grant_ttl_sec,
                assignment_secret=args.assignment_secret,
                assignment_crypto=args.assignment_crypto,
                network=args.network,
                discovery_backend=args.discovery_backend,
                discovery_heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
                stress_emit=args.stress_emit,
                stress_emit_interval=args.stress_emit_interval,
                stress_epoch_base=args.stress_epoch_base,
                stress_pin_weights_epoch=args.stress_pin_weights_epoch,
                stress_skip_bootstrap_if_present=not args.stress_force_bootstrap,
                epoch_timeout_sec=args.epoch_timeout_sec,
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
            owner_signer=owner_signer,
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
    resolve_hotkey_arg(args)
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
            owner_hotkey=args.owner_hotkey,
        ),
    )
    try:
        miner.loop()
    except KeyboardInterrupt:
        miner.stop()
    return 0


def cmd_validator(args: argparse.Namespace) -> int:
    require_run_id(args)
    if not args.validator_hotkey:
        resolve_hotkey_arg(args, attr="validator_hotkey")
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
            wallet_path=args.wallet_path,
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
    owner_signer = load_hotkey_signer(args)
    if not args.validator_hotkey and owner_signer is not None:
        args.validator_hotkey = owner_signer.identity
    bucket = build_bucket(args)
    manager = AuditJobManager(
        bucket=bucket,
        config=AuditJobConfig(
            netuid=args.netuid,
            run_id=args.run_id,
            validator_hotkey=args.validator_hotkey,
            owner_secret=args.owner_secret,
            owner_signer=owner_signer,
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
        validator_hotkey=args.validator_hotkey,
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


def cmd_dashboard_backend(args: argparse.Namespace) -> int:
    from teuton_core.dashboard_backend import DashboardConfig, serve_dashboard_backend

    bucket = build_bucket(args)
    serve_dashboard_backend(
        bucket=bucket,
        config=DashboardConfig(
            netuid=args.netuid,
            run_id=args.run_id or None,
            db_path=args.db_path,
            host=args.host,
            port=args.port,
            refresh_sec=args.refresh_sec,
            heartbeat_ttl_sec=discovery_heartbeat_ttl(args),
            max_jobs=args.max_jobs,
            bucket_poll_sec=args.bucket_poll_sec,
            chain_poll_sec=args.chain_poll_sec,
            network=args.network,
            open_browser=args.open_browser,
            max_inflight_per_hotkey=args.max_inflight_per_hotkey,
        ),
    )
    return 0


def _resolve_ls_run_id(args: argparse.Namespace) -> str | None:
    """Translate `--run-id` / `--all-runs` into the value fetch_* expects.

    Defaults to ``$TEUTON_RUN_ID`` if set so the CLI matches the runtime
    entrypoint's resolution order. Use ``--all-runs`` to force a network-wide
    view.
    """
    if getattr(args, "all_runs", False):
        return None
    explicit = (getattr(args, "run_id", "") or "").strip()
    if explicit:
        return explicit
    env = (os.environ.get("TEUTON_RUN_ID") or "").strip()
    return env or None


def cmd_ls_runs(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    rows = cli_views.fetch_runs(bucket=bucket, netuid=args.netuid)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if args.json:
        print(cli_views.to_json(rows))
        return 0
    print(cli_views.render_runs_table(rows))
    return 0


def cmd_ls_miners(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    run_id = _resolve_ls_run_id(args)
    rows = cli_views.fetch_miners(
        bucket=bucket,
        netuid=args.netuid,
        run_id=run_id,
        heartbeat_ttl_sec=args.heartbeat_ttl_sec,
        include_stale=not args.live_only,
        limit=args.limit if args.limit and args.limit > 0 else None,
    )
    if args.json:
        print(cli_views.to_json(rows))
        return 0
    print(cli_views.render_miners_table(rows))
    return 0


def cmd_ls_jobs(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    run_id = _resolve_ls_run_id(args)
    if run_id is None:
        raise SystemExit("error: ls jobs requires --run-id (or $TEUTON_RUN_ID); --all-runs is not supported here")
    rows = cli_views.fetch_jobs(
        bucket=bucket,
        netuid=args.netuid,
        run_id=run_id,
        kind=args.kind or None,
        status=args.status or None,
        limit=args.limit,
    )
    if args.json:
        print(cli_views.to_json(rows))
        return 0
    print(cli_views.render_jobs_table(rows))
    return 0


def cmd_ls_job(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    run_id = _resolve_ls_run_id(args)
    if run_id is None:
        raise SystemExit("error: ls job requires --run-id (or $TEUTON_RUN_ID)")
    detail = cli_views.fetch_job_detail(
        bucket=bucket,
        netuid=args.netuid,
        run_id=run_id,
        job_id=args.job_id,
    )
    if args.json:
        print(cli_views.to_json(detail))
        return 0
    print(cli_views.render_job_detail(detail))
    return 0


def _maybe_wait_for_receipt(
    args: argparse.Namespace,
    *,
    bucket,
    netuid: int,
    run_id: str,
    hotkey: str,
    job_id: str,
) -> tuple[Any, bool]:
    """Honour the shared ``--wait`` / ``--timeout-sec`` / ``--poll-interval``
    flag set. Returns ``(receipt_or_none, timed_out)`` so the caller can decide
    how to render the outcome.
    """
    if not getattr(args, "wait", False):
        return None, False
    receipt = cli_jobs.wait_for_receipt(
        bucket=bucket,
        netuid=netuid,
        run_id=run_id,
        hotkey=hotkey,
        job_id=job_id,
        timeout_sec=args.timeout_sec,
        poll_interval=args.poll_interval,
    )
    return receipt, receipt is None


def cmd_send_job(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    run_id = (args.run_id or os.environ.get("TEUTON_RUN_ID") or "").strip()
    if not run_id:
        raise SystemExit("error: send-job requires --run-id (or $TEUTON_RUN_ID)")
    owner_signer = load_hotkey_signer(args)
    try:
        manifest = cli_jobs.send_pipe_forward_job(
            bucket=bucket,
            netuid=args.netuid,
            run_id=run_id,
            hotkey=args.hotkey,
            worker_id=args.worker_id or None,
            stage=args.stage,
            mb=args.mb,
            epoch=args.epoch if args.epoch > 0 else None,
            owner_secret=args.owner_secret,
            owner_signer=owner_signer,
            grant_mode=args.grant_mode,
            grant_ttl_sec=args.grant_ttl_sec,
            assignment_crypto=args.assignment_crypto,
            assignment_secret=args.assignment_secret,
            network=args.network,
            heartbeat_ttl_sec=args.heartbeat_ttl_sec,
        )
    except LookupError as e:
        raise SystemExit(f"error: {e}") from e
    except FileNotFoundError as e:
        raise SystemExit(f"error: {e}") from e

    receipt, timed_out = _maybe_wait_for_receipt(
        args,
        bucket=bucket,
        netuid=args.netuid,
        run_id=run_id,
        hotkey=manifest.assigned_hotkey,
        job_id=manifest.job_id,
    )
    if args.json:
        result = cli_jobs.SendJobResult(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            assigned_hotkey=manifest.assigned_hotkey,
            assigned_worker=manifest.assigned_worker or "",
            manifest=manifest.to_dict(),
            receipt=receipt.to_dict() if receipt is not None else None,
            timed_out=timed_out,
        )
        print(cli_views.to_json(result))
        return 124 if timed_out else 0
    print(
        cli_jobs.render_send_job_summary(
            job_id=manifest.job_id,
            assigned_hotkey=manifest.assigned_hotkey,
            assigned_worker=manifest.assigned_worker or "",
            receipt=receipt,
            timed_out=timed_out,
            timeout_sec=args.timeout_sec,
        )
    )
    return 124 if timed_out else 0


def cmd_health_check(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    run_id = (args.run_id or os.environ.get("TEUTON_RUN_ID") or "").strip()
    if not run_id:
        raise SystemExit("error: health-check requires --run-id (or $TEUTON_RUN_ID)")
    owner_signer = load_hotkey_signer(args)
    hotkeys = [s.strip() for s in (args.hotkeys or "").split(",") if s.strip()] or None
    total_holder: dict[str, int] = {}

    def _on_done(done: int, total: int, row) -> None:
        total_holder["t"] = total
        if not args.quiet:
            print(
                f"  [{done}/{total}] {row.hotkey_ss58[:8]}... -> {row.status} "
                f"({row.time_to_receipt_sec:.1f}s)" if row.time_to_receipt_sec else
                f"  [{done}/{total}] {row.hotkey_ss58[:8]}... -> {row.status}",
                flush=True,
            )

    rows = cli_jobs.health_check(
        bucket=bucket,
        netuid=args.netuid,
        run_id=run_id,
        hotkeys=hotkeys,
        owner_secret=args.owner_secret,
        owner_signer=owner_signer,
        grant_mode=args.grant_mode,
        grant_ttl_sec=args.grant_ttl_sec,
        assignment_crypto=args.assignment_crypto,
        assignment_secret=args.assignment_secret,
        network=args.network,
        heartbeat_ttl_sec=args.heartbeat_ttl_sec,
        per_miner_timeout_sec=args.per_miner_timeout_sec,
        receipt_poll_interval=args.poll_interval,
        concurrency=args.concurrent,
        progress=_on_done,
    )
    if args.json:
        print(cli_views.to_json(rows))
        return 0
    print(cli_jobs.render_health_check_table(rows))
    return 0


def cmd_stress_stream(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    run_id = (args.run_id or os.environ.get("TEUTON_RUN_ID") or "").strip()
    if not run_id:
        raise SystemExit("error: stress-stream requires --run-id (or $TEUTON_RUN_ID)")
    owner_signer = load_hotkey_signer(args)
    hotkeys = [s.strip() for s in (args.hotkeys or "").split(",") if s.strip()]
    if not hotkeys:
        from teuton_runtime.discovery import scan_bucket_discovery_records
        records = scan_bucket_discovery_records(
            bucket,
            netuid=args.netuid,
            run_id=run_id,
            heartbeat_ttl_sec=60.0,
        )
        hotkeys = sorted({r.worker.hotkey_ss58 for r in records})
        if not hotkeys:
            raise SystemExit("error: no live miners found; pass --hotkeys explicitly")

    def _progress(kind: str, sample, emitted: int, errored: int) -> None:
        if args.quiet:
            return
        if kind == "emit":
            note = f"error: {sample.error}" if sample.error else "emitted"
            print(f"  emit  [{emitted}] {sample.job_id[:30]:<30} -> {sample.hotkey[:8]}... {note}", flush=True)
        else:
            lat = sample.latency_sec or 0.0
            print(f"  recv  {sample.job_id[:30]:<30} latency={lat:.1f}s compute={sample.compute_sec:.3f}s", flush=True)

    report = cli_jobs.stress_stream(
        bucket=bucket,
        netuid=args.netuid,
        run_id=run_id,
        rate_per_min=args.rate_per_min,
        duration_sec=args.duration_sec,
        hotkeys=hotkeys,
        owner_secret=args.owner_secret,
        owner_signer=owner_signer,
        grant_mode=args.grant_mode,
        grant_ttl_sec=args.grant_ttl_sec,
        assignment_crypto=args.assignment_crypto,
        assignment_secret=args.assignment_secret,
        network=args.network,
        receipt_timeout_sec=args.receipt_timeout_sec,
        progress=_progress,
    )
    if args.json:
        print(cli_views.to_json(report))
        return 0
    print(cli_jobs.render_stream_report(report))
    return 0


def cmd_submit_manifest(args: argparse.Namespace) -> int:
    bucket = build_bucket(args)
    owner_signer = load_hotkey_signer(args)
    try:
        manifest = cli_jobs.submit_manifest_file(
            bucket=bucket,
            manifest_path=args.manifest,
            owner_secret=args.owner_secret,
            owner_signer=owner_signer,
            grant_mode=args.grant_mode,
            grant_ttl_sec=args.grant_ttl_sec,
            assignment_crypto=args.assignment_crypto,
            assignment_secret=args.assignment_secret,
            netuid=args.netuid if args.netuid > 0 else None,
            network=args.network,
            resign=args.resign,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"error: {e}") from e

    netuid = args.netuid if args.netuid > 0 else None
    if netuid is None:
        # cli_jobs.submit_manifest_file already inferred it; recover the value
        # from the manifest graph_ref URI for the wait poll.
        for part in (manifest.graph_ref.uri or "").split("/"):
            if part.startswith("netuid="):
                netuid = int(part[len("netuid=") :])
                break
    if netuid is None:
        raise SystemExit("error: could not determine netuid for receipt poll")

    receipt, timed_out = _maybe_wait_for_receipt(
        args,
        bucket=bucket,
        netuid=netuid,
        run_id=manifest.run_id,
        hotkey=manifest.assigned_hotkey,
        job_id=manifest.job_id,
    )
    if args.json:
        result = cli_jobs.SendJobResult(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            assigned_hotkey=manifest.assigned_hotkey,
            assigned_worker=manifest.assigned_worker or "",
            manifest=manifest.to_dict(),
            receipt=receipt.to_dict() if receipt is not None else None,
            timed_out=timed_out,
        )
        print(cli_views.to_json(result))
        return 124 if timed_out else 0
    print(
        cli_jobs.render_send_job_summary(
            job_id=manifest.job_id,
            assigned_hotkey=manifest.assigned_hotkey,
            assigned_worker=manifest.assigned_worker or "",
            receipt=receipt,
            timed_out=timed_out,
            timeout_sec=args.timeout_sec,
        )
    )
    return 124 if timed_out else 0


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
    p.add_argument("--assignment-secret", default="teuton-dev-assignment")
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


def add_wallet_args(p: argparse.ArgumentParser, *, default_wallet: str | None = "") -> None:
    p.add_argument("--wallet-path", default=os.environ.get("BT_WALLET_PATH", "~/.bittensor/wallets"))
    p.add_argument("--wallet-name", default=os.environ.get("BT_WALLET_NAME", default_wallet or ""))
    p.add_argument("--hotkey-name", default=os.environ.get("BT_HOTKEY_NAME", ""))


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
    smoke.add_argument("--owner-secret", default="owner-dev-secret")
    smoke.add_argument("--miner-secret", default="miner-dev-secret")
    smoke.add_argument("--validator-secret", default="validator-dev-secret")
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
    orch.add_argument(
        "--stress-emit",
        action="store_true",
        help="Continuously emit streaming jobs without waiting for epoch completion; intended for load testing.",
    )
    orch.add_argument(
        "--stress-emit-interval",
        type=float,
        default=0.0,
        help="Optional sleep between stress-emitted epochs.",
    )
    orch.add_argument(
        "--stress-epoch-base",
        type=int,
        default=1_000_000,
        help="Synthetic epoch counter starts here; keeps stress job_ids out of any historical j-e* range.",
    )
    orch.add_argument(
        "--stress-pin-weights-epoch",
        type=int,
        default=0,
        help="Pin every stress job's per-stage weight URI to this epoch (default 0 = bootstrap).",
    )
    orch.add_argument(
        "--stress-force-bootstrap",
        action="store_true",
        help="Force task.bootstrap() to run even in stress mode (default: skip when manifest config already present).",
    )
    orch.add_argument(
        "--epoch-timeout-sec",
        type=float,
        default=300.0,
        help="Per-epoch wait deadline so wait_epoch can never silently hang.",
    )
    orch.add_argument("--network", default="finney")
    orch.add_argument("--owner-secret", default="owner-dev-secret")
    add_wallet_args(orch)
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
    miner.add_argument("--hotkey", default=os.environ.get("MINER_HOTKEY_SS58", ""))
    miner.add_argument("--devices", default="cpu")
    miner.add_argument("--device-group", default="", help="Comma-separated GPUs that should execute as one multi-GPU worker group.")
    miner.add_argument("--poll-interval", type=float, default=0.1)
    miner.add_argument("--fault-mode", default="")
    miner.add_argument("--fault-rate", type=float, default=1.0)
    miner.add_argument("--miner-secret", default="miner-dev-secret")
    miner.add_argument("--owner-secret", default="owner-dev-secret")
    miner.add_argument("--encryption-secret", default=os.environ.get("TEUTON_ENCRYPTION_SECRET", "teuton-dev-encryption"))
    add_wallet_args(miner)
    miner.add_argument("--owner-hotkey", default=os.environ.get("TEUTON_OWNER_HOTKEY", ""))
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
    val.add_argument("--validator-hotkey", default=os.environ.get("VALIDATOR_HOTKEY_SS58", ""))
    val.add_argument("--device", default="cpu")
    val.add_argument("--sample-rate", type=float, default=1.0)
    val.add_argument("--max-receipts", type=int, default=None)
    val.add_argument("--publish-weights", action="store_true")
    val.add_argument("--set-weights", action="store_true")
    add_wallet_args(val, default_wallet=None)
    val.add_argument("--network", default=None)
    val.add_argument("--owner-secret", default="owner-dev-secret")
    val.add_argument("--miner-secret", default="miner-dev-secret")
    val.add_argument("--validator-secret", default="validator-dev-secret")
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
    audit.add_argument("--validator-hotkey", default=os.environ.get("VALIDATOR_HOTKEY_SS58", ""))
    audit.add_argument("--sample-rate", type=float, default=1.0)
    audit.add_argument("--max-jobs", type=int, default=None)
    audit.add_argument("--network", default="finney")
    audit.add_argument("--owner-secret", default="owner-dev-secret")
    add_wallet_args(audit, default_wallet=os.environ.get("VALIDATOR_WALLET_NAME", ""))
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
    ledger.add_argument("--validator-secret", default="validator-dev-secret")
    ledger.add_argument("--validator-hotkey", default=os.environ.get("VALIDATOR_HOTKEY_SS58", ""))
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

    dashboard = sub.add_parser("dashboard-backend")
    add_bucket_args(dashboard)
    dashboard.add_argument("--netuid", type=int, default=0)
    dashboard.add_argument("--run-id", default=os.environ.get("TEUTON_RUN_ID", ""), help="Optional run filter.")
    dashboard.add_argument("--db-path", default=os.environ.get("TEUTON_DASHBOARD_DB_PATH", "/var/lib/teuton-dashboard/dashboard.sqlite3"))
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--open-browser", action="store_true")
    dashboard.add_argument("--refresh-sec", type=float, default=3.0)
    dashboard.add_argument("--max-jobs", type=int, default=int(os.environ.get("TEUTON_DASHBOARD_MAX_JOBS") or "200"))
    dashboard.add_argument(
        "--max-inflight-per-hotkey",
        type=int,
        default=int(os.environ.get("TEUTON_MAX_INFLIGHT_PER_HOTKEY") or "8"),
        help="Per-hotkey queue cap used to render the 'at-cap' badge; mirror the orchestrator's value.",
    )
    dashboard.add_argument("--bucket-poll-sec", type=float, default=float(os.environ.get("TEUTON_DASHBOARD_BUCKET_POLL_SEC") or "5"))
    dashboard.add_argument("--chain-poll-sec", type=float, default=float(os.environ.get("TEUTON_DASHBOARD_CHAIN_POLL_SEC") or "30"))
    dashboard.add_argument("--network", default=os.environ.get("BT_NETWORK", "finney"))
    add_discovery_args(dashboard)
    dashboard.set_defaults(fn=cmd_dashboard_backend)

    ls = sub.add_parser(
        "ls",
        help="Read-only views of the live network: runs, miners, jobs (kubectl-style).",
    )
    ls_sub = ls.add_subparsers(dest="ls_target", required=True)

    ls_runs = ls_sub.add_parser("runs", help="List runs visible on the bucket for the given netuid.")
    add_bucket_args(ls_runs)
    ls_runs.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    ls_runs.add_argument("--limit", type=int, default=0, help="Cap rows (0 = no cap).")
    ls_runs.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    ls_runs.set_defaults(fn=cmd_ls_runs)

    ls_miners = ls_sub.add_parser("miners", help="List heartbeating miners on the network.")
    add_bucket_args(ls_miners)
    ls_miners.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    ls_miners.add_argument(
        "--run-id",
        default="",
        help="Filter by run_id. Defaults to $TEUTON_RUN_ID. Use --all-runs to span every run.",
    )
    ls_miners.add_argument("--all-runs", action="store_true", help="Show miners from every run.")
    ls_miners.add_argument(
        "--heartbeat-ttl-sec",
        type=float,
        default=30.0,
        help="A miner is reported as 'live' if its heartbeat is fresher than this.",
    )
    ls_miners.add_argument("--live-only", action="store_true", help="Hide stale miners.")
    ls_miners.add_argument("--limit", type=int, default=0, help="Cap rows (0 = no cap).")
    ls_miners.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    ls_miners.set_defaults(fn=cmd_ls_miners)

    ls_jobs = ls_sub.add_parser("jobs", help="List recent jobs for a given run.")
    add_bucket_args(ls_jobs)
    ls_jobs.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    ls_jobs.add_argument(
        "--run-id",
        default="",
        help="Run to inspect. Defaults to $TEUTON_RUN_ID.",
    )
    ls_jobs.add_argument(
        "--all-runs",
        action="store_true",
        help="Not supported for ls jobs (errors); included for symmetry with ls miners.",
    )
    ls_jobs.add_argument("--kind", default="", help="Filter by job kind (e.g. pipe_forward).")
    ls_jobs.add_argument(
        "--status",
        default="",
        choices=["", "created", "completed", "verified", "failed", "stale"],
        help="Filter by job status.",
    )
    ls_jobs.add_argument("--limit", type=int, default=50, help="Cap rows (default 50).")
    ls_jobs.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    ls_jobs.set_defaults(fn=cmd_ls_jobs)

    ls_job = ls_sub.add_parser("job", help="Show details for a single job (manifest, receipts, verdicts, audits).")
    add_bucket_args(ls_job)
    ls_job.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    ls_job.add_argument(
        "--run-id",
        default="",
        help="Run to inspect. Defaults to $TEUTON_RUN_ID.",
    )
    ls_job.add_argument("job_id", help="The job_id to inspect.")
    ls_job.add_argument("--json", action="store_true", help="Emit JSON instead of a formatted block.")
    ls_job.set_defaults(fn=cmd_ls_job)

    send = sub.add_parser(
        "send-job",
        help="Emit one synthetic pipe_forward job pinned to a specific miner hotkey.",
    )
    add_bucket_args(send)
    send.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    send.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Existing run to attach to (must be bootstrapped). Defaults to $TEUTON_RUN_ID.",
    )
    send.add_argument("--hotkey", required=True, help="Target miner hotkey (SS58).")
    send.add_argument(
        "--worker-id",
        default="",
        help="Specific worker_id to pin to. Defaults to the miner's freshest heartbeating worker.",
    )
    send.add_argument("--stage", type=int, default=0, help="Pipeline stage to forward.")
    send.add_argument("--mb", type=int, default=0, help="Microbatch index for the job_id.")
    send.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Synthetic epoch (0 = auto-pick in the 9_000_000+ range so we never collide).",
    )
    send.add_argument(
        "--heartbeat-ttl-sec",
        type=float,
        default=120.0,
        help="Reject the send if the miner's heartbeat is older than this.",
    )
    send.add_argument(
        "--owner-secret",
        default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"),
        help="Owner secret used to sign the manifest when no --wallet-name is provided.",
    )
    send.add_argument("--network", default=os.environ.get("BT_NETWORK", "finney"))
    add_wallet_args(send)
    add_grant_args(send)
    _add_wait_args(send)
    send.add_argument("--json", action="store_true", help="Emit JSON instead of a human summary.")
    send.set_defaults(fn=cmd_send_job)

    hc = sub.add_parser(
        "health-check",
        help="Send one synthetic pipe_forward to every (or selected) live miner and report per-miner latency.",
    )
    add_bucket_args(hc)
    hc.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    hc.add_argument(
        "--run-id",
        default=os.environ.get("TEUTON_RUN_ID", ""),
        help="Existing run to attach to. Defaults to $TEUTON_RUN_ID.",
    )
    hc.add_argument(
        "--hotkeys",
        default="",
        help="Comma-separated list of hotkeys to probe. Empty = every live miner on the run.",
    )
    hc.add_argument(
        "--heartbeat-ttl-sec",
        type=float,
        default=60.0,
        help="Only probe miners whose heartbeat is fresher than this.",
    )
    hc.add_argument(
        "--per-miner-timeout-sec",
        type=float,
        default=300.0,
        help="Max time to wait for a single receipt before flagging the miner as 'timeout'.",
    )
    hc.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Per-receipt poll interval.",
    )
    hc.add_argument(
        "--concurrent",
        type=int,
        default=8,
        help="Parallel send/wait fanout.",
    )
    hc.add_argument(
        "--owner-secret",
        default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"),
    )
    hc.add_argument("--network", default=os.environ.get("BT_NETWORK", "finney"))
    hc.add_argument("--quiet", action="store_true", help="Suppress per-miner progress lines.")
    hc.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    add_wallet_args(hc)
    add_grant_args(hc)
    hc.set_defaults(fn=cmd_health_check)

    stream = sub.add_parser(
        "stress-stream",
        help="Emit jobs at a controlled rate across N miners and report per-job latency.",
    )
    add_bucket_args(stream)
    stream.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", "3")))
    stream.add_argument("--run-id", default=os.environ.get("TEUTON_RUN_ID", ""))
    stream.add_argument(
        "--rate-per-min",
        type=float,
        default=10.0,
        help="Target jobs per minute (round-robined across --hotkeys).",
    )
    stream.add_argument(
        "--duration-sec",
        type=float,
        default=120.0,
        help="Emit phase length in seconds.",
    )
    stream.add_argument(
        "--receipt-timeout-sec",
        type=float,
        default=300.0,
        help="After emit phase ends, wait this long for pending receipts before declaring stale.",
    )
    stream.add_argument(
        "--hotkeys",
        default="",
        help="Comma-separated SS58 hotkeys to target. Empty = every live miner.",
    )
    stream.add_argument(
        "--owner-secret",
        default=os.environ.get("TEUTON_OWNER_SECRET", "owner-dev-secret"),
    )
    stream.add_argument("--network", default=os.environ.get("BT_NETWORK", "finney"))
    stream.add_argument("--quiet", action="store_true")
    stream.add_argument("--json", action="store_true")
    add_wallet_args(stream)
    add_grant_args(stream)
    stream.set_defaults(fn=cmd_stress_stream)

    submit = sub.add_parser(
        "submit-manifest",
        help="Submit a manifest JSON file as a single job on the network.",
    )
    add_bucket_args(submit)
    submit.add_argument(
        "--netuid",
        type=int,
        default=0,
        help="netuid for the bucket layout. 0 = infer from the manifest's graph_ref URI.",
    )
    submit.add_argument("--manifest", required=True, help="Path to a JobManifestV3 JSON file.")
    submit.add_argument(
        "--owner-secret",
        default=os.environ.get("TEUTON_OWNER_SECRET", ""),
        help="Re-sign the manifest with this secret if --resign is set.",
    )
    submit.add_argument(
        "--resign",
        action="store_true",
        help="Re-sign the manifest before writing (otherwise the existing owner_signature is kept).",
    )
    submit.add_argument("--network", default=os.environ.get("BT_NETWORK", "finney"))
    add_wallet_args(submit)
    add_grant_args(submit)
    _add_wait_args(submit)
    submit.add_argument("--json", action="store_true", help="Emit JSON instead of a human summary.")
    submit.set_defaults(fn=cmd_submit_manifest)

    return p


def _add_wait_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wait",
        action="store_true",
        help="Block until the receipt for the emitted job lands on the bucket (or timeout).",
    )
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=60.0,
        help="How long to wait for the receipt when --wait is set.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Receipt-poll interval (seconds).",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
