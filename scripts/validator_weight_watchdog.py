"""Validator weight-setting watchdog.

Polls the Bittensor finney chain and verifies that the Teuton validator
hotkey has set weights on its netuid within the last N blocks
(default 360 blocks = ~1 hour at 12s/block). If the gap exceeds the
threshold the watchdog signs and submits ``set_weights(uids=[0],
weights=[1.0])`` from the validator's own hotkey, sending the entire
weight vector to the burn UID (UID 0 on the subnet).

The watchdog only fires when the validator has *fallen behind* on its
weight-setting schedule. As long as the validator process is healthy
and meeting its own deadlines, the watchdog stays quiet.

Run from the repo root:

    source .venv/bin/activate
    python scripts/validator_weight_watchdog.py \
        --netuid 3 \
        --wallet-name teutonic --hotkey-name default \
        --network finney \
        --threshold-blocks 360 \
        --poll-sec 60

Or as a pm2 daemon (recommended):

    pm2 start scripts/validator_weight_watchdog.py \
        --name validator-weight-watchdog \
        --interpreter "$(pwd)/.venv/bin/python" \
        --no-autorestart-cron \
        -- --netuid 3 --wallet-name teutonic --hotkey-name default

Notes:
- ``set_weights`` is a hotkey-signed extrinsic; no coldkey password is
  required. The hotkey on disk must be unencrypted.
- The chain enforces ``WeightsSetRateLimit`` (netuid 3 = 100 blocks).
  We only fire when ``blocks_since_last_update > threshold (360)`` so
  the rate limit is never the bottleneck.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from typing import Any

LOG = logging.getLogger("validator_watchdog")


def configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(lvl)
    LOG.handlers = [handler]
    LOG.setLevel(lvl)
    # Don't propagate to root: bittensor / loguru may have lowered the root
    # logger to WARNING which would otherwise drop our INFO records.
    LOG.propagate = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--netuid", type=int, default=int(os.environ.get("TEUTON_NETUID", 3)))
    p.add_argument(
        "--wallet-name",
        default=os.environ.get("VALIDATOR_WALLET_NAME", "teutonic"),
        help="Bittensor wallet name that owns the validator hotkey.",
    )
    p.add_argument(
        "--hotkey-name",
        default=os.environ.get("VALIDATOR_HOTKEY_NAME", "default"),
        help="Hotkey name inside the wallet (the on-chain validator hotkey).",
    )
    p.add_argument(
        "--validator-hotkey",
        default=os.environ.get("VALIDATOR_HOTKEY_SS58") or None,
        help="SS58 to monitor. Defaults to the loaded wallet's hotkey ss58_address.",
    )
    p.add_argument("--network", default=os.environ.get("BT_NETWORK", "finney"))
    p.add_argument(
        "--threshold-blocks",
        type=int,
        default=int(os.environ.get("WATCHDOG_THRESHOLD_BLOCKS", 360)),
        help="Trigger burn fallback once blocks_since_last_update exceeds this.",
    )
    p.add_argument("--burn-uid", type=int, default=int(os.environ.get("WATCHDOG_BURN_UID", 0)))
    p.add_argument("--burn-weight", type=float, default=float(os.environ.get("WATCHDOG_BURN_WEIGHT", 1.0)))
    p.add_argument("--poll-sec", type=int, default=int(os.environ.get("WATCHDOG_POLL_SEC", 60)))
    p.add_argument(
        "--error-backoff-sec",
        type=int,
        default=int(os.environ.get("WATCHDOG_ERROR_BACKOFF_SEC", 30)),
    )
    p.add_argument(
        "--burn-cooldown-sec",
        type=int,
        default=int(os.environ.get("WATCHDOG_BURN_COOLDOWN_SEC", 60 * 20)),
        help="Minimum wall-clock seconds between consecutive burn submissions from this watchdog.",
    )
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    p.add_argument("--once", action="store_true", help="Run a single check then exit (for cron/testing).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(int(os.environ.get("WATCHDOG_DRY_RUN", "0") or "0")),
        help="Skip the actual set_weights submission; only log what would happen.",
    )
    return p.parse_args(argv)


def _import_bt():
    import bittensor as bt
    return bt


def build_subtensor(bt_module, network: str):
    return bt_module.Subtensor(network=network)


def lookup_validator_uid(metagraph: Any, hotkey_ss58: str) -> int | None:
    try:
        return int(metagraph.hotkeys.index(hotkey_ss58))
    except ValueError:
        return None


def submit_burn_weights(
    *,
    subtensor: Any,
    wallet: Any,
    netuid: int,
    burn_uid: int,
    burn_weight: float,
    dry_run: bool,
) -> dict:
    payload = {
        "netuid": int(netuid),
        "uids": [int(burn_uid)],
        "weights": [float(burn_weight)],
    }
    if dry_run:
        LOG.warning("DRY-RUN: would submit set_weights %s", payload)
        return {"dry_run": True, **payload}
    LOG.warning("submitting BURN set_weights %s", payload)
    result = subtensor.set_weights(
        wallet=wallet,
        netuid=int(netuid),
        uids=[int(burn_uid)],
        weights=[float(burn_weight)],
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    if isinstance(result, tuple) and len(result) == 2:
        success, msg = result
    else:
        success = bool(getattr(result, "success", result is True))
        msg = str(getattr(result, "error_message", "") or getattr(result, "message", "") or result)
    LOG.warning("burn set_weights result success=%s msg=%s", success, msg)
    return {"success": bool(success), "msg": str(msg), **payload}


def check_once(
    *,
    subtensor: Any,
    wallet: Any,
    validator_hotkey: str,
    netuid: int,
    threshold_blocks: int,
    burn_uid: int,
    burn_weight: float,
    dry_run: bool,
    burn_cooldown_sec: int,
    last_burn_unix: float | None,
) -> tuple[dict, float | None]:
    metagraph = subtensor.metagraph(int(netuid))
    current_block = int(subtensor.get_current_block())
    uid = lookup_validator_uid(metagraph, validator_hotkey)
    if uid is None:
        LOG.error("validator hotkey %s is NOT in netuid=%s metagraph", validator_hotkey, netuid)
        return (
            {
                "status": "not_registered",
                "validator_hotkey": validator_hotkey,
                "netuid": int(netuid),
                "current_block": current_block,
            },
            last_burn_unix,
        )

    last_update = int(metagraph.last_update[uid])
    blocks_since = current_block - last_update
    permit = bool(metagraph.validator_permit[uid]) if len(metagraph.validator_permit) > uid else False
    LOG.info(
        "uid=%s ss58=%s current_block=%s last_update=%s blocks_since=%s threshold=%s permit=%s",
        uid, validator_hotkey, current_block, last_update, blocks_since, threshold_blocks, permit,
    )

    if blocks_since <= threshold_blocks:
        return (
            {
                "status": "ok",
                "uid": uid,
                "blocks_since": blocks_since,
                "current_block": current_block,
                "last_update": last_update,
                "validator_permit": permit,
            },
            last_burn_unix,
        )

    now = time.time()
    if last_burn_unix is not None and (now - last_burn_unix) < burn_cooldown_sec:
        cooldown_left = burn_cooldown_sec - (now - last_burn_unix)
        LOG.warning(
            "validator behind by %s blocks but burn cooldown active (%.0fs left); skipping",
            blocks_since, cooldown_left,
        )
        return (
            {
                "status": "cooldown",
                "uid": uid,
                "blocks_since": blocks_since,
                "current_block": current_block,
                "last_update": last_update,
                "cooldown_left_sec": cooldown_left,
            },
            last_burn_unix,
        )

    LOG.warning(
        "validator behind by %s blocks (> %s); firing burn fallback to UID %s",
        blocks_since, threshold_blocks, burn_uid,
    )
    result = submit_burn_weights(
        subtensor=subtensor,
        wallet=wallet,
        netuid=int(netuid),
        burn_uid=int(burn_uid),
        burn_weight=float(burn_weight),
        dry_run=bool(dry_run),
    )
    return (
        {
            "status": "burned" if result.get("success") or result.get("dry_run") else "burn_failed",
            "uid": uid,
            "blocks_since": blocks_since,
            "current_block": current_block,
            "last_update": last_update,
            "result": result,
        },
        now if (result.get("success") or result.get("dry_run")) else last_burn_unix,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Import bittensor first; it installs its own logging handlers / level
    # on the root logger which would otherwise filter our INFO records.
    bt = _import_bt()
    configure_logging(args.log_level)

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey_name)
    validator_hotkey = args.validator_hotkey or wallet.hotkey.ss58_address

    print(
        f"[watchdog] starting: netuid={args.netuid} wallet={args.wallet_name} "
        f"hotkey={args.hotkey_name} ss58={validator_hotkey} network={args.network} "
        f"threshold_blocks={args.threshold_blocks} poll_sec={args.poll_sec} "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    LOG.info(
        "watchdog up: netuid=%s wallet=%s hotkey=%s ss58=%s network=%s threshold=%s poll=%ss dry_run=%s",
        args.netuid, args.wallet_name, args.hotkey_name, validator_hotkey,
        args.network, args.threshold_blocks, args.poll_sec, args.dry_run,
    )

    stop_flag = {"v": False}

    def _on_sig(signum, _frame):
        LOG.info("received signal %s; shutting down after current iteration", signum)
        stop_flag["v"] = True

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    subtensor = None
    last_burn_unix: float | None = None

    while not stop_flag["v"]:
        try:
            if subtensor is None:
                subtensor = build_subtensor(bt, args.network)

            result, last_burn_unix = check_once(
                subtensor=subtensor,
                wallet=wallet,
                validator_hotkey=validator_hotkey,
                netuid=args.netuid,
                threshold_blocks=args.threshold_blocks,
                burn_uid=args.burn_uid,
                burn_weight=args.burn_weight,
                dry_run=args.dry_run,
                burn_cooldown_sec=args.burn_cooldown_sec,
                last_burn_unix=last_burn_unix,
            )
            LOG.debug("check result %s", result)

            if args.once:
                print(result)
                return 0

            for _ in range(args.poll_sec):
                if stop_flag["v"]:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:  # broad on purpose: keep the loop alive
            LOG.exception("watchdog loop error: %s; backing off %ss", e, args.error_backoff_sec)
            subtensor = None  # force reconnect next iter
            for _ in range(args.error_backoff_sec):
                if stop_flag["v"]:
                    break
                time.sleep(1)

    LOG.info("watchdog exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
