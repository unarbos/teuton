#!/usr/bin/env bash
# Register N ed25519 miner hotkeys on netuid 3 finney under coldkey
# `teuton_mining`. Idempotent: skips any hotkey whose SS58 is already in the
# netuid 3 metagraph.
#
# Usage:
#   ./scripts/register_miners.sh                      # default: 10 miners
#   ./scripts/register_miners.sh --n 5                # only 5
#   ./scripts/register_miners.sh --prefix teuton_miner_sn3_ --start 1 --n 10
#
# Requires:
#   - bittensor + bittensor-wallet installed (already in .venv via uv)
#   - btcli on PATH
#   - teuton_mining coldkey unlocked or password supplied via env
#
# Cost (burn): printed before any extrinsic; explicit "yes" prompt required.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NETUID=${NETUID:-3}
NETWORK=${NETWORK:-finney}
WALLET=${WALLET:-teuton_mining}
PREFIX="teuton_miner_sn3_"
START=1
N=10
DRY_RUN=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n) N="$2"; shift 2;;
        --start) START="$2"; shift 2;;
        --prefix) PREFIX="$2"; shift 2;;
        --netuid) NETUID="$2"; shift 2;;
        --network) NETWORK="$2"; shift 2;;
        --wallet) WALLET="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        --yes|-y) ASSUME_YES=1; shift;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

source "$REPO_ROOT/.venv/bin/activate"

GENERATOR="$REPO_ROOT/scripts/generate_ed25519_hotkey.py"
if [ ! -f "$GENERATOR" ]; then
    echo "missing $GENERATOR" >&2; exit 1
fi

# 1. Generate any missing keys.
generated=()
for ((i=START; i<START+N; i++)); do
    hk_name="${PREFIX}${i}"
    hk_path="$HOME/.bittensor/wallets/${WALLET}/hotkeys/${hk_name}"
    if [ -s "$hk_path" ]; then
        echo "[gen] keep existing ${hk_name}"
    else
        echo "[gen] creating ${hk_name}"
        python "$GENERATOR" --wallet-name "$WALLET" --hotkey "$hk_name"
        generated+=("$hk_name")
    fi
done

# 2. Print SS58s + check on-chain registration.
echo
echo "=== Resolved miner hotkeys ==="
python - <<PY
import json, os, sys
from pathlib import Path
import bittensor as bt
wallet_root = Path.home() / ".bittensor" / "wallets" / "${WALLET}" / "hotkeys"
prefix = "${PREFIX}"
start = ${START}
n = ${N}
st = bt.Subtensor(network="${NETWORK}")
mg = st.metagraph(${NETUID})
registered = set(mg.hotkeys)
rows = []
for i in range(start, start + n):
    name = f"{prefix}{i}"
    pub = wallet_root / f"{name}pub.txt"
    info = {}
    if pub.exists():
        try:
            info = json.loads(pub.read_text())
        except Exception:
            pass
    ss58 = info.get("ss58Address") or "?"
    rows.append((name, ss58, ss58 in registered))
print(f"netuid={${NETUID}} burn_cost(recycle) = {st.recycle(netuid=${NETUID})}")
for name, ss58, on_chain in rows:
    mark = "ALREADY REGISTERED" if on_chain else "needs register"
    print(f"  {name:32s} {ss58}  {mark}")
# Save remaining for later steps
remaining = [(name, ss58) for name, ss58, on_chain in rows if not on_chain]
Path("/tmp/teuton_miner_register_queue.json").write_text(json.dumps(remaining))
print(f"\n{len(remaining)} hotkeys queued for registration")
PY

QUEUE_LEN=$(python -c 'import json; print(len(json.load(open("/tmp/teuton_miner_register_queue.json"))))')
if [ "$QUEUE_LEN" -eq 0 ]; then
    echo "All target hotkeys already registered."
    exit 0
fi

# 3. Cost confirmation.
TOTAL_COST=$(python - <<PY
import bittensor as bt
st = bt.Subtensor(network="${NETWORK}")
burn = st.recycle(netuid=${NETUID})
# bt.Balance has float() coercion
rao_per_one = int(burn.rao)
print(rao_per_one * ${QUEUE_LEN})
PY
)
TOTAL_TAO=$(python -c "print(${TOTAL_COST}/1e9)")
echo
echo "=== About to register ${QUEUE_LEN} hotkeys on netuid ${NETUID} (${NETWORK}) ==="
echo "    coldkey: ${WALLET}"
printf "    total burn: %s rao = τ%s\n" "${TOTAL_COST}" "${TOTAL_TAO}"
echo "    (cost is irreversible)"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] not registering. Re-run without --dry-run to spend."
    exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    read -p "Type 'yes' to proceed: " ans
    [ "$ans" = "yes" ] || { echo "aborted."; exit 1; }
fi

# 4. Register each queued hotkey via the Python bittensor API. We need the
#    coldkey password; the script reads it from $TEUTON_MINING_COLDKEY_PW and
#    hands it to bittensor-wallet via the Rust-side `save_password_to_env`
#    (which is what the official examples use).
if [ -z "${TEUTON_MINING_COLDKEY_PW:-}" ]; then
    echo "error: TEUTON_MINING_COLDKEY_PW must be set (the teuton_mining coldkey password)." >&2
    echo "       e.g.  TEUTON_MINING_COLDKEY_PW='...' bash scripts/register_miners.sh --yes" >&2
    exit 3
fi

TEUTON_MINING_COLDKEY_PW="$TEUTON_MINING_COLDKEY_PW" python - <<PY
import json, os, time
import bittensor as bt
import bittensor_wallet

pw = os.environ["TEUTON_MINING_COLDKEY_PW"]
queue = json.load(open("/tmp/teuton_miner_register_queue.json"))
st = bt.Subtensor(network="${NETWORK}")

# Sanity-check the password against the coldkey before spending anything.
probe = bittensor_wallet.Wallet(name="${WALLET}")
probe.coldkey_file.save_password_to_env(pw)
try:
    probe.unlock_coldkey()
except Exception as e:
    raise SystemExit(f"coldkey unlock failed (wrong password?): {e}")
print(f"coldkey unlocked: {probe.coldkey.ss58_address}")

failures = []
for name, ss58 in queue:
    print(f"\n=== registering {name} ({ss58}) ===")
    for attempt in range(4):
        try:
            w = bittensor_wallet.Wallet(name="${WALLET}", hotkey=name)
            w.coldkey_file.save_password_to_env(pw)
            resp = st.burned_register(
                wallet=w,
                netuid=${NETUID},
                wait_for_inclusion=True,
                wait_for_finalization=True,
                raise_error=False,
            )
            ok = (
                bool(getattr(resp, "success", False))
                or bool(getattr(resp, "is_success", False))
                or (resp is True)
                or str(getattr(resp, "status", "")).lower() in {"success", "ok"}
            )
            print(
                f"  resp: success={ok} "
                f"block_hash={getattr(resp,'block_hash',None)} "
                f"extrinsic_hash={getattr(resp,'extrinsic_hash',None)} "
                f"error={getattr(resp,'error_message',None) or getattr(resp,'error',None)}"
            )
            # double-check on chain
            mg = st.metagraph(${NETUID})
            if ss58 in set(mg.hotkeys):
                print(f"  CONFIRMED on chain: {name}")
                break
            if ok:
                print("  resp said ok but ss58 not yet on metagraph; sleeping briefly and continuing")
                time.sleep(2)
                continue
        except Exception as e:
            print(f"  attempt {attempt+1} raised: {e!r}")
        backoff = 5 * (2 ** attempt)
        print(f"  retrying in {backoff}s")
        time.sleep(backoff)
    else:
        print(f"  failed permanently: {name}")
        failures.append(name)

if failures:
    print(f"\nfailed: {failures}")
    raise SystemExit(1)
PY

# 5. Verify.
echo
echo "=== final metagraph check ==="
python - <<PY
import json
import bittensor as bt
st = bt.Subtensor(network="${NETWORK}")
mg = st.metagraph(${NETUID})
hotkeys = set(mg.hotkeys)
queue = json.load(open("/tmp/teuton_miner_register_queue.json"))
ok, missing = [], []
for name, ss58 in queue:
    if ss58 in hotkeys:
        ok.append(name)
    else:
        missing.append(name)
print(f"registered now: {len(ok)} / {len(queue)}")
for m in missing:
    print(f"  MISSING: {m}")
PY
