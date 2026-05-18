#!/usr/bin/env bash
# Teuton container entrypoint.
#
# The role to run is selected by $TEUTON_ROLE (or positional arg). The CLI
# subcommands `validator` and `audit-jobs` are one-shot per invocation, so we
# wrap them in a supervised loop. Long-running roles (`miner`, `orchestrator`)
# exec their command directly; if they crash, Docker `restart: always`
# brings them back.
#
# NOTE: the legacy `auditor-worker` role has been retired in favour of the
# trusted-anchor peer-audit model -- audit-eligible miners (flagged via
# TEUTON_AUDIT_ELIGIBLE_HOTKEYS) now pick up audit_replay jobs from inside
# the regular miner loop.
set -uo pipefail

if [ -n "${TEUTON_ROLE:-}" ]; then
    ROLE="$TEUTON_ROLE"
    if [ "${1:-}" = "banner" ]; then
        shift
    fi
else
    ROLE="${1:-banner}"
    if [ "$#" -gt 0 ]; then
        shift
    fi
fi
# Build marker.
: "${TEUTON_PHASE15_MARKER:=quota-fix-1}"

# Resolve effective run id. Precedence (highest wins):
#   1. TEUTON_RUN_ID        -- explicit runtime override (compose env or .env)
#   2. RUN_ID              -- legacy host env name (kept for backward compat)
#   3. TEUTON_BAKED_RUN_ID  -- value baked into the image at build time
# The result is re-exported as TEUTON_RUN_ID so downstream CLIs (which read
# it via os.environ) inherit the resolved value automatically.
TEUTON_RUN_ID="${TEUTON_RUN_ID:-${RUN_ID:-${TEUTON_BAKED_RUN_ID:-}}}"
export TEUTON_RUN_ID

# Resolve public fleet defaults baked into the image. Runtime env still wins,
# which lets operators temporarily point a host at a different bucket/run.
S3_BUCKET="${S3_BUCKET:-${TEUTON_BAKED_S3_BUCKET:-}}"
S3_REGION="${S3_REGION:-${TEUTON_BAKED_S3_REGION:-us-east-1}}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-${TEUTON_BAKED_S3_ENDPOINT_URL:-}}"
TEUTON_NETUID="${TEUTON_NETUID:-${TEUTON_BAKED_NETUID:-3}}"
BT_NETWORK="${BT_NETWORK:-${TEUTON_BAKED_BT_NETWORK:-finney}}"
TEUTON_GRANT_MODE="${TEUTON_GRANT_MODE:-${TEUTON_BAKED_GRANT_MODE:-presigned}}"
TEUTON_ASSIGNMENT_CRYPTO="${TEUTON_ASSIGNMENT_CRYPTO:-${TEUTON_BAKED_ASSIGNMENT_CRYPTO:-ed25519}}"
TEUTON_DISCOVERY_BACKEND="${TEUTON_DISCOVERY_BACKEND:-${TEUTON_BAKED_DISCOVERY_BACKEND:-bucket}}"
TEUTON_MINER_POLL_INTERVAL="${TEUTON_MINER_POLL_INTERVAL:-${TEUTON_BAKED_MINER_POLL_INTERVAL:-0.05}}"
TEUTON_OWNER_HOTKEY="${TEUTON_OWNER_HOTKEY:-${TEUTON_BAKED_OWNER_HOTKEY:-}}"
export S3_BUCKET S3_REGION S3_ENDPOINT_URL TEUTON_NETUID BT_NETWORK
export TEUTON_GRANT_MODE TEUTON_ASSIGNMENT_CRYPTO TEUTON_DISCOVERY_BACKEND TEUTON_MINER_POLL_INTERVAL TEUTON_OWNER_HOTKEY

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "error: \$$name must be set" >&2
        exit 2
    fi
}

exec_default_miner() {
    require_env S3_BUCKET

    local wallet_name="${MINER_COLDKEY:-${MINER_WALLET_NAME:-${BT_WALLET_NAME:-teuton_mining}}}"
    local wallet_path="${BT_WALLET_PATH:-/root/.bittensor/wallets}"
    local hotkey_name="${MINER_HOTKEY:-${MINER_HOTKEY_NAME:-${BT_HOTKEY_NAME:-}}}"
    if [ -z "$hotkey_name" ]; then
        echo "error: \$MINER_HOTKEY, \$MINER_HOTKEY_NAME, or \$BT_HOTKEY_NAME must be set" >&2
        exit 2
    fi

    local args=(
        --netuid="${TEUTON_NETUID}"
        --devices="${MINER_DEVICES:-cuda}"
        --wallet-path="${wallet_path}"
        --wallet-name="${wallet_name}"
        --hotkey-name="${hotkey_name}"
        --grant-mode="${TEUTON_GRANT_MODE}"
        --assignment-crypto="${TEUTON_ASSIGNMENT_CRYPTO}"
        --discovery-backend="${TEUTON_DISCOVERY_BACKEND}"
        --poll-interval="${TEUTON_MINER_POLL_INTERVAL}"
        --s3-bucket="${S3_BUCKET}"
        --s3-region="${S3_REGION}"
        --audit-eligible-hotkeys="${TEUTON_AUDIT_ELIGIBLE_HOTKEYS:-}"
    )
    if [ -n "${MINER_HOTKEY_SS58:-}" ]; then
        args+=(--hotkey="${MINER_HOTKEY_SS58}")
    fi
    if [ -n "${TEUTON_OWNER_HOTKEY}" ]; then
        args+=(--owner-hotkey="${TEUTON_OWNER_HOTKEY}")
    fi
    if [ -n "${S3_ENDPOINT_URL}" ]; then
        args+=(--s3-endpoint-url="${S3_ENDPOINT_URL}")
    fi
    exec teuton-v3 miner "${args[@]}"
}

print_banner() {
    cat <<EOF
=================================================================
Teuton v3 container
  role        : ${ROLE}
  git_sha     : ${TEUTON_GIT_SHA:-unknown}
  build_time  : ${TEUTON_BUILD_TIME:-unknown}
  run_id      : ${TEUTON_RUN_ID:-<unset>} (baked=${TEUTON_BAKED_RUN_ID:-<none>})
  netuid      : ${TEUTON_NETUID:-<unset>}
  bucket      : ${S3_BUCKET:-<unset>}
  hostname    : $(hostname)
  python      : $(python --version 2>&1)
  cwd         : $(pwd)
=================================================================
EOF
}

loop_forever() {
    local label="$1"; shift
    local sleep_sec="${TEUTON_LOOP_SLEEP_SEC:-30}"
    while true; do
        echo "[${label}] starting at $(date -u +%FT%TZ)"
        "$@"
        local rc=$?
        echo "[${label}] exited rc=${rc}, sleeping ${sleep_sec}s before next pass" >&2
        sleep "$sleep_sec"
    done
}

print_banner

case "$ROLE" in
    banner)
        echo "set TEUTON_ROLE to one of: miner | orchestrator | validator | audit-jobs"
        exit 0
        ;;
    miner)
        if [ "$#" -eq 0 ]; then
            exec_default_miner
        fi
        exec teuton-v3 miner "$@"
        ;;
    orchestrator)
        exec teuton-v3 orchestrator "$@"
        ;;
    validator)
        loop_forever "validator" teuton-v3 validator "$@"
        ;;
    audit-jobs)
        loop_forever "audit-jobs" teuton-v3 audit-jobs "$@"
        ;;
    shell)
        exec bash "$@"
        ;;
    *)
        exec "$ROLE" "$@"
        ;;
esac
