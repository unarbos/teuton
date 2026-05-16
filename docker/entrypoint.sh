#!/usr/bin/env bash
# Teuton container entrypoint.
#
# The role to run is selected by $TEUTON_ROLE (or positional arg). The CLI
# subcommands `validator` and `audit-jobs` are one-shot per invocation, so we
# wrap them in a supervised loop. Long-running roles (`miner`, `orchestrator`,
# `auditor-worker`) exec their command directly; if they crash, Docker
# `restart: always` brings them back.
set -uo pipefail

ROLE="${TEUTON_ROLE:-${1:-banner}}"
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

print_banner() {
    cat <<EOF
=================================================================
Teuton v3 container
  role        : ${ROLE}
  git_sha     : ${TEUTON_GIT_SHA:-unknown}
  build_time  : ${TEUTON_BUILD_TIME:-unknown}
  run_id      : ${TEUTON_RUN_ID:-<unset>} (baked=${TEUTON_BAKED_RUN_ID:-<none>})
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
        echo "set TEUTON_ROLE to one of: miner | orchestrator | validator | audit-jobs | auditor-worker"
        exit 0
        ;;
    miner)
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
    auditor-worker)
        exec python -m bench.auditor_worker "$@"
        ;;
    shell)
        exec bash "$@"
        ;;
    *)
        exec "$ROLE" "$@"
        ;;
esac
