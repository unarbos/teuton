#!/usr/bin/env bash
# Locus container entrypoint.
#
# The role to run is selected by $LOCUS_ROLE (or positional arg). The CLI
# subcommands `validator` and `audit-jobs` are one-shot per invocation, so we
# wrap them in a supervised loop. Long-running roles (`miner`, `orchestrator`,
# `auditor-worker`) exec their command directly; if they crash, Docker
# `restart: always` brings them back.
set -uo pipefail

ROLE="${LOCUS_ROLE:-${1:-banner}}"
# Build marker.
: "${LOCUS_PHASE15_MARKER:=quota-fix-1}"

# Resolve effective run id. Precedence (highest wins):
#   1. LOCUS_RUN_ID        -- explicit runtime override (compose env or .env)
#   2. RUN_ID              -- legacy host env name (kept for backward compat)
#   3. LOCUS_BAKED_RUN_ID  -- value baked into the image at build time
# The result is re-exported as LOCUS_RUN_ID so downstream CLIs (which read
# it via os.environ) inherit the resolved value automatically.
LOCUS_RUN_ID="${LOCUS_RUN_ID:-${RUN_ID:-${LOCUS_BAKED_RUN_ID:-}}}"
export LOCUS_RUN_ID

print_banner() {
    cat <<EOF
=================================================================
Locus v3 container
  role        : ${ROLE}
  git_sha     : ${LOCUS_GIT_SHA:-unknown}
  build_time  : ${LOCUS_BUILD_TIME:-unknown}
  run_id      : ${LOCUS_RUN_ID:-<unset>} (baked=${LOCUS_BAKED_RUN_ID:-<none>})
  hostname    : $(hostname)
  python      : $(python --version 2>&1)
  cwd         : $(pwd)
=================================================================
EOF
}

loop_forever() {
    local label="$1"; shift
    local sleep_sec="${LOCUS_LOOP_SLEEP_SEC:-30}"
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
        echo "set LOCUS_ROLE to one of: miner | orchestrator | validator | audit-jobs | auditor-worker"
        exit 0
        ;;
    miner)
        exec locus-v3 miner "$@"
        ;;
    orchestrator)
        exec locus-v3 orchestrator "$@"
        ;;
    validator)
        loop_forever "validator" locus-v3 validator "$@"
        ;;
    audit-jobs)
        loop_forever "audit-jobs" locus-v3 audit-jobs "$@"
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
