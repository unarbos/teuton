#!/usr/bin/env bash
# Bring up the local validator stack: orchestrator + validator-loop +
# audit-jobs-loop + watchtower, with the teutonic wallet bind-mounted ro.
#
# Sources:
#   - TEUTON_*, AWS_*, S3_* from /home/const/teuton/.env
#   - DOCKER_USER, DOCKER_PAT from Doppler arbos/dev
#   - RUN_ID from /tmp/teuton_sn3_run_id (or env)
#
# Usage:
#   ./scripts/deploy_validator.sh [up|down|logs|ps]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-up}"

set -a
source .env
set +a

export RUN_ID="${RUN_ID:-$(cat /tmp/teuton_sn3_run_id 2>/dev/null)}"
if [ -z "$RUN_ID" ]; then
    echo "RUN_ID is empty; set it or write /tmp/teuton_sn3_run_id" >&2
    exit 2
fi
export TEUTON_RUN_ID="$RUN_ID"

export TEUTON_NETUID="${TEUTON_NETUID:-3}"
export VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-teutonic}"
export VALIDATOR_HOTKEY_NAME="${VALIDATOR_HOTKEY_NAME:-default}"
export VALIDATOR_HOTKEY_SS58="${VALIDATOR_HOTKEY_SS58:-5E6yHkmZmSpBT5aa2rNZcmeYa1y3N9jw1h7g53oNPzMUpnqG}"
export AUDIT_MODE="${AUDIT_MODE:-consume}"
export AUDIT_SAMPLE_RATE="${AUDIT_SAMPLE_RATE:-0.2}"
export AUDIT_MAX_JOBS="${AUDIT_MAX_JOBS:-50}"
export AUDIT_JOBS_SLEEP_SEC="${AUDIT_JOBS_SLEEP_SEC:-15}"
export TEUTON_LOOP_SLEEP_SEC="${TEUTON_LOOP_SLEEP_SEC:-30}"
export ORCHESTRATOR_STEPS="${ORCHESTRATOR_STEPS:-1000000}"
export ORCHESTRATOR_TIMEOUT_SEC="${ORCHESTRATOR_TIMEOUT_SEC:-31536000}"
if [ -z "${TEUTON_AUDIT_ELIGIBLE_HOTKEYS:-}" ] && [ -f bench/fleet.json ]; then
    source .venv/bin/activate
    export TEUTON_AUDIT_ELIGIBLE_HOTKEYS="$(
        python - <<'PY'
import json
from pathlib import Path
fleet = json.loads(Path("bench/fleet.json").read_text())
print(",".join(fleet.get("audit_eligible_hotkeys") or []))
PY
    )"
fi

echo "RUN_ID=$RUN_ID  netuid=$TEUTON_NETUID  validator=$VALIDATOR_WALLET_NAME/$VALIDATOR_HOTKEY_NAME ($VALIDATOR_HOTKEY_SS58)"

doppler run --project arbos --config dev -- bash -lc '
    set -euo pipefail
    test -n "${DOCKER_USER:-}" || { echo "DOCKER_USER missing from doppler" >&2; exit 1; }
    test -n "${DOCKER_PAT:-}"  || { echo "DOCKER_PAT missing from doppler"  >&2; exit 1; }
    echo "$DOCKER_PAT" | docker login -u "$DOCKER_USER" --password-stdin >/dev/null

    export DOCKER_USER S3_BUCKET S3_REGION S3_ENDPOINT_URL \
           AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
           TEUTON_OWNER_HOTKEY TEUTON_ASSIGNMENT_CRYPTO \
           TEUTON_AUDIT_ELIGIBLE_HOTKEYS \
           TEUTON_NETUID RUN_ID TEUTON_RUN_ID \
           VALIDATOR_WALLET_NAME VALIDATOR_HOTKEY_NAME VALIDATOR_HOTKEY_SS58 \
           AUDIT_MODE AUDIT_SAMPLE_RATE AUDIT_MAX_JOBS AUDIT_JOBS_SLEEP_SEC \
           TEUTON_LOOP_SLEEP_SEC TEUTON_GRANT_TTL_SEC ORCHESTRATOR_STEPS ORCHESTRATOR_TIMEOUT_SEC

    case "'"$ACTION"'" in
        up)    docker compose -f docker/compose.validator.yml pull
               docker compose -f docker/compose.validator.yml up -d
               docker compose -f docker/compose.validator.yml ps ;;
        down)  docker compose -f docker/compose.validator.yml down ;;
        logs)  docker compose -f docker/compose.validator.yml logs --tail=80 ;;
        ps)    docker compose -f docker/compose.validator.yml ps ;;
        *)     echo "unknown action: '"$ACTION"'" >&2; exit 2 ;;
    esac
'
