#!/usr/bin/env bash
# End-to-end deploy of the Teuton public dashboard:
#
#   1. Provision the Cloudflare Tunnel + DNS via setup_cloudflare_dashboard.py
#      (idempotent; reuses an existing tunnel of the same name).
#   2. SSH to the chosen host, write /root/teuton-dashboard/.env, scp the compose file,
#      docker login, and `docker compose up -d` the dashboard stack.
#
# Usage:
#   doppler run --project arbos --config dev -- \
#       ./scripts/deploy_dashboard.sh \
#           --host root@95.133.252.33 --port 10311 \
#           --hostname dashboard.teutonic.ai
#
# Required env (from Doppler arbos/dev or your shell):
#   CLOUDFLARE_API_TOKEN       (Account: Tunnel Edit, Zone: Read+DNS Edit)
#   DOCKER_USER, DOCKER_PAT    (so the host can pull the teuton image)
#   S3_BUCKET, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#   TEUTON_NETUID               (default 3)
#
# Optional:
#   TEUTON_RUN_ID          (optional; when empty, dashboard auto-selects the
#                           latest run discovered in the bucket)
#   --tunnel-name NAME    (default: teuton-dashboard)
#   --no-proxy            (grey-cloud DNS instead of orange-cloud)
#   --skip-cloudflare     (reuse the TEUTON_DASHBOARD_TUNNEL_TOKEN already set)
#   --skip-host           (only do the Cloudflare setup, print the token)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOSTNAME_PUBLIC="dashboard.teutonic.ai"
SSH_HOST=""
SSH_PORT="22"
TUNNEL_NAME="teuton-dashboard"
NO_PROXY=0
SKIP_CF=0
SKIP_HOST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostname)        HOSTNAME_PUBLIC="$2"; shift 2;;
        --host)            SSH_HOST="$2";        shift 2;;
        --port)            SSH_PORT="$2";        shift 2;;
        --tunnel-name)     TUNNEL_NAME="$2";     shift 2;;
        --no-proxy)        NO_PROXY=1;           shift;;
        --skip-cloudflare) SKIP_CF=1;            shift;;
        --skip-host)       SKIP_HOST=1;          shift;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

require() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "error: \$$name must be set (use Doppler or your env)" >&2
        exit 1
    fi
}

if [ "$SKIP_CF" -ne 1 ]; then
    require CLOUDFLARE_API_TOKEN
fi
if [ "$SKIP_HOST" -ne 1 ]; then
    require DOCKER_USER
    require DOCKER_PAT
    require S3_BUCKET
    require AWS_ACCESS_KEY_ID
    require AWS_SECRET_ACCESS_KEY
    if [ -z "$SSH_HOST" ]; then
        echo "error: --host user@ip is required (skip with --skip-host)" >&2
        exit 1
    fi
fi

TUNNEL_TOKEN_FILE=""
if [ "$SKIP_CF" -ne 1 ]; then
    TUNNEL_TOKEN_FILE=$(mktemp)
    trap 'rm -f "$TUNNEL_TOKEN_FILE"' EXIT

    NO_PROXY_FLAG=()
    [ "$NO_PROXY" -eq 1 ] && NO_PROXY_FLAG+=("--no-proxy")

    if ! command -v python >/dev/null; then
        echo "error: python not on PATH (activate the venv first)" >&2; exit 1
    fi

    python "$REPO_ROOT/scripts/setup_cloudflare_dashboard.py" \
        --hostname "$HOSTNAME_PUBLIC" \
        --tunnel-name "$TUNNEL_NAME" \
        --token-out "$TUNNEL_TOKEN_FILE" \
        "${NO_PROXY_FLAG[@]}"

    TEUTON_DASHBOARD_TUNNEL_TOKEN=$(tr -d '[:space:]' < "$TUNNEL_TOKEN_FILE")
    export TEUTON_DASHBOARD_TUNNEL_TOKEN
fi

if [ "$SKIP_HOST" -eq 1 ]; then
    echo
    echo "[deploy_dashboard] --skip-host set; not touching any remote host."
    echo "Tunnel token has been printed by setup_cloudflare_dashboard.py."
    exit 0
fi

if [ -z "${TEUTON_DASHBOARD_TUNNEL_TOKEN:-}" ]; then
    echo "error: TEUTON_DASHBOARD_TUNNEL_TOKEN is empty (set it or drop --skip-cloudflare)" >&2
    exit 1
fi

TEUTON_RUN_ID="${TEUTON_RUN_ID:-}"
export TEUTON_RUN_ID
if [ -z "$TEUTON_RUN_ID" ]; then
    echo "[deploy_dashboard] TEUTON_RUN_ID empty; dashboard will auto-select latest bucket run"
else
    echo "[deploy_dashboard] pinning dashboard to TEUTON_RUN_ID=$TEUTON_RUN_ID"
fi

# During the Locus -> Teuton rename the live fleet data can still reside in
# the original bucket. Prefer an existing bucket so the dashboard never boots
# into a NoSuchBucket blank page just because the env name moved ahead of the
# S3 migration.
if [ -n "${S3_BUCKET:-}" ]; then
    RESOLVED_S3_BUCKET="$(
        python - <<'PY'
import os

import boto3
from botocore.exceptions import ClientError

current = os.environ.get("S3_BUCKET", "")
fallback = os.environ.get("TEUTON_DASHBOARD_FALLBACK_BUCKET", "locus-decentralized-training-797648826017")
region = os.environ.get("S3_REGION", "us-east-1")
s3 = boto3.client(
    "s3",
    region_name=region,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
)

def exists(name: str) -> bool:
    if not name:
        return False
    try:
        s3.head_bucket(Bucket=name)
        return True
    except ClientError:
        return False

print(current if exists(current) or not exists(fallback) else fallback)
PY
    )"
    if [ -n "$RESOLVED_S3_BUCKET" ] && [ "$RESOLVED_S3_BUCKET" != "$S3_BUCKET" ]; then
        echo "[deploy_dashboard] S3_BUCKET=$S3_BUCKET does not exist; using $RESOLVED_S3_BUCKET"
        S3_BUCKET="$RESOLVED_S3_BUCKET"
        export S3_BUCKET
    fi
fi

SSH_OPTS=(
    -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=15
    -o ServerAliveInterval=30
    -p "$SSH_PORT"
)
SCP_OPTS=(
    -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=15
    -P "$SSH_PORT"
)

remote() { ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"; }
push_inline() {
    local remote_path="$1"; local mode="${2:-600}"
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
        "mkdir -p $(dirname "$remote_path") && cat > '$remote_path' && chmod $mode '$remote_path'"
}

echo
REMOTE_DIR="/root/teuton-dashboard"

echo "=== deploying dashboard stack to $SSH_HOST:$SSH_PORT ==="
remote "mkdir -p '$REMOTE_DIR' /root/.docker"

remote "echo '$DOCKER_PAT' | docker login -u '$DOCKER_USER' --password-stdin"

push_inline "$REMOTE_DIR/.env" 600 <<EOF
DOCKER_USER=$DOCKER_USER
S3_BUCKET=$S3_BUCKET
S3_REGION=${S3_REGION:-us-east-1}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-}
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
TEUTON_NETUID=${TEUTON_NETUID:-3}
TEUTON_RUN_ID=$TEUTON_RUN_ID
TEUTON_DASHBOARD_TUNNEL_TOKEN=$TEUTON_DASHBOARD_TUNNEL_TOKEN
TEUTON_DASHBOARD_REFRESH_SEC=${TEUTON_DASHBOARD_REFRESH_SEC:-3.0}
TEUTON_DASHBOARD_MAX_JOBS=${TEUTON_DASHBOARD_MAX_JOBS:-200}
TEUTON_MAX_INFLIGHT_PER_HOTKEY=${TEUTON_MAX_INFLIGHT_PER_HOTKEY:-8}
TEUTON_DASHBOARD_DB_PATH=${TEUTON_DASHBOARD_DB_PATH:-/var/lib/teuton-dashboard/dashboard.sqlite3}
TEUTON_DASHBOARD_BUCKET_POLL_SEC=${TEUTON_DASHBOARD_BUCKET_POLL_SEC:-5.0}
TEUTON_DASHBOARD_CHAIN_POLL_SEC=${TEUTON_DASHBOARD_CHAIN_POLL_SEC:-30.0}
BT_NETWORK=${BT_NETWORK:-finney}
EOF

scp "${SCP_OPTS[@]}" \
    "$REPO_ROOT/docker/compose.dashboard.yml" \
    "$SSH_HOST:$REMOTE_DIR/compose.yml"

remote "cd '$REMOTE_DIR' && docker compose pull && docker compose up -d"
remote "cd '$REMOTE_DIR' && docker compose ps --format 'table {{.Service}}\t{{.State}}\t{{.Image}}'"

echo
echo "=================================================================="
echo "  https://$HOSTNAME_PUBLIC should be live in <60 s."
echo "  Watch the tunnel:  ssh $SSH_HOST 'docker logs -f teuton-dashboard-tunnel'"
echo "  Watch the UI:      ssh $SSH_HOST 'docker logs -f teuton-dashboard-ui'"
echo "=================================================================="
