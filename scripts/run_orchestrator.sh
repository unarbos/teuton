#!/usr/bin/env bash
# Launch the locus orchestrator from THIS repo against the live fleet.
#
# The orchestrator is the dev-side piece that decides what work to throw on
# the network. It owns the task definition (locus_tasks/gpt_pipe.py) and
# submits manifests + assignment grants to the bucket. Miners, validators,
# and auditors pick the work up from the bucket without needing to know what
# the task is.
#
# Usage:
#   ./scripts/run_orchestrator.sh                            # gpt_pipe, defaults
#   GPT_PIPE_N_BLOCKS_PER_STAGE=8 GPT_PIPE_D=1024 ./scripts/run_orchestrator.sh
#   RUN_ID=sn3-other ./scripts/run_orchestrator.sh
#
# Reads RUN_ID from $RUN_ID, else /tmp/locus_sn3_run_id, else exits.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source .env
set +a

if [ -z "${RUN_ID:-}" ]; then
    if [ -f /tmp/locus_sn3_run_id ]; then
        RUN_ID="$(cat /tmp/locus_sn3_run_id)"
    fi
fi
[ -n "${RUN_ID:-}" ] || { echo "RUN_ID is empty; set env or /tmp/locus_sn3_run_id" >&2; exit 2; }
export RUN_ID

export LOCUS_NETUID="${LOCUS_NETUID:-3}"
export GPT_PIPE_N_STAGES="${GPT_PIPE_N_STAGES:-4}"
export GPT_PIPE_N_MICROBATCHES="${GPT_PIPE_N_MICROBATCHES:-16}"
export GPT_PIPE_N_BLOCKS_PER_STAGE="${GPT_PIPE_N_BLOCKS_PER_STAGE:-4}"
export GPT_PIPE_D="${GPT_PIPE_D:-768}"
export GPT_PIPE_N_HEAD="${GPT_PIPE_N_HEAD:-12}"
export GPT_PIPE_D_FF="${GPT_PIPE_D_FF:-3072}"
export GPT_PIPE_B="${GPT_PIPE_B:-8}"
export GPT_PIPE_T="${GPT_PIPE_T:-256}"

STEPS="${ORCHESTRATOR_STEPS:-1000000}"
TIMEOUT_SEC="${ORCHESTRATOR_TIMEOUT_SEC:-31536000}"
POLL="${ORCHESTRATOR_POLL_INTERVAL:-0.1}"

source .venv/bin/activate

echo "[orchestrator] run_id=$RUN_ID netuid=$LOCUS_NETUID task=gpt_pipe"
echo "  n_stages=$GPT_PIPE_N_STAGES n_microbatches=$GPT_PIPE_N_MICROBATCHES n_blocks_per_stage=$GPT_PIPE_N_BLOCKS_PER_STAGE"
echo "  D=$GPT_PIPE_D N_HEAD=$GPT_PIPE_N_HEAD D_FF=$GPT_PIPE_D_FF B=$GPT_PIPE_B T=$GPT_PIPE_T"

exec python -u -m locus_core.cli orchestrator \
    --run-id "$RUN_ID" \
    --netuid "$LOCUS_NETUID" \
    --task gpt_pipe \
    --steps "$STEPS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --poll-interval "$POLL" \
    --grant-mode presigned \
    --assignment-crypto ed25519 \
    --discovery-backend bucket \
    --s3-bucket "$S3_BUCKET" \
    --s3-region "${S3_REGION:-us-east-1}" \
    --owner-secret "$LOCUS_OWNER_SECRET"
