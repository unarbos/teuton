#!/usr/bin/env bash
# Launch the teuton orchestrator from THIS repo against the live fleet.
#
# The orchestrator is the dev-side piece that decides what work to throw on
# the network. It owns the task definition (teuton_tasks/gpt_pipe.py) and
# submits manifests + assignment grants to the bucket. Miners, validators,
# and auditors pick the work up from the bucket without needing to know what
# the task is.
#
# Usage:
#   ./scripts/run_orchestrator.sh                            # gpt_pipe, defaults
#   GPT_PIPE_N_BLOCKS_PER_STAGE=8 GPT_PIPE_D=1024 ./scripts/run_orchestrator.sh
#   RUN_ID=sn3-other ./scripts/run_orchestrator.sh
#
# Reads RUN_ID from $RUN_ID, else /tmp/teuton_sn3_run_id, else exits.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source .env
set +a

if [ -z "${RUN_ID:-}" ]; then
    if [ -f /tmp/teuton_sn3_run_id ]; then
        RUN_ID="$(cat /tmp/teuton_sn3_run_id)"
    fi
fi
[ -n "${RUN_ID:-}" ] || { echo "RUN_ID is empty; set env or /tmp/teuton_sn3_run_id" >&2; exit 2; }
export RUN_ID

export TEUTON_NETUID="${TEUTON_NETUID:-3}"
export GPT_PIPE_N_STAGES="${GPT_PIPE_N_STAGES:-4}"
# Bigger microbatch waves keep the dependency pipeline full for longer. With
# 4 stages and training enabled, each epoch emits roughly:
#   n_microbatches * (4 forward + 4 backward) + 4 outer jobs.
# Stress mode emits 4 stages * n_microbatches forward jobs per iteration. With
# 8 we get 32 jobs / iter, which keeps the dashboard's 50-job cap populated
# with a healthy mix of created/completed/verified jobs (instead of being
# filled entirely with pending creations that scroll off before miners can
# process them). Override via GPT_PIPE_N_MICROBATCHES.
export GPT_PIPE_N_MICROBATCHES="${GPT_PIPE_N_MICROBATCHES:-8}"
export GPT_PIPE_N_BLOCKS_PER_STAGE="${GPT_PIPE_N_BLOCKS_PER_STAGE:-4}"
export GPT_PIPE_D="${GPT_PIPE_D:-768}"
export GPT_PIPE_N_HEAD="${GPT_PIPE_N_HEAD:-12}"
export GPT_PIPE_D_FF="${GPT_PIPE_D_FF:-3072}"
export GPT_PIPE_B="${GPT_PIPE_B:-8}"
export GPT_PIPE_T="${GPT_PIPE_T:-256}"
# Live multi-host runs need enough optimistic quota to keep the pipeline fed
# while receipts land asynchronously. The scheduler still releases quota when
# terminal outputs appear; this just prevents the local orchestrator from
# crashing during the initial fan-out.
export TEUTON_BASE_QUOTA="${TEUTON_BASE_QUOTA:-50000}"

STEPS="${ORCHESTRATOR_STEPS:-1000000}"
TIMEOUT_SEC="${ORCHESTRATOR_TIMEOUT_SEC:-31536000}"
POLL="${ORCHESTRATOR_POLL_INTERVAL:-0.1}"
# Stress mode continuously emits forward-only jobs against the existing run.
# Job IDs use a synthetic epoch counter starting at ORCHESTRATOR_STRESS_EPOCH_BASE
# so they never collide with the historic j-e* IDs the original orchestrator
# wrote. Weights are pinned to ORCHESTRATOR_STRESS_PIN_WEIGHTS_EPOCH (default 0).
# Default to stress mode: continuously emit forward-only jobs against the live
# run so the network stays in churn even when no human is driving training.
# Flip via ORCHESTRATOR_STRESS_EMIT=0 to run the normal training loop instead.
STRESS_EMIT="${ORCHESTRATOR_STRESS_EMIT:-1}"
STRESS_EMIT_INTERVAL="${ORCHESTRATOR_STRESS_EMIT_INTERVAL:-0}"
STRESS_EPOCH_BASE="${ORCHESTRATOR_STRESS_EPOCH_BASE:-1000000}"
STRESS_PIN_WEIGHTS_EPOCH="${ORCHESTRATOR_STRESS_PIN_WEIGHTS_EPOCH:-0}"
EPOCH_TIMEOUT_SEC="${ORCHESTRATOR_EPOCH_TIMEOUT_SEC:-300}"

source .venv/bin/activate

echo "[orchestrator] run_id=$RUN_ID netuid=$TEUTON_NETUID task=gpt_pipe"
echo "  n_stages=$GPT_PIPE_N_STAGES n_microbatches=$GPT_PIPE_N_MICROBATCHES n_blocks_per_stage=$GPT_PIPE_N_BLOCKS_PER_STAGE"
echo "  D=$GPT_PIPE_D N_HEAD=$GPT_PIPE_N_HEAD D_FF=$GPT_PIPE_D_FF B=$GPT_PIPE_B T=$GPT_PIPE_T"
echo "  stress_emit=$STRESS_EMIT stress_emit_interval=$STRESS_EMIT_INTERVAL poll=$POLL quota=$TEUTON_BASE_QUOTA"

stress_args=(--epoch-timeout-sec "$EPOCH_TIMEOUT_SEC")
if [ "$STRESS_EMIT" != "0" ] && [ "$STRESS_EMIT" != "false" ] && [ "$STRESS_EMIT" != "False" ]; then
    stress_args+=(
        --stress-emit
        --stress-emit-interval "$STRESS_EMIT_INTERVAL"
        --stress-epoch-base "$STRESS_EPOCH_BASE"
        --stress-pin-weights-epoch "$STRESS_PIN_WEIGHTS_EPOCH"
    )
fi

exec python -u -m teuton_core.cli orchestrator \
    --run-id "$RUN_ID" \
    --netuid "$TEUTON_NETUID" \
    --task gpt_pipe \
    --steps "$STEPS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --poll-interval "$POLL" \
    "${stress_args[@]}" \
    --grant-mode presigned \
    --grant-ttl-sec "${TEUTON_GRANT_TTL_SEC:-86400}" \
    --assignment-crypto ed25519 \
    --discovery-backend bucket \
    --s3-bucket "$S3_BUCKET" \
    --s3-region "${S3_REGION:-us-east-1}" \
    --wallet-path "${BT_WALLET_PATH:-$HOME/.bittensor/wallets}" \
    --wallet-name "${VALIDATOR_WALLET_NAME:-${BT_WALLET_NAME:-teutonic}}" \
    --hotkey-name "${VALIDATOR_HOTKEY_NAME:-${BT_HOTKEY_NAME:-default}}"
