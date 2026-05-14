#!/usr/bin/env bash
# Spawn N detached workers on the local box, each writing to /tmp/locus-w<i>.log.
# Args: <N_WORKERS> <RUN_ID> <BOX_TAG>
set -euo pipefail

N=$1
RUN_ID=$2
BOX_TAG=$3

mkdir -p /tmp/locus_logs
rm -f /tmp/locus_logs/${BOX_TAG}-w*.log

for i in $(seq 0 $((N-1))); do
    WID="${BOX_TAG}-w${i}"
    LOG="/tmp/locus_logs/${WID}.log"
    setsid nohup /root/.venv/bin/python -u -m bench.dist worker \
        --run-id "$RUN_ID" \
        --worker-id "$WID" \
        --poll-interval 1.0 \
        --heartbeat-interval 2.0 \
        --max-idle-iters 60 \
        > "$LOG" 2>&1 < /dev/null &
done

sleep 1
echo "started: $(pgrep -fa "bench.dist worker --run-id $RUN_ID" | wc -l) workers"
pgrep -fa "bench.dist worker --run-id $RUN_ID" | head -5
