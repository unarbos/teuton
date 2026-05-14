#!/usr/bin/env bash
# Pool management for the 5-box / 23-GPU Lium pool.
#
# Subcommands:
#   sync             rsync local source + .env to every box, pip install -e .
#   bootstrap TASK RUN_ID ROUNDS    bootstrap a run on the master (D) only
#   start TASK RUN_ID NUB DEVICE    start orchestrator on D + workers everywhere
#                                   pinned to UB by index. DEVICE = cpu | cuda
#   tail RUN_ID ROUNDS              live tail metrics + telemetry from D
#   stop                            kill all bench.dist processes on every box
#   wipe RUN_ID                     wipe an S3 prefix
#   status                          who is alive, how many procs per box
#
# Inventory (TAG, USER, HOST, PORT, N_WORKERS, GPU_CLASS) lives below as the
# `BOXES` array. Workers are launched in pin-id order across boxes so UB
# assignments are reproducible.

set -euo pipefail

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
SSH() { ssh "${SSH_OPTS[@]}" "$@"; }
SCP() { scp "${SSH_OPTS[@]}" "$@"; }

# tag user host port n_workers gpu_class
BOXES=(
  "A root 95.133.252.32 10300 8 H200"
  "B root 144.31.250.244 2009 8 RTX3090"
  "C root 144.31.250.244 1009 5 RTX4090"
  "D root 91.224.44.207 20600 1 RTX3090"
  "E root 91.224.44.226 30199 1 RTX3090"
)

# D is the orchestrator. Listing here so other commands know.
MASTER_TAG="D"

_for_each_box() {
  local fn=$1
  for spec in "${BOXES[@]}"; do
    read -r tag user host port nwork gpuclass <<< "$spec"
    "$fn" "$tag" "$user" "$host" "$port" "$nwork" "$gpuclass"
  done
}

_master() {
  for spec in "${BOXES[@]}"; do
    read -r tag user host port nwork gpuclass <<< "$spec"
    if [ "$tag" = "$MASTER_TAG" ]; then
      echo "$user $host $port"
      return
    fi
  done
  echo ""
}

# -------- sync ---------------------------------------------------------------

_sync_one() {
  local tag=$1 user=$2 host=$3 port=$4
  echo "[sync $tag] -> $user@$host:$port"
  rsync -az --delete \
    --exclude '__pycache__' --exclude '.venv' --exclude '*.egg-info' \
    --exclude '.pytest_cache' \
    -e "ssh ${SSH_OPTS[*]} -p $port" \
    /Users/const/Locus/Locus/ "$user@$host:/root/Locus/" >/dev/null
  SCP -P "$port" /Users/const/Locus/.env "$user@$host:/root/.env" >/dev/null
  SSH -p "$port" "$user@$host" \
    "cd /root/Locus && /root/.venv/bin/pip install --quiet -e ." 2>&1 | tail -1
}

cmd_sync() { _for_each_box _sync_one; echo "[sync] all 5 boxes synced"; }

# -------- bootstrap ----------------------------------------------------------

cmd_bootstrap() {
  local task=$1 run=$2 rounds=$3
  read -r u h p <<< "$(_master)"
  SSH -p "$p" "$u@$h" \
    "cd /root/Locus && /root/.venv/bin/python -m bench.legacy_dist bootstrap --task $task --run-id $run --max-rounds $rounds"
}

# -------- start --------------------------------------------------------------
#
# Assigns workers to UBs in a globally-determined order so the same call always
# gives the same pin layout. Order of boxes is BOXES[]; within each box w0..wN-1.
# UB id = (worker_index % NUB).

_start_box_workers() {
  local tag=$1 user=$2 host=$3 port=$4 nwork=$5 gpuclass=$6
  local run=$7 nub=$8 device=$9 base_idx=${10}

  SSH -p "$port" "$user@$host" "cat > /root/start_local.sh" <<'EOF'
#!/usr/bin/env bash
# args: TAG RUN NUB DEVICE GPU_CLASS BASE_IDX N_WORKERS
set -e
TAG=$1; RUN=$2; NUB=$3; DEVICE=$4; GPUCLASS=$5; BASE_IDX=$6; N=$7
mkdir -p /tmp/locus_logs
rm -f /tmp/locus_logs/${TAG}-w*.log
for i in $(seq 0 $((N-1))); do
    GLOBAL_IDX=$((BASE_IDX + i))
    UB=$((GLOBAL_IDX % NUB))
    WID="${TAG}-w${i}"
    LOG="/tmp/locus_logs/${WID}.log"
    if [ "$DEVICE" = "cuda" ]; then
        DEV_ARG="cuda:${i}"
    else
        DEV_ARG="cpu"
    fi
    setsid nohup /root/.venv/bin/python -u -m bench.dist worker \
        --run-id "$RUN" \
        --worker-id "$WID" \
        --resident-ub "$UB" \
        --gpu-class "$GPUCLASS" \
        --device "$DEV_ARG" \
        --poll-interval 0.5 \
        --heartbeat-interval 1.0 \
        --max-idle-iters 1200 \
        > "$LOG" 2>&1 < /dev/null &
done
sleep 1
echo "$TAG: $(pgrep -fa 'bench.dist worker' | wc -l) workers (base_idx=$BASE_IDX, nub=$NUB, device=$DEVICE)"
EOF

  SSH -p "$port" "$user@$host" \
    "bash /root/start_local.sh $tag $run $nub $device $gpuclass $base_idx $nwork"
}

cmd_start() {
  local task=$1 run=$2 nub=$3 device=$4
  echo "[start] task=$task run=$run nub=$nub device=$device"
  read -r u h p <<< "$(_master)"
  echo "[orch] starting on master ($MASTER_TAG = $u@$h:$p)"

  # Write a tiny start-orch script remotely, then invoke it. The double-fork
  # via setsid + bash -c keeps the python process from holding ssh's stdio.
  SSH -p "$p" "$u@$h" "cat > /root/start_orch.sh" <<'EOF'
#!/usr/bin/env bash
set -e
TASK=$1; RUN=$2
cd /root/Locus
rm -f /tmp/locus-orch.log
# 90s startup_grace_sec: ssh+setsid serial spawning across 5 boxes takes
# ~30-60s; the orchestrator must wait long enough that all workers heartbeat
# before round 0 emits, otherwise pin assignments get stuck on the few
# workers that beat the grace.
setsid bash -c "/root/.venv/bin/python -u -m bench.dist orchestrator --task $TASK --run-id $RUN --poll-interval 0.2 --startup-grace-sec 90 >/tmp/locus-orch.log 2>&1 </dev/null" >/dev/null 2>&1 </dev/null &
disown $! 2>/dev/null || true
sleep 1
pgrep -fa "bench.dist orchestrator" | head -1
EOF
  SSH -p "$p" "$u@$h" "bash /root/start_orch.sh $task $run"

  local base_idx=0
  for spec in "${BOXES[@]}"; do
    read -r tag user host port nwork gpuclass <<< "$spec"
    _start_box_workers "$tag" "$user" "$host" "$port" "$nwork" "$gpuclass" \
      "$run" "$nub" "$device" "$base_idx"
    base_idx=$((base_idx + nwork))
  done
  echo "[start] launched $base_idx workers across 5 boxes for run=$run"
}

# -------- start_pipe (streaming mode) ----------------------------------------
# args: TASK RUN_ID N_STAGES DEVICE
# Pins workers to STAGES (not UBs). Uses streaming-orchestrator subcommand.

_start_pipe_box_workers() {
  local tag=$1 user=$2 host=$3 port=$4 nwork=$5 gpuclass=$6
  local run=$7 nstages=$8 device=$9 base_idx=${10}

  SSH -p "$port" "$user@$host" "cat > /root/start_pipe_local.sh" <<'EOF'
#!/usr/bin/env bash
set -e
TAG=$1; RUN=$2; NSTAGES=$3; DEVICE=$4; GPUCLASS=$5; BASE_IDX=$6; N=$7
mkdir -p /tmp/locus_logs
rm -f /tmp/locus_logs/${TAG}-w*.log
for i in $(seq 0 $((N-1))); do
    GLOBAL_IDX=$((BASE_IDX + i))
    STAGE=$((GLOBAL_IDX % NSTAGES))
    WID="${TAG}-w${i}"
    LOG="/tmp/locus_logs/${WID}.log"
    if [ "$DEVICE" = "cuda" ]; then
        DEV_ARG="cuda:${i}"
    else
        DEV_ARG="cpu"
    fi
    setsid nohup /root/.venv/bin/python -u -m bench.dist worker \
        --run-id "$RUN" \
        --worker-id "$WID" \
        --pipe-stage "$STAGE" \
        --gpu-class "$GPUCLASS" \
        --device "$DEV_ARG" \
        --poll-interval 0.3 \
        --heartbeat-interval 1.0 \
        --max-idle-iters 10000 \
        > "$LOG" 2>&1 < /dev/null &
done
sleep 1
echo "$TAG: $(pgrep -fa 'bench.dist worker' | wc -l) workers (base_idx=$BASE_IDX, nstages=$NSTAGES, device=$DEVICE)"
EOF

  SSH -p "$port" "$user@$host" \
    "bash /root/start_pipe_local.sh $tag $run $nstages $device $gpuclass $base_idx $nwork"
}

cmd_start_pipe() {
  local task=$1 run=$2 nstages=$3 device=$4
  echo "[start-pipe] task=$task run=$run nstages=$nstages device=$device"
  read -r u h p <<< "$(_master)"
  echo "[stream-orch] starting on master ($MASTER_TAG = $u@$h:$p)"

  SSH -p "$p" "$u@$h" "cat > /root/start_stream_orch.sh" <<'EOF'
#!/usr/bin/env bash
set -e
TASK=$1; RUN=$2
cd /root/Locus
rm -f /tmp/locus-orch.log
setsid bash -c "/root/.venv/bin/python -u -m bench.dist streaming-orchestrator --task $TASK --run-id $RUN --poll-interval 0.2 --startup-grace-sec 90 >/tmp/locus-orch.log 2>&1 </dev/null" >/dev/null 2>&1 </dev/null &
disown $! 2>/dev/null || true
sleep 1
pgrep -fa "bench.dist streaming-orchestrator" | head -1
EOF
  SSH -p "$p" "$u@$h" "bash /root/start_stream_orch.sh $task $run"

  local base_idx=0
  for spec in "${BOXES[@]}"; do
    read -r tag user host port nwork gpuclass <<< "$spec"
    _start_pipe_box_workers "$tag" "$user" "$host" "$port" "$nwork" "$gpuclass" \
      "$run" "$nstages" "$device" "$base_idx"
    base_idx=$((base_idx + nwork))
  done
  echo "[start-pipe] launched $base_idx workers across 5 boxes for run=$run"
}

# -------- tail ---------------------------------------------------------------

cmd_tail() {
  local run=$1 rounds=$2
  read -r u h p <<< "$(_master)"
  SSH -p "$p" "$u@$h" \
    "cd /root/Locus && /root/.venv/bin/python -m bench.legacy_dist tail \
       --run-id $run --max-rounds $rounds --poll-interval 2.0 --timeout-sec 1800.0"
}

# -------- stop ---------------------------------------------------------------

_stop_box() {
  local tag=$1 user=$2 host=$3 port=$4
  SSH -p "$port" "$user@$host" \
    "ps aux | awk '/bench\\.dist (worker|orchestrator)/ && !/awk/ {print \$2}' | xargs -r kill -9 2>/dev/null; \
     sleep 1; \
     n=\$(ps aux | awk '/bench\\.dist (worker|orchestrator)/ && !/awk/' | wc -l); \
     echo \"$tag: alive=\$n\""
}

cmd_stop() { _for_each_box _stop_box; }

# -------- wipe ---------------------------------------------------------------

cmd_wipe() {
  local run=$1
  read -r u h p <<< "$(_master)"
  SSH -p "$p" "$u@$h" \
    "cd /root/Locus && /root/.venv/bin/python -m bench.legacy_dist wipe --run-id $run"
}

# -------- status -------------------------------------------------------------

_status_box() {
  local tag=$1 user=$2 host=$3 port=$4 nwork=$5 gpuclass=$6
  local out
  out=$(SSH -p "$port" "$user@$host" \
    "ps aux | awk '/bench\\.dist worker/ && !/awk/' | wc -l; \
     ps aux | awk '/bench\\.dist orchestrator/ && !/awk/' | wc -l; \
     ls /tmp/locus_logs/${tag}-w*.log 2>/dev/null | wc -l" 2>&1)
  awk -v tag="$tag" -v expect="$nwork" -v gpu="$gpuclass" \
    'BEGIN{nw=0;no=0;nl=0} NR==1{nw=$1} NR==2{no=$1} NR==3{nl=$1} \
     END{printf "%-3s %-10s expect=%d alive_workers=%d alive_orch=%d log_files=%d\n", \
                 tag, gpu, expect, nw, no, nl}' <<< "$out"
}

cmd_status() { _for_each_box _status_box; }

# -------- main ---------------------------------------------------------------

case "${1:-}" in
  sync)       cmd_sync ;;
  bootstrap)  shift; cmd_bootstrap "$@" ;;
  start)      shift; cmd_start "$@" ;;
  start-pipe) shift; cmd_start_pipe "$@" ;;
  tail)       shift; cmd_tail "$@" ;;
  stop)       cmd_stop ;;
  wipe)       shift; cmd_wipe "$@" ;;
  status)     cmd_status ;;
  *) cat <<USAGE
Usage:
  bench/pool.sh sync
  bench/pool.sh bootstrap TASK RUN_ID ROUNDS
  bench/pool.sh start TASK RUN_ID N_UB DEVICE
  bench/pool.sh start-pipe TASK RUN_ID N_STAGES DEVICE
  bench/pool.sh tail RUN_ID ROUNDS
  bench/pool.sh stop
  bench/pool.sh wipe RUN_ID
  bench/pool.sh status
USAGE
  exit 1 ;;
esac
