#!/usr/bin/env bash
set -e
cd /root/teuton
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
pip install --quiet boto3
pkill -f viz_demo_heartbeat.py 2>/dev/null || true
sleep 1
nohup bash -lc 'source /root/teuton/.viz_demo.env && source /root/teuton/.venv/bin/activate && exec python -u /root/teuton/viz_demo_heartbeat.py' > /tmp/viz_demo_heartbeat.log 2>&1 &
disown
sleep 3
pgrep -fa viz_demo_heartbeat.py || echo NOPID
echo --- log ---
tail -n 8 /tmp/viz_demo_heartbeat.log 2>/dev/null || echo NOLOG
