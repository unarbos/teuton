#!/usr/bin/env bash
# One-shot health snapshot of the live Teuton fleet. Designed to be invoked
# in a loop while babysitting a stress run.
#
#   for i in $(seq 1 N); do scripts/monitor_full.sh; sleep 30; done
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source .env
set +a

DASHBOARD_URL="${DASHBOARD_URL:-http://91.224.44.227:40041}"
ts() { date -u +'%H:%M:%S'; }

# Bucket probe (fast)
source .venv/bin/activate
python scripts/probe_network.py

# Orchestrator pulse: last iteration line + uptime
last_iter=$(tail -n 1 /home/const/.pm2/logs/teuton-orchestrator-out.log 2>/dev/null | head -c 200)
pm2_uptime=$(pm2 jlist 2>/dev/null | python3 -c "import sys,json,time; data=json.load(sys.stdin); o=[x for x in data if x['name']=='teuton-orchestrator']; sys.stdout.write('na' if not o else str(int((time.time()*1000-o[0]['pm2_env']['pm_uptime'])/1000))+'s')" 2>/dev/null || echo na)
printf '%s [orch] uptime=%s last_iter=%s\n' "$(ts)" "$pm2_uptime" "${last_iter:-<no log>}"

# Dashboard snapshot summary
timeout 90 curl -fsS "$DASHBOARD_URL/api/snapshot" -o /tmp/_dash_snap.json
python3 - <<'EOF'
import json, sys, time
try:
    s = json.load(open('/tmp/_dash_snap.json'))
except Exception as e:
    print(f"[dash] read err {e}"); raise SystemExit(0)
now = time.time()
j = s.get('jobs') or []
sts = {}
for x in j: sts[x.get('status','?')] = sts.get(x.get('status','?'),0)+1
bk = s['meta']['health']['states'].get('bucket') or {}
print(f"{time.strftime('%H:%M:%S')} [dash] jobs={len(j)} status={sts} machines={len(s.get('machines',[]))} bucket_age={now-bk.get('updated_unix',0):.0f}s")
EOF

# Validator pulse
docker logs --since=90s teuton-validator 2>&1 | grep -E '"submitted":|"reason":|starting at' | tail -3 | sed "s/^/$(ts) [val] /"
