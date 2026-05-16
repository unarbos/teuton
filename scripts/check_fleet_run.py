"""Poll every non-protected fleet pod and report which RUN_ID it is on now.

For each host:
  - reads `TEUTON_BAKED_RUN_ID` (from `docker inspect` of the running container)
  - greps `run_id` from the most recent banner in `docker logs`

Run repeatedly to watch a roll-out finish.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.lium_protected import PROTECTED_POD_IDS, PROTECTED_SSH_HOSTS

REPO_ROOT = Path(__file__).resolve().parent.parent
FLEET = json.loads((REPO_ROOT / "bench" / "fleet.json").read_text())

CONTAINER_HINTS = {
    "miner": ["teuton-miner"],
    "multi-miner": [
        "teuton-miner-gpu0", "teuton-miner-gpu1", "teuton-miner-gpu2", "teuton-miner-gpu3"
    ],
    "auditor": ["teuton-auditor-gpu0"],
}


def all_hosts() -> list[tuple[str, str, dict]]:
    out: list[tuple[str, str, dict]] = []
    if FLEET.get("auditor"):
        out.append((FLEET["auditor"]["huid"], "auditor", FLEET["auditor"]["ssh"]))
    for m in FLEET.get("miners", []):
        out.append((m["huid"], "miner", m["ssh"]))
    for m in FLEET.get("multi_miner", []):
        out.append((m["huid"], "multi-miner", m["ssh"]))
    return out


def ssh_run(ssh: dict, remote_cmd: str, timeout: int = 15) -> tuple[int, str]:
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", str(ssh["port"]),
        f"{ssh['user']}@{ssh['host']}",
        remote_cmd,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "ssh timeout"


def probe(huid: str, role: str, ssh: dict) -> dict:
    if ssh["host"] in PROTECTED_SSH_HOSTS:
        return {"huid": huid, "role": "PROTECTED", "skip": True}
    container = CONTAINER_HINTS[role][0]
    rc, out = ssh_run(
        ssh,
        f"docker inspect --format '{{{{.Image}}}} {{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' {container} 2>&1 | head -50 ; echo ---LOGS--- ; docker logs {container} --tail 20 2>&1 | grep -E 'run_id|build_time' | head -3"
    )
    info = {"huid": huid, "role": role, "rc": rc, "container": container}
    if rc != 0:
        info["error"] = out.strip()[:200]
        return info
    image_line, _, env_block_logs = out.partition("\n")
    info["image_sha"] = image_line.strip().split("@")[-1] if "@" in image_line else image_line.strip()
    env_block, _, logs = env_block_logs.partition("---LOGS---")
    baked = ""
    for line in env_block.splitlines():
        if line.startswith("TEUTON_BAKED_RUN_ID="):
            baked = line.split("=", 1)[1].strip()
        if line.startswith("TEUTON_RUN_ID="):
            info["env_TEUTON_RUN_ID"] = line.split("=", 1)[1].strip()
    info["env_TEUTON_BAKED_RUN_ID"] = baked
    m = re.search(r"run_id\s*:\s*(\S+)", logs)
    info["banner_run_id"] = m.group(1) if m else ""
    return info


def fmt(info: dict, expected: str) -> str:
    if info.get("skip"):
        return f"  {info['huid']:22s} PROTECTED"
    if "error" in info:
        return f"  {info['huid']:22s} ERROR  {info['error'][:80]}"
    banner = info.get("banner_run_id") or info.get("env_TEUTON_RUN_ID") or info.get("env_TEUTON_BAKED_RUN_ID") or "?"
    mark = "OK " if banner == expected else "OLD"
    return (
        f"  {info['huid']:22s} {mark}  run_id={banner:30s} "
        f"baked={info.get('env_TEUTON_BAKED_RUN_ID','')[:30]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect", default=None, help="Expected run_id (default: /tmp/teuton_sn3_run_id)")
    ap.add_argument("--watch", type=int, default=0, help="Poll every N seconds, until all match --expect (0 = single pass)")
    args = ap.parse_args()
    expected = args.expect or Path("/tmp/teuton_sn3_run_id").read_text().strip()

    hosts = all_hosts()
    print(f"Expecting run_id = {expected}")
    while True:
        results = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(probe, h, r, s): (h, r) for h, r, s in hosts}
            for fut in as_completed(futs):
                results.append(fut.result())
        ok = 0
        total = 0
        results.sort(key=lambda d: d.get("huid", ""))
        for r in results:
            line = fmt(r, expected)
            print(line)
            if "OLD" in line or "ERROR" in line:
                pass
            elif "OK" in line:
                ok += 1
            if not r.get("skip"):
                total += 1
        print(f"--- {ok}/{total} on {expected} ---")
        if args.watch <= 0 or ok == total:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
