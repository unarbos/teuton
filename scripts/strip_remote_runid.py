"""Remove the `RUN_ID=...` line from each non-protected remote pod's
/root/teuton/.env so the image-baked TEUTON_BAKED_RUN_ID wins when Watchtower
recreates the container after a fresh `scripts/build_push.sh --run-id ...`.

Reads bench/fleet.json for ssh endpoints, respects scripts/lium_protected.

Usage:
    python -m scripts.strip_remote_runid
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.lium_protected import PROTECTED_POD_IDS, PROTECTED_SSH_HOSTS

REPO_ROOT = Path(__file__).resolve().parent.parent
FLEET = json.loads((REPO_ROOT / "bench" / "fleet.json").read_text())


def all_hosts() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if FLEET.get("auditor"):
        out.append((FLEET["auditor"]["huid"], FLEET["auditor"]["ssh"]))
    for m in FLEET.get("miners", []):
        out.append((m["huid"], m["ssh"]))
    for m in FLEET.get("multi_miner", []):
        out.append((m["huid"], m["ssh"]))
    return out


def strip(huid: str, ssh: dict) -> tuple[str, int, str]:
    if ssh["host"] in PROTECTED_SSH_HOSTS:
        return (huid, 0, "PROTECTED host, skipped")
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", str(ssh["port"]),
        f"{ssh['user']}@{ssh['host']}",
        # idempotent: remove any RUN_ID= line; harmless if already absent.
        "test -f /root/teuton/.env && sed -i '/^RUN_ID=/d' /root/teuton/.env && "
        "echo 'stripped' || echo 'no .env'",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    msg = (p.stdout or p.stderr or "").strip()
    return (huid, p.returncode, msg)


def main() -> int:
    hosts = all_hosts()
    print(f"stripping RUN_ID from {len(hosts)} hosts")
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(strip, h, s): h for h, s in hosts}
        for fut in as_completed(futs):
            huid, rc, msg = fut.result()
            mark = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"  {huid:22s} {mark}: {msg}")
            if rc != 0:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
