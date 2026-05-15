"""Switch every non-protected Lium pod from the current template to the
verified `daturaai/dind` template via the underlying API.

The SDK's `Lium.switch_template` has a parsing bug (`ExecutorInfo.ip` is now
required but the API doesn't return it). This script bypasses that by calling
the raw HTTP endpoint directly and waits for each pod to come back to
`RUNNING` on the new template.

Usage:
    doppler run --project arbos --config dev -- \\
        python scripts/lium_switch_to_dind.py --confirm

Without --confirm the script only prints what it would do.
"""
from __future__ import annotations

import argparse
import time

import lium

from scripts.lium_protected import PROTECTED_POD_IDS, PROTECTED_SSH_HOSTS

DIND_TEMPLATE_ID = "f6f54e1a-88aa-4868-906f-7a8c874e05f9"
DIND_TEMPLATE_NAME = "daturaai/dind"


def ssh_host(pod) -> str:
    ssh = pod.ssh_cmd or ""
    return ssh.split("@", 1)[-1].split(" ", 1)[0] if "@" in ssh else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true", help="actually perform the switch")
    ap.add_argument("--include-pod-id", action="append", default=[], help="restrict to specific pod_ids")
    ap.add_argument("--wait-each", type=float, default=180.0, help="seconds to wait per pod for RUNNING")
    args = ap.parse_args()

    c = lium.Lium()
    pods = list(c.ps())

    selected = []
    for p in pods:
        host = ssh_host(p)
        if p.id in PROTECTED_POD_IDS or host in PROTECTED_SSH_HOSTS:
            print(f"  skip PROTECTED {p.huid} ({p.id})")
            continue
        if args.include_pod_id and p.id not in args.include_pod_id:
            continue
        # Pull the current template name so we don't re-switch DinD pods.
        try:
            meta = c.pod(p.id)
            tmpl_image = (meta.get("template") or {}).get("docker_image", "") if isinstance(meta, dict) else ""
        except Exception:
            tmpl_image = ""
        if DIND_TEMPLATE_NAME in tmpl_image:
            print(f"  skip already-DinD {p.huid}")
            continue
        selected.append((p, tmpl_image))

    if not selected:
        print("no pods to switch.")
        return 0

    print(f"\nWill switch these {len(selected)} pod(s) to DinD ({DIND_TEMPLATE_ID}):")
    for p, img in selected:
        exe = p.executor
        gpu = f"{exe.gpu_count}x {exe.gpu_type}" if exe else "?"
        print(f"  - {p.huid:24s}  {p.id}  {gpu:24s}  was:{img}")

    if not args.confirm:
        print("\nDry-run only. Re-run with --confirm to actually switch.")
        return 0

    for p, _img in selected:
        print(f"\n=== switching {p.huid} ({p.id}) ===")
        try:
            resp = c._request(
                "PUT",
                f"/pods/{p.id}/switch-template",
                json={"template_id": DIND_TEMPLATE_ID},
            )
            data = resp.json()
            print(f"  HTTP {resp.status_code}  status={data.get('status')}")
        except Exception as e:
            print(f"  switch failed: {e!r}")
            continue

        deadline = time.time() + args.wait_each
        last_status = None
        while time.time() < deadline:
            time.sleep(8)
            try:
                fresh = {x.id: x for x in c.ps()}.get(p.id)
                if not fresh:
                    print("  pod disappeared (still being reprovisioned)")
                    continue
                st = fresh.status
                if st != last_status:
                    print(f"  status={st} ssh={fresh.ssh_cmd}")
                    last_status = st
                if st == "RUNNING":
                    break
            except Exception as e:
                print(f"  poll error: {e!r}")
        else:
            print(f"  WARNING: still not RUNNING after {args.wait_each}s")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
