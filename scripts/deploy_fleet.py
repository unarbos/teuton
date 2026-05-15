"""Deploy the Locus Docker stack to every assigned Lium pod in bench/fleet.json.

For each pod the script:
  1. Verifies it is not in the protected list (scripts/lium_protected.py).
  2. SSHes in, ensures dockerd is up (idempotent on DinD pods).
  3. SCPs the assigned wallet hotkeys to /root/.bittensor/wallets/locus_mining/hotkeys/.
  4. Writes /root/locus/.env with bucket creds + shared secrets + per-host
     hotkey assignments.
  5. SCPs the role-appropriate compose file.
  6. docker login via the Doppler-supplied PAT (so Watchtower can pull).
  7. docker compose pull && up -d.

Usage:
    doppler run --project arbos --config dev -- \\
        python scripts/deploy_fleet.py [--only miner|auditor|multi-miner] \\
                                       [--host-filter <huid_or_pod_id>]

Set RUN_ID via --run-id or it falls back to /tmp/locus_sn3_run_id.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.lium_protected import PROTECTED_POD_IDS, PROTECTED_SSH_HOSTS

REPO_ROOT = Path(__file__).resolve().parent.parent
FLEET_JSON = REPO_ROOT / "bench" / "fleet.json"
WALLET_ROOT = Path("/home/const/.bittensor/wallets/locus_mining/hotkeys")

REQUIRED_ENV_KEYS = [
    "DOCKER_USER",
    "DOCKER_PAT",
    "S3_BUCKET",
    "S3_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "LOCUS_OWNER_SECRET",
    "LOCUS_MINER_SECRET",
    "LOCUS_VALIDATOR_SECRET",
    "LOCUS_ASSIGNMENT_SECRET",
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_env() -> dict[str, str]:
    """Merge process env + .env values (.env wins for AWS/LOCUS keys)."""
    env = dict(os.environ)
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    missing = [k for k in REQUIRED_ENV_KEYS if not env.get(k)]
    if missing:
        fail(f"missing env keys: {missing}")
    env.setdefault("S3_REGION", "us-east-1")
    env.setdefault("S3_ENDPOINT_URL", "")
    return env


def ssh_cmd(host_cfg: dict) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-p", str(host_cfg["port"]),
        f"{host_cfg['user']}@{host_cfg['host']}",
    ]


def scp_cmd(host_cfg: dict, local: str, remote: str) -> list[str]:
    return [
        "scp",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        "-P", str(host_cfg["port"]),
        local,
        f"{host_cfg['user']}@{host_cfg['host']}:{remote}",
    ]


def assert_unprotected(pod_id: str, host: str) -> None:
    if pod_id in PROTECTED_POD_IDS:
        fail(f"refusing to deploy to PROTECTED pod_id={pod_id}")
    if host in PROTECTED_SSH_HOSTS:
        fail(f"refusing to deploy to PROTECTED ssh host={host}")


def ssh_exec(host_cfg: dict, remote_cmd: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ssh_cmd(host_cfg) + [remote_cmd]
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def scp_file(host_cfg: dict, local: Path, remote: str) -> None:
    if not local.exists():
        fail(f"local file missing: {local}")
    cmd = scp_cmd(host_cfg, str(local), remote)
    subprocess.run(cmd, check=True)


def scp_inline(host_cfg: dict, content: str, remote: str, *, chmod: str | None = None) -> None:
    """Pipe content to the remote path via ssh tee (avoids temp files locally)."""
    cmd = ssh_cmd(host_cfg) + [
        f"mkdir -p {shlex.quote(os.path.dirname(remote) or '/')} && cat > {shlex.quote(remote)}"
        + (f" && chmod {chmod} {shlex.quote(remote)}" if chmod else "")
    ]
    p = subprocess.run(cmd, input=content.encode("utf-8"), check=True)


def render_dotenv(env: dict[str, str], run_id: str | None, extras: dict[str, str]) -> str:
    # NOTE: RUN_ID is intentionally NOT written here. Each image carries a
    # `LOCUS_BAKED_RUN_ID` ARG that the entrypoint resolves; that's the source
    # of truth so a single `scripts/build_push.sh --run-id X` flips the whole
    # fleet on the next Watchtower pull without re-scp'ing .env files.
    # Callers that need a host-pinned override can write `RUN_ID=` themselves
    # after this function returns (rare; usually for one-off debugging only).
    base = {
        "DOCKER_USER": env["DOCKER_USER"],
        "S3_BUCKET": env["S3_BUCKET"],
        "S3_REGION": env["S3_REGION"],
        "S3_ENDPOINT_URL": env.get("S3_ENDPOINT_URL", ""),
        "AWS_ACCESS_KEY_ID": env["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": env["AWS_SECRET_ACCESS_KEY"],
        "LOCUS_OWNER_SECRET": env["LOCUS_OWNER_SECRET"],
        "LOCUS_MINER_SECRET": env["LOCUS_MINER_SECRET"],
        "LOCUS_VALIDATOR_SECRET": env["LOCUS_VALIDATOR_SECRET"],
        "LOCUS_ASSIGNMENT_SECRET": env["LOCUS_ASSIGNMENT_SECRET"],
        "LOCUS_ASSIGNMENT_CRYPTO": env.get("LOCUS_ASSIGNMENT_CRYPTO", "ed25519"),
        "LOCUS_NETUID": env.get("LOCUS_NETUID", "3"),
    }
    base.update(extras)
    lines = [f"{k}={shlex.quote(v)}" for k, v in base.items()]
    return "\n".join(lines) + "\n"


def docker_login_remote(host_cfg: dict, env: dict[str, str]) -> None:
    cmd = (
        f'echo {shlex.quote(env["DOCKER_PAT"])} | '
        f'docker login -u {shlex.quote(env["DOCKER_USER"])} --password-stdin'
    )
    ssh_exec(host_cfg, cmd)


def ensure_remote_dirs(host_cfg: dict) -> None:
    ssh_exec(
        host_cfg,
        "mkdir -p /root/locus && "
        "mkdir -p /root/.bittensor/wallets/locus_mining/hotkeys && "
        "mkdir -p /root/.docker && "
        "chmod 700 /root/.bittensor/wallets/locus_mining/hotkeys",
    )


def scp_hotkey(host_cfg: dict, hotkey_name: str) -> None:
    local_priv = WALLET_ROOT / hotkey_name
    local_pub = WALLET_ROOT / f"{hotkey_name}pub.txt"
    local_sidecar = WALLET_ROOT / f"{hotkey_name}.ed25519.json"
    remote_dir = "/root/.bittensor/wallets/locus_mining/hotkeys"
    if local_priv.exists():
        scp_inline(host_cfg, local_priv.read_text(), f"{remote_dir}/{hotkey_name}", chmod="600")
    else:
        fail(f"missing local hotkey: {local_priv}")
    if local_pub.exists():
        scp_inline(host_cfg, local_pub.read_text(), f"{remote_dir}/{hotkey_name}pub.txt", chmod="644")
    if local_sidecar.exists():
        scp_inline(
            host_cfg, local_sidecar.read_text(), f"{remote_dir}/{hotkey_name}.ed25519.json", chmod="600"
        )


def deploy_single(host_cfg: dict, *, role: str, env: dict[str, str], run_id: str, dotenv_extras: dict[str, str], hotkeys: list[str], compose_local: Path) -> None:
    assert_unprotected(host_cfg["pod_id"], host_cfg["ssh"]["host"])
    ssh = host_cfg["ssh"]
    print(f"\n=== {role}: {host_cfg.get('huid','?')} ({ssh['host']}:{ssh['port']}) ===")

    ensure_remote_dirs(ssh)
    docker_login_remote(ssh, env)
    for hk in hotkeys:
        scp_hotkey(ssh, hk)

    # Push compose file
    remote_compose = "/root/locus/compose.yml"
    scp_inline(ssh, compose_local.read_text(), remote_compose, chmod="644")

    # Push the .env (must NOT be world-readable: 600)
    scp_inline(
        ssh,
        render_dotenv(env, run_id, dotenv_extras),
        "/root/locus/.env",
        chmod="600",
    )

    # Bring up the stack
    ssh_exec(
        ssh,
        "cd /root/locus && docker compose pull && docker compose up -d",
    )
    out = ssh_exec(
        ssh,
        "cd /root/locus && docker compose ps --format 'table {{.Service}}\\t{{.State}}\\t{{.Image}}'",
        capture=True,
        check=False,
    )
    print(out.stdout)
    if out.stderr:
        print(out.stderr, file=sys.stderr)


def deploy(fleet: dict[str, Any], *, env: dict[str, str], run_id: str, only_role: str | None, host_filter: str | None) -> None:
    def match(h: dict) -> bool:
        if host_filter and host_filter not in (h.get("huid", ""), h.get("pod_id", "")):
            return False
        return True

    # Miners (single-miner pods)
    if only_role in (None, "miner"):
        for m in fleet.get("miners", []):
            if not match(m):
                continue
            ssh = m["ssh"]
            host_cfg = {**ssh, "pod_id": m["pod_id"], "ssh": ssh, "huid": m["huid"]}
            extras = {
                "MINER_HOTKEY_SS58": m["hotkey"]["ss58"],
                "MINER_HOTKEY_NAME": m["hotkey"]["name"],
                "MINER_WALLET_NAME": "locus_mining",
                "MINER_DEVICES": m.get("miner_devices", "cuda"),
            }
            deploy_single(
                host_cfg,
                role="miner",
                env=env,
                run_id=run_id,
                dotenv_extras=extras,
                hotkeys=[m["hotkey"]["name"]],
                compose_local=REPO_ROOT / m["compose"],
            )

    # Multi-miner pods
    if only_role in (None, "multi-miner"):
        for m in fleet.get("multi_miner", []):
            if not match(m):
                continue
            ssh = m["ssh"]
            host_cfg = {**ssh, "pod_id": m["pod_id"], "ssh": ssh, "huid": m["huid"]}
            extras: dict[str, str] = {"MINER_WALLET_NAME": "locus_mining"}
            for hk in m["hotkeys"]:
                i = hk["gpu_index"]
                extras[f"MINER_HK_SS58_{i}"] = hk["ss58"]
                extras[f"MINER_HK_NAME_{i}"] = hk["name"]
            deploy_single(
                host_cfg,
                role="multi-miner",
                env=env,
                run_id=run_id,
                dotenv_extras=extras,
                hotkeys=[hk["name"] for hk in m["hotkeys"]],
                compose_local=REPO_ROOT / m["compose"],
            )

    # Auditor
    if only_role in (None, "auditor"):
        a = fleet.get("auditor")
        if a and match(a):
            ssh = a["ssh"]
            host_cfg = {**ssh, "pod_id": a["pod_id"], "ssh": ssh, "huid": a["huid"]}
            extras: dict[str, str] = {"AUDITOR_WALLET_NAME": "locus_mining"}
            for hk in a["hotkeys"]:
                i = hk["index"]
                extras[f"AUDITOR_HK_{i}"] = hk["ss58"]
                extras[f"AUDITOR_HK_NAME_{i}"] = hk["name"]
            deploy_single(
                host_cfg,
                role="auditor",
                env=env,
                run_id=run_id,
                dotenv_extras=extras,
                hotkeys=[hk["name"] for hk in a["hotkeys"]],
                compose_local=REPO_ROOT / a["compose"],
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["miner", "multi-miner", "auditor"], default=None)
    ap.add_argument("--host-filter", default=None, help="match by huid or pod_id")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fleet", default=str(FLEET_JSON))
    args = ap.parse_args()

    fleet = json.loads(Path(args.fleet).read_text())
    env = load_env()

    run_id = args.run_id
    if not run_id:
        rid_file = Path(fleet.get("run_id_file", "/tmp/locus_sn3_run_id"))
        if not rid_file.exists():
            fail(f"run-id file missing: {rid_file}")
        run_id = rid_file.read_text().strip()
    if not run_id:
        fail("empty run id")

    print(f"deploy plan:")
    print(f"  netuid    : {fleet.get('netuid')}")
    print(f"  network   : {fleet.get('network')}")
    print(f"  run_id    : {run_id}")
    print(f"  miners    : {len(fleet.get('miners', []))} single-pod + {sum(len(m['hotkeys']) for m in fleet.get('multi_miner', []))} multi")
    print(f"  auditor   : {fleet['auditor']['huid'] if fleet.get('auditor') else 'none'}")
    if args.only:
        print(f"  filter    : only={args.only}")
    if args.host_filter:
        print(f"  filter    : host={args.host_filter}")
    print()
    deploy(fleet, env=env, run_id=run_id, only_role=args.only, host_filter=args.host_filter)
    print("\nFLEET DEPLOY DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
