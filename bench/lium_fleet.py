"""Session-safe Lium fleet helper for Teuton v3.

No credentials are embedded. Set LIUM_API_KEY via Doppler or the environment.
Only pods recorded in SESSION_FILE may be terminated by this script.

All `rent` flows default to the verified daturaai/dind template so the rented
pod boots with a working Docker daemon, allowing the Teuton role containers and
Watchtower to run inside it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

SESSION_FILE = Path(os.environ.get("TEUTON_V3_LIUM_SESSION", "/tmp/teuton_v3_lium_session.json"))

# Pre-verified Lium template that ships docker-in-docker (DinD). Each rented
# pod boots its own dockerd, so we can `docker compose up -d` our teuton images
# + Watchtower inside.
DEFAULT_TEMPLATE_ID = os.environ.get(
    "TEUTON_LIUM_TEMPLATE_ID", "f6f54e1a-88aa-4868-906f-7a8c874e05f9"
)
DEFAULT_TEMPLATE_IMAGE = os.environ.get("TEUTON_LIUM_TEMPLATE_IMAGE", "daturaai/dind")


def client():
    if not os.environ.get("LIUM_API_KEY"):
        raise SystemExit("LIUM_API_KEY is required; load it from Doppler/env")
    import lium
    return lium.Lium()


def resolve_template_id(c, args) -> str:
    """Resolve --template-id / --template-image / DEFAULT_TEMPLATE_ID to an id."""
    if getattr(args, "template_id", None):
        return args.template_id
    image = getattr(args, "template_image", None) or DEFAULT_TEMPLATE_IMAGE
    tag = getattr(args, "template_tag", None) or ""
    tmpl = c.get_template_by_image_name(image_name=image, image_tag=tag or None)
    if tmpl is not None:
        return tmpl.id
    return DEFAULT_TEMPLATE_ID


def load_session() -> dict:
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return {"rented": []}


def save_session(session: dict) -> None:
    SESSION_FILE.write_text(json.dumps(session, indent=2, sort_keys=True))


def cmd_available(args) -> int:
    c = client()
    executors = c.ls(gpu_type=args.gpu)
    executors.sort(key=lambda e: e.price_per_gpu)
    for e in executors[: args.top]:
        net = (e.specs or {}).get("network", {}) or {}
        upload = net.get("ema_verifyx_upload_speed") or net.get("ema_upload_speed", 0)
        print(f"{e.huid} {e.gpu_count}x {e.gpu_type} ${e.price_per_hour:.2f}/hr upload={upload:.0f}Mbps id={e.id}")
    return 0


def cmd_rent(args) -> int:
    c = client()
    template_id = resolve_template_id(c, args)
    session = load_session()
    candidates = []
    for e in c.ls(gpu_type=args.gpu):
        net = (e.specs or {}).get("network", {}) or {}
        upload = net.get("ema_verifyx_upload_speed") or net.get("ema_upload_speed", 0) or 0
        cuda = (e.specs or {}).get("gpu", {}).get("cuda_driver", 0) or 0
        if args.gpu_count is not None and e.gpu_count != args.gpu_count:
            continue
        if upload >= args.min_upload_mbps and cuda >= args.min_cuda:
            candidates.append(e)
    candidates.sort(key=lambda e: e.price_per_gpu)
    for e in candidates[: args.n]:
        res = c.up(
            executor_id=e.id,
            name=f"teuton-v3-{e.huid[:12]}",
            template_id=template_id,
        )
        pod_id = res.get("id") or (res.get("pod") or {}).get("id")
        session.setdefault("rented", []).append({
            "pod_id": pod_id,
            "executor_id": e.id,
            "huid": e.huid,
            "gpu_type": e.gpu_type,
            "gpu_count": e.gpu_count,
            "price_per_hour": float(e.price_per_hour or 0),
            "rented_at": time.time(),
            "template_id": template_id,
        })
        save_session(session)
        print(f"rented {e.huid} pod_id={pod_id} template={template_id}")
    return 0


def cmd_rent_mix(args) -> int:
    """Rent N machines per GPU class, each on the DinD template.

    Example:
        --mix "H100=2,A100=2,RTX4090=2,RTX3090=2,A6000=2"
    """
    if not args.mix:
        raise SystemExit("--mix is required, e.g. H100=2,A100=2,RTX4090=2")
    pairs: list[tuple[str, int]] = []
    for chunk in args.mix.split(","):
        gpu, _, count = chunk.partition("=")
        if not gpu or not count:
            raise SystemExit(f"bad --mix entry: {chunk!r} (expected GPU=N)")
        pairs.append((gpu.strip(), int(count)))

    print(f"[rent-mix] template_id will resolve to {args.template_id or DEFAULT_TEMPLATE_ID}")
    for gpu, n in pairs:
        print(f"\n[rent-mix] === renting {n}x {gpu} ===")
        sub = argparse.Namespace(
            gpu=gpu,
            n=n,
            min_upload_mbps=args.min_upload_mbps,
            min_cuda=args.min_cuda,
            template_id=args.template_id,
            template_image=args.template_image,
            template_tag=args.template_tag,
            gpu_count=args.gpu_count,
        )
        cmd_rent(sub)
    return 0


def cmd_templates(args) -> int:
    c = client()
    filt = args.filter or None
    for tmpl in c.templates(filter=filt, only_my=args.only_my):
        img = getattr(tmpl, "docker_image", "?")
        tag = getattr(tmpl, "docker_image_tag", "?")
        status = getattr(tmpl, "status", "?")
        print(f"  id={tmpl.id} name={tmpl.name} image={img}:{tag} status={status}")
    return 0


def cmd_wait_ready(args) -> int:
    c = client()
    wanted = {r["pod_id"] for r in load_session().get("rented", []) if r.get("pod_id")}
    for pod in c.ps():
        if pod.id in wanted:
            c.wait_ready(pod, timeout=args.timeout, poll_interval=10)
            print(f"ready {pod.huid} {pod.ssh_cmd}")
    return 0


def cmd_write_hosts(args) -> int:
    c = client()
    wanted = {r["pod_id"] for r in load_session().get("rented", []) if r.get("pod_id")}
    pods = [p for p in c.ps() if p.id in wanted]
    lines = ["# Teuton v3 session-only Lium hosts", "# tag user host port n_workers gpu_class price_per_hour"]
    for i, pod in enumerate(sorted(pods, key=lambda p: p.huid)):
        parts = (pod.ssh_cmd or "").split()
        host = parts[1].split("@")[-1] if len(parts) > 1 else "?"
        port = parts[3] if len(parts) > 3 else "22"
        tag = chr(ord("A") + i)
        e = pod.executor
        lines.append(f"{tag} root {host} {port} {e.gpu_count} {e.gpu_type} {float(e.price_per_hour or 0):.2f}")
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(pods)} hosts to {args.output}")
    return 0


def cmd_terminate_mine(args) -> int:
    c = client()
    session = load_session()
    wanted = [r["pod_id"] for r in session.get("rented", []) if r.get("pod_id")]
    pods = {p.id: p for p in c.ps()}
    remaining = []
    for record in session.get("rented", []):
        pod_id = record.get("pod_id")
        if not pod_id:
            continue
        pod = pods.get(pod_id)
        if pod is None:
            continue
        if pod_id not in wanted:
            remaining.append(record)
            continue
        c.down(pod)
        print(f"terminated {pod.huid} ({pod_id})")
    session["rented"] = remaining
    save_session(session)
    return 0


def _add_template_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--template-id", default=None, help=f"Lium template id; default {DEFAULT_TEMPLATE_ID}")
    parser.add_argument("--template-image", default=None, help=f"Lookup template by docker image (default {DEFAULT_TEMPLATE_IMAGE})")
    parser.add_argument("--template-tag", default=None, help="Optional docker tag for --template-image lookup")
    parser.add_argument("--gpu-count", type=int, default=None, help="If set, only rent executors with exactly this GPU count")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    av = sub.add_parser("available")
    av.add_argument("--gpu", required=True)
    av.add_argument("--top", type=int, default=10)
    av.set_defaults(fn=cmd_available)
    rent = sub.add_parser("rent")
    rent.add_argument("--gpu", required=True)
    rent.add_argument("--n", type=int, required=True)
    rent.add_argument("--min-upload-mbps", type=float, default=150.0)
    rent.add_argument("--min-cuda", type=int, default=12000)
    _add_template_args(rent)
    rent.set_defaults(fn=cmd_rent)

    mix = sub.add_parser("rent-mix")
    mix.add_argument("--mix", required=True, help="GPU=N,GPU=N,... e.g. H100=2,A100=2,RTX4090=2")
    mix.add_argument("--min-upload-mbps", type=float, default=150.0)
    mix.add_argument("--min-cuda", type=int, default=12000)
    _add_template_args(mix)
    mix.set_defaults(fn=cmd_rent_mix)

    tmpl = sub.add_parser("templates")
    tmpl.add_argument("--filter", default=None)
    tmpl.add_argument("--only-my", action="store_true")
    tmpl.set_defaults(fn=cmd_templates)

    wait = sub.add_parser("wait-ready")
    wait.add_argument("--timeout", type=int, default=900)
    wait.set_defaults(fn=cmd_wait_ready)
    hosts = sub.add_parser("write-hosts")
    hosts.add_argument("--output", default="bench/hosts.env")
    hosts.set_defaults(fn=cmd_write_hosts)
    sub.add_parser("terminate-mine").set_defaults(fn=cmd_terminate_mine)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
