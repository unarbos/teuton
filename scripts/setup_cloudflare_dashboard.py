#!/usr/bin/env python3
"""Provision the Teuton public dashboard via Cloudflare Tunnel.

This script is idempotent. It creates (or reuses) one Cloudflare Tunnel,
points a public hostname like ``dashboard.teutonic.ai`` at the local
``discovery-ui`` container, sets the matching DNS CNAME, and prints the
tunnel token. Drop that token into ``TEUTON_DASHBOARD_TUNNEL_TOKEN`` on the
host running ``docker/compose.dashboard.yml`` and the dashboard becomes
globally reachable over HTTPS — no inbound port required on the box.

Architecture:

    public                 cloudflare edge                 host
    -----                  ---------------                 ----
    https://dashboard.teutonic.ai
        |
        v
    Cloudflare Tunnel  <----- outbound ----- cloudflared
                                                |
                                   (docker network: dashboard)
                                                v
                                       discovery-ui:8765
                                                |
                                                v
                                              S3 bucket

Required scopes on the API token (create one at
https://dash.cloudflare.com/profile/api-tokens):
    - Account ▸ Cloudflare Tunnel ▸ Edit
    - Account ▸ Account Settings ▸ Read    (to look up the account id)
    - Zone    ▸ Zone ▸ Read                 (to look up the zone id)
    - Zone    ▸ DNS ▸ Edit                  (to write the CNAME)

Usage:
    export CLOUDFLARE_API_TOKEN=...
    python scripts/setup_cloudflare_dashboard.py \
        --hostname dashboard.teutonic.ai \
        --service  http://discovery-ui:8765 \
        --tunnel-name teuton-dashboard

Convenient via Doppler (auto-fills CLOUDFLARE_API_TOKEN if available):
    doppler run --project arbos --config dev -- \
        python scripts/setup_cloudflare_dashboard.py \
            --hostname dashboard.teutonic.ai
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


CF_API_BASE = "https://api.cloudflare.com/client/v4"


def fail(msg: str) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def cf_request(
    method: str,
    path: str,
    *,
    token: str,
    body: dict | None = None,
    query: dict | None = None,
) -> dict:
    """Call the Cloudflare API and return the parsed `result` payload."""
    url = CF_API_BASE + path
    if query:
        url = url + "?" + urllib.parse.urlencode(query)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"raw": str(e)}
        fail(f"Cloudflare API {method} {path} -> {e.code}: {json.dumps(err)}")
    if not payload.get("success", False):
        fail(f"Cloudflare API {method} {path} returned success=false: {json.dumps(payload)}")
    return payload


def find_account_id(*, token: str, override: str | None) -> str:
    if override:
        return override
    payload = cf_request("GET", "/accounts", token=token, query={"per_page": 50})
    accounts = payload.get("result") or []
    if not accounts:
        fail("API token has no readable accounts; add the 'Account Settings: Read' scope.")
    if len(accounts) == 1:
        return accounts[0]["id"]
    names = ", ".join(f"{a.get('name')!r} ({a['id']})" for a in accounts)
    fail(f"multiple accounts visible to this token, pass --account-id explicitly. Visible: {names}")
    return ""


def find_zone(*, token: str, hostname: str, zone_override: str | None) -> tuple[str, str]:
    """Return ``(zone_id, zone_name)`` for the apex domain of ``hostname``."""
    candidates: list[str] = []
    parts = hostname.split(".")
    if zone_override:
        candidates.append(zone_override)
    for i in range(len(parts) - 1):
        candidates.append(".".join(parts[i:]))
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        payload = cf_request("GET", "/zones", token=token, query={"name": cand, "per_page": 1})
        zones = payload.get("result") or []
        if zones:
            return zones[0]["id"], zones[0]["name"]
    fail(
        f"could not find a Cloudflare zone for {hostname!r}. "
        "Add the domain to Cloudflare and grant the token Zone:Read on it."
    )
    return "", ""


def find_tunnel(*, token: str, account_id: str, name: str) -> dict | None:
    payload = cf_request(
        "GET",
        f"/accounts/{account_id}/cfd_tunnel",
        token=token,
        query={"name": name, "is_deleted": "false", "per_page": 50},
    )
    tunnels = payload.get("result") or []
    for t in tunnels:
        if t.get("name") == name and not t.get("deleted_at"):
            return t
    return None


def create_tunnel(*, token: str, account_id: str, name: str) -> dict:
    tunnel_secret = secrets.token_bytes(32).hex()
    payload = cf_request(
        "POST",
        f"/accounts/{account_id}/cfd_tunnel",
        token=token,
        body={
            "name": name,
            "tunnel_secret": tunnel_secret,
            "config_src": "cloudflare",
        },
    )
    return payload["result"]


def get_tunnel_token(*, token: str, account_id: str, tunnel_id: str) -> str:
    payload = cf_request(
        "GET",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/token",
        token=token,
    )
    raw = payload.get("result")
    if not isinstance(raw, str) or not raw:
        fail("Cloudflare did not return a tunnel token; check the API token scopes.")
    return raw  # type: ignore[return-value]


def put_tunnel_config(
    *,
    token: str,
    account_id: str,
    tunnel_id: str,
    hostname: str,
    service: str,
) -> None:
    cf_request(
        "PUT",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations",
        token=token,
        body={
            "config": {
                "ingress": [
                    {
                        "hostname": hostname,
                        "service": service,
                        "originRequest": {
                            "noTLSVerify": True,
                            "connectTimeout": "30s",
                        },
                    },
                    {"service": "http_status:404"},
                ],
            }
        },
    )


def upsert_dns(
    *,
    token: str,
    zone_id: str,
    name: str,
    target: str,
    proxied: bool,
) -> None:
    existing = cf_request(
        "GET",
        f"/zones/{zone_id}/dns_records",
        token=token,
        query={"name": name, "type": "CNAME", "per_page": 1},
    )
    records = existing.get("result") or []
    body = {
        "type": "CNAME",
        "name": name,
        "content": target,
        "ttl": 1,
        "proxied": proxied,
        "comment": "teuton dashboard tunnel",
    }
    if records:
        rec_id = records[0]["id"]
        cf_request("PUT", f"/zones/{zone_id}/dns_records/{rec_id}", token=token, body=body)
    else:
        cf_request("POST", f"/zones/{zone_id}/dns_records", token=token, body=body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hostname", default=os.environ.get("TEUTON_DASHBOARD_HOSTNAME", "dashboard.teutonic.ai"),
                   help="Public hostname to publish (default dashboard.teutonic.ai)")
    p.add_argument("--service", default=os.environ.get("TEUTON_DASHBOARD_SERVICE", "http://discovery-ui:8765"),
                   help="Internal service URL the tunnel points at (default http://discovery-ui:8765)")
    p.add_argument("--tunnel-name", default=os.environ.get("TEUTON_DASHBOARD_TUNNEL_NAME", "teuton-dashboard"),
                   help="Cloudflare Tunnel name (default teuton-dashboard)")
    p.add_argument("--account-id", default=os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
                   help="Cloudflare account id (auto-detected if the token sees exactly one)")
    p.add_argument("--zone", default=os.environ.get("CLOUDFLARE_ZONE"),
                   help="Force a specific zone name (otherwise inferred from --hostname)")
    p.add_argument("--no-proxy", action="store_true",
                   help="Create a grey-cloud DNS record instead of orange-cloud (defaults to proxied)")
    p.add_argument("--token-out", default=None,
                   help="Optional file to write the tunnel token to (chmod 600). Default: stdout only.")
    p.add_argument("--api-token", default=os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CF_API_TOKEN"),
                   help="Cloudflare API token (default $CLOUDFLARE_API_TOKEN)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_token:
        fail("CLOUDFLARE_API_TOKEN is empty. Set it in env or pass --api-token.")

    print(f"[cf] hostname    : {args.hostname}")
    print(f"[cf] tunnel name : {args.tunnel_name}")
    print(f"[cf] service     : {args.service}")

    account_id = find_account_id(token=args.api_token, override=args.account_id)
    print(f"[cf] account id  : {account_id}")

    zone_id, zone_name = find_zone(
        token=args.api_token,
        hostname=args.hostname,
        zone_override=args.zone,
    )
    print(f"[cf] zone        : {zone_name} ({zone_id})")

    tunnel = find_tunnel(token=args.api_token, account_id=account_id, name=args.tunnel_name)
    if tunnel is None:
        print(f"[cf] creating tunnel {args.tunnel_name!r}")
        tunnel = create_tunnel(token=args.api_token, account_id=account_id, name=args.tunnel_name)
    else:
        print(f"[cf] reusing tunnel {tunnel['id']}")
    tunnel_id = tunnel["id"]

    print(f"[cf] writing ingress {args.hostname} -> {args.service}")
    put_tunnel_config(
        token=args.api_token,
        account_id=account_id,
        tunnel_id=tunnel_id,
        hostname=args.hostname,
        service=args.service,
    )

    cname_target = f"{tunnel_id}.cfargotunnel.com"
    print(f"[cf] upserting DNS {args.hostname} CNAME -> {cname_target} (proxied={not args.no_proxy})")
    upsert_dns(
        token=args.api_token,
        zone_id=zone_id,
        name=args.hostname,
        target=cname_target,
        proxied=not args.no_proxy,
    )

    tunnel_token = get_tunnel_token(token=args.api_token, account_id=account_id, tunnel_id=tunnel_id)
    if args.token_out:
        with open(args.token_out, "w", encoding="utf-8") as fh:
            fh.write(tunnel_token + "\n")
        os.chmod(args.token_out, 0o600)
        print(f"[cf] token written to {args.token_out} (chmod 600)")

    print()
    print("=" * 70)
    print(f"  Public URL : https://{args.hostname}")
    print(f"  Tunnel id  : {tunnel_id}")
    print("  Drop the following into the host's /root/teuton/.env, then run")
    print("  `docker compose -f docker/compose.dashboard.yml up -d`:")
    print()
    print(f"  TEUTON_DASHBOARD_TUNNEL_TOKEN={tunnel_token}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
