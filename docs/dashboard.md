# Public Dashboard

The Teuton discovery UI (`teuton-v3 discovery-ui`) ships with a Docker stack
that exposes it on a public hostname through a Cloudflare Tunnel — no
inbound port has to be opened on the host.

Default deployment:

| Resource          | Value                                |
| ----------------- | ------------------------------------ |
| Public URL        | `https://dashboard.teutonic.ai`      |
| Backend           | one container running the Python UI  |
| Edge              | Cloudflare Tunnel (orange-cloud)     |
| Inbound on host   | none (cloudflared dials out to CF)   |

```text
       internet                cloudflare edge                 docker host
       --------                ---------------                 -----------

     dashboard.teutonic.ai
              |                              outbound only
              v                                   ^
       Cloudflare Tunnel  <----- WSS tunnel ----- cloudflared
                                                       |
                                            (network: teuton-dashboard)
                                                       |
                                                       v
                                             discovery-ui :8765
                                                       |
                                                       v
                                                    S3 bucket
```

## Files

- `docker/compose.dashboard.yml` — `discovery-ui` + `cloudflared` +
  Watchtower. The host only ever talks to S3 (for state) and Cloudflare
  (for the tunnel).
- `scripts/setup_cloudflare_dashboard.py` — idempotent Cloudflare API
  driver that creates/reuses a tunnel, writes the public hostname route,
  upserts the DNS CNAME, and prints the tunnel token.
- `scripts/deploy_dashboard.sh` — wraps the Cloudflare setup, then SSHes
  to the host, drops `/root/teuton/.env`, scps the compose file, logs in
  to Docker Hub, and brings the stack up.

## Prerequisites

A Cloudflare API token with these scopes (create at
<https://dash.cloudflare.com/profile/api-tokens>):

- **Account ▸ Cloudflare Tunnel ▸ Edit**
- **Account ▸ Account Settings ▸ Read** (so the script can resolve the
  account id)
- **Zone ▸ Zone ▸ Read** on the `teutonic.ai` zone
- **Zone ▸ DNS ▸ Edit** on the `teutonic.ai` zone

Add the token to Doppler:

```bash
doppler secrets set CLOUDFLARE_API_TOKEN=... --project arbos --config dev
```

The host you deploy to needs:

- Docker (≥ 24) — no NVIDIA driver required, the dashboard is CPU-only.
- Outbound HTTPS to `*.cloudflare.com` and `cloudflared`'s tunnel
  endpoints (open by default on every Lium pod we use).
- A Docker Hub login (the host pulls `${DOCKER_USER}/teuton:miner`,
  which already contains `teuton-v3`).

## One-shot Deploy

From this repo, with the venv active:

```bash
source .venv/bin/activate

doppler run --project arbos --config dev -- \
    bash scripts/deploy_dashboard.sh \
        --host root@<host-ip> --port <ssh-port> \
        --hostname dashboard.teutonic.ai
```

The script will:

1. Call the Cloudflare API to create (or reuse) a tunnel named
   `teuton-dashboard`, route `dashboard.teutonic.ai → http://discovery-ui:8765`,
   and upsert the proxied CNAME on the `teutonic.ai` zone.
2. Capture the tunnel token, SSH to the host, write `/root/teuton/.env`
   with the token + bucket creds, scp `docker/compose.dashboard.yml`,
   `docker login`, then `docker compose pull && up -d`.
3. Print the public URL and `docker logs` follow commands.

The first launch usually goes live in under 60 seconds (DNS propagates
through the Cloudflare edge and the tunnel is registered on connect).

## Step-by-step (manual)

If you'd rather drive this by hand:

```bash
# 1. Provision Cloudflare side only and capture the token.
doppler run --project arbos --config dev -- \
    python scripts/setup_cloudflare_dashboard.py \
        --hostname dashboard.teutonic.ai \
        --tunnel-name teuton-dashboard

# Copy the printed `TEUTON_DASHBOARD_TUNNEL_TOKEN=...` line.

# 2. On the host, drop /root/teuton/.env (chmod 600):
cat > /root/teuton/.env <<'EOF'
DOCKER_USER=...
S3_BUCKET=...
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
TEUTON_NETUID=3
TEUTON_DASHBOARD_TUNNEL_TOKEN=<token from step 1>
EOF
chmod 600 /root/teuton/.env

# 3. Push the compose file and bring it up.
scp docker/compose.dashboard.yml root@<host>:/root/teuton/compose.yml
ssh root@<host> 'cd /root/teuton && docker compose pull && docker compose up -d'
```

## Verifying

```bash
# 1. Tunnel handshake (look for "Registered tunnel connection ...")
ssh root@<host> docker logs --tail=200 teuton-dashboard-tunnel

# 2. UI process (look for "[discovery-ui] serving http://0.0.0.0:8765")
ssh root@<host> docker logs --tail=200 teuton-dashboard-ui

# 3. Edge view from anywhere on the internet
curl -I https://dashboard.teutonic.ai/
# HTTP/2 200, server: cloudflare

# 4. Live snapshot (same data the page polls)
curl https://dashboard.teutonic.ai/api/snapshot | jq '.meta'
```

## Operations

- **Refresh cadence** — change `TEUTON_DASHBOARD_REFRESH_SEC` /
  `TEUTON_DASHBOARD_CACHE_SEC` in `/root/teuton/.env` and `docker compose
  up -d` to apply. Defaults are 3 s page refresh on a 1.5 s server-side
  cache, which keeps the bucket-list cost negligible while still feeling
  live.
- **Run filter** — by default the UI shows the latest run found in the
  bucket. To pin to a specific run, append `--run-id ...` to the
  `discovery-ui` command in `compose.dashboard.yml` and redeploy.
- **Restart the tunnel** — `docker restart teuton-dashboard-tunnel`.
  Cloudflare keeps the tunnel id stable across restarts, so the public
  URL doesn't change.
- **Rotate the tunnel token** — re-run `setup_cloudflare_dashboard.py`,
  then update `TEUTON_DASHBOARD_TUNNEL_TOKEN` and `docker compose up -d`.
- **Take it down** — `docker compose down` on the host. The Cloudflare
  Tunnel stays registered (in case you want to bring it back); to fully
  delete it, remove the tunnel from the Cloudflare Zero Trust dashboard.

## Optional: gate it behind Cloudflare Access

The current setup is public read-only. If you want it limited to a list
of allowed identities, add a Cloudflare Access self-hosted application:

1. Cloudflare Dashboard ▸ Zero Trust ▸ Access ▸ Applications ▸ Add
   application ▸ self-hosted.
2. Application domain: `dashboard.teutonic.ai`.
3. Add an Access policy ("Email is *@example.com" or whichever rule
   suits the team).

cloudflared in the compose stack stays untouched — the gate sits on the
edge before the tunnel.

## Troubleshooting

- **`Unable to reach the origin service`** in the cloudflared logs — the
  `discovery-ui` container is not on the same `teuton-dashboard` docker
  network, or it crashed. `docker compose ps` and `docker logs
  teuton-dashboard-ui`.
- **`522` from Cloudflare** — the origin (`discovery-ui`) is up but slow.
  Bump `TEUTON_DASHBOARD_CACHE_SEC` so the snapshot is served from cache
  rather than a fresh bucket scan on every poll.
- **`530 1014 CNAME flattening to ...cfargotunnel.com`** — the DNS record
  exists but isn't proxied. Re-run `setup_cloudflare_dashboard.py` (or
  flip the orange cloud on in the Cloudflare DNS panel).
- **`access denied` from the API setup** — the token is missing one of
  the scopes above; the easiest fix is to recreate it with the **Edit
  Cloudflare Tunnel** template plus a Zone scope for `teutonic.ai`.
