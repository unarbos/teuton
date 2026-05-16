# Mining

This guide walks a third-party operator through onboarding a GPU host as a
Teuton miner: generating a Bittensor wallet and an ED25519 hotkey, registering
that hotkey on subnet 3 (Finney), and starting the miner so the orchestrator
can hand it work.

The fleet runs on:

- **network**: `finney` (Bittensor mainnet)
- **netuid**: `3`
- **hotkey type**: native ED25519 (`crypto_type=0`) — required for encrypted
  assignment grants
- **runtime**: a single `teuton:miner` Docker image; one container per GPU

## What A Miner Does

A miner runs one worker per GPU under a single hotkey. Each worker:

1. Polls the shared S3 bucket for new job manifests addressed to its hotkey.
2. Decrypts the assignment grant with its hotkey's ED25519 → X25519 key, which
   yields presigned S3 GET/PUT URLs for the inputs/outputs of that job only.
3. Executes the manifest on its assigned GPU.
4. Writes the output tensors and a signed receipt back through those URLs.
5. Emits a heartbeat with its run id, capabilities (GPU class, count, RTT), and
   identity so the orchestrator can keep scheduling it.

The validator later replays a sampled subset of those receipts and pushes
weights on chain. Honest, fast workers earn weight; missing, wrong, or
corrupted receipts get scored down.

## Prerequisites

Hardware:

- One or more NVIDIA GPUs with CUDA 12-capable drivers (verified with
  `nvidia-smi`). A6000 / RTX3090 / RTX4090 / A100 / H100 / B200 are all in use.
- Reasonable upload bandwidth (≥150 Mbps recommended) and a public-egress
  network — workers stream tensors to/from S3.

Host software:

- Linux (Ubuntu 22.04 is what the official image is built on).
- Docker (≥ 24) with the NVIDIA Container Toolkit so the daemon can expose
  GPUs to containers (`docker run --gpus all ...`).
- `git`, `curl`, `python3.11+`. `uv` only if you want a non-Docker setup.

Bittensor:

- A funded coldkey with enough TAO to cover the registration burn on netuid 3
  (the current burn is printed before any extrinsic is submitted).

Operator-supplied credentials (you will get these from the Teuton team):

- An S3 bucket name, region, and a pair of AWS keys with read/write to the
  `v3/netuid=3/...` prefix.
- The three shared HMAC secrets used by the dev signature scheme:
  `TEUTON_OWNER_SECRET`, `TEUTON_MINER_SECRET`, `TEUTON_ASSIGNMENT_SECRET`.

> Without those four credentials the miner cannot read manifests or write
> receipts. Reach out to coordinate before you generate keys so you do not
> burn TAO on a hotkey that will sit idle.

## 1. Install Teuton And Bittensor

Clone the repo:

```bash
git clone https://github.com/unarbos/teuton.git
cd teuton
```

Install with `uv` (this builds the patched `bittensor-wallet` from source, so
the first run takes a few minutes):

```bash
uv sync --all-extras
source .venv/bin/activate
```

Verify the toolchain:

```bash
btcli --version
teuton-v3 --help
python -c "import bittensor as bt; print(bt.__version__)"
```

If `btcli --version` fails, run `uv pip install bittensor-cli` and retry.

## 2. Create A Bittensor Coldkey

The coldkey holds funds and authorizes registration. Pick any name you like;
this guide uses `teuton_mining`.

```bash
btcli wallet new-coldkey \
    --wallet-name teuton_mining \
    --n-words 24
```

You will be prompted for a password and shown a 24-word mnemonic. **Write
the mnemonic down offline.** If you lose it, the funds are unrecoverable.

The coldkey now lives at `~/.bittensor/wallets/teuton_mining/coldkey`.

Fund the coldkey's SS58 address with enough TAO to cover the registration
burn for every hotkey you plan to register, plus a small fee buffer
(`btcli wallet overview --wallet-name teuton_mining` will print the address).

## 3. Generate ED25519 Miner Hotkeys

Teuton encrypts assignment grants to each miner's hotkey using ED25519 →
X25519. The default `btcli wallet new_hotkey` command produces an `sr25519`
hotkey, which **cannot** decrypt those grants. Use the helper script
instead — it calls `bittensor_wallet.Wallet.create_new_hotkey(crypto_type=0)`
and self-tests the resulting key with a sealed-box round trip.

For one hotkey:

```bash
python scripts/generate_ed25519_hotkey.py \
    --wallet-name teuton_mining \
    --hotkey      teuton_miner_sn3_1
```

For N hotkeys at once (also handles registration in a single pass; covered
in the next step):

```bash
./scripts/register_miners.sh --wallet teuton_mining \
    --prefix teuton_miner_sn3_ --start 1 --n 4 --dry-run
```

The generator writes:

```text
~/.bittensor/wallets/teuton_mining/hotkeys/teuton_miner_sn3_1        (private)
~/.bittensor/wallets/teuton_mining/hotkeys/teuton_miner_sn3_1pub.txt (public ss58 + cryptoType=0)
```

It prints the SS58 address and confirms `encryption_self_test: ok`. Save
that SS58 — the orchestrator addresses jobs to it.

> If the script aborts with "Installed bittensor-wallet does not expose
> native ED25519 hotkey generation", `uv sync --all-extras` did not pick up
> the patched `bittensor-wallet`. Re-run `uv sync --all-extras --reinstall`
> and confirm `pyproject.toml` still pins
> `bittensor-wallet @ git+https://github.com/latent-to/btwallet.git@feat/roman/add-ed25519-support`.

## 4. Register Hotkeys On Netuid 3 (Finney)

You can register one at a time with `btcli`:

```bash
btcli subnets register \
    --netuid 3 \
    --network finney \
    --wallet-name teuton_mining \
    --wallet-hotkey teuton_miner_sn3_1
```

`btcli` prints the recycle (burn) cost in TAO and asks for confirmation
before submitting the extrinsic.

For batch registration, use the helper script (it skips hotkeys already on
the metagraph and does an explicit cost check before spending anything):

```bash
# Confirm the plan and the total burn cost without registering:
./scripts/register_miners.sh --wallet teuton_mining \
    --prefix teuton_miner_sn3_ --start 1 --n 4 --dry-run

# Register for real (needs the coldkey password to unlock once):
TEUTON_MINING_COLDKEY_PW='your-coldkey-password' \
    ./scripts/register_miners.sh --wallet teuton_mining \
        --prefix teuton_miner_sn3_ --start 1 --n 4 --yes
```

After it finishes, verify:

```bash
btcli subnets metagraph --netuid 3 --network finney \
    | grep -i "$(cat ~/.bittensor/wallets/teuton_mining/hotkeys/teuton_miner_sn3_1pub.txt | jq -r .ss58Address)"
```

You should see your hotkey listed with a UID. Send the SS58s to the
operator so they can add them to the assignment plan.

## 5. Get Operator Credentials

From the Teuton operator you will need:

| Variable                  | Why                                                          |
| ------------------------- | ------------------------------------------------------------ |
| `S3_BUCKET`               | Bucket the orchestrator writes manifests/grants into.         |
| `S3_REGION`               | Bucket region (defaults to `us-east-1`).                      |
| `AWS_ACCESS_KEY_ID`       | Read jobs/grants, write receipts.                             |
| `AWS_SECRET_ACCESS_KEY`   | Pair for the access key.                                      |
| `TEUTON_OWNER_SECRET`      | Verify orchestrator signatures on manifests.                  |
| `TEUTON_MINER_SECRET`      | Sign your receipts.                                           |
| `TEUTON_ASSIGNMENT_SECRET` | Authenticate encrypted assignment envelopes.                  |
| `DOCKER_USER` / `DOCKER_PAT` | (Docker path) pull the prebuilt `teuton:miner` image.        |

The miner only needs read+write under `v3/netuid=3/`. Operators usually
hand out a scoped IAM key — do not commit the secret anywhere.

## 6. Run The Miner (Docker, recommended)

The repo ships docker compose stacks for single-GPU and multi-GPU hosts.
Watchtower polls Docker Hub every 60 s, so you do not need to redeploy when
the operator pushes a new image.

### Single-GPU host

Create `/root/teuton/.env` on the host:

```bash
mkdir -p /root/teuton
cat > /root/teuton/.env <<'EOF'
DOCKER_USER=<operator-supplied>
S3_BUCKET=<operator-supplied>
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=<operator-supplied>
AWS_SECRET_ACCESS_KEY=<operator-supplied>

TEUTON_NETUID=3
TEUTON_OWNER_SECRET=<operator-supplied>
TEUTON_MINER_SECRET=<operator-supplied>
TEUTON_ASSIGNMENT_SECRET=<operator-supplied>
TEUTON_ASSIGNMENT_CRYPTO=ed25519

MINER_WALLET_NAME=teuton_mining
MINER_HOTKEY_NAME=teuton_miner_sn3_1
MINER_HOTKEY_SS58=<ss58 from step 3>
MINER_DEVICES=cuda
EOF
chmod 600 /root/teuton/.env
```

Make sure your hotkey files are at
`/root/.bittensor/wallets/teuton_mining/hotkeys/<MINER_HOTKEY_NAME>{,pub.txt}`.
The compose file mounts that directory read-only into the container; the
files never leave the host.

Copy `docker/compose.miner.yml` from this repo to `/root/teuton/compose.yml`
and start the stack:

```bash
echo "$DOCKER_PAT" | docker login -u "$DOCKER_USER" --password-stdin
cd /root/teuton
docker compose pull
docker compose up -d
docker compose logs -f miner
```

You should see the miner banner print its baked `run_id`, then a stream of
`[worker] heartbeat …` lines.

### Multi-GPU host

For hosts with multiple GPUs you want to expose as independent miners
(one hotkey per GPU), use `docker/compose.multi-miner.yml` and add
`MINER_HK_SS58_<i>` / `MINER_HK_NAME_<i>` to the .env for each GPU index.
The included file declares four GPU services; copy the `miner-gpuN` block
to add more.

### Run id

The orchestrator coordinates everyone with a `run_id`. Two ways to set it:

1. **Image-baked (default).** The operator builds and pushes the image with
   `docker build --build-arg TEUTON_RUN_ID=<id>`. Watchtower picks it up and
   the entrypoint resolves `TEUTON_RUN_ID` from `TEUTON_BAKED_RUN_ID`.
2. **Per-host override.** Set `TEUTON_RUN_ID=<id>` (or `RUN_ID=<id>`) in
   `/root/teuton/.env` to pin a single host to a specific run.

If neither is set, the miner refuses to start with:

```text
error: --run-id is empty. Provide --run-id, set TEUTON_RUN_ID/RUN_ID
in the environment, or rebuild the image with --build-arg TEUTON_RUN_ID=...
```

## 6 (alt). Run The Miner Without Docker

For development or hosts where Docker isn't an option:

```bash
source .venv/bin/activate
export S3_BUCKET=...
export S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export TEUTON_OWNER_SECRET=...
export TEUTON_MINER_SECRET=...
export TEUTON_ASSIGNMENT_SECRET=...

teuton-v3 miner \
    --netuid 3 \
    --run-id   "$TEUTON_RUN_ID" \
    --hotkey   "<miner ss58>" \
    --hotkey-name teuton_miner_sn3_1 \
    --wallet-name teuton_mining \
    --wallet-path "$HOME/.bittensor/wallets" \
    --devices cuda \
    --grant-mode presigned \
    --assignment-crypto ed25519 \
    --discovery-backend bucket \
    --poll-interval 0.5
```

For a host with 4 GPUs and 4 separate hotkeys, run four processes with
`--devices cuda:0`, `cuda:1`, etc. (and a different `--hotkey/--hotkey-name`
per process). Use `--devices cuda:0,cuda:1,cuda:2,cuda:3` only if you want
**one** hotkey to span all four GPUs as a single multi-GPU worker.

## 7. Verify The Miner Is Live

Three signals to check, in order:

1. **Container health (Docker path):**

   ```bash
   docker compose ps
   docker compose logs --tail=200 miner
   ```

2. **Heartbeat in the bucket.** Each worker writes one heartbeat object per
   poll cycle:

   ```text
   v3/netuid=3/miners/<hotkey-ss58>/workers/<worker-id>/heartbeat.json
   ```

   The public **dashboard** at <https://dashboard.teutonic.ai> is the
   easiest way to confirm — your hotkey should appear in the *Miners*
   table with a recent `last seen` once the heartbeat lands. (See
   [docs/dashboard.md](dashboard.md) for how it's hosted.)

   You can also run the same UI locally:

   ```bash
   teuton-v3 discovery-ui --netuid 3 --port 8765 --open-browser
   ```

3. **Receipts.** Once the orchestrator schedules a job to your hotkey, you
   will see new receipts under:

   ```text
   v3/netuid=3/receipts/<run_id>/hotkey=<H>/<job_id>/attempt=<A>.json
   ```

   They include the manifest hash, IO digests, timing telemetry, and your
   miner signature. The validator replays these before crediting work.

## Grant Modes

The `--grant-mode` flag controls how the miner authenticates against S3.
Production miners always use `presigned`:

- `direct` — process already holds bucket credentials and uses them for
  every read/write (dev only).
- `local` — encrypted grants resolve to local/direct ops; useful for
  `teuton-v3 local-smoke` runs that exercise the grant code path without S3.
- `presigned` — encrypted grants embed presigned S3 GET/PUT URLs scoped to
  the exact input/output URIs of one job. The miner decrypts the envelope,
  verifies it matches the public manifest, and uses those URLs only.

Even in `presigned` mode the miner still needs basic bucket access to
*list and read* the manifest/grant objects themselves. Operators typically
issue a scoped IAM key with read on `v3/netuid=3/jobs/`,
`v3/netuid=3/assignments/`, and read+write on `v3/netuid=3/receipts/` and
`v3/netuid=3/miners/` (heartbeats).

## Heartbeats

Each worker writes a heartbeat under:

```text
v3/netuid=3/miners/<hotkey>/workers/<worker_id>/heartbeat.json
```

Heartbeats include `run_id`, so an orchestrator only schedules workers for
the current run and treats anything older than `--discovery-heartbeat-ttl-sec`
(default `30`) as stale.

## Receipts

After each successful job a worker writes a signed receipt under:

```text
v3/netuid=3/receipts/<run_id>/hotkey=<H>/<job_id>/attempt=<A>.json
```

Receipts include the manifest hash, input/output digests, worker identity,
timing/byte telemetry, and the miner signature. The validator replays the
job and only credits the receipt if its outputs match.

## Adversarial Test Modes (Dev Only)

The runtime supports a handful of fault-injection modes that intentionally
fail validation. Do **not** run these against a real bucket — your hotkey
will get scored to zero.

```bash
teuton-v3 miner \
    --run-id   RUN_ID \
    --hotkey   BAD_MINER \
    --devices  cuda \
    --fault-mode partial_corrupt \
    --fault-rate 1.0
```

Supported modes: `partial_corrupt`, `wrong_output`, `skip_compute`.

## Troubleshooting

- **`error: --run-id is empty`** — the entrypoint could not resolve a run id.
  Set `TEUTON_RUN_ID=` (or `RUN_ID=`) in `/root/teuton/.env`, or pull a fresh
  image whose `TEUTON_BAKED_RUN_ID` is populated.
- **`AssignmentDecryptError` / `cannot decrypt grant`** — the hotkey on disk
  is not ED25519. Re-generate with `scripts/generate_ed25519_hotkey.py` and
  re-register. Confirm `pub.txt` shows `"cryptoType": 0`.
- **No heartbeats appearing in the bucket** — usually a credentials problem.
  Re-check the four AWS variables and confirm the IAM principal can `PutObject`
  under `v3/netuid=3/miners/<your-ss58>/`.
- **Container exits with `nvidia-container-cli: requirement error`** —
  install the NVIDIA Container Toolkit and restart Docker
  (`sudo systemctl restart docker`).
- **Watchtower never updates** — confirm the Docker Hub credentials in
  `/root/.docker/config.json` are valid and the container has the
  `com.centurylinklabs.watchtower.enable=true` label
  (the included compose files set it).
- **Hotkey not on metagraph after `register`** — wait one block and
  `btcli subnets metagraph --netuid 3 --network finney`. The
  `register_miners.sh` helper retries up to four times automatically.
