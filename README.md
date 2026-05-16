# Teuton v3

Teuton v3 is the subnet-ready version of Teuton: a bucket-native distributed
training runtime with explicit roles for an owner orchestrator, miners, and a
validator. The mainnet deployment lives on Bittensor **netuid 3 (Finney)**.

The current v3 implementation supports:

- signed v3 job manifests, miner receipts, and validator verdicts
- hotkey-scoped miner workers, with one worker process per GPU
- local/no-chain smoke tests
- shared-bucket fleet runs with encrypted assignment grants
- ED25519 hotkeys with per-job presigned S3 URLs
- replay validation and compute-unit scoring
- dry-run or real Bittensor `set_weights` adapter
- round-style MLP jobs and a v3-native streaming bridge for `gpt_pipe`
- a live discovery UI for inspecting active workers and receipts

## Run A Miner

If you came here to contribute compute on netuid 3, jump straight to the
**[Mining Guide](docs/mining.md)** — it covers wallet creation, ED25519 hotkey
generation, on-chain registration, and starting the prebuilt
`teuton:miner` Docker image.

## Dashboard

Live fleet view (active workers, runs, recent receipts) is hosted at
**<https://dashboard.teutonic.ai>** — anyone can watch the network in real
time.

To run a local copy against the same bucket:

```bash
teuton-v3 discovery-ui --netuid 3 --port 8765 --open-browser
```

The public site is the same UI, fronted by a Cloudflare Tunnel. See the
[Dashboard Deployment](docs/dashboard.md) doc for the compose stack and the
one-shot `scripts/deploy_dashboard.sh` provisioner.

## Install

From this directory:

```bash
uv sync
source .venv/bin/activate
```

For the full install, including Bittensor, drand timelock, Lium, dataset
helpers, and tests:

```bash
uv sync --all-extras
source .venv/bin/activate
```

For only subnet dependencies:

```bash
uv sync --extra subnet
source .venv/bin/activate
```

## Quick Smoke

Honest local run:

```bash
teuton-v3 local-smoke --steps 1 --miners 4
```

Adversarial local run:

```bash
teuton-v3 local-smoke \
  --steps 1 \
  --miners 4 \
  --bad-miner-index 0 \
  --fault-mode partial_corrupt \
  --sample-rate 1.0
```

Expected behavior: the honest miner set receives positive dry-run weights; the
corrupt miner receives score and weight `0.0`.

## Main Roles

- **Orchestrator**: owner-operated process that writes signed job manifests to
  the bucket.
- **Miner**: hotkey-bound process that supervises one or more GPU workers,
  executes assigned jobs, and writes signed receipts.
- **Validator**: owner-operated process that samples receipts, replays jobs,
  writes signed verdicts, computes scores, and optionally calls Bittensor
  `set_weights`.

## Shared Bucket Mode

The CLI reads bucket credentials from flags or environment variables:

```bash
export S3_BUCKET=...
export S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Secrets should come from Doppler or your environment. Do not commit `.env`
files.

## Docker Deployment

The repo ships a single image with three role tags, plus compose stacks for
each role:

```text
$DOCKER_USER/teuton:miner      docker/compose.miner.yml
$DOCKER_USER/teuton:miner      docker/compose.multi-miner.yml   (multi-GPU host)
$DOCKER_USER/teuton:validator  docker/compose.validator.yml
$DOCKER_USER/teuton:auditor    docker/compose.auditor.yml
$DOCKER_USER/teuton:miner      docker/compose.dashboard.yml     (public dashboard via Cloudflare Tunnel)
```

Watchtower (included in each stack) polls Docker Hub every 60 s, so a
`scripts/build_push.sh --run-id <id>` from the operator side rolls the whole
fleet onto a new image and run id without touching any host. Miners only need
to populate `/root/teuton/.env` and the wallet hotkeys once; see the
[Mining Guide](docs/mining.md) for the full step-by-step.

## Docs

- [Mining](docs/mining.md) — generate keys, register on netuid 3, run the
  miner stack
- [Dashboard Deployment](docs/dashboard.md) — host the discovery UI on a
  public hostname via Cloudflare Tunnel
- [Running the Validator](docs/validator.md)
- [SDK Usage](docs/sdk.md)
- [V2 Architecture Preserved In V3](docs/v2-architecture.md)
- [Scaling Lessons](docs/scaling-lessons.md)
- [Fleet Operations Notes](docs/fleet-operations.md)

## Current Caveats

- The `gpt_pipe` streaming bridge is a compact v3-native pipeline smoke task,
  not the full historical GPT workload.
- Real subnet operation requires a valid Bittensor wallet, registered hotkeys,
  and validator permissions for `set_weights`.
