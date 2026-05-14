# Locus v3

Locus v3 is the subnet-ready version of Locus: a bucket-native distributed
training runtime with explicit roles for an owner orchestrator, miners, and a
validator.

The current v3 implementation supports:

- signed v3 job manifests, miner receipts, and validator verdicts
- hotkey-scoped miner workers, with one worker process per GPU
- local/no-chain smoke tests
- shared-bucket fleet runs
- replay validation and compute-unit scoring
- dry-run or real Bittensor `set_weights` adapter
- round-style MLP jobs and a v3-native streaming bridge for `gpt_pipe`

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
locus-v3 local-smoke --steps 1 --miners 4
```

Adversarial local run:

```bash
locus-v3 local-smoke \
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

## Docs

- [Mining](docs/mining.md)
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
