# Running The Validator

The validator samples miner receipts, replays the corresponding job manifests,
writes signed verdicts, computes score windows, and optionally publishes
Bittensor weights.

In the current owner-operated v1 design, the validator and orchestrator can run
on the same machine with full bucket access.

## Environment

Bucket access:

```bash
export S3_BUCKET=...
export S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Dev signature secrets:

```bash
export TEUTON_OWNER_SECRET=owner-dev-secret
export TEUTON_MINER_SECRET=miner-dev-secret
export TEUTON_VALIDATOR_SECRET=validator-dev-secret
```

All three must match the secrets used by the orchestrator, miners, and
validator during a dev run. Production subnet mode should replace this with
wallet-backed signatures.

## Dry-Run Validator

Dry-run mode computes scores and prints a weight payload without calling the
chain:

```bash
teuton-v3 validator \
  --run-id RUN_ID \
  --sample-rate 1.0 \
  --publish-weights
```

With explicit S3 flags:

```bash
teuton-v3 validator \
  --run-id RUN_ID \
  --sample-rate 1.0 \
  --publish-weights \
  --s3-bucket "$S3_BUCKET" \
  --s3-region "$S3_REGION"
```

## Real `set_weights`

Install the subnet extra:

```bash
uv sync --extra subnet
source .venv/bin/activate
```

Then run:

```bash
teuton-v3 validator \
  --run-id RUN_ID \
  --validator-hotkey VALIDATOR_HOTKEY \
  --sample-rate 1.0 \
  --publish-weights \
  --set-weights \
  --wallet-name WALLET_NAME \
  --hotkey-name HOTKEY_NAME \
  --network finney \
  --netuid NETUID
```

The adapter normalizes non-negative scores before publishing. Hotkeys that do
not map to current metagraph UIDs are dropped and reported.

## Crypto-Aware Validation

If a manifest output has an `ArtifactCryptoPolicy`, the validator enforces it:

- `signed`: verify the artifact envelope signature before comparing tensors.
- `encrypted`: decrypt the envelope before tensor decoding.
- `drand_timelock`: decrypt only after the configured drand round is available.

If a timelocked artifact is not yet revealable, the verdict is `inconclusive`
rather than `fail`. Re-run the validator after the reveal round to settle the
receipt.

## What Is Scored

The validator builds score windows from receipts and verdicts:

```text
score = pass_cu + unsampled_cu * trust_multiplier
```

If sampled failures dominate, `trust_multiplier` drops to `0.0`, which also
discounts unsampled work from that hotkey.

## Verification Flow

For each sampled receipt:

1. Load the receipt.
2. Load the referenced manifest.
3. Verify owner manifest signature.
4. Verify miner receipt signature.
5. Verify the manifest hash matches the receipt.
6. Decode inputs and check input digests.
7. Replay the graph on the validator device.
8. Compare outputs using the manifest verification policy.
9. Write a signed verdict.

Verdicts are stored under:

```text
v3/netuid=<N>/verdicts/<run_id>/validator=<V>/<receipt_id>.json
```

Score windows are stored under:

```text
v3/netuid=<N>/scores/window=run=<run_id>/scores.json
```

## Late Receipts

Receipts may arrive after the first validator pass. It is safe to run the
validator again:

```bash
teuton-v3 validator --run-id RUN_ID --sample-rate 1.0 --publish-weights
```

Already-verdict receipts are skipped; new receipts are replayed and added to
the score window.
