# SDK Usage

The v3 Python packages are designed so tests, fleet scripts, and future subnet
services can use the same core classes as the CLI.

## Package Boundaries

```text
locus_core          protocol, paths, signatures, v2 IR import surface
locus_runtime       storage, tensor codec, evaluator, job executor
locus_orchestrator  run managers and schedulers
locus_miner         miner neuron facade and worker supervisor
locus_validator     replay verifier, scoring, Bittensor adapter
locus_tasks         task plugins
```

## Local Round Run

```python
import threading
import tempfile

from locus_miner.neuron import MinerNeuron, MinerNeuronConfig
from locus_orchestrator.run_manager import RunConfig, RunManager
from locus_runtime.storage import open_local_bucket
from locus_validator.neuron import ValidatorNeuron, ValidatorNeuronConfig

root = tempfile.mkdtemp(prefix="locus-v3-")
bucket = open_local_bucket(root, "sdk")
run_id = "sdk-round"

miners = [
    MinerNeuron(
        bucket=bucket,
        config=MinerNeuronConfig(
            netuid=0,
            run_id=run_id,
            hotkey_ss58=f"miner{i}",
            devices=["cpu"],
        ),
    )
    for i in range(4)
]

threads = [threading.Thread(target=m.loop, daemon=True) for m in miners]
for thread in threads:
    thread.start()

manager = RunManager(
    bucket=bucket,
    config=RunConfig(netuid=0, run_id=run_id, task="mlp", max_steps=1),
)
manager.run_loop(timeout_sec=60)

for miner in miners:
    miner.stop()

validator = ValidatorNeuron(
    bucket=bucket,
    config=ValidatorNeuronConfig(
        netuid=0,
        run_id=run_id,
        validator_hotkey="validator0",
        sample_rate=1.0,
        dry_run_weights=True,
    ),
)

result = validator.run_once(max_receipts=10_000, publish_weights=True)
print(result["scores"])
```

## Local Adversarial Run

```python
bad_miner = MinerNeuron(
    bucket=bucket,
    config=MinerNeuronConfig(
        netuid=0,
        run_id=run_id,
        hotkey_ss58="bad-miner",
        devices=["cpu"],
        fault_mode="partial_corrupt",
        fault_rate=1.0,
    ),
)
```

The validator should assign this hotkey a score of `0.0` after replay.

## Shared S3 Bucket

```python
import os

from locus_runtime.storage import S3Bucket

bucket = S3Bucket(
    bucket=os.environ["S3_BUCKET"],
    region=os.environ.get("S3_REGION", "us-east-1"),
    access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
    secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
)
```

Use the same `bucket` object with `RunManager`, `MinerNeuron`, and
`ValidatorNeuron`.

## Streaming GPT Bridge

`gpt_pipe` currently uses v2 graph builders and v2 internal tensor paths, but
v3 emits signed manifests and collects v3 receipts/verdicts.

```python
from locus_orchestrator.streaming import StreamingRunConfig, StreamingRunManager

manager = StreamingRunManager(
    bucket=bucket,
    config=StreamingRunConfig(
        netuid=0,
        run_id="sdk-stream",
        task="gpt_pipe",
        max_epochs=1,
    ),
)
manager.run_loop(timeout_sec=180)
```

For small smoke tests, you can patch the imported `locus.tasks.gpt_pipe`
globals before creating the manager, then run tiny stage/microbatch graphs.

## Signatures

Dev mode uses HMAC secrets:

```python
RunConfig(..., owner_secret="owner-dev-secret")
MinerNeuronConfig(..., miner_secret="miner-dev-secret")
ValidatorNeuronConfig(
    ...,
    owner_secret="owner-dev-secret",
    miner_secret="miner-dev-secret",
    validator_secret="validator-dev-secret",
)
```

The validator rejects manifests or receipts whose signatures do not verify.
Production subnet mode should bind these checks to Bittensor wallet signatures.

## Artifact Crypto Policies

Every `ArtifactRef` can carry an optional `ArtifactCryptoPolicy`. If no policy
is set, bytes are stored exactly as before.

Signed output:

```python
from locus_core.protocol import ArtifactCryptoPolicy, ArtifactRef, CryptoMode

delta = ArtifactRef(
    name="delta",
    uri=delta_uri,
    crypto=ArtifactCryptoPolicy(
        mode=CryptoMode.SIGNED.value,
        required_signer="assigned_hotkey",
    ),
)
```

Encrypted output in dev mode:

```python
private_update = ArtifactRef(
    name="private_update",
    uri=update_uri,
    crypto=ArtifactCryptoPolicy(
        mode=CryptoMode.ENCRYPTED.value,
        key_id="dev-shared-key",
    ),
)
```

drand timelock output:

```python
timelocked = ArtifactRef(
    name="future_update",
    uri=update_uri,
    crypto=ArtifactCryptoPolicy(
        mode=CryptoMode.DRAND_TIMELOCK.value,
        drand_round=123456,
    ),
)
```

The executor wraps signed/encrypted/timelocked outputs in an `ArtifactEnvelope`.
The validator verifies signatures, decrypts where possible, and reports
timelocked artifacts as inconclusive until their reveal round is available.

For CLI runs, set an owner default:

```bash
locus-v3 orchestrator \
  --run-id RUN_ID \
  --task mlp \
  --crypto signed \
  --required-signer assigned_hotkey
```

or:

```bash
locus-v3 orchestrator \
  --run-id RUN_ID \
  --task mlp \
  --crypto drand-timelock \
  --drand-round 123456
```

## Encrypted Assignment Grants

For production-style bucket access, keep the manifest public and put sensitive
bearer URLs in an encrypted assignment grant.

```bash
locus-v3 orchestrator \
  --run-id RUN_ID \
  --task mlp \
  --grant-mode presigned \
  --grant-ttl-sec 600
```

The orchestrator writes:

```text
v3/netuid=<N>/jobs/<run_id>/<job_id>/manifest.json
v3/netuid=<N>/assignments/<run_id>/<job_id>/hotkey=<H>.json
```

The manifest contains canonical `s3://` input/output URIs. The assignment file
contains encrypted GET/PUT grants for those exact URIs plus the receipt URI.

Local tests can use the same path without S3:

```bash
locus-v3 orchestrator --run-id RUN_ID --grant-mode local
locus-v3 miner --run-id RUN_ID --hotkey miner0 --grant-mode local
```

## Lifecycle

Use the v3 lifecycle helper to clean all prefixes owned by a run:

```python
from locus_runtime.lifecycle import wipe_run

deleted = wipe_run(bucket, netuid=0, run_id="sdk-round")
print(deleted)
```
