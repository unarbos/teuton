# V2 Architecture Preserved In V3

V2 established the bucket-native tensor language that v3 keeps: content-addressed IR graphs, typed tensor blobs, pure worker interpretation, and bucket-backed recovery. V3 adds signed manifests, miner/validator identities, grants, artifact crypto, and subnet scoring.

## Preserved In Code

- `teuton_core.ir`, `teuton_runtime.eval`, `teuton_runtime.tensor_io`, and `teuton_runtime.storage` carry the core IR/evaluator/wire/storage behavior forward.
- `teuton_legacy_v2` preserves the full v2 runtime, scheduler, worker, validator, data utilities, and task catalog under a v3-owned package name.
- `teuton_tasks` exposes wrappers for the v2 task catalog while keeping the v3 `mlp` smoke path as the default subnet task.

## Protocol Difference

V2 manifests used `assigned_to` worker IDs and unsigned receipts. V3 manifests bind work to hotkeys/workers, can carry artifact crypto policies, and produce signed receipts/verdicts. Legacy v2 code is retained for reproducibility and comparison, not as the security model for subnet operation.
