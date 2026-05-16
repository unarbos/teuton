# V2 Migration Inventory

Classification for `teuton_v2` assets:

## Ported

- `teuton/*.py`, `teuton/tasks/*.py`, and `teuton/data/*.py` are preserved under `teuton_legacy_v2`.
- Public task wrappers are available under `teuton_tasks` for legacy task names.
- Maintained benchmark and fleet utilities are under `bench/`.

## Curated

- `SPEC.md`, `scaling_check.md`, `MEGAFLEET_RESULTS.md`, and `PIPE_TRAIN_RESULTS.md` are condensed into `docs/v2-architecture.md`, `docs/scaling-lessons.md`, and `docs/fleet-operations.md`.

## Superseded

- Unsigned v2 protocol records are superseded for subnet operation by `teuton_core.protocol`, `teuton_core.signatures`, and v3 artifact crypto.
- The v2 package name `teuton` is superseded by v3 package splits and `teuton_legacy_v2` for preserved legacy execution.

## Discarded

- `.pytest_cache`, `teuton.egg-info`, generated host env files, and raw one-off experiment logs are not carried as maintained source.
