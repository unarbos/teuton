# V2 Migration Inventory

Classification for `locus_v2` assets:

## Ported

- `locus/*.py`, `locus/tasks/*.py`, and `locus/data/*.py` are preserved under `locus_legacy_v2`.
- Public task wrappers are available under `locus_tasks` for legacy task names.
- Maintained benchmark and fleet utilities are under `bench/`.

## Curated

- `SPEC.md`, `scaling_check.md`, `MEGAFLEET_RESULTS.md`, and `PIPE_TRAIN_RESULTS.md` are condensed into `docs/v2-architecture.md`, `docs/scaling-lessons.md`, and `docs/fleet-operations.md`.

## Superseded

- Unsigned v2 protocol records are superseded for subnet operation by `locus_core.protocol`, `locus_core.signatures`, and v3 artifact crypto.
- The v2 package name `locus` is superseded by v3 package splits and `locus_legacy_v2` for preserved legacy execution.

## Discarded

- `.pytest_cache`, `locus.egg-info`, generated host env files, and raw one-off experiment logs are not carried as maintained source.
