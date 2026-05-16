# Fleet Operations Notes

V2 fleet tooling has been curated into `teuton_v3/bench`. The maintained path is:

- `bench/lium_fleet.py` for session-safe Lium rental management. It requires `LIUM_API_KEY` from Doppler or the environment and embeds no key.
- `bench/legacy_dist.py` for running the preserved v2-style runtime against an S3 bucket while comparing behavior during migration.
- `bench/legacy_fleet_pool.sh` and `bench/legacy_pool.sh` for old pool launch flows, rewritten to call the v3-owned legacy package.
- `bench/dashboard.py`, `bench/util_smoke.py`, and `bench/probe_s3.py` for utilization and S3 diagnostics.

Host files and one-off experiment runners were intentionally not copied as maintained assets. Generate host files from the active Lium session instead of committing them.
