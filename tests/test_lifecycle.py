from __future__ import annotations

from teuton_core import paths
from teuton_runtime.lifecycle import wipe_run


def test_wipe_run_removes_only_owned_v3_prefixes(local_bucket) -> None:
    run_id = "wipe-me"
    other_run = "keep-me"
    owned = [
        paths.run_root(0, run_id) + "/state.json",
        paths.jobs_prefix(0, run_id) + "index.json",
        f"{paths.root(0)}/artifacts/{run_id}/job/out.bin",
        paths.receipts_prefix(0, run_id) + "hotkey=m/job/attempt=0.json",
        paths.verdicts_prefix(0, run_id) + "validator=v/receipt.json",
    ]
    keep = [
        paths.run_root(0, other_run) + "/state.json",
        paths.jobs_prefix(0, other_run) + "index.json",
    ]
    for key in owned + keep:
        local_bucket.put(local_bucket.uri_for_key(key), b"x")

    deleted = wipe_run(local_bucket, netuid=0, run_id=run_id)

    assert deleted == len(owned)
    assert all(not local_bucket.exists(local_bucket.uri_for_key(key)) for key in owned)
    assert all(local_bucket.exists(local_bucket.uri_for_key(key)) for key in keep)
