from __future__ import annotations

from teuton_core import paths
from teuton_core.job_index import list_job_ids


def test_list_job_ids_prefers_aggregate_index(local_bucket, run_id) -> None:
    netuid = 0
    idx = paths.job_index_key(netuid, run_id)
    prefix = paths.jobs_prefix(netuid, run_id)
    local_bucket.put_json(
        local_bucket.uri_for_key(idx),
        ["j-a", "j-b"],
    )
    # Manifests not required when index is authoritative
    assert list_job_ids(local_bucket, index_key=idx, jobs_prefix_key=prefix) == ["j-a", "j-b"]


def test_list_job_ids_falls_back_to_manifest_scan(local_bucket, run_id) -> None:
    netuid = 0
    idx = paths.job_index_key(netuid, run_id)
    prefix = paths.jobs_prefix(netuid, run_id)
    mid = "orphan-job"
    man_uri = local_bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, mid))
    local_bucket.put_json(
        man_uri,
        {
            "job_id": mid,
            "run_id": run_id,
            "step_id": 0,
            "kind": "forward",
            "graph_ref": {"sha256": "0" * 64, "uri": "s3://test-bucket/g.json"},
            "params": {},
            "inputs": [],
            "outputs": [],
            "assigned_hotkey": "m",
            "attempt": 0,
            "deadline_unix": 9**9,
            "created_unix": 0,
            "resource_requirements": {},
            "verification_policy": {"critical": False},
        },
    )
    assert list_job_ids(local_bucket, index_key=idx, jobs_prefix_key=prefix) == [mid]


def test_list_job_ids_empty_index_triggers_fallback(local_bucket, run_id) -> None:
    netuid = 0
    idx = paths.job_index_key(netuid, run_id)
    prefix = paths.jobs_prefix(netuid, run_id)
    local_bucket.put_json(local_bucket.uri_for_key(idx), [])
    mid = "only-manifest"
    man_uri = local_bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, mid))
    local_bucket.put_json(
        man_uri,
        {
            "job_id": mid,
            "run_id": run_id,
            "step_id": 0,
            "kind": "forward",
            "graph_ref": {"sha256": "0" * 64, "uri": "s3://test-bucket/g.json"},
            "params": {},
            "inputs": [],
            "outputs": [],
            "assigned_hotkey": "m",
            "attempt": 0,
            "deadline_unix": 9**9,
            "created_unix": 0,
            "resource_requirements": {},
            "verification_policy": {"critical": False},
        },
    )
    assert list_job_ids(local_bucket, index_key=idx, jobs_prefix_key=prefix) == [mid]


def test_list_job_ids_manifest_list_cap_zero_skips_scan_when_no_index(local_bucket, run_id) -> None:
    netuid = 0
    idx = paths.job_index_key(netuid, run_id)
    prefix = paths.jobs_prefix(netuid, run_id)
    mid = "solo"
    man_uri = local_bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, mid))
    local_bucket.put_json(
        man_uri,
        {
            "job_id": mid,
            "run_id": run_id,
            "step_id": 0,
            "kind": "forward",
            "graph_ref": {"sha256": "0" * 64, "uri": "s3://test-bucket/g.json"},
            "params": {},
            "inputs": [],
            "outputs": [],
            "assigned_hotkey": "m",
            "attempt": 0,
            "deadline_unix": 9**9,
            "created_unix": 0,
            "resource_requirements": {},
            "verification_policy": {"critical": False},
        },
    )
    assert (
        list_job_ids(
            local_bucket,
            index_key=idx,
            jobs_prefix_key=prefix,
            manifest_list_max_uris=0,
        )
        == []
    )
