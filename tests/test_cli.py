from __future__ import annotations

import json

from teuton_core.cli import main


def test_local_smoke_cli_outputs_scores(tmp_path, capsys) -> None:
    rc = main([
        "local-smoke",
        "--run-id",
        "cli-smoke",
        "--local-root",
        str(tmp_path),
        "--bucket",
        "cli",
        "--steps",
        "1",
        "--miners",
        "2",
        "--timeout-sec",
        "30",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index('{\n  "checked"'):])
    assert payload["checked"] > 0
    assert payload["scores"]
    assert payload["weight_update"]


def test_wipe_run_cli_uses_local_bucket(tmp_path, capsys) -> None:
    rc = main([
        "wipe-run",
        "--run-id",
        "missing",
        "--local-root",
        str(tmp_path),
        "--bucket",
        "cli",
    ])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"] == 0
    assert payload["run_id"] == "missing"


def test_local_smoke_cli_with_signed_crypto(tmp_path, capsys) -> None:
    rc = main([
        "local-smoke",
        "--run-id",
        "cli-signed-smoke",
        "--local-root",
        str(tmp_path),
        "--bucket",
        "cli-signed",
        "--steps",
        "1",
        "--miners",
        "2",
        "--timeout-sec",
        "30",
        "--crypto",
        "signed",
        "--required-signer",
        "assigned_hotkey",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index('{\n  "checked"'):])
    assert payload["checked"] > 0
    assert payload["scores"]
