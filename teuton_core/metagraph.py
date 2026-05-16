"""Metagraph helpers for assignment-grant encryption."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from .signatures import sha256_hex


@dataclass(frozen=True)
class HotkeyMetagraphInfo:
    hotkey: str
    uid: int
    coldkey: str | None
    public_key: bytes
    metagraph_hash: str
    block: int | None = None


class MetagraphHotkeyResolver:
    def resolve(self, hotkey: str) -> HotkeyMetagraphInfo:
        raise NotImplementedError


class StaticHotkeyResolver(MetagraphHotkeyResolver):
    def __init__(self, mapping: dict[str, HotkeyMetagraphInfo]) -> None:
        self.mapping = dict(mapping)

    def resolve(self, hotkey: str) -> HotkeyMetagraphInfo:
        try:
            return self.mapping[hotkey]
        except KeyError as e:
            raise KeyError(f"hotkey {hotkey} not found in static metagraph mapping") from e


class BtcliMetagraphHotkeyResolver(MetagraphHotkeyResolver):
    """Resolve registered hotkey public keys via `btcli subnets show --json-output`."""

    def __init__(self, *, netuid: int, network: str = "finney", mechid: int = 0) -> None:
        self.netuid = int(netuid)
        self.network = network
        self.mechid = int(mechid)
        self._rows: list[dict] | None = None
        self._hash: str | None = None

    def resolve(self, hotkey: str) -> HotkeyMetagraphInfo:
        rows = self._load_rows()
        for row in rows:
            if row.get("hotkey") != hotkey:
                continue
            public_key = public_key_from_hotkey_ss58(hotkey)
            return HotkeyMetagraphInfo(
                hotkey=hotkey,
                uid=int(row["uid"]),
                coldkey=row.get("coldkey"),
                public_key=public_key,
                metagraph_hash=self._hash or sha256_hex(json.dumps(rows, sort_keys=True).encode("utf-8")),
            )
        raise KeyError(f"hotkey {hotkey} not found in netuid {self.netuid} metagraph")

    def _load_rows(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        output = subprocess.check_output(
            [
                "btcli",
                "subnets",
                "show",
                "--netuid",
                str(self.netuid),
                "--network",
                self.network,
                "--mechid",
                str(self.mechid),
                "--no-prompt",
                "--json-output",
                "--quiet",
            ],
            text=True,
        )
        data = json.loads(output, strict=False)
        rows = list(data.get("uids") or [])
        self._hash = sha256_hex(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self._rows = rows
        return rows


def public_key_from_hotkey_ss58(hotkey: str) -> bytes:
    try:
        from substrateinterface.utils.ss58 import ss58_decode
    except Exception:
        try:
            from scalecodec.utils.ss58 import ss58_decode
        except Exception as e:
            raise RuntimeError("substrateinterface or scalecodec is required to decode hotkey public keys") from e
    return bytes.fromhex(ss58_decode(hotkey))
