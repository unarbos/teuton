"""Small deterministic signing helpers.

For no-chain tests we use shared-secret HMAC signatures. In subnet mode these
records can additionally be signed by Bittensor wallets; the payload hashing
here remains the canonical bytes-to-sign surface.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Protocol
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_dict(value: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(value))


def sign_dict(value: dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_json(value), hashlib.sha256).hexdigest()


def verify_dict(value: dict[str, Any], secret: str, signature: str) -> bool:
    return hmac.compare_digest(sign_dict(value, secret), signature)


class Signer(Protocol):
    identity: str

    def sign(self, payload: bytes) -> str: ...


class Verifier(Protocol):
    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool: ...


@dataclass
class HmacSigner:
    secret: str
    identity: str = "dev-hmac"

    def sign(self, payload: bytes) -> str:
        return hmac.new(self.secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


class BittensorHotkeySigner:
    """Wallet-backed signer placeholder.

    The exact wallet signing API varies across Bittensor releases. Keep this
    adapter isolated so production subnet code can pin the SDK version while
    the rest of the runtime uses the Signer/Verifier protocol.
    """

    def __init__(self, *, wallet, identity: str | None = None) -> None:
        self.wallet = wallet
        self.identity = identity or getattr(getattr(wallet, "hotkey", None), "ss58_address", "bittensor-hotkey")

    def sign(self, payload: bytes) -> str:
        signer = getattr(getattr(self.wallet, "hotkey", None), "sign", None)
        if signer is None:
            raise RuntimeError("wallet hotkey does not expose sign(payload)")
        sig = signer(payload)
        return sig.hex() if isinstance(sig, bytes) else str(sig)

    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool:
        verifier = getattr(getattr(self.wallet, "hotkey", None), "verify", None)
        if verifier is None:
            raise RuntimeError("wallet hotkey does not expose verify(payload, signature)")
        sig = bytes.fromhex(signature) if all(c in "0123456789abcdefABCDEF" for c in signature) else signature
        return bool(verifier(payload, sig))
