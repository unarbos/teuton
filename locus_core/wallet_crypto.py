"""Assignment-grant crypto adapters.

Bittensor hotkeys always provide sign/verify. Some Substrate keypair builds
also expose authenticated message encryption helpers; when available we use
those for assignment grants. The dev adapter provides encrypted grant
round-trips for local and CI tests.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import NoReturn

from .protocol import AssignmentGrantV3, EncryptedAssignmentGrantV3
from .signatures import canonical_json, sha256_hex
from locus_runtime.crypto import xor_crypt


class AssignmentEncryptor:
    scheme: str

    def encrypt_for_hotkey(
        self,
        grant: AssignmentGrantV3,
        *,
        recipient_hotkey: str,
        recipient_uid: int | None = None,
        metagraph_block: int | None = None,
        metagraph_hash: str | None = None,
    ) -> EncryptedAssignmentGrantV3:
        raise NotImplementedError


class AssignmentDecryptor:
    scheme: str

    def decrypt(self, encrypted: EncryptedAssignmentGrantV3, *, expected_hotkey: str) -> AssignmentGrantV3:
        raise NotImplementedError


@dataclass
class DevAssignmentCrypto(AssignmentEncryptor, AssignmentDecryptor):
    secret: str = "locus-dev-assignment"
    scheme: str = "dev-xor-v1"

    def encrypt_for_hotkey(
        self,
        grant: AssignmentGrantV3,
        *,
        recipient_hotkey: str,
        recipient_uid: int | None = None,
        metagraph_block: int | None = None,
        metagraph_hash: str | None = None,
    ) -> EncryptedAssignmentGrantV3:
        plaintext = canonical_json(grant.to_dict())
        ciphertext = xor_crypt(plaintext, key=f"{self.secret}:{recipient_hotkey}")
        return EncryptedAssignmentGrantV3(
            job_id=grant.job_id,
            run_id=grant.run_id,
            recipient_hotkey=recipient_hotkey,
            recipient_uid=recipient_uid,
            metagraph_block=metagraph_block,
            metagraph_hash=metagraph_hash or sha256_hex(recipient_hotkey.encode("utf-8")),
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
            crypto_scheme=self.scheme,
        )

    def decrypt(self, encrypted: EncryptedAssignmentGrantV3, *, expected_hotkey: str) -> AssignmentGrantV3:
        if encrypted.crypto_scheme != self.scheme:
            raise ValueError(f"unsupported assignment scheme {encrypted.crypto_scheme!r}")
        if encrypted.recipient_hotkey != expected_hotkey:
            raise ValueError("assignment grant recipient hotkey mismatch")
        ciphertext = base64.b64decode(encrypted.ciphertext_b64.encode("ascii"))
        plaintext = xor_crypt(ciphertext, key=f"{self.secret}:{expected_hotkey}")
        return AssignmentGrantV3.from_dict(json.loads(plaintext.decode("utf-8")))


class BittensorWalletCrypto(AssignmentEncryptor, AssignmentDecryptor):
    """Bittensor hotkey adapter.

    Bittensor wallet releases vary: some expose only sign(data) and
    verify(data, signature), while Substrate keypairs also expose
    encrypt_message/decrypt_message. Encryption is therefore feature-detected
    and fails explicitly when the wrapped keypair does not support it.
    """

    scheme = "substrate-keypair-box-v1"
    sign_scheme = "bittensor-hotkey-sign-v1"

    def __init__(self, keypair) -> None:
        self.keypair = keypair
        self.identity = getattr(keypair, "ss58_address", "unknown-hotkey")

    def sign(self, payload: bytes) -> str:
        sig = self.keypair.sign(payload)
        return sig.hex() if isinstance(sig, bytes) else str(sig)

    def verify(self, payload: bytes, signature: str) -> bool:
        sig = bytes.fromhex(signature) if _is_hex(signature) else signature
        return bool(self.keypair.verify(payload, sig))

    def encrypt_for_hotkey(
        self,
        grant: AssignmentGrantV3,
        *,
        recipient_hotkey: str,
        recipient_uid: int | None = None,
        metagraph_block: int | None = None,
        metagraph_hash: str | None = None,
        recipient_public_key: bytes | str | None = None,
    ) -> EncryptedAssignmentGrantV3:
        encrypt = getattr(self.keypair, "encrypt_message", None)
        if encrypt is None:
            raise NotImplementedError(
                "Installed Bittensor Keypair exposes sign/verify but not sealed encryption. "
                "Use a Substrate keypair with encrypt_message/decrypt_message or DevAssignmentCrypto for tests."
            )
        recipient_public_key_bytes = (
            _key_material_to_bytes(recipient_public_key, "recipient_public_key")
            if recipient_public_key is not None
            else _public_key_bytes_from_hotkey(recipient_hotkey)
        )
        sender_public_key = _keypair_public_key_bytes(self.keypair)
        plaintext = canonical_json(grant.to_dict())
        try:
            ciphertext = encrypt(plaintext, recipient_public_key_bytes)
        except Exception as e:
            _raise_keypair_crypto_error(e)
        return EncryptedAssignmentGrantV3(
            job_id=grant.job_id,
            run_id=grant.run_id,
            recipient_hotkey=recipient_hotkey,
            recipient_uid=recipient_uid,
            metagraph_block=metagraph_block,
            metagraph_hash=metagraph_hash or sha256_hex(recipient_hotkey.encode("utf-8")),
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
            crypto_scheme=self.scheme,
            sender_hotkey=self.identity,
            sender_public_key_hex=sender_public_key.hex(),
            recipient_public_key_hex=recipient_public_key_bytes.hex(),
        )

    def decrypt(self, encrypted: EncryptedAssignmentGrantV3, *, expected_hotkey: str) -> AssignmentGrantV3:
        if encrypted.crypto_scheme != self.scheme:
            raise ValueError(f"unsupported assignment scheme {encrypted.crypto_scheme!r}")
        if encrypted.recipient_hotkey != expected_hotkey:
            raise ValueError("assignment grant recipient hotkey mismatch")
        decrypt = getattr(self.keypair, "decrypt_message", None)
        if decrypt is None:
            raise NotImplementedError(
                "Installed Bittensor Keypair exposes sign/verify but not sealed decryption. "
                "Use a Substrate keypair with encrypt_message/decrypt_message."
            )
        if not encrypted.sender_public_key_hex:
            raise ValueError("encrypted assignment grant missing sender public key")
        ciphertext = base64.b64decode(encrypted.ciphertext_b64.encode("ascii"))
        try:
            plaintext = decrypt(ciphertext, bytes.fromhex(encrypted.sender_public_key_hex))
        except Exception as e:
            _raise_keypair_crypto_error(e)
        return AssignmentGrantV3.from_dict(json.loads(plaintext.decode("utf-8")))


def _is_hex(value: str) -> bool:
    try:
        bytes.fromhex(value)
        return True
    except ValueError:
        return False


def _keypair_public_key_bytes(keypair) -> bytes:
    public_key = getattr(keypair, "public_key", None)
    if public_key is None:
        raise ValueError("Bittensor keypair does not expose public_key bytes for encrypted grants")
    return _key_material_to_bytes(public_key, "keypair.public_key")


def _public_key_bytes_from_hotkey(hotkey: str) -> bytes:
    try:
        from substrateinterface.utils.ss58 import ss58_decode
    except Exception:
        try:
            from scalecodec.utils.ss58 import ss58_decode
        except Exception as e:
            raise RuntimeError(
                "substrateinterface or scalecodec is required to decode recipient hotkey public keys"
            ) from e
    decoded = ss58_decode(hotkey)
    return _key_material_to_bytes(decoded, "recipient_hotkey")


def _key_material_to_bytes(value: bytes | bytearray | str, field_name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        text = value[2:] if value.startswith("0x") else value
        try:
            return bytes.fromhex(text)
        except ValueError as e:
            raise ValueError(f"{field_name} must be bytes or a hex string") from e
    raise TypeError(f"{field_name} must be bytes or a hex string")


def _raise_keypair_crypto_error(error: Exception) -> NoReturn:
    if "Only ed25519 keypair type supported" in str(error):
        raise NotImplementedError(
            "Substrate keypair encryption is only supported for ed25519 keypairs; "
            "default Bittensor SR25519 hotkeys can sign/verify but cannot encrypt/decrypt this way."
        ) from error
    raise error
