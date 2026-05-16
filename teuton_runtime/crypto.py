"""Artifact crypto envelopes for Teuton v3."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass

from teuton_core.protocol import ArtifactCryptoPolicy, ArtifactEnvelope, CryptoMode
from teuton_core.signatures import HmacSigner, Signer, Verifier, canonical_json, sha256_hex


class TimelockPending(Exception):
    """Raised when a timelocked artifact cannot be decrypted yet."""


class DrandTimelockProvider:
    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        raise NotImplementedError

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        raise NotImplementedError


DRAND_QUICKNET_CHAIN_HASH = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
DRAND_QUICKNET_API = "https://api.drand.sh/"


@dataclass
class MockDrandTimelockProvider(DrandTimelockProvider):
    revealed_round: int = 0
    secret: str = "mock-drand"

    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        return xor_crypt(plaintext, key=f"{self.secret}:{round_number}")

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        if self.revealed_round < round_number:
            raise TimelockPending(f"drand round {round_number} is not revealed")
        return xor_crypt(ciphertext, key=f"{self.secret}:{round_number}")


@dataclass
class DrandTlockProvider(DrandTimelockProvider):
    """Real drand timelock provider backed by the official `tle` CLI.

    The Python timelock wheel is currently platform-limited, while the drand
    Go CLI is the reference implementation and supports macOS/Linux. The CLI
    produces standard tlock ciphertext and decrypts by fetching the drand
    signature for the target round.
    """

    network: str = DRAND_QUICKNET_API
    chain_hash: str = DRAND_QUICKNET_CHAIN_HASH
    tle_path: str | None = None
    timeout_sec: int = 30
    force_past_round_encrypt: bool = False

    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        with tempfile.TemporaryDirectory(prefix="teuton-drand-") as tmp:
            in_path = os.path.join(tmp, "plain.bin")
            out_path = os.path.join(tmp, "cipher.tle")
            with open(in_path, "wb") as f:
                f.write(plaintext)
            cmd = [
                self._tle(),
                "--encrypt",
                "--network",
                self._network(policy),
                "--chain",
                self._chain_hash(policy),
                "--round",
                str(int(round_number)),
                "--output",
                out_path,
            ]
            if self.force_past_round_encrypt:
                cmd.append("--force")
            cmd.append(in_path)
            self._run(cmd)
            with open(out_path, "rb") as f:
                return f.read()

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        with tempfile.TemporaryDirectory(prefix="teuton-drand-") as tmp:
            in_path = os.path.join(tmp, "cipher.tle")
            out_path = os.path.join(tmp, "plain.bin")
            with open(in_path, "wb") as f:
                f.write(ciphertext)
            cmd = [
                self._tle(),
                "--decrypt",
                "--network",
                self._network(policy),
                "--chain",
                self._chain_hash(policy),
                "--output",
                out_path,
                in_path,
            ]
            try:
                self._run(cmd)
            except TimelockPending:
                raise
            with open(out_path, "rb") as f:
                return f.read()

    def latest_round(self, policy: ArtifactCryptoPolicy | None = None) -> int:
        url = self._network(policy or ArtifactCryptoPolicy()).rstrip("/") + f"/{self._chain_hash(policy or ArtifactCryptoPolicy())}/public/latest"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_sec) as resp:
                body = resp.read()
        except urllib.error.HTTPError:
            # Some relays expose the default chain at /public/latest.
            fallback = self._network(policy or ArtifactCryptoPolicy()).rstrip("/") + "/public/latest"
            with urllib.request.urlopen(fallback, timeout=self.timeout_sec) as resp:
                body = resp.read()
        return int(json.loads(body.decode("utf-8"))["round"])

    def _network(self, policy: ArtifactCryptoPolicy) -> str:
        return policy.key_id or self.network

    def _chain_hash(self, policy: ArtifactCryptoPolicy) -> str:
        return policy.drand_chain_hash or self.chain_hash

    def _tle(self) -> str:
        path = self.tle_path or shutil.which("tle")
        if path:
            return path
        go_path = shutil.which("go")
        if go_path:
            try:
                gopath = subprocess.check_output([go_path, "env", "GOPATH"], text=True, timeout=5).strip()
                candidate = os.path.join(gopath, "bin", "tle")
                if os.path.exists(candidate):
                    return candidate
            except Exception:
                pass
        raise RuntimeError(
            "drand timelock requires the official `tle` CLI. "
            "Install it with: go install github.com/drand/tlock/cmd/tle@latest"
        )

    def _run(self, cmd: list[str]) -> None:
        proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout_sec, check=False)
        if proc.returncode == 0:
            return
        stderr = proc.stderr.decode("utf-8", errors="replace")
        stdout = proc.stdout.decode("utf-8", errors="replace")
        msg = stderr or stdout
        if "too early" in msg.lower() or "no beacon" in msg.lower() or "not found" in msg.lower():
            raise TimelockPending(msg.strip() or "drand round is not revealed")
        raise RuntimeError(f"tle command failed: {msg.strip()}")


class BittensorDrandTimelockProvider(DrandTimelockProvider):
    """Real drand timelock provider from the `bittensor-drand` package.

    This is the default production path because `uv sync --extra subnet`
    installs the dependency alongside Bittensor. It uses drand timelock
    encryption and returns `TimelockPending` until the reveal round is
    available.
    """

    def __init__(self) -> None:
        try:
            import bittensor_drand
        except Exception as e:
            raise RuntimeError(
                "bittensor-drand is required for drand timelock. "
                "Install with `uv sync --extra subnet` or `uv sync --all-extras`."
            ) from e
        self._drand = bittensor_drand

    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        self._validate_policy(policy)
        ciphertext, reveal_round = self._drand.encrypt_at_round(plaintext, int(round_number))
        if int(reveal_round) != int(round_number):
            raise RuntimeError(f"bittensor-drand returned reveal round {reveal_round}, expected {round_number}")
        return ciphertext

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        self._validate_policy(policy)
        plaintext = self._drand.decrypt(ciphertext, no_errors=True)
        if plaintext is None:
            raise TimelockPending(f"drand round {round_number} is not revealed")
        return bytes(plaintext)

    def latest_round(self) -> int:
        return int(self._drand.get_latest_round())

    @staticmethod
    def _validate_policy(policy: ArtifactCryptoPolicy) -> None:
        if policy.drand_chain_hash and policy.drand_chain_hash != DRAND_QUICKNET_CHAIN_HASH:
            raise ValueError("bittensor-drand provider supports only the default quicknet chain")
        if policy.drand_public_key:
            raise ValueError("bittensor-drand provider does not accept custom drand public keys")


class BittensorTimelockProvider(BittensorDrandTimelockProvider):
    """Backward-compatible name for the production Bittensor drand provider."""


def default_policy(policy: ArtifactCryptoPolicy | None) -> ArtifactCryptoPolicy:
    return policy or ArtifactCryptoPolicy()


def encode_envelope(
    plaintext: bytes,
    policy: ArtifactCryptoPolicy | None,
    *,
    signer: Signer | None = None,
    encryption_secret: str = "teuton-dev-encryption",
    timelock_provider: DrandTimelockProvider | None = None,
) -> bytes:
    policy = default_policy(policy)
    mode = str(policy.mode)
    if mode == CryptoMode.NONE.value:
        return plaintext

    ciphertext = plaintext
    if mode == CryptoMode.ENCRYPTED.value:
        ciphertext = xor_crypt(plaintext, key=policy.key_id or encryption_secret)
    elif mode == CryptoMode.DRAND_TIMELOCK.value:
        if policy.drand_round is None:
            raise ValueError("drand_timelock policy requires drand_round")
        provider = timelock_provider or BittensorDrandTimelockProvider()
        ciphertext = provider.encrypt(plaintext, round_number=int(policy.drand_round), policy=policy)
    elif mode != CryptoMode.SIGNED.value:
        raise ValueError(f"unknown crypto mode: {mode}")

    envelope = ArtifactEnvelope(
        crypto_mode=mode,
        payload_b64=base64.b64encode(ciphertext).decode("ascii"),
        plaintext_sha256=sha256_hex(plaintext),
        ciphertext_sha256=sha256_hex(ciphertext),
        signer=getattr(signer, "identity", None),
        cipher_suite=policy.cipher_suite,
        key_id=policy.key_id,
        drand_round=policy.drand_round,
        drand_chain_hash=policy.drand_chain_hash,
        drand_public_key=policy.drand_public_key,
    )
    if signer is not None:
        envelope.signature = signer.sign(canonical_json(envelope.signed_payload_dict()))
    return json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_envelope(
    blob: bytes,
    policy: ArtifactCryptoPolicy | None,
    *,
    verifier: Verifier | None = None,
    encryption_secret: str = "teuton-dev-encryption",
    timelock_provider: DrandTimelockProvider | None = None,
) -> bytes:
    policy = default_policy(policy)
    mode = str(policy.mode)
    if mode == CryptoMode.NONE.value:
        return blob

    envelope = ArtifactEnvelope.from_dict(json.loads(blob.decode("utf-8")))
    verify_artifact_envelope(envelope, policy, verifier=verifier)
    ciphertext = base64.b64decode(envelope.payload_b64.encode("ascii"))
    if sha256_hex(ciphertext) != envelope.ciphertext_sha256:
        raise ValueError("artifact ciphertext digest mismatch")

    if mode == CryptoMode.SIGNED.value:
        plaintext = ciphertext
    elif mode == CryptoMode.ENCRYPTED.value:
        plaintext = xor_crypt(ciphertext, key=policy.key_id or encryption_secret)
    elif mode == CryptoMode.DRAND_TIMELOCK.value:
        if policy.drand_round is None:
            raise ValueError("drand_timelock policy requires drand_round")
        provider = timelock_provider or BittensorDrandTimelockProvider()
        plaintext = provider.decrypt(ciphertext, round_number=int(policy.drand_round), policy=policy)
    else:
        raise ValueError(f"unknown crypto mode: {mode}")

    if sha256_hex(plaintext) != envelope.plaintext_sha256:
        raise ValueError("artifact plaintext digest mismatch")
    return plaintext


def verify_artifact_envelope(
    envelope: ArtifactEnvelope,
    expected_policy: ArtifactCryptoPolicy,
    *,
    verifier: Verifier | None = None,
) -> None:
    if envelope.crypto_mode != str(expected_policy.mode):
        raise ValueError(f"artifact crypto mode mismatch: {envelope.crypto_mode} != {expected_policy.mode}")
    if expected_policy.required_signer and envelope.signer != expected_policy.required_signer:
        raise ValueError("artifact signer mismatch")
    if expected_policy.mode in (CryptoMode.SIGNED.value, CryptoMode.ENCRYPTED.value, CryptoMode.DRAND_TIMELOCK.value):
        if not envelope.signature:
            raise ValueError("artifact signature missing")
        if verifier is not None and not verifier.verify(
            canonical_json(envelope.signed_payload_dict()),
            envelope.signature,
            envelope.signer,
        ):
            raise ValueError("artifact signature verification failed")


def artifact_digest_from_blob(name: str, uri: str, blob: bytes, policy: ArtifactCryptoPolicy | None) -> dict:
    policy = default_policy(policy)
    mode = str(policy.mode)
    if mode == CryptoMode.NONE.value:
        return {
            "sha256": sha256_hex(blob),
            "size_bytes": len(blob),
            "crypto_mode": mode,
        }
    envelope = ArtifactEnvelope.from_dict(json.loads(blob.decode("utf-8")))
    return {
        "sha256": sha256_hex(blob),
        "size_bytes": len(blob),
        "plaintext_sha256": envelope.plaintext_sha256,
        "ciphertext_sha256": envelope.ciphertext_sha256,
        "envelope_sha256": sha256_hex(blob),
        "signature": envelope.signature,
        "crypto_mode": envelope.crypto_mode,
    }


def xor_crypt(data: bytes, *, key: str) -> bytes:
    stream = hashlib.sha256(key.encode("utf-8")).digest()
    out = bytearray()
    for i, b in enumerate(data):
        if i and i % len(stream) == 0:
            stream = hashlib.sha256(stream).digest()
        out.append(b ^ stream[i % len(stream)])
    return bytes(out)
