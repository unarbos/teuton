#!/usr/bin/env python3
"""Generate a native Bittensor ED25519 miner hotkey for Teuton."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from bittensor_wallet import Wallet  # type: ignore[import-not-found]
from nacl.public import SealedBox  # type: ignore[import-not-found]
from nacl.signing import SigningKey, VerifyKey  # type: ignore[import-not-found]
from scalecodec.utils.ss58 import ss58_decode  # type: ignore[import-not-found]


DEFAULT_WALLET_PATH = Path.home() / ".bittensor" / "wallets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a native ED25519 hotkey for Teuton miners using bittensor-wallet "
            "crypto_type=0. The generated hotkey can be registered normally with btcli, "
            "and its metagraph SS58 hotkey can be used to encrypt presigned links."
        )
    )
    parser.add_argument("--wallet-name", required=True, help="Bittensor wallet/coldkey name, e.g. teuton_mining")
    parser.add_argument("--hotkey", required=True, help="Hotkey name to create, e.g. teuton_miner_ed25519_1")
    parser.add_argument("--wallet-path", type=Path, default=DEFAULT_WALLET_PATH, help=f"Wallet root path (default: {DEFAULT_WALLET_PATH})")
    parser.add_argument("--n-words", type=int, default=12, help="Mnemonic length for the native wallet generator")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files")
    return parser.parse_args()


def hotkey_file(wallet_path: Path, wallet_name: str, hotkey: str) -> Path:
    return wallet_path / wallet_name / "hotkeys" / hotkey


def hotkey_pub_file(wallet_path: Path, wallet_name: str, hotkey: str) -> Path:
    return wallet_path / wallet_name / "hotkeys" / f"{hotkey}pub.txt"


def require_native_ed25519_wallet() -> None:
    create_new_hotkey = inspect.signature(Wallet.create_new_hotkey)
    if "crypto_type" not in create_new_hotkey.parameters:
        raise RuntimeError(
            "Installed bittensor-wallet does not expose native ED25519 hotkey generation. "
            "Install a version that includes latent-to/btwallet#195, then rerun this command."
        )


def self_test(seed: bytes, ss58_address: str) -> None:
    public_key = bytes.fromhex(ss58_decode(ss58_address))
    signing_key = SigningKey(seed)
    if signing_key.verify_key.encode() != public_key:
        raise RuntimeError("generated private seed does not match SS58 public key")

    plaintext = b"https://example.invalid/teuton-presigned-link-self-test"
    ciphertext = SealedBox(VerifyKey(public_key).to_curve25519_public_key()).encrypt(plaintext)
    decrypted = SealedBox(signing_key.to_curve25519_private_key()).decrypt(ciphertext)
    if decrypted != plaintext:
        raise RuntimeError("ED25519-to-X25519 encryption self-test failed")


def main() -> int:
    args = parse_args()
    require_native_ed25519_wallet()

    wallet = Wallet(name=args.wallet_name, hotkey=args.hotkey, path=str(args.wallet_path))
    wallet.create_new_hotkey(
        crypto_type=0,
        n_words=args.n_words,
        overwrite=args.overwrite,
        suppress=True,
        use_password=False,
    )

    private_path = hotkey_file(args.wallet_path, args.wallet_name, args.hotkey)
    public_path = hotkey_pub_file(args.wallet_path, args.wallet_name, args.hotkey)
    private_payload = json.loads(private_path.read_text(encoding="utf-8"))
    public_payload = json.loads(public_path.read_text(encoding="utf-8"))
    if private_payload.get("cryptoType") != 0 or public_payload.get("cryptoType") != 0:
        raise RuntimeError("native wallet generated a non-ED25519 hotkey")

    ss58_address = str(private_payload["ss58Address"])
    public_key_hex = str(private_payload["publicKey"]).removeprefix("0x")
    seed = bytes.fromhex(str(private_payload["secretSeed"]).removeprefix("0x"))
    if bytes.fromhex(ss58_decode(ss58_address)).hex() != public_key_hex:
        raise RuntimeError("generated SS58 address does not decode to the hotkey public key")
    if public_payload.get("ss58Address") != ss58_address:
        raise RuntimeError("hotkey public file does not match the private hotkey file")
    self_test(seed, ss58_address)

    print("Created ED25519 Teuton miner hotkey")
    print(f"wallet: {args.wallet_name}")
    print(f"hotkey: {args.hotkey}")
    print(f"ss58_address: {ss58_address}")
    print(f"public_key_hex: {public_key_hex}")
    print(f"native_hotkey_file: {private_path}")
    print(f"native_hotkey_pub_file: {public_path}")
    print("encryption_self_test: ok")
    print()
    print("Register with:")
    print(
        "btcli subnets register "
        f"--netuid <NETUID> --wallet-name {args.wallet_name} --hotkey {args.hotkey} --wallet-path {args.wallet_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
