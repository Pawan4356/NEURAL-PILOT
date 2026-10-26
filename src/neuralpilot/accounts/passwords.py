"""Password hashing with the standard library's scrypt.

scrypt is memory-hard, so offline guessing against a leaked database is slow.
It is used here instead of a third-party package to keep dependencies unchanged.
Stored format: `scrypt$<n>$<r>$<p>$<salt hex>$<hash hex>`, so the cost
parameters travel with each hash and can be raised later without breaking
existing accounts.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_N = 2**14
_R = 8
_P = 1
_DKLEN = 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        expected = bytes.fromhex(digest_hex)
        candidate = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except ValueError:
        return False  # malformed or foreign hash: treat as a failed login, never as a crash
    return hmac.compare_digest(candidate, expected)
