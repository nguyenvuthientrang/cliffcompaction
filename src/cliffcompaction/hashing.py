"""Canonical digests and the prefix hash chain.

h_0 = H(digest_0); h_i = H(h_{i-1} || digest_i). chain[i] identifies
messages[0..i] independent of volatile serialization details, so prefix
identity is a single dict lookup.
"""

from __future__ import annotations

import hashlib
import json


def canonical_json(obj) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_obj(obj) -> str:
    return digest_bytes(canonical_json(obj).encode("utf-8"))


def chain_hashes(message_digests: list[str]) -> list[str]:
    """chain[i] = hash identifying messages[0..i] as a sequence."""
    chain: list[str] = []
    prev = ""
    for d in message_digests:
        prev = hashlib.sha256((prev + d).encode("utf-8")).hexdigest()
        chain.append(prev)
    return chain
