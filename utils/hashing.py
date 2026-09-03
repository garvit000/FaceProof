"""
Deterministic hashing and cryptographic fingerprinting module for FaceProof.
Produces deterministic JSON representations and computes SHA-256 digests.
"""

import hashlib
import json
from typing import Any, Dict, Union


def serialize_deterministic_json(data: Dict[str, Any]) -> str:
    """
    Produce a deterministic JSON string representation with sorted keys,
    compact separators, and UTF-8 encoding.

    Args:
        data: Python dictionary to serialize.

    Returns:
        Deterministic JSON string.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def generate_sha256_fingerprint(data: Union[Dict[str, Any], str, bytes]) -> str:
    """
    Generate a deterministic SHA-256 hex digest for a given data dictionary, string, or bytes.

    If a dictionary is passed, it is first serialized into deterministic JSON format.

    Args:
        data: Dictionary, string, or byte string to hash.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.
    """
    if isinstance(data, dict):
        canonical_str = serialize_deterministic_json(data)
        byte_data = canonical_str.encode("utf-8")
    elif isinstance(data, str):
        byte_data = data.encode("utf-8")
    elif isinstance(data, bytes):
        byte_data = data
    else:
        raise TypeError(f"Unsupported data type for hashing: {type(data)}")

    return hashlib.sha256(byte_data).hexdigest()


def verify_fingerprint(data: Dict[str, Any], expected_fingerprint: str) -> bool:
    """
    Verify whether the deterministic SHA-256 fingerprint of `data` matches `expected_fingerprint`.

    Args:
        data: Payload dictionary to verify.
        expected_fingerprint: 64-char SHA-256 hex string.

    Returns:
        True if the computed fingerprint matches expected_fingerprint, False otherwise.
    """
    computed = generate_sha256_fingerprint(data)
    return computed.lower() == expected_fingerprint.lower()
