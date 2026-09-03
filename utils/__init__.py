"""
Utility package for FaceProof.
Provides deterministic JSON serialization, SHA-256 fingerprinting, and metadata packaging.
"""

from .hashing import serialize_deterministic_json, generate_sha256_fingerprint, verify_fingerprint
from .metadata import build_handoff_payload, save_result_payload

__all__ = [
    "serialize_deterministic_json",
    "generate_sha256_fingerprint",
    "verify_fingerprint",
    "build_handoff_payload",
    "save_result_payload",
]
