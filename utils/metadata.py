"""
Metadata packaging and blockchain handoff preparation for FaceProof.
Formats the final result payload and saves it deterministically to output/result.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .hashing import generate_sha256_fingerprint, serialize_deterministic_json


def build_canonical_payload(match_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract immutable discovered content metadata into a clean dictionary for deterministic hashing.
    Excludes volatile fields like execution timestamps to ensure reproducibility.
    """
    if not match_data:
        return {
            "source": "None",
            "post_url": "",
            "title": "No match found",
            "status": "unverified",
        }

    payload = {
        "source": str(match_data.get("source", "")),
        "post_url": str(match_data.get("post_url", "")),
        "title": str(match_data.get("title", "")),
    }

    if match_data.get("image_url"):
        payload["image_url"] = str(match_data["image_url"])
    if match_data.get("face_similarity") is not None:
        payload["face_similarity"] = round(float(match_data["face_similarity"]), 4)

    return payload


def build_handoff_payload(
    input_info: Dict[str, Any],
    search_info: Dict[str, Any],
    best_match: Optional[Dict[str, Any]],
    all_candidates: List[Dict[str, Any]],
    crop_path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Construct the complete structured output payload for FaceProof.

    Args:
        input_info: Details about the input image and detected faces.
        search_info: Details about the search query and engine.
        best_match: Selected best match candidate dictionary.
        all_candidates: List of all evaluated candidates.
        crop_path: Path to the processed face crop.
        dry_run: Boolean indicating if this was a dry-run execution.

    Returns:
        Structured Python dictionary ready for saving to output/result.json.
    """
    # Build the immutable canonical payload for hashing
    canonical_payload = build_canonical_payload(best_match)
    fingerprint = generate_sha256_fingerprint(canonical_payload)

    result = {
        "faceproof_version": "1.0.0",
        "dry_run": dry_run,
        "input": input_info,
        "search": search_info,
        "match": best_match,
        "all_candidates": all_candidates,
        "canonical_payload": canonical_payload,
        "fingerprint": fingerprint,
        "blockchain_handoff": {
            "fingerprint_to_register": fingerprint,
            "hash_algorithm": "SHA-256",
            "canonical_payload_json": serialize_deterministic_json(canonical_payload),
            "instructions": (
                "For blockchain integration: read 'fingerprint' (or 'fingerprint_to_register') "
                "and store it on-chain (e.g. in a smart contract event or state storage). "
                "To re-verify at any future time: compute SHA-256 over 'canonical_payload' "
                "using deterministic JSON serialization and compare against the on-chain hash."
            ),
        },
    }

    if crop_path:
        result["input"]["face_crop_path"] = str(crop_path)

    return result


def save_result_payload(
    payload: Dict[str, Any],
    output_path: Union[str, Path] = "output/result.json",
    indent: int = 2,
) -> Path:
    """
    Save the result payload to disk formatted with readable indentation.

    Args:
        payload: The complete structured result dictionary.
        output_path: Target path (default 'output/result.json').
        indent: Indentation level for pretty printing (default 2).

    Returns:
        Path of the written JSON file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
    return out_p
