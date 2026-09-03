"""
Unit and integration tests for FaceProof pipeline components.
"""

import json
import os
from pathlib import Path
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from face.detector import detect_faces, FaceDetectionResult, ensure_yunet_model
from face.cropper import crop_face, save_face_crop
from face.encoder import encode_face, compute_similarity, ensure_sface_model
from utils.hashing import (
    serialize_deterministic_json,
    generate_sha256_fingerprint,
    verify_fingerprint,
)
from utils.metadata import build_canonical_payload, build_handoff_payload, save_result_payload
from search.lens import LensCandidate, LensSearchResult
from search.matcher import CandidateMatcher


def create_synthetic_face_image() -> np.ndarray:
    """Create an image with a clear synthetic human face drawing."""
    img = np.full((400, 400, 3), (210, 210, 210), dtype=np.uint8)
    # Head contour
    cv2.ellipse(img, (200, 200), (90, 120), 0, 0, 360, (180, 195, 230), -1)
    # Eyes
    cv2.ellipse(img, (165, 175), (16, 9), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (235, 175), (16, 9), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (165, 175), 6, (70, 40, 20), -1)
    cv2.circle(img, (235, 175), 6, (70, 40, 20), -1)
    # Eyebrows
    cv2.ellipse(img, (165, 155), (20, 5), 0, 180, 360, (50, 30, 15), 3)
    cv2.ellipse(img, (235, 155), (20, 5), 0, 180, 360, (50, 30, 15), 3)
    # Nose
    cv2.line(img, (200, 175), (200, 215), (140, 150, 180), 2)
    cv2.ellipse(img, (200, 215), (9, 4), 0, 0, 180, (140, 150, 180), 2)
    # Mouth
    cv2.ellipse(img, (200, 255), (28, 14), 0, 0, 180, (90, 90, 180), -1)
    return img


def test_models_download():
    """Verify that ONNX models are downloaded and accessible."""
    yunet = ensure_yunet_model()
    sface = ensure_sface_model()
    assert yunet.exists() and yunet.stat().st_size > 100_000
    assert sface.exists() and sface.stat().st_size > 10_000_000


def test_deterministic_hashing():
    """Verify that deterministic JSON serialization and SHA-256 produce consistent hashes."""
    payload_a = {"source": "Instagram", "post_url": "https://instagram.com/p/123", "title": "Profile Post"}
    # Reverse key insertion order
    payload_b = {"title": "Profile Post", "post_url": "https://instagram.com/p/123", "source": "Instagram"}

    json_a = serialize_deterministic_json(payload_a)
    json_b = serialize_deterministic_json(payload_b)

    assert json_a == json_b
    assert json_a == '{"post_url":"https://instagram.com/p/123","source":"Instagram","title":"Profile Post"}'

    hash_a = generate_sha256_fingerprint(payload_a)
    hash_b = generate_sha256_fingerprint(payload_b)

    assert hash_a == hash_b
    assert len(hash_a) == 64
    assert verify_fingerprint(payload_a, hash_a)


def test_matcher_ranking():
    """Verify candidate evaluation and ranking logic."""
    search_result = LensSearchResult(
        engine="Google Lens",
        query_type="reverse_image",
        total_results=2,
        candidates=[
            LensCandidate(
                title="Post A",
                post_url="https://example.com/a",
                source="Web",
                position=1,
            ),
            LensCandidate(
                title="Post B",
                post_url="https://example.com/b",
                source="Web",
                position=2,
            ),
        ],
    )

    matcher = CandidateMatcher(top_k_validate=0)
    match_result = matcher.evaluate_candidates(search_result)

    assert match_result.best_candidate is not None
    assert match_result.best_candidate.candidate.title == "Post A"
    assert match_result.total_candidates == 2


def test_handoff_payload_structure(tmp_path):
    """Verify structured output payload creation and saving."""
    input_info = {"filename": "test.jpg", "face_detected": True, "face_count": 1}
    search_info = {"engine": "Google Lens via SerpAPI", "query_type": "reverse_image"}
    best_match = {
        "source": "Instagram",
        "post_url": "https://instagram.com/p/xyz",
        "title": "Verified Post",
        "image_url": "https://example.com/img.jpg",
        "face_similarity": 0.88,
    }

    payload = build_handoff_payload(
        input_info=input_info,
        search_info=search_info,
        best_match=best_match,
        all_candidates=[best_match],
        crop_path="output/processed/face_crop.jpg",
        dry_run=True,
    )

    assert "canonical_payload" in payload
    assert "fingerprint" in payload
    assert "blockchain_handoff" in payload
    assert len(payload["fingerprint"]) == 64

    # Verify fingerprint matches canonical payload
    assert verify_fingerprint(payload["canonical_payload"], payload["fingerprint"])

    # Test saving
    out_file = tmp_path / "result.json"
    save_result_payload(payload, out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["fingerprint"] == payload["fingerprint"]


def test_serpapi_response_parsing():
    """Verify parsing of actual SerpAPI Google Lens response structure."""
    from search.lens import GoogleLensSearcher

    mock_serpapi_data = {
        "search_metadata": {"status": "Success"},
        "visual_matches": [
            {
                "position": 1,
                "title": "Hrithik Roshan Instagram Reel",
                "link": "https://www.instagram.com/reel/xyz123/",
                "source": "Instagram",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/image?q=test1",
            },
            {
                "position": 2,
                "title": "Hrithik Roshan Facebook Post",
                "link": "https://www.facebook.com/posts/456",
                "source": "Facebook",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/image?q=test2",
            },
        ],
        "knowledge_graph": {
            "title": "Hrithik Roshan",
            "link": "https://en.wikipedia.org/wiki/Hrithik_Roshan",
            "source": "Wikipedia",
        },
    }

    searcher = GoogleLensSearcher(api_key="test_key")
    result = searcher._parse_lens_response(mock_serpapi_data, "https://example.com/test.jpg")

    assert result.total_results == 3
    # Check knowledge graph extracted
    assert any(c.source == "Wikipedia" and "wikipedia.org" in c.post_url for c in result.candidates)
    # Check visual matches extracted
    insta_cand = next(c for c in result.candidates if c.source == "Instagram")
    assert insta_cand.title == "Hrithik Roshan Instagram Reel"
    assert insta_cand.post_url == "https://www.instagram.com/reel/xyz123/"


def test_no_match_handoff_behavior():
    """Verify that when 0 matches are found, no false fingerprint is generated."""
    input_info = {"filename": "unknown.jpg", "face_detected": True, "face_count": 1}
    search_info = {"engine": "Google Lens via SerpAPI", "query_type": "reverse_image"}

    payload = build_handoff_payload(
        input_info=input_info,
        search_info=search_info,
        best_match=None,
        all_candidates=[],
        dry_run=False,
    )

    assert payload["search_status"] == "NO_MATCH"
    assert payload["canonical_payload"] is None
    assert payload["fingerprint"] is None
    assert payload["blockchain_handoff"]["status"] == "no_match_found"
    assert payload["blockchain_handoff"]["fingerprint_to_register"] is None


if __name__ == "__main__":
    print("Running tests...")
    test_models_download()
    print("✓ Models downloaded")
    test_deterministic_hashing()
    print("✓ Deterministic hashing validated")
    test_matcher_ranking()
    print("✓ Matcher ranking validated")
    test_serpapi_response_parsing()
    print("✓ SerpAPI parser validated")
    test_no_match_handoff_behavior()
    print("✓ No-match handoff behavior validated")
    import tempfile
    test_handoff_payload_structure(Path(tempfile.gettempdir()))
    print("✓ Handoff payload validated")
    print("ALL TESTS PASSED!")
