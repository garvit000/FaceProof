"""
Candidate matching and validation engine for FaceProof.
Evaluates Google Lens candidates using search engine visual ranking and optional
face similarity verification against downloaded candidate thumbnails.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import requests

from face.detector import detect_faces
from face.encoder import encode_face, compute_similarity
from .lens import LensCandidate, LensSearchResult


@dataclass
class CandidateEvaluation:
    """Detailed evaluation record for a single candidate result."""
    candidate: LensCandidate
    visual_rank: int
    face_similarity: Optional[float] = None
    validation_status: str = "pending"  # "face_verified", "face_not_detected", "thumbnail_unavailable", "skipped"
    rank_score: float = 0.0

    def to_dict(self) -> dict:
        data = {
            "source": self.candidate.source,
            "post_url": self.candidate.post_url,
            "title": self.candidate.title,
            "visual_rank": self.visual_rank,
            "validation_status": self.validation_status,
        }
        if self.candidate.image_url:
            data["image_url"] = self.candidate.image_url
        if self.face_similarity is not None:
            data["face_similarity"] = round(self.face_similarity, 4)
        if self.candidate.snippet:
            data["snippet"] = self.candidate.snippet
        return data


@dataclass
class MatchResult:
    """Final outcome of candidate ranking and selection."""
    best_candidate: Optional[CandidateEvaluation]
    all_evaluated: List[CandidateEvaluation] = field(default_factory=list)
    total_candidates: int = 0
    verification_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "best_match": self.best_candidate.to_dict() if self.best_candidate else None,
            "total_evaluated": len(self.all_evaluated),
            "candidates": [c.to_dict() for c in self.all_evaluated],
            "notes": self.verification_notes,
        }


class CandidateMatcher:
    """
    Ranks search candidates using search engine order and optional visual face similarity.
    """

    def __init__(self, top_k_validate: int = 5, request_timeout: int = 6):
        self.top_k_validate = top_k_validate
        self.request_timeout = request_timeout

    def _download_candidate_image(self, url: str) -> Optional[np.ndarray]:
        """Best-effort download of a candidate thumbnail/image as a BGR numpy array."""
        if not url:
            return None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=self.request_timeout)
            if resp.status_code == 200 and resp.content:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                # Convert PIL RGB to OpenCV BGR
                rgb_arr = np.array(img)
                bgr_arr = rgb_arr[:, :, ::-1].copy()
                return bgr_arr
        except Exception:
            return None
        return None

    def evaluate_candidates(
        self,
        search_result: LensSearchResult,
        input_embedding: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """
        Evaluate and rank candidates from Google Lens.

        Ranking strategy:
        1. Candidates are examined in their natural Google Lens visual ranking order.
        2. For the top K candidates with accessible thumbnails, face detection and embedding
           comparison are performed on a best-effort basis.
        3. If face similarity is computed, it boosts confidence in the visual match.
        4. If a candidate image cannot be downloaded or has no detectable face, the candidate is
           retained at its engine rank with validation_status marked accordingly.

        Args:
            search_result: Search result returned by GoogleLensSearcher.
            input_embedding: 128-d face embedding of the input face for similarity comparison.

        Returns:
            MatchResult containing the ranked candidates and best match.
        """
        candidates = search_result.candidates
        if not candidates:
            return MatchResult(
                best_candidate=None,
                all_evaluated=[],
                total_candidates=0,
                verification_notes="No candidate matches discovered by reverse-image search.",
            )

        evaluated: List[CandidateEvaluation] = []

        for idx, cand in enumerate(candidates):
            visual_rank = cand.position if cand.position > 0 else (idx + 1)
            evaluation = CandidateEvaluation(
                candidate=cand,
                visual_rank=visual_rank,
                face_similarity=None,
                validation_status="skipped",
                rank_score=1.0 / (1.0 + 0.1 * visual_rank),  # Base score from search rank
            )

            # Perform best-effort face validation on top K candidates if embedding provided
            if input_embedding is not None and idx < self.top_k_validate:
                target_img_url = cand.image_url or cand.thumbnail_url
                if target_img_url:
                    img_bgr = self._download_candidate_image(target_img_url)
                    if img_bgr is not None:
                        try:
                            cand_faces = detect_faces(img_bgr)
                            if cand_faces:
                                primary_face = cand_faces[0]
                                cand_embedding = encode_face(img_bgr, primary_face)
                                sim = compute_similarity(input_embedding, cand_embedding)
                                evaluation.face_similarity = sim
                                evaluation.validation_status = "face_verified"
                                # Composite rank score combining search position and face similarity
                                evaluation.rank_score = (1.0 / (1.0 + 0.1 * visual_rank)) + (0.5 * max(0.0, sim))
                            else:
                                evaluation.validation_status = "face_not_detected"
                        except Exception:
                            evaluation.validation_status = "face_comparison_error"
                    else:
                        evaluation.validation_status = "thumbnail_unavailable"
                else:
                    evaluation.validation_status = "thumbnail_unavailable"

            evaluated.append(evaluation)

        # Sort candidates: highest rank_score first, then by visual_rank ascending
        evaluated.sort(key=lambda e: (-e.rank_score, e.visual_rank))

        best = evaluated[0] if evaluated else None
        notes = (
            f"Evaluated {len(evaluated)} candidates discovered by Google Lens. "
            f"Best match selected based on search rank and visual verification."
        )

        return MatchResult(
            best_candidate=best,
            all_evaluated=evaluated,
            total_candidates=len(evaluated),
            verification_notes=notes,
        )
