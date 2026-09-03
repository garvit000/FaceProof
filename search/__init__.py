"""
Search package for FaceProof.
Provides Google Lens reverse image search and candidate matching.
"""

from .lens import GoogleLensSearcher, LensCandidate, LensSearchResult
from .matcher import CandidateMatcher, MatchResult, CandidateEvaluation

__all__ = [
    "GoogleLensSearcher",
    "LensCandidate",
    "LensSearchResult",
    "CandidateMatcher",
    "MatchResult",
    "CandidateEvaluation",
]
