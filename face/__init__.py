"""
Face processing package for FaceProof.
Provides face detection, cropping, and encoding modules.
"""

from .detector import detect_faces, FaceDetectionResult
from .cropper import crop_face, save_face_crop
from .encoder import encode_face, compute_similarity

__all__ = [
    "detect_faces",
    "FaceDetectionResult",
    "crop_face",
    "save_face_crop",
    "encode_face",
    "compute_similarity",
]
