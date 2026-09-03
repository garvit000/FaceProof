"""
Face encoding module for FaceProof using OpenCV SFace (ONNX).
Extracts 128-dimensional L2-normalized facial embeddings and computes similarity metrics.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import urllib.request
import cv2
import numpy as np

from .detector import FaceDetectionResult, load_image

# Official SFace ONNX model URL and storage path
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def ensure_sface_model(model_path: Optional[Path] = None) -> Path:
    """Ensure the SFace ONNX model is present locally, downloading if necessary."""
    path = Path(model_path) if model_path else DEFAULT_SFACE_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download SFace ONNX model
        urllib.request.urlretrieve(SFACE_MODEL_URL, str(path))
    return path


def get_face_recognizer(model_path: Optional[Path] = None) -> cv2.FaceRecognizerSF:
    """Instantiate and return the OpenCV FaceRecognizerSF model."""
    model_file = ensure_sface_model(model_path)
    return cv2.FaceRecognizerSF.create(
        model=str(model_file),
        config="",
    )


def encode_face(
    image_input: Union[str, Path, np.ndarray],
    face: FaceDetectionResult,
    model_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Extract a normalized 128-dimensional face embedding for the detected face.

    Args:
        image_input: Original image (path or numpy array) from which the face was detected.
        face: The FaceDetectionResult containing the face bounding box and landmarks.
        model_path: Optional custom path to the SFace ONNX model.

    Returns:
        128-dimensional numpy float32 array (L2 normalized).
    """
    image = load_image(image_input)
    recognizer = get_face_recognizer(model_path)

    # Align and crop face using detected 5-point facial landmarks
    aligned_face = recognizer.alignCrop(image, face.raw_array)

    # Extract feature vector
    embedding = recognizer.feature(aligned_face)

    # Flatten and return
    embedding = embedding.flatten().astype(np.float32)
    # Ensure L2 normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 128-d face embeddings.

    Returns:
        Cosine similarity in range [-1.0, 1.0], typically [0.0, 1.0] for faces.
        Higher means greater visual facial similarity.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    f1 = embedding1.flatten()
    f2 = embedding2.flatten()
    denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
    if denom == 0:
        return 0.0
    similarity = float(np.dot(f1, f2) / denom)
    # Bound to valid range
    return max(-1.0, min(1.0, round(similarity, 4)))
