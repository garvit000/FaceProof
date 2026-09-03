"""
Face detection module for FaceProof using OpenCV YuNet (ONNX).
Detects faces, extracts bounding boxes, confidences, and 5-point facial landmarks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import urllib.request
import cv2
import numpy as np

# Default YuNet ONNX model URL and storage path
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceDetectionResult:
    """Structured representation of a single detected face."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: List[Tuple[float, float]]
    raw_array: np.ndarray  # 15-element array required by SFace
    is_primary: bool = False

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "box": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "confidence": round(float(self.confidence), 4),
            "is_primary": self.is_primary,
        }


def ensure_yunet_model(model_path: Optional[Path] = None) -> Path:
    """Ensure the YuNet ONNX model is present locally, downloading if necessary."""
    path = Path(model_path) if model_path else DEFAULT_YUNET_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download YuNet ONNX model
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(path))
    return path


def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load an image from a file path or validate an existing numpy array (BGR).
    Raises FileNotFoundError or ValueError if invalid.
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to decode image from path: {img_path}")
        return image
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0 or len(image_input.shape) < 2:
            raise ValueError("Empty or invalid image numpy array provided.")
        return image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def detect_faces(
    image_input: Union[str, Path, np.ndarray],
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    model_path: Optional[Path] = None,
) -> List[FaceDetectionResult]:
    """
    Detect all faces in an image using OpenCV YuNet.

    Returns:
        List of FaceDetectionResult sorted by face area in descending order.
        If multiple faces are detected, the first element has is_primary=True.
    """
    image = load_image(image_input)
    img_h, img_w = image.shape[:2]

    model_file = ensure_yunet_model(model_path)
    detector = cv2.FaceDetectorYN.create(
        model=str(model_file),
        config="",
        input_size=(img_w, img_h),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return []

    results: List[FaceDetectionResult] = []
    for face in faces:
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        # YuNet provides 5 landmarks: right_eye, left_eye, nose_tip, right_mouth, left_mouth
        landmarks = [
            (float(face[4]), float(face[5])),
            (float(face[6]), float(face[7])),
            (float(face[8]), float(face[9])),
            (float(face[10]), float(face[11])),
            (float(face[12]), float(face[13])),
        ]
        confidence = float(face[14])

        results.append(
            FaceDetectionResult(
                x=max(0, x),
                y=max(0, y),
                width=max(1, w),
                height=max(1, h),
                confidence=confidence,
                landmarks=landmarks,
                raw_array=face,
                is_primary=False,
            )
        )

    # Sort faces by bounding box area descending (largest first)
    results.sort(key=lambda f: f.area, reverse=True)
    if results:
        results[0].is_primary = True

    return results
