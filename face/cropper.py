"""
Face cropping module for FaceProof.
Produces padded face crops around detected face regions, preserving contextual features.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import cv2
import numpy as np

from .detector import FaceDetectionResult, load_image


def crop_face(
    image_input: Union[str, Path, np.ndarray],
    face: FaceDetectionResult,
    padding_ratio: float = 0.25,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a padded crop of the detected face from the image.

    Args:
        image_input: Input image path or numpy array.
        face: The FaceDetectionResult containing the face bounding box.
        padding_ratio: Fraction of width/height to add as padding on each side (default 0.25).

    Returns:
        Tuple of (cropped_image_array, (crop_x1, crop_y1, crop_x2, crop_y2))
    """
    image = load_image(image_input)
    img_h, img_w = image.shape[:2]

    # Calculate padding amounts
    pad_w = int(face.width * padding_ratio)
    pad_h = int(face.height * padding_ratio)

    # Calculate clamped bounding box with padding
    x1 = max(0, face.x - pad_w)
    y1 = max(0, face.y - pad_h)
    x2 = min(img_w, face.x + face.width + pad_w)
    y2 = min(img_h, face.y + face.height + pad_h)

    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def save_face_crop(
    crop_image: np.ndarray,
    output_path: Union[str, Path] = "output/processed/face_crop.jpg",
) -> Path:
    """
    Save the cropped face image to disk.

    Args:
        crop_image: Cropped face image numpy array (BGR).
        output_path: Target file path.

    Returns:
        Path object of the saved file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_p), crop_image)
    if not success:
        raise IOError(f"Failed to write face crop to: {out_p}")
    return out_p
