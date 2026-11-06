"""
DeepGuard Pretrained Deepfake Detector (MS-EffGCViT-B0).

Integrates the official DeepGuard architecture (Multi-Scale Efficient Global Context
Vision Transformer b0 trained on FaceForensics++) for standalone deepfake evaluation.

Architecture & Preprocessing:
- Backbone: tf_efficientnet_b0.ns_jft_in1k + Multi-Scale GCViT (Local & Global Window Attention)
- Expected Resolution: 224x224 (Isotropic resize with constant zero padding)
- Preprocessing: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Output Mapping:
    Class 0 = REAL
    Class 1 = FAKE
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DEEPGUARD_MODEL_PATH = PROJECT_ROOT / "models" / "deepguard"

LABEL_MAPPING = {
    0: "REAL",
    1: "FAKE",
}


class InvalidCropError(ValueError):
    """Raised when an input crop is empty, corrupted, or has degenerate dimensions."""


class DeepGuardDetector:
    """
    Standalone inference wrapper for the DeepGuard MS-EffGCViT-B0 model.

    Loads model weights once and exposes high-level evaluation methods
    for isolated facial crops or full images (using SCRFD face detection).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_DEEPGUARD_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DeepGuard model weights not found at: {self.model_path}. "
                "Ensure Phase 1 download has completed."
            )

        # Device selection: CUDA if available unless overridden
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Initializing DeepGuardDetector on device: %s", self.device)

        # Load transformers model with custom architecture from local directory
        self.model = AutoModelForImageClassification.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Load official DeepGuard isotropic test transform
        from deepguard.data.transforms import get_test_transforms

        self.transform = get_test_transforms(img_size=[224, 224])
        logger.info("DeepGuardDetector loaded successfully from %s", self.model_path)

    def preprocess_crop(self, face_crop: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Validate and preprocess an isolated face crop using official DeepGuard transforms.

        Args:
            face_crop: RGB numpy array or PIL Image.

        Returns:
            Preprocessed 4D PyTorch tensor [1, 3, 224, 224] on target device.
        """
        if isinstance(face_crop, Image.Image):
            crop_rgb = np.array(face_crop.convert("RGB"))
        elif isinstance(face_crop, np.ndarray):
            if face_crop.size == 0 or face_crop.ndim not in (2, 3):
                raise InvalidCropError(f"Invalid crop dimensions: {getattr(face_crop, 'shape', None)}")
            if face_crop.ndim == 2:
                crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
            elif face_crop.shape[2] == 4:
                crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_RGBA2RGB)
            else:
                crop_rgb = face_crop
        else:
            raise InvalidCropError(f"Unsupported crop type: {type(face_crop)}")

        h, w = crop_rgb.shape[:2]
        if h < 5 or w < 5:
            raise InvalidCropError(f"Crop dimensions ({w}x{h}) are too small for forensic evaluation.")

        try:
            tensor = self.transform(image=crop_rgb)["image"].unsqueeze(0)
            return tensor.to(self.device)
        except Exception as exc:
            raise InvalidCropError(f"Failed to apply DeepGuard transforms: {exc}") from exc

    def predict_crop(self, face_crop: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Classify a single facial crop as REAL or FAKE.

        Args:
            face_crop: Extracted facial crop (RGB numpy array or PIL Image).

        Returns:
            Dictionary containing predicted label, probabilities, and scores.
        """
        try:
            tensor = self.preprocess_crop(face_crop)
        except InvalidCropError as err:
            logger.warning("DeepGuard prediction failed on crop: %s", err)
            return {
                "status": "error",
                "error": str(err),
                "label": "ERROR",
                "score": 0.0,
                "real_score": 0.0,
                "fake_score": 0.0,
                "probabilities": {"REAL": 0.0, "FAKE": 0.0},
                "device": str(self.device),
            }

        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
            logits = outputs.logits  # [1, 2] = [real, fake]
            probs = torch.softmax(logits, dim=1)[0]

        real_prob = float(probs[0].item())
        fake_prob = float(probs[1].item())
        label = "FAKE" if fake_prob >= real_prob else "REAL"
        confidence = max(real_prob, fake_prob)

        return {
            "status": "success",
            "label": label,
            "score": round(confidence, 4),
            "real_score": round(real_prob, 4),
            "fake_score": round(fake_prob, 4),
            "probabilities": {
                "REAL": round(real_prob, 4),
                "FAKE": round(fake_prob, 4),
            },
            "device": str(self.device),
        }

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        face_detector: Optional[Any] = None,
        margin_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Run deepfake analysis on an image file or loaded image array.

        If a face detector is provided (e.g. SCRFDFaceDetector), faces are detected
        and isolated before classification. If no face detector is provided, the input
        is treated as a pre-cropped face.

        Args:
            image_input: Path, PIL Image, or BGR numpy array.
            face_detector: Optional SCRFDFaceDetector instance.
            margin_ratio: Context margin around the face bounding box (default: 0.20).

        Returns:
            Dictionary containing prediction outcomes, detected faces count, and per-face scores.
        """
        # Load image array (BGR standard for OpenCV detection)
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                return {
                    "status": "error",
                    "error": f"Image file not found: {img_path}",
                    "label": "ERROR",
                }
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                return {
                    "status": "error",
                    "error": f"Failed to decode image file: {img_path}",
                    "label": "ERROR",
                }
        elif isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input.convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image_bgr = image_input
        else:
            return {
                "status": "error",
                "error": f"Unsupported image input type: {type(image_input)}",
                "label": "ERROR",
            }

        # If no face detector provided, evaluate image directly as a face crop
        if face_detector is None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = self.predict_crop(image_rgb)
            result["faces_detected"] = 1
            return result

        # Detect faces using SCRFD
        detections = face_detector.detect(image_bgr)
        if not detections:
            return {
                "status": "no_face_detected",
                "faces_detected": 0,
                "label": "INCONCLUSIVE",
                "message": "No face detected in the image.",
                "score": 0.0,
                "real_score": 0.0,
                "fake_score": 0.0,
                "probabilities": {"REAL": 0.0, "FAKE": 0.0},
                "device": str(self.device),
            }

        # Crop and evaluate primary detected face
        primary_box = detections[0]["box"]
        x, y, w, h = primary_box
        img_h, img_w = image_bgr.shape[:2]

        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        crop_bgr = image_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        crop_res = self.predict_crop(crop_rgb)
        crop_res["faces_detected"] = len(detections)
        crop_res["primary_box"] = [x, y, w, h]
        crop_res["padded_box"] = [x1, y1, x2, y2]
        crop_res["detection_score"] = float(detections[0].get("score", 0.0))
        return crop_res
