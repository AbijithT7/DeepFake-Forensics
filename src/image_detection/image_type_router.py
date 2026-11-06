"""Image-Type Router Classifier using frozen DINOv2 embeddings.

Forensic Typology Classifier:
This module classifies image manipulation and generation typology into:
- REAL (authentic portrait)
- STYLEGAN_GAN (StyleGAN generative portrait)
- DIFFUSION (latent / pixel diffusion generation)
- FACE_SWAP (identity swap manipulation)
- INPAINTING (local content synthesis / removal)
- OTHER_SYNTHETIC (non-photorealistic or alternative synthetic generators)
- UNKNOWN (out-of-distribution, ambiguous, or low-confidence inputs)

CRITICAL ARCHITECTURAL BOUNDARY:
- Image-type classification determines generator/manipulation typology only.
- It does NOT determine whether an image is authentic (REAL vs. FAKE).
- It does NOT modify or replace specialist detector models or fusion logic.
- Importing this module does NOT load or call any detection models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from PIL import Image

from src.image_detection.image_type_encoder import ImageInput, ImageTypeEncoder

DEFAULT_ROUTER_DIR = Path(__file__).resolve().parents[2] / "models" / "image_type_router"
DEFAULT_MODEL_PATH = DEFAULT_ROUTER_DIR / "router_model.joblib"
DEFAULT_CONFIG_PATH = DEFAULT_ROUTER_DIR / "router_config.json"
DEFAULT_LABEL_MAPPING_PATH = DEFAULT_ROUTER_DIR / "label_mapping.json"


@dataclass
class RouterPrediction:
    """Structured routing outcome for a single input image."""

    predicted_class: str
    raw_predicted_class: str
    confidence: float
    raw_confidence: float
    probabilities: dict[str, float]
    is_rejected: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "raw_predicted_class": self.raw_predicted_class,
            "confidence": float(self.confidence),
            "raw_confidence": float(self.raw_confidence),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "is_rejected": bool(self.is_rejected),
            "rejection_reason": self.rejection_reason,
        }


class ImageTypeRouter:
    """Routes images to manipulation/generator categories based on DINOv2 features.

    Args:
        encoder: Pre-initialized ImageTypeEncoder instance, or None to construct on demand.
        model_path: Path to serialized trained classifier bundle (.joblib).
        config_path: Path to router configuration (.json).
        device: Device for the encoder ('cpu', 'cuda', or None for auto).
        confidence_threshold: Minimum predicted probability to accept a specific class.
            Predictions below this threshold are routed to UNKNOWN.
            If None, loaded from configuration.
    """

    def __init__(
        self,
        encoder: ImageTypeEncoder | None = None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | torch.device | None = None,
        confidence_threshold: float | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Router model bundle not found at {self.model_path}")

        bundle = joblib.load(self.model_path)
        self.classifier = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.feature_preprocessing = bundle.get("feature_preprocessing", "norm")
        self.classes: list[str] = list(bundle["class_names"])
        self.unknown_class_name: str = bundle.get("unknown_class_name", "UNKNOWN")

        default_threshold = float(bundle.get("confidence_threshold", 0.45))
        self.confidence_threshold = float(confidence_threshold) if confidence_threshold is not None else default_threshold

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ImageTypeEncoder(device=device, local_files_only=local_files_only)

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Apply feature transformation (L2 normalization or scaling) matching training."""
        if self.feature_preprocessing == "norm":
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return features / norms
        elif self.feature_preprocessing == "scaled" and self.scaler is not None:
            return self.scaler.transform(features)
        return features

    def predict_from_features(self, features: np.ndarray | torch.Tensor) -> list[RouterPrediction]:
        """Predict image types from precomputed (N, 768) DINOv2 features."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        if features.ndim == 1:
            features = features[np.newaxis, :]

        if features.shape[1] != 768:
            raise ValueError(f"Expected 768-dimensional features, got {features.shape[1]}")

        proc_features = self._preprocess_features(features)
        probs_matrix = self.classifier.predict_proba(proc_features)

        results: list[RouterPrediction] = []
        for i in range(len(features)):
            probs = probs_matrix[i]
            raw_idx = int(np.argmax(probs))
            raw_class = self.classes[raw_idx]
            raw_conf = float(probs[raw_idx])

            prob_dict = {c: float(probs[j]) for j, c in enumerate(self.classes)}

            if raw_conf < self.confidence_threshold:
                # Low confidence rejection: safely route to UNKNOWN
                pred_class = self.unknown_class_name
                conf = float(prob_dict.get(self.unknown_class_name, raw_conf))
                is_rejected = True
                rejection_reason = (
                    f"Confidence {raw_conf:.3f} for '{raw_class}' is below rejection threshold "
                    f"{self.confidence_threshold:.3f}; routed to {self.unknown_class_name}."
                )
            else:
                pred_class = raw_class
                conf = raw_conf
                is_rejected = False
                rejection_reason = None

            results.append(
                RouterPrediction(
                    predicted_class=pred_class,
                    raw_predicted_class=raw_class,
                    confidence=conf,
                    raw_confidence=raw_conf,
                    probabilities=prob_dict,
                    is_rejected=is_rejected,
                    rejection_reason=rejection_reason,
                )
            )

        return results

    def route_image(
        self,
        image: ImageInput,
        *,
        color_order: str = "RGB",
    ) -> RouterPrediction:
        """Route a single image input (file path, PIL Image, or NumPy array)."""
        return self.route_batch([image], color_order=color_order)[0]

    def route_batch(
        self,
        images: Sequence[ImageInput],
        *,
        batch_size: int = 16,
        color_order: str = "RGB",
    ) -> list[RouterPrediction]:
        """Route a batch of images through feature extraction and classification."""
        features = self.encoder.extract_features(
            images,
            batch_size=batch_size,
            color_order=color_order,
            normalize=False,
        )
        return self.predict_from_features(features)
