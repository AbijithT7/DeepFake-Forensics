"""
Community Forensics ViT Detector (CVPR 2025).

Integrates the official Vision Transformer (ViT-Small) from
buildborderless/CommunityForensics-DeepfakeDet-ViT trained on 2.7M samples
across 4,803 generators for deepfake and generative image detection.

Architecture & Preprocessing:
- Architecture: ViT-Small (patch 16, image_size 384x384, 6 attention heads, intermediate_size 1536)
- Preprocessing: CLIP-style normalization (mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
  Shortest-edge resize to 440, followed by center-crop to 384x384.
- Output: Single-unit logit, sigmoid activation produces probability of FAKE.
  prob_fake = sigmoid(logit)
  prob_real = 1.0 - prob_fake
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_COMMUNITY_VIT_CACHE = PROJECT_ROOT / "models" / "community_forensics"
REPO_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


class CommunityForensicsViTDetector:
    """Standalone inference module for Community Forensics ViT-Small detector."""

    def __init__(
        self,
        model_name_or_path: str = REPO_ID,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Union[str, Path]] = DEFAULT_COMMUNITY_VIT_CACHE,
        local_files_only: bool = False,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_COMMUNITY_VIT_CACHE
        self.model_name_or_path = model_name_or_path
        self.local_files_only = local_files_only

        logger.info(
            "Initializing CommunityForensicsViTDetector on %s (cache: %s)...",
            self.device,
            self.cache_dir,
        )

        try:
            # First attempt: load with local_files_only if requested or if local cache exists
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name_or_path,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )
            self.model = ViTForImageClassification.from_pretrained(
                self.model_name_or_path,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )
        except Exception as err:
            logger.warning(
                "Could not load Community Forensics ViT from local cache (%s). Attempting online download...",
                err,
            )
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name_or_path,
                cache_dir=str(self.cache_dir),
                local_files_only=False,
            )
            self.model = ViTForImageClassification.from_pretrained(
                self.model_name_or_path,
                cache_dir=str(self.cache_dir),
                local_files_only=False,
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info("CommunityForensicsViTDetector loaded successfully.")

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Classify a facial crop or image.

        Args:
            image: PIL Image (RGB) or numpy array (BGR or RGB).

        Returns:
            Dict containing label, uncalibrated score, probabilities, and raw logit.
        """
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty.")
            # Convert BGR to RGB if 3 channels
            if image.ndim == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]
            elif image.ndim == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            else:
                image_rgb = image
            pil_img = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape [1, 1]
            raw_logit = float(logits[0][0].item())
            prob_fake = float(torch.sigmoid(logits[0][0]).item())

        prob_real = 1.0 - prob_fake
        label = "FAKE" if prob_fake >= 0.5 else "REAL"
        score = max(prob_fake, prob_real)

        return {
            "label": label,
            "score": round(score, 4),
            "probabilities": {
                "REAL": round(prob_real, 4),
                "FAKE": round(prob_fake, 4),
            },
            "raw_logits": {
                "logit": round(raw_logit, 4),
            },
            "loaded": True,
            "preprocessing": "CLIP normalization, shortest-edge 440, center crop 384x384",
        }
