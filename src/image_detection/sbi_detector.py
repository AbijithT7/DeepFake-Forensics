"""
Self-Blended Images (SBI) EfficientNet-B4 Detector (CVPR 2022).

Integrates the official SBI deepfake detection architecture from:
https://github.com/mapooon/SelfBlendedImages
(Kaede Shiohara & Toshihiko Yamasaki, "Detecting Deepfakes with Self-Blended Images", CVPR 2022 Oral)

Architecture & Preprocessing:
- Backbone: EfficientNet-B4 (efficientnet_pytorch)
- Number of classes: 2 (Class 0: REAL, Class 1: FAKE)
- Resolution: 380x380 (RGB)
- Preprocessing: Resize to 380x380, float division by 255.0 (values in [0.0, 1.0]).
- Checkpoint: Official FF-raw checkpoint (FFraw.tar) trained on Self-Blended Images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SBI_WEIGHTS_PATH = PROJECT_ROOT / "models" / "sbi" / "FFraw.tar"


class SBIEfficientNetB4(nn.Module):
    """PyTorch wrapper for SBI's EfficientNet-B4 architecture."""

    def __init__(self):
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        # Use from_name to instantiate architecture without downloading ImageNet weights
        self.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SBIDetector:
    """
    Standalone inference module for Self-Blended Images (SBI) EfficientNet-B4 detector.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weights_path = Path(weights_path) if weights_path else DEFAULT_SBI_WEIGHTS_PATH

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"SBI pretrained weights not found at: {self.weights_path}. "
                "Ensure FFraw.tar has been downloaded into models/sbi/."
            )

        logger.info(
            "Initializing SBIDetector on %s using weights from %s...",
            self.device,
            self.weights_path,
        )

        self.model = SBIEfficientNetB4()
        checkpoint = torch.load(str(self.weights_path), map_location=self.device)

        # SBI checkpoints store state dict inside dict key 'model'
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("SBIDetector loaded successfully.")

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Resize image to 380x380 and normalize to [0, 1] range as done in SBI inference.

        Args:
            image: PIL Image or OpenCV numpy array.

        Returns:
            torch.Tensor of shape [1, 3, 380, 380] on target device.
        """
        if isinstance(image, Image.Image):
            rgb_arr = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty.")
            if image.ndim == 3 and image.shape[2] == 3:
                # Assume OpenCV BGR if passed as ndarray
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_arr = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # SBI uses 380x380 resolution with bilinear interpolation
        resized = cv2.resize(rgb_arr, (380, 380), interpolation=cv2.INTER_LINEAR)
        # Scaled to [0, 1] as float32
        normalized = resized.astype(np.float32) / 255.0
        # Transpose to (C, H, W) and add batch dimension -> [1, 3, 380, 380]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run SBI deepfake detection on an isolated facial crop.

        Class 0 = REAL
        Class 1 = FAKE

        Returns:
            Standardized dict containing label, score, probabilities, and logits.
        """
        tensor = self.preprocess_image(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item())

        label = "FAKE" if prob_fake >= prob_real else "REAL"
        score = max(prob_real, prob_fake)

        raw_logits_dict = {
            "REAL": round(float(logits[0][0].item()), 4),
            "FAKE": round(float(logits[0][1].item()), 4),
        }

        return {
            "label": label,
            "score": round(score, 4),
            "probabilities": {
                "REAL": round(prob_real, 4),
                "FAKE": round(prob_fake, 4),
            },
            "raw_logits": raw_logits_dict,
            "loaded": True,
            "preprocessing": "380x380 RGB, linear resize, [0, 1] scaling",
        }
