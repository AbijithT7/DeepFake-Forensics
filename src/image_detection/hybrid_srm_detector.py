"""
Hybrid Deepfake Detector (Phase 2): EfficientNet-B4 + SRM Xception Dual-Stream CNN.

Pretrained model from Hugging Face:
https://huggingface.co/AdityaManojShinde/deepfake-detector
(Aditya Manoj Shinde, "Deepfake Face Detector")

Architecture & Preprocessing:
- Spatial Stream: EfficientNet-B4 pretrained on ImageNet (outputs 1792 features).
- Frequency Stream: Xception with Steganalysis Rich Model (SRM) fixed 5x5 filters
  applied to grayscale pixel residuals to isolate high-frequency GAN noise (outputs 2048 features).
- Fusion: Concatenation of spatial + frequency features (1792 + 2048 = 3840 features).
- Classifier: Linear(3840, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 1) -> Sigmoid.
- Resolution: 224x224 RGB.
- Normalization: ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
- Label Mapping:
    prob = model(tensor).item()
    Output: Sigmoid probability in range [0.0, 1.0].
    Class 0 (prob -> 0) = FAKE
    Class 1 (prob -> 1) = REAL
- Training Dataset: xhlulu/140k-real-and-fake-faces (StyleGAN2 vs FFHQ).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_detector" / "deepfake_detector_phase2.pth"
HF_REPO_ID = "AdityaManojShinde/deepfake-detector"
HF_FILENAME = "deepfake_detector_phase2.pth"


class SRMLayer(nn.Module):
    """
    Steganalysis Rich Model (SRM) filtering layer for frequency/residual feature extraction.
    Applies three fixed 5x5 high-pass filters to a grayscale image to capture noise residuals.
    """

    def __init__(self):
        super().__init__()
        f1 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        f2 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        )
        f3 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        filters = torch.stack([f1, f2, f3]).unsqueeze(1)
        self.conv = nn.Conv2d(1, 3, kernel_size=5, padding=2, bias=False)
        self.conv.weight.data = filters
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard ITU-R BT.601 luma conversion
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return self.conv(gray)


class HybridDeepfakeDetectorModel(nn.Module):
    """
    Dual-stream hybrid CNN combining spatial (EfficientNet-B4) and frequency (SRM + Xception)
    streams for facial deepfake detection.
    """

    def __init__(self):
        super().__init__()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="timm.models._factory")
            self.spatial = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
            self.srm = SRMLayer()
            self.frequency = timm.create_model("xception", pretrained=False, num_classes=0, in_chans=3)

        self.classifier = nn.Sequential(
            nn.Linear(3840, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial(x)
        noise_maps = self.srm(x)
        freq_features = self.frequency(noise_maps)
        combined = torch.cat([spatial_features, freq_features], dim=1)
        return self.classifier(combined)


class HybridDeepfakeDetector:
    """
    Standalone inference module for the Hybrid SRM Deepfake Detector
    (AdityaManojShinde/deepfake-detector).
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        local_files_only: bool = False,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weights_path = Path(weights_path) if weights_path else DEFAULT_HYBRID_MODEL_PATH
        self.local_files_only = local_files_only

        logger.info("Initializing HybridDeepfakeDetector on %s...", self.device)

        if not self.weights_path.is_file():
            if self.local_files_only:
                raise FileNotFoundError(
                    f"Hybrid detector weights not found locally at: {self.weights_path}"
                )
            logger.info(
                "Checkpoint not found at %s. Attempting download from %s...",
                self.weights_path,
                HF_REPO_ID,
            )
            try:
                from huggingface_hub import hf_hub_download

                downloaded_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_FILENAME,
                )
                self.weights_path = Path(downloaded_path)
            except Exception as dl_err:
                raise FileNotFoundError(
                    f"Failed to find or download Hybrid detector weights from {HF_REPO_ID}: {dl_err}"
                ) from dl_err

        # Instantiate architecture
        self.model = HybridDeepfakeDetectorModel()

        # Load weights
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info("HybridDeepfakeDetector loaded successfully from %s.", self.weights_path)

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Validate, resize to 224x224 and normalize using ImageNet parameters.

        Args:
            image: PIL Image or OpenCV numpy array.

        Returns:
            torch.Tensor of shape [1, 3, 224, 224] on target device.
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty.")
            if image.ndim == 3 and image.shape[2] == 3:
                # Standard OpenCV BGR -> RGB
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[2] == 4:
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_arr = image
            pil_img = Image.fromarray(rgb_arr)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = self.transform(pil_img).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Run deepfake detection on an isolated facial crop.

        Model outputs sigmoid probability (prob):
        - prob -> 1.0 indicates REAL
        - prob -> 0.0 indicates FAKE

        Returns:
            Standardized dict containing label, score, probabilities, and raw prob.
        """
        tensor = self.preprocess_image(image)

        with torch.no_grad():
            prob = float(self.model(tensor).item())

        prob_real = prob
        prob_fake = 1.0 - prob

        label = "REAL" if prob_real > 0.5 else "FAKE"
        score = max(prob_real, prob_fake)

        return {
            "label": label,
            "score": round(score, 4),
            "probabilities": {
                "REAL": round(prob_real, 4),
                "FAKE": round(prob_fake, 4),
            },
            "raw_logits": {
                "prob": round(prob, 4),
            },
            "loaded": True,
            "preprocessing": "224x224 RGB, ImageNet mean/std normalization",
        }
