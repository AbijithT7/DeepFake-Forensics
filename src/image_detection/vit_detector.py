"""
Hugging Face ViT secondary detector for DeepFake Forensics.

The model configuration exposes human-readable class names (for example
``{0: "Realism", 1: "Deepfake"}``). This module derives the REAL/FAKE index
mapping from the model configuration instead of relying on hardcoded indices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_VIT_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_VIT_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

LABEL_ALIASES = {
    "realism": "REAL",
    "real": "REAL",
    "deepfake": "FAKE",
    "fake": "FAKE",
}


def resolve_label_mapping(id2label: Dict[Union[int, str], str]) -> Dict[int, str]:
    """
    Map the model configuration's ``id2label`` entries to canonical REAL/FAKE labels.

    Matching is case-insensitive. Exactly one REAL class and one FAKE class must be
    present, and the mapping is validated before being returned.
    """
    mapping: Dict[int, str] = {}
    for raw_index, raw_label in id2label.items():
        index = int(raw_index)
        canonical = LABEL_ALIASES.get(str(raw_label).strip().lower())
        if canonical is None:
            raise ValueError(
                f"Unrecognized ViT class label {raw_label!r} at index {index}. "
                f"Expected a REAL/FAKE equivalent."
            )
        mapping[index] = canonical

    real_indices = [index for index, label in mapping.items() if label == "REAL"]
    fake_indices = [index for index, label in mapping.items() if label == "FAKE"]
    if len(real_indices) != 1 or len(fake_indices) != 1:
        raise ValueError(
            "ViT label mapping must contain exactly one REAL and one FAKE class; "
            f"got {mapping}"
        )

    return mapping


class ViTDetector:
    """Load the Hugging Face ViT model and classify images as REAL or FAKE."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_VIT_MODEL_NAME,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Union[str, Path]] = DEFAULT_VIT_CACHE_DIR,
        local_files_only: bool = False,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.processor = ViTImageProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        self.model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = resolve_label_mapping(self.model.config.id2label)
        self.real_index = next(
            index for index, label in self.id2label.items() if label == "REAL"
        )
        self.fake_index = next(
            index for index, label in self.id2label.items() if label == "FAKE"
        )

    def predict(self, image: Image.Image) -> Dict[str, object]:
        """Classify a PIL image and return the canonical label with probabilities."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0]

        fake_prob = float(probs[self.fake_index].item())
        real_prob = float(probs[self.real_index].item())
        label = "REAL" if real_prob >= fake_prob else "FAKE"

        return {
            "label": label,
            "score": round(max(real_prob, fake_prob), 4),
            "probabilities": {
                "FAKE": round(fake_prob, 4),
                "REAL": round(real_prob, 4),
            },
            "id2label": dict(self.id2label),
        }
