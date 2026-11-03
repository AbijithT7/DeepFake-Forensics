"""
Occlusion Sensitivity Module for DeepFake Forensics.

Provides a gradient-free, perturbation-based explainability method (Zeiler & Fergus, 2014).
Slides an occlusion patch (neutral gray or black) across the image and records the drop
in target class probability.

Useful for:
1. Validating Grad-CAM heatmaps without gradient/hook artifacts.
2. Determining whether the model truly relies on the face or on background/corners.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

CLASS_NAMES = ["FAKE", "REAL"]


@dataclass
class OcclusionResult:
    """Container for occlusion sensitivity test results."""
    heatmap: np.ndarray  # 2D float32 array in [0.0, 1.0], shape (H, W)
    base_probability: float  # Original probability of target class (0-100)
    max_drop: float  # Maximum observed drop in probability (0-100)
    target_idx: int
    target_class: str
    model_name: str
    patch_size: int
    stride: int
    region_impacts: Optional[Dict[str, float]] = None  # Percentage contribution per anatomical region
    most_influential_region: Optional[str] = None


class OcclusionSensitivity:
    """
    Computes spatial occlusion sensitivity maps by measuring how much
    the model's confidence drops when specific image regions are obscured.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: transforms.Compose,
        device: Optional[torch.device] = None,
        patch_size: int = 32,
        stride: int = 16,
        fill_color: int = 128,  # Neutral gray
    ):
        """
        Args:
            model: PyTorch model in evaluation mode.
            transform: Model-specific preprocessing transform.
            device: Execution device (CPU or CUDA).
            patch_size: Square size of the occluding box in pixels.
            stride: Step size between sliding occlusion patches.
            fill_color: Gray value [0-255] used to mask pixels.
        """
        self.model = model
        self.model.eval()
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.fill_color = fill_color

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    def generate(
        self,
        image: Image.Image,
        target_class: Union[str, int] = "predicted",
        target_size: Optional[Tuple[int, int]] = None,
    ) -> OcclusionResult:
        """
        Run the sliding-window occlusion sensitivity test on a PIL Image.

        Args:
            image: Original PIL Image.
            target_class: 'predicted', 'fake', 'real', 0, or 1.
            target_size: Optional (width, height) to resize resulting heatmap.

        Returns:
            OcclusionResult with normalized sensitivity heatmap and drop statistics.
        """
        rgb_img = image.convert("RGB")
        w_orig, h_orig = rgb_img.size

        # Downscale moderately if image is very large for fast interactive execution
        max_dim = 256
        scale = min(max_dim / max(w_orig, h_orig), 1.0)
        proc_w, proc_h = max(int(w_orig * scale), 64), max(int(h_orig * scale), 64)
        proc_img = rgb_img.resize((proc_w, proc_h), resample=Image.Resampling.BILINEAR)

        # Baseline forward pass
        t_orig = self.transform(proc_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            base_logits = self.model(t_orig)
            base_probs = torch.softmax(base_logits, dim=1)[0]

        # Determine target index
        if str(target_class).lower() in ("0", "fake"):
            target_idx = 0
        elif str(target_class).lower() in ("1", "real"):
            target_idx = 1
        else:
            target_idx = torch.argmax(base_probs).item()

        target_name = CLASS_NAMES[target_idx]
        base_target_prob = base_probs[target_idx].item()

        # Scale patch size proportionally to the processing size
        actual_patch = max(int(self.patch_size * scale), 16)
        actual_stride = max(int(self.stride * scale), 8)

        # Accumulation maps
        sensitivity_map = np.zeros((proc_h, proc_w), dtype=np.float32)
        overlap_counts = np.zeros((proc_h, proc_w), dtype=np.float32)

        img_np = np.array(proc_img)
        max_observed_drop = 0.0

        # Slide occlusion patch across the grid
        for y in range(0, proc_h - actual_patch + 1, actual_stride):
            for x in range(0, proc_w - actual_patch + 1, actual_stride):
                # Mask out patch
                occluded_np = img_np.copy()
                occluded_np[y : y + actual_patch, x : x + actual_patch] = self.fill_color
                occluded_pil = Image.fromarray(occluded_np)

                t_occ = self.transform(occluded_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    occ_logits = self.model(t_occ)
                    occ_probs = torch.softmax(occ_logits, dim=1)[0]
                    occ_prob = occ_probs[target_idx].item()

                # Sensitivity = drop in target probability when this region is obscured
                drop = max(0.0, base_target_prob - occ_prob)
                if drop > max_observed_drop:
                    max_observed_drop = drop

                sensitivity_map[y : y + actual_patch, x : x + actual_patch] += drop
                overlap_counts[y : y + actual_patch, x : x + actual_patch] += 1.0

        # Normalize by overlap counts
        overlap_counts = np.maximum(overlap_counts, 1.0)
        heatmap = sensitivity_map / overlap_counts

        # Min-max normalize to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        # Resize to desired output size (or original image size)
        out_w, out_h = target_size if target_size else (w_orig, h_orig)
        if (heatmap.shape[1], heatmap.shape[0]) != (out_w, out_h):
            pil_hm = Image.fromarray(heatmap)
            pil_hm_resized = pil_hm.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
            heatmap = np.array(pil_hm_resized)

        model_name = getattr(self.model, "__class__", type(self.model)).__name__
        region_impacts, top_region = self._compute_facial_region_impacts(heatmap)

        return OcclusionResult(
            heatmap=heatmap,
            base_probability=base_target_prob * 100.0,
            max_drop=max_observed_drop * 100.0,
            target_idx=target_idx,
            target_class=target_name,
            model_name=model_name,
            patch_size=actual_patch,
            stride=actual_stride,
            region_impacts=region_impacts,
            most_influential_region=top_region,
        )

    def _compute_facial_region_impacts(
        self,
        heatmap: np.ndarray,
    ) -> Tuple[Dict[str, float], str]:
        """
        Segment the face crop into standard anatomical sectors and compute percentage impact.
        """
        h, w = heatmap.shape
        regions_def = {
            "Forehead & Hairline": (0.00, 0.25, 0.15, 0.85),
            "Eyes & Brow Area": (0.25, 0.46, 0.15, 0.85),
            "Nose & Mid-Face": (0.46, 0.68, 0.20, 0.80),
            "Mouth & Lips": (0.68, 0.84, 0.22, 0.78),
            "Chin & Jawline": (0.84, 1.00, 0.18, 0.82),
            "Outer Boundaries & Cheeks": (0.15, 0.85, 0.00, 0.20),
        }

        raw_scores = {}
        for name, (y1_f, y2_f, x1_f, x2_f) in regions_def.items():
            y1, y2 = int(h * y1_f), max(int(h * y2_f), int(h * y1_f) + 1)
            x1, x2 = int(w * x1_f), max(int(w * x2_f), int(w * x1_f) + 1)
            sub_map = heatmap[y1:y2, x1:x2]
            raw_scores[name] = float(np.mean(sub_map)) if sub_map.size > 0 else 0.0

        total = sum(raw_scores.values()) + 1e-6
        percentages = {k: round((v / total) * 100.0, 1) for k, v in raw_scores.items()}
        top_name = max(percentages.items(), key=lambda item: item[1])[0]
        return percentages, top_name
