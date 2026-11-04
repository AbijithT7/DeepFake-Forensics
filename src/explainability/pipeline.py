"""
DeepFake Forensics: Face-Based Explainability Pipeline & CLI.

Uses the unified DeepFakeInferenceEngine:
1. SCRFD Face Detection with controlled margin cropping.
2. Pre-classification face quality and visibility validation.
3. EfficientNet-B0 inference on detected face crops.
4. Standard Grad-CAM spatial attribution heatmap and clean overlay generation.
5. Never falls back to classifying the full original image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.visualizer import GradCAMVisualizer
from src.image_detection.inference import (
    DEFAULT_UNCERTAINTY_THRESHOLD,
    DeepFakeInferenceEngine,
    ImageInferenceResult,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"


class ExplainabilityPipeline:
    """
    Command-line and module interface for face-based Grad-CAM explainability.
    Wraps DeepFakeInferenceEngine for 100% pipeline consistency.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "auto",
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        colormap: str = "turbo",
        alpha: float = 0.5,
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    ):
        self.output_dir = Path(output_dir)
        self.colormap = colormap
        self.alpha = alpha
        self.uncertainty_threshold = uncertainty_threshold

        torch_device = None
        if isinstance(device, str) and device != "auto":
            torch_device = torch.device(device)
        elif isinstance(device, torch.device):
            torch_device = device

        self.engine = DeepFakeInferenceEngine(device=torch_device, auto_load=True)
        self.visualizer = GradCAMVisualizer(default_colormap=self.colormap, default_alpha=self.alpha)

    def run_image(
        self,
        image_input: Union[str, Path, Image.Image],
        output_subdir: Optional[Union[str, Path]] = None,
        save_artifacts: bool = True,
    ) -> ImageInferenceResult:
        """
        Execute unified SCRFD detection, EfficientNet-B0 prediction, and Grad-CAM generation.
        """
        stem = "sample_image"
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image not found: {p}")
            stem = p.stem
            with Image.open(p) as img:
                orig_pil = img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        target_dir = Path(output_subdir) if output_subdir else self.output_dir / stem
        if save_artifacts:
            target_dir.mkdir(parents=True, exist_ok=True)

        # Run unified inference with Grad-CAM
        result = self.engine.predict_image(
            image_input=image_input,
            uncertainty_threshold=self.uncertainty_threshold,
            include_gradcam=True,
        )

        if save_artifacts:
            # Save original
            orig_path = target_dir / "original.png"
            orig_pil.save(orig_path, format="PNG")

            for pred in result.predictions:
                prefix = f"face_{pred.face_id}_" if result.faces_detected > 1 else ""
                if pred.face_id in self.engine.crop_artifacts:
                    art = self.engine.crop_artifacts[pred.face_id]
                    # Save face crop
                    art.crop_pil.save(target_dir / f"{prefix}face_crop.png", format="PNG")

                    # If Grad-CAM was generated, decode and save
                    if pred.gradcam_heatmap_base64 and pred.gradcam_overlay_base64:
                        # Extract from visualizer using crop artifact
                        target_layer = self.engine.primary_model.get_target_layer_for_gradcam()
                        from src.explainability.gradcam import GradCAM
                        from src.image_detection.efficientnet_model import get_eval_transform

                        transform = get_eval_transform()
                        tensor = transform(art.crop_pil).unsqueeze(0).to(self.engine.device)
                        cam = GradCAM(self.engine.primary_model, target_layer, self.engine.device)
                        cam_res = cam.generate(tensor, target_class="predicted", target_size=art.crop_pil.size)

                        hm_rgb = self.visualizer.apply_colormap(cam_res.heatmap)
                        hm_pil = Image.fromarray(hm_rgb)
                        ov_pil = self.visualizer.create_overlay(art.crop_pil, cam_res.heatmap)

                        hm_pil.save(target_dir / f"{prefix}heatmap.png", format="PNG")
                        ov_pil.save(target_dir / f"{prefix}overlay.png", format="PNG")

                        sbs_pil = self.visualizer.create_side_by_side(art.crop_pil, hm_pil, ov_pil, cam_res)
                        sbs_pil.save(target_dir / f"{prefix}side_by_side.png", format="PNG")

        self._print_summary(result, stem)
        return result

    def _print_summary(self, result: ImageInferenceResult, image_name: str) -> None:
        """Print a structured console summary."""
        print("\n" + "=" * 70)
        print(" DEEPFAKE FORENSICS: EFFICIENTNET-B0 INFERENCE & GRAD-CAM REPORT")
        print("=" * 70)
        print(f" Image:             {image_name}")
        print(f" Faces Detected:    {result.faces_detected} (SCRFD)")
        print(f" Status:            {result.status.upper()}")
        print(f" Device:            {result.execution_device.upper()}")
        print(f" Classifier:        {result.model_name}")
        print("-" * 70)

        for p in result.predictions:
            print(f" Face #{p.face_id}:")
            print(f"   Outcome:         {p.label}")
            if p.score is not None:
                print(f"   Confidence:      {p.score * 100.0:.2f}% ({p.score_type})")
            if p.probabilities:
                print(f"   Softmax Scores:  REAL: {p.probabilities.get('REAL', 0.0) * 100.0:.1f}%, FAKE: {p.probabilities.get('FAKE', 0.0) * 100.0:.1f}%")
            if p.reason:
                print(f"   Status Reason:   {p.reason}")
            if p.box:
                print(f"   Bounding Box:    {p.box}")
            print("-" * 70)
        print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepFake Forensics: SCRFD Face Detection & EfficientNet-B0 Grad-CAM CLI.",
    )
    parser.add_argument("image_pos", nargs="?", default=None, help="Input image path.")
    parser.add_argument("--image", "-i", dest="image_flag", default=None, help="Input image path.")
    parser.add_argument("--device", "-d", dest="device", default="auto", help="Compute device (auto, cpu, cuda).")
    parser.add_argument("--output-dir", "-o", dest="output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--colormap", dest="colormap", default="turbo")
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.5)
    parser.add_argument("--threshold", dest="threshold", type=float, default=DEFAULT_UNCERTAINTY_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    target_image = args.image_flag or args.image_pos

    if not target_image:
        print("[ERROR] Please provide an image to analyze. Example: python -m src.explainability.pipeline --image tests/sample_image_real.jpg")
        sys.exit(1)

    pipeline = ExplainabilityPipeline(
        device=args.device,
        output_dir=args.output_dir,
        colormap=args.colormap,
        alpha=args.alpha,
        uncertainty_threshold=args.threshold,
    )
    pipeline.run_image(image_input=target_image)


if __name__ == "__main__":
    main()
