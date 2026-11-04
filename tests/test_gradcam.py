"""
Automated Test Suite for EfficientNet-B0 Grad-CAM Explainability Module.

Tests:
1. EfficientNet-B0 checkpoint loading without modification (eval mode).
2. Target convolutional layer detection (backbone.features[8]).
3. Grad-CAM mathematical execution: activations, gradients, global pooling, ReLU, min-max norm.
4. Target class selection ('predicted', 'real', 'fake') producing valid attributions.
5. Hook cleanup and absence of memory leaks.
6. Spatial heatmap alignment across multiple resolutions and aspect ratios.
7. Visualizer outputs: colormap, intensity-modulated overlay (transparent inactive regions), side-by-side.
8. End-to-end explainability pipeline execution on REAL and FAKE test faces.
9. End-to-end CLI execution via python -m src.explainability.pipeline.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import GradCAM, GradCAMResult, find_target_layer
from src.explainability.visualizer import GradCAMVisualizer
from src.explainability.pipeline import ExplainabilityPipeline
from src.image_detection.efficientnet_model import (
    CLASS_NAMES,
    DeepfakeEfficientNetB0,
    get_eval_transform,
)
from src.image_detection.predict_efficientnet import load_efficientnet_model


class TestGradCAMModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model, _ = load_efficientnet_model(device=cls.device)
        cls.transform = get_eval_transform()
        cls.real_img_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        cls.fake_img_path = PROJECT_ROOT / "tests" / "sample_image_fake.jpg"

    def test_01_checkpoint_loading_and_eval_mode(self):
        """Verify EfficientNet-B0 checkpoint loads cleanly and is in evaluation mode."""
        self.assertIsInstance(self.model, DeepfakeEfficientNetB0)
        self.assertFalse(self.model.training, "EfficientNet-B0 must be in eval mode.")

    def test_02_target_layer_identification(self):
        """Verify target convolutional layer is accurately resolved to features[8]."""
        target_layer, layer_name = find_target_layer(self.model)
        self.assertIs(target_layer, self.model.get_target_layer_for_gradcam())
        self.assertIn("features", layer_name)

    def test_03_gradcam_gradient_calculation(self):
        """Verify Grad-CAM forward/backward hooks, global average pooling, and ReLU."""
        cam_engine = GradCAM(self.model, device=self.device)
        x = torch.randn(1, 3, 224, 224)
        res = cam_engine.generate(x, target_class="predicted", target_size=(224, 224))

        self.assertIsInstance(res, GradCAMResult)
        self.assertEqual(res.heatmap.shape, (224, 224))
        self.assertGreaterEqual(float(res.heatmap.min()), 0.0)
        self.assertLessEqual(float(res.heatmap.max()), 1.0)
        self.assertTrue(np.any(res.heatmap > 0.0), "Heatmap must contain non-zero activations.")
        self.assertIn(res.predicted_class, CLASS_NAMES)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 100.0)

    def test_04_target_class_selection(self):
        """Verify target class selection ('predicted', 'real', 'fake') operates correctly."""
        cam_engine = GradCAM(self.model, device=self.device)
        x = torch.randn(1, 3, 224, 224)

        res_fake = cam_engine.generate(x, target_class="fake", target_size=(100, 100))
        self.assertEqual(res_fake.target_class, "FAKE")
        self.assertEqual(res_fake.target_idx, 0)

        res_real = cam_engine.generate(x, target_class="real", target_size=(100, 100))
        self.assertEqual(res_real.target_class, "REAL")
        self.assertEqual(res_real.target_idx, 1)

    def test_05_hook_cleanup_and_no_memory_leak(self):
        """Verify forward and backward hooks are deregistered and caches cleared."""
        cam_engine = GradCAM(self.model, device=self.device)
        x = torch.randn(1, 3, 224, 224)
        _ = cam_engine.generate(x, target_class="predicted", target_size=(100, 100))

        self.assertEqual(len(cam_engine._hook_handles), 0, "Hook handles must be emptied.")
        self.assertIsNone(cam_engine.activations, "Activations must be cleared.")
        self.assertIsNone(cam_engine.gradients, "Gradients must be cleared.")

        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.all(param.grad == 0), "Gradients must be zeroed.")

    def test_06_spatial_alignment_and_resolutions(self):
        """Verify Grad-CAM resizes correctly across arbitrary aspect ratios."""
        cam_engine = GradCAM(self.model, device=self.device)
        test_shapes = [(128, 128), (200, 150), (320, 240), (512, 384)]

        for w, h in test_shapes:
            x = torch.randn(1, 3, 224, 224)
            res = cam_engine.generate(x, target_class="predicted", target_size=(w, h))
            self.assertEqual(res.heatmap.shape, (h, w))

    def test_07_visualizer_components_and_transparent_overlay(self):
        """Verify visualizer creates colormaps, side-by-side, and transparent inactive overlay."""
        viz = GradCAMVisualizer(default_colormap="turbo", default_alpha=0.5)

        base_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        # Create heatmap with zero region and active peak region
        heatmap = np.zeros((200, 200), dtype=np.float32)
        heatmap[50:150, 50:150] = 1.0  # active center

        # Test colormap
        rgb = viz.apply_colormap(heatmap)
        self.assertEqual(rgb.shape, (200, 200, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        # Test overlay transparency on inactive regions
        overlay = viz.create_overlay(base_img, heatmap, alpha=0.5)
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, (200, 200))

        ov_arr = np.array(overlay)
        base_arr = np.array(base_img)

        # At corner (10, 10), heatmap == 0.0 -> overlay MUST match base image exactly (zero tint!)
        np.testing.assert_array_equal(
            ov_arr[10, 10],
            base_arr[10, 10],
            err_msg="Inactive heatmap region must remain 100% transparent with no purple tint.",
        )

        # At center (100, 100), heatmap == 1.0 -> overlay must be blended with active colormap
        self.assertFalse(
            np.array_equal(ov_arr[100, 100], base_arr[100, 100]),
            "Active region must show blended attention.",
        )

        # Test side-by-side
        res = GradCAMResult(
            heatmap=heatmap,
            raw_cam=heatmap,
            predicted_idx=0,
            predicted_class="FAKE",
            target_idx=0,
            target_class="FAKE",
            probabilities={"FAKE": 95.0, "REAL": 5.0},
            confidence=95.0,
            target_score=3.5,
            layer_name="backbone.features[8]",
            model_name="DeepfakeEfficientNetB0",
        )
        sbs = viz.create_side_by_side(base_img, Image.fromarray(rgb), overlay, res)
        self.assertIsInstance(sbs, Image.Image)
        self.assertGreater(sbs.size[0], sbs.size[1])

    def test_08_pipeline_execution_on_test_faces(self):
        """Verify ExplainabilityPipeline executes cleanly on REAL and FAKE test faces."""
        pipeline = ExplainabilityPipeline(device="cpu", colormap="turbo", alpha=0.5)

        # Test Real face
        res_real = pipeline.run_image(self.real_img_path, save_artifacts=False)
        self.assertEqual(res_real.status, "success")
        self.assertEqual(res_real.predictions[0].label, "REAL")

        # Test Fake face
        res_fake = pipeline.run_image(self.fake_img_path, save_artifacts=False)
        self.assertEqual(res_fake.status, "inconclusive")
        self.assertEqual(res_fake.predictions[0].label, "INCONCLUSIVE")
        self.assertEqual(res_fake.predictions[0].fusion["reason"], "model_disagreement")

    def test_09_cli_execution(self):
        """Verify CLI executes cleanly via python -m src.explainability.pipeline."""
        cmd = [
            sys.executable,
            "-m",
            "src.explainability.pipeline",
            "--image",
            str(self.real_img_path),
            "--device",
            "cpu",
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI command failed:\n{proc.stderr}")
        self.assertIn("DEEPFAKE FORENSICS", proc.stdout)
        self.assertIn("EFFICIENTNET-B0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
