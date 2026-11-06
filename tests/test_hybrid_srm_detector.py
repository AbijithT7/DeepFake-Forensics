"""
Unit tests for Hybrid SRM Deepfake Detector (AdityaManojShinde/deepfake-detector).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.hybrid_srm_detector import (
    HybridDeepfakeDetector,
    HybridDeepfakeDetectorModel,
    SRMLayer,
)


class TestHybridSRMDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = HybridDeepfakeDetector()
        cls.sample_real_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        cls.sample_fake_path = PROJECT_ROOT / "tests" / "sample_image_fake.jpg"

    def test_01_initialization_and_device(self):
        """Verify model loads onto the detected device."""
        self.assertIsNotNone(self.detector.model)
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(self.detector.device.type, expected_type)

    def test_02_predict_pil_image(self):
        """Verify predict works with a PIL image and returns valid schema."""
        img = Image.open(self.sample_real_path).convert("RGB")
        result = self.detector.predict(img)

        self.assertIn(result["label"], ["REAL", "FAKE"])
        self.assertGreaterEqual(result["score"], 0.50)
        self.assertLessEqual(result["score"], 1.00)
        self.assertIn("REAL", result["probabilities"])
        self.assertIn("FAKE", result["probabilities"])
        self.assertAlmostEqual(
            result["probabilities"]["REAL"] + result["probabilities"]["FAKE"],
            1.0,
            places=3,
        )
        self.assertTrue(result["loaded"])
        self.assertIn("raw_logits", result)
        self.assertIn("prob", result["raw_logits"])

    def test_03_predict_numpy_array(self):
        """Verify predict works with OpenCV-style BGR numpy arrays."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[40:180, 40:180] = 150
        result = self.detector.predict(arr)

        self.assertIn(result["label"], ["REAL", "FAKE"])
        self.assertIn("probabilities", result)
        self.assertTrue(result["loaded"])

    def test_04_preprocessing_tensor_shape(self):
        """Verify preprocessing outputs [1, 3, 224, 224] tensor on expected device."""
        dummy = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        tensor = self.detector.preprocess_image(dummy)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.device.type, self.detector.device.type)

    def test_05_srm_layer_filters(self):
        """Verify SRM layer filters are fixed with frozen weights."""
        srm = SRMLayer()
        self.assertEqual(srm.conv.weight.shape, (3, 1, 5, 5))
        self.assertFalse(srm.conv.weight.requires_grad)

        # Forward pass on 3-channel dummy image
        dummy_img = torch.randn(2, 3, 64, 64)
        noise_res = srm(dummy_img)
        self.assertEqual(noise_res.shape, (2, 3, 64, 64))

    def test_06_model_architecture_streams(self):
        """Verify dual streams produce 3840 concatenated features."""
        model = HybridDeepfakeDetectorModel()
        dummy_input = torch.randn(1, 3, 224, 224)

        spatial_feat = model.spatial(dummy_input)
        self.assertEqual(spatial_feat.shape, (1, 1792))

        noise_maps = model.srm(dummy_input)
        freq_feat = model.frequency(noise_maps)
        self.assertEqual(freq_feat.shape, (1, 2048))

        out = model(dummy_input)
        self.assertEqual(out.shape, (1, 1))
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_07_invalid_inputs(self):
        """Verify error handling on invalid input types and empty arrays."""
        with self.assertRaises(ValueError):
            self.detector.predict(np.array([]))

        with self.assertRaises(TypeError):
            self.detector.predict("not_an_image")  # type: ignore

    def test_08_missing_weights_handling(self):
        """Verify FileNotFoundError is raised when local weights do not exist and local_files_only=True."""
        with self.assertRaises(FileNotFoundError):
            HybridDeepfakeDetector(
                weights_path=PROJECT_ROOT / "non_existent_weights.pth",
                local_files_only=True,
            )

    def test_09_predict_sample_real_and_fake(self):
        """Verify inference on benchmark real and fake images."""
        if self.sample_real_path.is_file():
            img_real = Image.open(self.sample_real_path).convert("RGB")
            res_real = self.detector.predict(img_real)
            self.assertEqual(res_real["label"], "REAL")
            self.assertGreater(res_real["probabilities"]["REAL"], 0.80)

        if self.sample_fake_path.is_file():
            img_fake = Image.open(self.sample_fake_path).convert("RGB")
            res_fake = self.detector.predict(img_fake)
            self.assertEqual(res_fake["label"], "FAKE")
            self.assertGreater(res_fake["probabilities"]["FAKE"], 0.80)


if __name__ == "__main__":
    unittest.main()
