"""
Unit tests for Community Forensics ViT Detector.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.community_forensics_vit import CommunityForensicsViTDetector


class TestCommunityForensicsViT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = CommunityForensicsViTDetector()
        cls.sample_real_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"

    def test_01_initialization_and_device(self):
        """Verify model loads onto the detected device."""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.processor)
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
        self.assertIn("logit", result["raw_logits"])

    def test_03_predict_numpy_array(self):
        """Verify predict works with OpenCV-style BGR numpy arrays."""
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[50:250, 50:250] = 128
        result = self.detector.predict(arr)

        self.assertIn(result["label"], ["REAL", "FAKE"])
        self.assertIn("probabilities", result)

    def test_04_invalid_inputs(self):
        """Verify error handling on invalid input types and empty arrays."""
        with self.assertRaises(ValueError):
            self.detector.predict(np.array([]))

        with self.assertRaises(TypeError):
            self.detector.predict("not_an_image")  # type: ignore


if __name__ == "__main__":
    unittest.main()
