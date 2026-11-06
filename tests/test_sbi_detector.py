"""
Unit tests for Self-Blended Images (SBI) EfficientNet-B4 Detector.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.sbi_detector import SBIDetector


class TestSBIDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = SBIDetector()
        cls.sample_real_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"

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
        self.assertIn("REAL", result["raw_logits"])
        self.assertIn("FAKE", result["raw_logits"])

    def test_03_preprocessing_tensor_shape(self):
        """Verify preprocessing outputs [1, 3, 380, 380] in range [0, 1]."""
        dummy = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        tensor = self.detector.preprocess_image(dummy)
        self.assertEqual(tensor.shape, (1, 3, 380, 380))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_04_missing_weights_handling(self):
        """Verify FileNotFoundError is raised when weights file does not exist."""
        with self.assertRaises(FileNotFoundError):
            SBIDetector(weights_path=PROJECT_ROOT / "non_existent_weights.pth")

    def test_05_invalid_inputs(self):
        """Verify error handling on invalid input types and empty arrays."""
        with self.assertRaises(ValueError):
            self.detector.predict(np.array([]))

        with self.assertRaises(TypeError):
            self.detector.predict(12345)  # type: ignore


if __name__ == "__main__":
    unittest.main()
