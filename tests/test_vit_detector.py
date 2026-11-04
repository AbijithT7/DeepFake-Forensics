"""
Tests for the Hugging Face ViT detector label mapping and model loading.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.vit_detector import ViTDetector, resolve_label_mapping


class TestViTLabelMapping(unittest.TestCase):
    def test_exact_model_configuration_mapping(self):
        """The model's id2label={0: Realism, 1: Deepfake} maps to REAL/FAKE."""
        mapping = resolve_label_mapping({0: "Realism", 1: "Deepfake"})

        self.assertEqual(mapping, {0: "REAL", 1: "FAKE"})

    def test_case_insensitive_mapping(self):
        mapping = resolve_label_mapping({"0": "rEaLiSm", "1": "dEePfAkE"})

        self.assertEqual(mapping, {0: "REAL", 1: "FAKE"})

    def test_rejects_duplicate_real_classes(self):
        with self.assertRaises(ValueError):
            resolve_label_mapping({0: "Realism", 1: "Real"})

    def test_rejects_duplicate_fake_classes(self):
        with self.assertRaises(ValueError):
            resolve_label_mapping({0: "Deepfake", 1: "Fake"})

    def test_rejects_unrecognized_label(self):
        with self.assertRaises(ValueError):
            resolve_label_mapping({0: "Realism", 1: "Unknown"})


class TestViTDetectorLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ViTDetector(local_files_only=True)

    def test_vit_loads_successfully(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.processor)

    def test_loaded_model_uses_configured_label_mapping(self):
        self.assertEqual(self.detector.id2label, {0: "REAL", 1: "FAKE"})
        self.assertEqual(self.detector.real_index, 0)
        self.assertEqual(self.detector.fake_index, 1)

    def test_prediction_returns_canonical_label(self):
        image_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        with Image.open(image_path) as image:
            result = self.detector.predict(image.convert("RGB"))

        self.assertIn(result["label"], ("REAL", "FAKE"))
        self.assertEqual(set(result["probabilities"]), {"REAL", "FAKE"})
        self.assertEqual(result["id2label"], {0: "REAL", 1: "FAKE"})


if __name__ == "__main__":
    unittest.main()
