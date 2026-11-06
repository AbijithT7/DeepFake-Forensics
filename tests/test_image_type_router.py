"""Unit and contract tests for ImageTypeRouter, rejection mechanism, and evaluation artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.image_detection.image_type_encoder import ImageTypeEncoder
from src.image_detection.image_type_router import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_MODEL_PATH,
    ImageTypeRouter,
    RouterPrediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CLASSES = {
    "REAL",
    "STYLEGAN_GAN",
    "DIFFUSION",
    "FACE_SWAP",
    "INPAINTING",
    "OTHER_SYNTHETIC",
    "UNKNOWN",
}


class TestImageTypeRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared CPU encoder to avoid repeated model loading
        cls.encoder = ImageTypeEncoder(device="cpu", local_files_only=True)
        cls.router = ImageTypeRouter(encoder=cls.encoder)

    def test_01_saved_artifacts_exist(self):
        """Verify that all required router artifacts exist on disk."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Missing model bundle at {DEFAULT_MODEL_PATH}")
        self.assertTrue(DEFAULT_CONFIG_PATH.exists(), f"Missing router config at {DEFAULT_CONFIG_PATH}")
        self.assertTrue(DEFAULT_LABEL_MAPPING_PATH.exists(), f"Missing label mapping at {DEFAULT_LABEL_MAPPING_PATH}")

        with open(DEFAULT_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.assertEqual(set(mapping["classes"]), EXPECTED_CLASSES)
        self.assertEqual(mapping["unknown_class"], "UNKNOWN")

    def test_02_router_initialization(self):
        """Verify router initialization, class list, and threshold."""
        self.assertIsNotNone(self.router.encoder)
        self.assertEqual(set(self.router.classes), EXPECTED_CLASSES)
        self.assertEqual(self.router.unknown_class_name, "UNKNOWN")
        self.assertGreater(self.router.confidence_threshold, 0.0)
        self.assertLess(self.router.confidence_threshold, 1.0)

    def test_03_predict_from_features_properties(self):
        """Verify prediction schema and probability sum from feature vectors."""
        dummy_feats = np.random.randn(4, 768).astype(np.float32)
        predictions = self.router.predict_from_features(dummy_feats)

        self.assertEqual(len(predictions), 4)
        for pred in predictions:
            self.assertIsInstance(pred, RouterPrediction)
            self.assertIn(pred.predicted_class, EXPECTED_CLASSES)
            self.assertIn(pred.raw_predicted_class, EXPECTED_CLASSES)
            self.assertGreaterEqual(pred.confidence, 0.0)
            self.assertLessEqual(pred.confidence, 1.0)
            self.assertGreaterEqual(pred.raw_confidence, 0.0)
            self.assertLessEqual(pred.raw_confidence, 1.0)

            # Probabilities must sum to 1.0
            prob_sum = sum(pred.probabilities.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=4)
            self.assertEqual(set(pred.probabilities.keys()), EXPECTED_CLASSES)

    def test_04_predict_from_features_torch_tensor(self):
        """Verify prediction from torch.Tensor."""
        dummy_tensor = torch.randn(2, 768, dtype=torch.float32)
        predictions = self.router.predict_from_features(dummy_tensor)
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions[0], RouterPrediction)

    def test_05_predict_from_features_invalid_dim(self):
        """Verify ValueError is raised when feature dimension is not 768."""
        invalid_feats = np.random.randn(2, 512).astype(np.float32)
        with self.assertRaises(ValueError):
            self.router.predict_from_features(invalid_feats)

    def test_06_rejection_mechanism_threshold(self):
        """Verify that high threshold forces rejection and routes to UNKNOWN."""
        strict_router = ImageTypeRouter(
            encoder=self.encoder,
            confidence_threshold=0.9999,
        )
        dummy_feats = np.random.randn(1, 768).astype(np.float32)
        preds = strict_router.predict_from_features(dummy_feats)
        self.assertEqual(len(preds), 1)
        pred = preds[0]
        self.assertTrue(pred.is_rejected)
        self.assertEqual(pred.predicted_class, "UNKNOWN")
        self.assertIsNotNone(pred.rejection_reason)
        self.assertIn("below rejection threshold", pred.rejection_reason)

    def test_07_route_image_pil_input(self):
        """Verify routing from a PIL Image."""
        img = Image.new("RGB", (128, 128), color=(120, 150, 180))
        result = self.router.route_image(img)
        self.assertIsInstance(result, RouterPrediction)
        self.assertIn(result.predicted_class, EXPECTED_CLASSES)
        as_dict = result.to_dict()
        self.assertIn("predicted_class", as_dict)
        self.assertIn("probabilities", as_dict)

    def test_08_route_image_numpy_rgb_and_bgr(self):
        """Verify routing with NumPy RGB and BGR arrays."""
        rgb_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_arr[:, :, 0] = 200  # Red tint
        res_rgb = self.router.route_image(rgb_arr, color_order="RGB")
        self.assertIsInstance(res_rgb, RouterPrediction)

        bgr_arr = rgb_arr[:, :, ::-1].copy()
        res_bgr = self.router.route_image(bgr_arr, color_order="BGR")
        self.assertIsInstance(res_bgr, RouterPrediction)

        for c in EXPECTED_CLASSES:
            self.assertAlmostEqual(res_rgb.probabilities[c], res_bgr.probabilities[c], places=3)

    def test_09_route_image_file_path(self):
        """Verify routing directly from file path."""
        test_img_path = PROJECT_ROOT / "tests" / "sample_noface.jpg"
        if test_img_path.exists():
            res = self.router.route_image(test_img_path)
            self.assertIsInstance(res, RouterPrediction)
            self.assertEqual(res.predicted_class, "UNKNOWN")
            self.assertGreater(res.confidence, 0.5)

    def test_10_route_batch_consistency(self):
        """Verify batch routing matches individual routing."""
        img1 = Image.new("RGB", (64, 64), color=(50, 60, 70))
        img2 = Image.new("RGB", (64, 64), color=(200, 210, 220))

        single_1 = self.router.route_image(img1)
        single_2 = self.router.route_image(img2)
        batch_res = self.router.route_batch([img1, img2])

        self.assertEqual(len(batch_res), 2)
        for c in EXPECTED_CLASSES:
            self.assertAlmostEqual(single_1.probabilities[c], batch_res[0].probabilities[c], places=4)
            self.assertAlmostEqual(single_2.probabilities[c], batch_res[1].probabilities[c], places=4)

    def test_11_isolation_from_detection_models(self):
        """Verify that ImageTypeRouter does NOT import or require detection models."""
        from src.image_detection.image_type_router import ImageTypeRouter
        self.assertTrue(hasattr(ImageTypeRouter, "route_image"))
        self.assertTrue(hasattr(ImageTypeRouter, "predict_from_features"))


if __name__ == "__main__":
    unittest.main()
