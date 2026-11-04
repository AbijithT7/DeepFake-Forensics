"""Regression tests for inference diagnostics and face-quality false positives."""

import unittest

import cv2
import numpy as np

from src.image_detection.face_detector import validate_face_quality
from src.image_detection.inference import fuse_model_predictions


class TestQualityGateRegressions(unittest.TestCase):
    def test_distributed_dark_pixels_are_not_uniform_black_occlusion(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(40, 210, size=(160, 160), dtype=np.uint8)
        gray[rng.random(gray.shape) < 0.20] = 0
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        usable, reason = validate_face_quality(
            crop, 0.9, (200, 200, 3), (20, 20, 160, 160), blur_threshold=5
        )

        self.assertTrue(usable)
        self.assertIsNone(reason)

    def test_contiguous_black_occlusion_is_still_rejected(self):
        rng = np.random.default_rng(8)
        gray = rng.integers(40, 210, size=(160, 160), dtype=np.uint8)
        gray[20:100, 20:100] = 0
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        usable, reason = validate_face_quality(
            crop, 0.9, (200, 200, 3), (20, 20, 160, 160), blur_threshold=5
        )

        self.assertFalse(usable)
        self.assertEqual(reason, "face_not_usable")


class TestFusionDiagnostics(unittest.TestCase):
    def test_fusion_reports_disagreement_without_forcing_a_label(self):
        result = fuse_model_predictions(
            {"label": "FAKE", "score": 0.999},
            {"label": "REAL", "score": 0.942},
            0.60,
        )

        self.assertEqual(result, {
            "label": "INCONCLUSIVE",
            "score": 0.942,
            "reason": "model_disagreement",
        })


if __name__ == "__main__":
    unittest.main()
