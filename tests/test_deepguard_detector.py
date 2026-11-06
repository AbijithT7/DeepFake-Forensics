"""Tests for the standalone DeepGuard detector module."""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.image_detection.deepguard_detector import DeepGuardDetector, InvalidCropError
from src.image_detection.face_detector import SCRFDFaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDeepGuardDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = DeepGuardDetector()
        cls.face_detector = SCRFDFaceDetector()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.transform)
        self.assertIn(str(self.detector.device), ["cpu", "cuda"])

    def test_predict_crop_valid_image(self):
        sample_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        pil_img = Image.open(sample_path)
        res = self.detector.predict_crop(pil_img)

        self.assertEqual(res["status"], "success")
        self.assertIn(res["label"], ["REAL", "FAKE"])
        self.assertIn("real_score", res)
        self.assertIn("fake_score", res)
        self.assertAlmostEqual(res["real_score"] + res["fake_score"], 1.0, places=3)

    def test_predict_crop_invalid_dimensions(self):
        tiny_array = np.zeros((2, 2, 3), dtype=np.uint8)
        res = self.detector.predict_crop(tiny_array)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["label"], "ERROR")

    def test_predict_crop_empty_array(self):
        empty_array = np.array([])
        res = self.detector.predict_crop(empty_array)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["label"], "ERROR")

    def test_predict_image_with_scrfd(self):
        sample_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        res = self.detector.predict_image(sample_path, face_detector=self.face_detector)

        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["faces_detected"], 1)
        self.assertIn("primary_box", res)
        self.assertIn(res["label"], ["REAL", "FAKE"])

    def test_predict_image_noface(self):
        sample_noface = PROJECT_ROOT / "tests" / "sample_noface.jpg"
        res = self.detector.predict_image(sample_noface, face_detector=self.face_detector)

        self.assertEqual(res["status"], "no_face_detected")
        self.assertEqual(res["label"], "INCONCLUSIVE")
        self.assertEqual(res["faces_detected"], 0)


if __name__ == "__main__":
    unittest.main()
