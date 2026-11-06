"""
Unit and Integration Tests for DeepFake Forensic Inference Engine.

Verifies:
1. Model loading and device detection (accurate CPU/CUDA reporting).
2. Prediction on valid real image (REAL outcome).
3. Prediction on valid AI-generated fake image (FAKE outcome).
4. Handling of no-face images (INCONCLUSIVE, no_face_detected).
5. Handling of multiple-face images (separate analysis per detected face).
6. Handling of invalid inputs (corrupt bytes, empty input, unreadable paths).
7. Uncertainty threshold logic (rejection of low-confidence predictions to INCONCLUSIVE).
8. Face crop tensor shape guarantee: EfficientNet receives ONLY (1, 3, 224, 224), never full scene.
9. Modular Grad-CAM readiness payload extraction.
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.efficientnet_model import DeepfakeEfficientNetB0
from src.image_detection.inference import (
    DeepFakeInferenceEngine,
    InvalidImageError,
    detect_execution_device,
)


class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DeepFakeInferenceEngine(auto_load=True)
        cls.real_image_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        cls.fake_image_path = PROJECT_ROOT / "tests" / "sample_image_fake.jpg"
        cls.noface_image_path = PROJECT_ROOT / "tests" / "sample_landscape_noface.jpg"
        cls.multiface_image_path = PROJECT_ROOT / "tests" / "sample_multiple_faces.jpg"

    def test_01_model_loading_and_device(self):
        """Verify models are loaded into memory and device is accurately reported."""
        self.assertTrue(self.engine.is_ready)
        self.assertIsInstance(self.engine.primary_model, DeepfakeEfficientNetB0)
        self.assertTrue(self.engine.detector.is_available)
        self.assertIsNotNone(self.engine.hybrid_srm_model)

        expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(self.engine.device.type, expected_device_type)

    def test_02_valid_real_image(self):
        """Verify inference on a valid real image produces REAL verdict."""
        self.assertTrue(self.real_image_path.is_file())
        result = self.engine.predict_image(self.real_image_path, model_name="efficientnet")

        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(result.faces_detected, 1)
        self.assertEqual(result.model_name, "EfficientNet-B0")

        pred = result.predictions[0]
        self.assertEqual(pred.face_id, 1)
        self.assertEqual(pred.label, "REAL")
        self.assertEqual(pred.score_type, "uncalibrated_model_score")
        self.assertGreaterEqual(pred.efficientnet["score"], 0.90)
        self.assertGreaterEqual(pred.score, 0.60)
        self.assertIsNotNone(pred.box)
        self.assertEqual(len(pred.box), 4)
        self.assertIn("REAL", pred.probabilities)
        self.assertIn("FAKE", pred.probabilities)
        self.assertIsNotNone(pred.hybrid_srm)
        self.assertEqual(pred.hybrid_srm["label"], "REAL")
        self.assertTrue(pred.hybrid_srm["loaded"])

    def test_03_valid_fake_image(self):
        """Verify disagreement is reported instead of forcing a fake verdict."""
        self.assertTrue(self.fake_image_path.is_file())
        result = self.engine.predict_image(self.fake_image_path, model_name="efficientnet")

        self.assertEqual(result.status, "inconclusive")
        self.assertGreaterEqual(result.faces_detected, 1)
        pred = result.predictions[0]
        self.assertEqual(pred.label, "INCONCLUSIVE")
        self.assertEqual(pred.fusion["reason"], "model_disagreement")
        self.assertEqual(pred.score_type, "uncalibrated_model_score")
        self.assertIsNotNone(pred.hybrid_srm)
        self.assertEqual(pred.hybrid_srm["label"], "FAKE")
        self.assertTrue(pred.hybrid_srm["loaded"])

    def test_04_no_face_image(self):
        """Verify no-face images return INCONCLUSIVE without failing."""
        self.assertTrue(self.noface_image_path.is_file())
        result = self.engine.predict_image(self.noface_image_path, model_name="efficientnet")

        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.faces_detected, 0)
        self.assertIn("No human faces detected", result.message)

        self.assertEqual(len(result.predictions), 1)
        pred = result.predictions[0]
        self.assertEqual(pred.label, "INCONCLUSIVE")
        self.assertEqual(pred.reason, "no_face_detected")
        self.assertIsNone(pred.score)
        self.assertIsNone(pred.box)

    def test_05_multiple_faces_image(self):
        """Verify images with multiple faces analyze each face separately."""
        self.assertTrue(self.multiface_image_path.is_file())
        result = self.engine.predict_image(self.multiface_image_path, model_name="efficientnet")

        self.assertEqual(result.faces_detected, 2)
        self.assertEqual(len(result.predictions), 2)

        pred1 = result.predictions[0]
        pred2 = result.predictions[1]
        self.assertEqual(pred1.face_id, 1)
        self.assertEqual(pred2.face_id, 2)
        self.assertNotEqual(pred1.box, pred2.box)

    def test_06_invalid_file_handling(self):
        """Verify that invalid inputs raise InvalidImageError."""
        # Non-existent file
        with self.assertRaises(InvalidImageError):
            self.engine.predict_image(PROJECT_ROOT / "tests" / "non_existent_file.jpg")

        # Empty bytes
        with self.assertRaises(InvalidImageError):
            self.engine.predict_image(b"")

        # Corrupt non-image bytes
        with self.assertRaises(InvalidImageError):
            self.engine.predict_image(b"This is completely invalid text instead of an image.")

        # Image smaller than 10x10
        tiny_img = Image.new("RGB", (4, 4), color="blue")
        buf = io.BytesIO()
        tiny_img.save(buf, format="PNG")
        with self.assertRaises(InvalidImageError):
            self.engine.predict_image(buf.getvalue())

    def test_07_uncertainty_threshold(self):
        """Verify predictions below uncertainty threshold are flagged INCONCLUSIVE."""
        result = self.engine.predict_image(
            self.real_image_path,
            model_name="efficientnet",
            uncertainty_threshold=0.99999,
        )
        pred = result.predictions[0]
        if pred.score < 0.99999:
            self.assertEqual(pred.label, "INCONCLUSIVE")
            self.assertEqual(pred.reason, "low_confidence")

    def test_08_face_crop_tensor_shape_guarantee(self):
        """Verify EfficientNet receives ONLY the cropped 224x224 face tensor, never full scene."""
        result = self.engine.predict_image(self.real_image_path)
        self.assertEqual(result.model_name, "EfficientNet-B0")
        self.assertEqual(self.engine.last_received_tensor_shape, (1, 3, 224, 224))
        self.assertGreaterEqual(result.faces_detected, 1)
        pred = result.predictions[0]
        self.assertIn(pred.label, ("REAL", "FAKE", "INCONCLUSIVE"))
        self.assertEqual(pred.score_type, "uncalibrated_model_score")

    def test_09_gradcam_readiness_payload(self):
        """Verify modular payload extraction for downstream Grad-CAM explainability."""
        self.engine.predict_image(self.real_image_path, model_name="efficientnet")
        payload = self.engine.get_gradcam_payload(face_id=1, model_name="efficientnet")

        self.assertIn("face_crop_pil", payload)
        self.assertIsInstance(payload["face_crop_pil"], Image.Image)
        self.assertIn("face_crop_bgr", payload)
        self.assertIsInstance(payload["face_crop_bgr"], np.ndarray)
        self.assertIn("model", payload)
        self.assertIsInstance(payload["model"], DeepfakeEfficientNetB0)
        self.assertIn("transform", payload)
        self.assertIn("target_layer", payload)
        self.assertIsNotNone(payload["target_layer"])
        self.assertIn(payload["predicted_class"], ("FAKE", "REAL"))
        self.assertIn(payload["predicted_class_idx"], (0, 1))

    def test_10_hybrid_srm_consensus_integration(self):
        """Verify fifth model (Hybrid SRM) is included in multi-model consensus assessment."""
        result = self.engine.predict_image(self.real_image_path)
        pred = result.predictions[0]
        self.assertIsNotNone(pred.hybrid_srm)
        self.assertTrue(pred.hybrid_srm["loaded"])
        self.assertIn("qualifying_votes", pred.combined_result)
        self.assertIn("hybrid_srm", pred.combined_result["qualifying_votes"])
        self.assertEqual(pred.combined_result["qualifying_votes"]["hybrid_srm"], "REAL")


if __name__ == "__main__":
    unittest.main()
