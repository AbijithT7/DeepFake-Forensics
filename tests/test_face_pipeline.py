"""
Unit and Integration Tests for Face-Only EfficientNet-B0 Pipeline.

Tests the 7 critical evaluation scenarios:
1. Full-scene image with one visible face (face cropped, full-scene never classified).
2. Portrait with one visible face.
3. Image with multiple faces (analyzed independently).
4. Image with no face (INCONCLUSIVE, no_face_detected).
5. Image with a very small face (INCONCLUSIVE, face_too_small).
6. Blurry face (INCONCLUSIVE, face_too_blurry).
7. Partially occluded face (INCONCLUSIVE, face_not_usable).

Verifies:
- EfficientNet receives ONLY the cropped face tensor of shape (1, 3, 224, 224), never the full image.
- Saves all extracted face crops to `outputs/test_crops/` for visual inspection.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.inference import DeepFakeInferenceEngine


class TestFaceOnlyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DeepFakeInferenceEngine(auto_load=True)
        cls.crops_dir = PROJECT_ROOT / "outputs" / "test_crops"
        cls.crops_dir.mkdir(parents=True, exist_ok=True)

        cls.fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
        cls.fixtures_dir.mkdir(parents=True, exist_ok=True)

        # Base sample images
        cls.real_face_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        cls.multiface_path = PROJECT_ROOT / "tests" / "sample_multiple_faces.jpg"
        cls.noface_path = PROJECT_ROOT / "tests" / "sample_landscape_noface.jpg"

        # Generate test fixtures if not present
        cls._create_test_fixtures()

    @classmethod
    def _create_test_fixtures(cls):
        real_img = cv2.imread(str(cls.real_face_path))
        h_f, w_f = real_img.shape[:2]

        # Scenario 1: Full-scene image with one visible face
        scene_path = cls.fixtures_dir / "case1_full_scene_one_face.jpg"
        if not scene_path.is_file():
            scene = np.full((600, 800, 3), (180, 200, 210), dtype=np.uint8)
            scene[400:, :] = (80, 100, 120)
            scene[50:250, 500:750] = (220, 240, 255)
            face_resized = cv2.resize(real_img, (180, 180))
            scene[150:330, 150:330] = face_resized
            cv2.imwrite(str(scene_path), scene)

        # Scenario 5: Image with a very small face (26x26 in a 400x400 canvas)
        small_path = cls.fixtures_dir / "case5_small_face.jpg"
        if not small_path.is_file():
            small_canvas = np.full((400, 400, 3), (200, 200, 200), dtype=np.uint8)
            tiny_face = cv2.resize(real_img, (26, 26))
            small_canvas[150:176, 150:176] = tiny_face
            cv2.imwrite(str(small_path), small_canvas)

        # Scenario 6: Blurry face (Gaussian blurred)
        blurry_path = cls.fixtures_dir / "case6_blurry_face.jpg"
        if not blurry_path.is_file():
            blurry = cv2.GaussianBlur(real_img, (35, 35), 15)
            cv2.imwrite(str(blurry_path), blurry)

        # Scenario 7: Partially occluded face (solid occlusion block)
        occluded_path = cls.fixtures_dir / "case7_occluded_face.jpg"
        if not occluded_path.is_file():
            occluded = real_img.copy()
            occluded[100:160, 50:180] = 0
            cv2.imwrite(str(occluded_path), occluded)

    def test_01_full_scene_with_one_visible_face(self):
        """Case 1: Full-scene image with one visible face."""
        scene_path = self.fixtures_dir / "case1_full_scene_one_face.jpg"
        full_img = cv2.imread(str(scene_path))
        full_h, full_w = full_img.shape[:2]

        result = self.engine.predict_image(scene_path)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.faces_detected, 1)

        pred = result.predictions[0]
        self.assertEqual(pred.face_id, 1)
        self.assertIn(pred.label, ("REAL", "FAKE"))
        self.assertEqual(pred.score_type, "uncalibrated_model_score")

        # Verify EfficientNet received ONLY the cropped face tensor (1, 3, 224, 224)
        self.assertEqual(self.engine.last_received_tensor_shape, (1, 3, 224, 224))
        self.assertNotEqual(self.engine.last_received_tensor_shape, (1, 3, full_h, full_w))

        # Save extracted crop for visual comparison
        crop_art = self.engine.crop_artifacts[1]
        cv2.imwrite(str(self.crops_dir / "case1_full_scene_face_crop.jpg"), crop_art.crop_bgr)
        self.assertTrue((self.crops_dir / "case1_full_scene_face_crop.jpg").is_file())

    def test_02_portrait_with_one_visible_face(self):
        """Case 2: Portrait with one visible face."""
        result = self.engine.predict_image(self.real_face_path)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.faces_detected, 1)

        pred = result.predictions[0]
        self.assertEqual(pred.face_id, 1)
        self.assertIn(pred.label, ("REAL", "FAKE"))
        self.assertEqual(pred.score_type, "uncalibrated_model_score")
        self.assertEqual(self.engine.last_received_tensor_shape, (1, 3, 224, 224))

        crop_art = self.engine.crop_artifacts[1]
        cv2.imwrite(str(self.crops_dir / "case2_portrait_face_crop.jpg"), crop_art.crop_bgr)
        self.assertTrue((self.crops_dir / "case2_portrait_face_crop.jpg").is_file())

    def test_03_image_with_multiple_faces(self):
        """Case 3: Image with multiple faces."""
        result = self.engine.predict_image(self.multiface_path)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.faces_detected, 2)
        self.assertEqual(len(result.predictions), 2)

        p1, p2 = result.predictions[0], result.predictions[1]
        self.assertEqual(p1.face_id, 1)
        self.assertEqual(p2.face_id, 2)
        self.assertNotEqual(p1.box, p2.box)

        # Save both crops for visual comparison
        for fid, art in self.engine.crop_artifacts.items():
            cv2.imwrite(str(self.crops_dir / f"case3_multiface_crop_{fid}.jpg"), art.crop_bgr)
            self.assertTrue((self.crops_dir / f"case3_multiface_crop_{fid}.jpg").is_file())

    def test_04_image_with_no_face(self):
        """Case 4: Image with no face."""
        # Reset tensor shape tracker before test
        self.engine.last_received_tensor_shape = None

        result = self.engine.predict_image(self.noface_path)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.faces_detected, 0)
        self.assertEqual(len(result.predictions), 1)

        pred = result.predictions[0]
        self.assertEqual(pred.label, "INCONCLUSIVE")
        self.assertEqual(pred.reason, "no_face_detected")
        self.assertIsNone(pred.score)
        self.assertEqual(pred.score_type, "uncalibrated_model_score")

        # Verify EfficientNet was NEVER called on the full image
        self.assertIsNone(self.engine.last_received_tensor_shape)

    def test_05_image_with_very_small_face(self):
        """Case 5: Image with a very small face."""
        small_path = self.fixtures_dir / "case5_small_face.jpg"
        result = self.engine.predict_image(small_path)

        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.faces_detected, 1)
        pred = result.predictions[0]
        self.assertEqual(pred.label, "INCONCLUSIVE")
        self.assertEqual(pred.reason, "face_too_small")
        self.assertIsNone(pred.score)

        if 1 in self.engine.crop_artifacts:
            crop_art = self.engine.crop_artifacts[1]
            cv2.imwrite(str(self.crops_dir / "case5_small_face_crop.jpg"), crop_art.crop_bgr)

    def test_06_blurry_face(self):
        """Case 6: Blurry face."""
        blurry_path = self.fixtures_dir / "case6_blurry_face.jpg"
        result = self.engine.predict_image(blurry_path)

        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.faces_detected, 1)
        pred = result.predictions[0]
        self.assertEqual(pred.label, "INCONCLUSIVE")
        self.assertEqual(pred.reason, "face_too_blurry")
        self.assertIsNone(pred.score)

        if 1 in self.engine.crop_artifacts:
            crop_art = self.engine.crop_artifacts[1]
            cv2.imwrite(str(self.crops_dir / "case6_blurry_face_crop.jpg"), crop_art.crop_bgr)

    def test_07_partially_occluded_face(self):
        """Case 7: Partially occluded face."""
        occluded_path = self.fixtures_dir / "case7_occluded_face.jpg"
        result = self.engine.predict_image(occluded_path)

        self.assertEqual(result.status, "inconclusive")
        pred = result.predictions[0]
        self.assertEqual(pred.label, "INCONCLUSIVE")
        self.assertIn(pred.reason, ("face_not_usable", "no_face_detected"))
        self.assertIsNone(pred.score)

        if 1 in self.engine.crop_artifacts:
            crop_art = self.engine.crop_artifacts[1]
            cv2.imwrite(str(self.crops_dir / "case7_occluded_face_crop.jpg"), crop_art.crop_bgr)


if __name__ == "__main__":
    unittest.main()
