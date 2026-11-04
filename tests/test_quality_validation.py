"""
Unit Tests for Face Quality and Usability Validation.

Verifies:
1. Usable faces with natural dark hair are NOT rejected (fixes uniform_black issue).
2. Usable faces against dark/black studio backdrops are NOT rejected.
3. Usable faces against bright/white studio backdrops are NOT rejected.
4. Genuinely unusable images are accurately flagged:
   - Severe artificial black censor bar occlusion (face_not_usable).
   - Severe artificial white censor bar occlusion (face_not_usable).
   - Severe artificial flat color occlusion (face_not_usable).
   - Degenerate pitch-black frame (face_not_usable).
   - Degenerate blown-out white frame (face_not_usable).
   - Degenerate flat/featureless frame (face_not_usable).
   - Severe boundary truncation (face_not_usable).
   - Excessive blur (face_too_blurry).
   - Sub-minimum resolution (face_too_small).
   - Low detection confidence (low_face_detection_confidence).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.face_detector import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    SCRFDFaceDetector,
    crop_face_with_controlled_margin,
    validate_face_quality,
)


class TestFaceQualityValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = SCRFDFaceDetector()
        cls.real_img_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        cls.real_bgr = cv2.imread(str(cls.real_img_path))
        detections = cls.detector.detect(cls.real_bgr)
        cls.base_det = detections[0]
        cls.base_box = cls.base_det["box"]
        cls.base_crop, cls.padded_box = crop_face_with_controlled_margin(cls.real_bgr, cls.base_box)

    def test_01_clean_face_is_usable(self):
        """Clean baseline face must pass quality validation."""
        is_usable, reason = validate_face_quality(
            crop_bgr=self.base_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
        )
        self.assertTrue(is_usable)
        self.assertIsNone(reason)

    def test_02_dark_hair_face_is_usable(self):
        """
        Face with natural dark hair (where >15% of pixels are dark) must NOT be rejected.
        Simulate natural dark hair by painting the hair region dark while preserving the face.
        """
        dark_hair_crop = self.base_crop.copy()
        # Darken the top 25% (hair region) to simulate thick black hair (< 15 intensity)
        h, w = dark_hair_crop.shape[:2]
        dark_hair_crop[: int(0.25 * h), :] = np.clip(
            dark_hair_crop[: int(0.25 * h), :].astype(np.int32) - 80, 0, 14
        ).astype(np.uint8)

        # Confirm that more than 15% of the crop pixels are now dark (< 15)
        gray = cv2.cvtColor(dark_hair_crop, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.sum(gray < 15)) / float(gray.size)
        self.assertGreater(dark_ratio, 0.15)

        is_usable, reason = validate_face_quality(
            crop_bgr=dark_hair_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
        )
        self.assertTrue(is_usable, f"Face with dark hair was wrongly rejected with reason: {reason}")
        self.assertIsNone(reason)

    def test_03_dark_background_face_is_usable(self):
        """
        Face photographed against a dark/black studio background must NOT be rejected.
        Simulate a black studio backdrop in the margins surrounding the face.
        """
        dark_bg_img = np.zeros_like(self.real_bgr)
        x, y, w, h = self.base_box
        # Paste face on black canvas
        dark_bg_img[y : y + h, x : x + w] = self.real_bgr[y : y + h, x : x + w]

        crop, _ = crop_face_with_controlled_margin(dark_bg_img, self.base_box)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.sum(gray < 15)) / float(gray.size)
        self.assertGreater(dark_ratio, 0.20)

        is_usable, reason = validate_face_quality(
            crop_bgr=crop,
            detection_score=self.base_det["score"],
            original_image_shape=dark_bg_img.shape,
            box=self.base_box,
        )
        self.assertTrue(is_usable, f"Face on dark background was wrongly rejected: {reason}")
        self.assertIsNone(reason)

    def test_04_bright_background_face_is_usable(self):
        """
        Face photographed against a white studio background must NOT be rejected.
        """
        white_bg_img = np.full_like(self.real_bgr, 255)
        x, y, w, h = self.base_box
        white_bg_img[y : y + h, x : x + w] = self.real_bgr[y : y + h, x : x + w]

        crop, _ = crop_face_with_controlled_margin(white_bg_img, self.base_box)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright_ratio = float(np.sum(gray > 245)) / float(gray.size)
        self.assertGreater(bright_ratio, 0.15)

        is_usable, reason = validate_face_quality(
            crop_bgr=crop,
            detection_score=self.base_det["score"],
            original_image_shape=white_bg_img.shape,
            box=self.base_box,
        )
        self.assertTrue(is_usable, f"Face on bright background was wrongly rejected: {reason}")
        self.assertIsNone(reason)

    def test_05_severe_black_censor_bar_is_rejected(self):
        """Face occluded with a solid black censor bar across the eyes must be rejected."""
        occluded_img = self.real_bgr.copy()
        x, y, w, h = self.base_box
        # Solid black censor rectangle across the middle of the face
        occluded_img[y + int(0.25 * h) : y + int(0.55 * h), x + int(0.05 * w) : x + int(0.95 * w)] = 0

        crop, _ = crop_face_with_controlled_margin(occluded_img, self.base_box)
        is_usable, reason = validate_face_quality(
            crop_bgr=crop,
            detection_score=self.base_det["score"],
            original_image_shape=occluded_img.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")

    def test_06_severe_white_censor_bar_is_rejected(self):
        """Face occluded with a solid white censor bar across the eyes must be rejected."""
        occluded_img = self.real_bgr.copy()
        x, y, w, h = self.base_box
        # Solid white censor rectangle across the middle of the face
        occluded_img[y + int(0.25 * h) : y + int(0.55 * h), x + int(0.05 * w) : x + int(0.95 * w)] = 255

        crop, _ = crop_face_with_controlled_margin(occluded_img, self.base_box)
        is_usable, reason = validate_face_quality(
            crop_bgr=crop,
            detection_score=self.base_det["score"],
            original_image_shape=occluded_img.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")

    def test_07_severe_flat_color_block_is_rejected(self):
        """Face occluded with a solid flat gray censor bar must be rejected."""
        occluded_img = self.real_bgr.copy()
        x, y, w, h = self.base_box
        # Solid flat gray censor rectangle
        occluded_img[y + int(0.25 * h) : y + int(0.55 * h), x + int(0.05 * w) : x + int(0.95 * w)] = (128, 128, 128)

        crop, _ = crop_face_with_controlled_margin(occluded_img, self.base_box)
        is_usable, reason = validate_face_quality(
            crop_bgr=crop,
            detection_score=self.base_det["score"],
            original_image_shape=occluded_img.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")

    def test_08_degenerate_pitch_black_frame_is_rejected(self):
        """Entirely dark/black frame must be rejected as face_not_usable."""
        black_crop = np.zeros_like(self.base_crop)
        is_usable, reason = validate_face_quality(
            crop_bgr=black_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")

    def test_09_degenerate_blown_out_white_frame_is_rejected(self):
        """Entirely blown-out white frame must be rejected as face_not_usable."""
        white_crop = np.full_like(self.base_crop, 255)
        is_usable, reason = validate_face_quality(
            crop_bgr=white_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")

    def test_10_degenerate_flat_solid_frame_is_rejected(self):
        """Solid flat gray frame with zero contrast must be rejected as face_not_usable."""
        flat_crop = np.full_like(self.base_crop, 128)
        is_usable, reason = validate_face_quality(
            crop_bgr=flat_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")

    def test_11_blurry_face_is_rejected(self):
        """Excessively blurred face must be rejected as face_too_blurry."""
        blurry_crop = cv2.GaussianBlur(self.base_crop, (35, 35), 15)
        is_usable, reason = validate_face_quality(
            crop_bgr=blurry_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_too_blurry")

    def test_12_tiny_face_is_rejected(self):
        """Face smaller than minimum dimensions must be rejected as face_too_small."""
        tiny_box = (self.base_box[0], self.base_box[1], 25, 25)
        tiny_crop = cv2.resize(self.base_crop, (25, 25))
        is_usable, reason = validate_face_quality(
            crop_bgr=tiny_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=tiny_box,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_too_small")

    def test_13_low_detection_confidence_is_rejected(self):
        """Face with detection score below threshold must be rejected as low_face_detection_confidence."""
        is_usable, reason = validate_face_quality(
            crop_bgr=self.base_crop,
            detection_score=0.25,
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
            min_confidence=DEFAULT_MIN_DETECTION_CONFIDENCE,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "low_face_detection_confidence")

    def test_14_severe_boundary_truncation_is_rejected(self):
        """Face truncated outside original image boundaries must be rejected as face_not_usable."""
        is_usable, reason = validate_face_quality(
            crop_bgr=self.base_crop,
            detection_score=self.base_det["score"],
            original_image_shape=self.real_bgr.shape,
            box=self.base_box,
            truncation_ratio=0.50,
        )
        self.assertFalse(is_usable)
        self.assertEqual(reason, "face_not_usable")


if __name__ == "__main__":
    unittest.main()
