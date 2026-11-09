"""
Unit and regression test suite for Verdict Engine v3 (Specialist-Any-FAKE).

Validates the 10 core requirements:
1. Any specialist crossing threshold produces FAKE (label="FAKE", primary trigger identified).
2. No specialist crossing threshold produces INCONCLUSIVE (reason="no_high_confidence_fake_signal").
3. No usable face produces INCONCLUSIVE (reason="no_usable_face").
4. All models unavailable produces INCONCLUSIVE (reason="all_models_unavailable").
5. One model unavailable while another detects FAKE still produces FAKE.
6. Missing model scores handled safely without crash.
7. Invalid thresholds rejected (raises ValueError if T_i < 0 or T_i > 1).
8. allow_real=false prevents REAL verdicts.
9. Legacy engine remains operational when VERDICT_ENGINE=legacy.
10. API response contains required debugging fields.

Regression tests:
- EffNet 1.00, ViT 0.209, CF ViT 0.817, SBI 0.003, SRM 1.00 -> FAKE (EffNet and SRM fire).
- All 5 models low -> INCONCLUSIVE (never REAL).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.image_detection.inference import (
    DeepFakeInferenceEngine,
    FacePrediction,
    ImageInferenceResult,
    evaluate_verdict_v3,
    load_verdict_config,
)
from src.api.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVerdictEngineV3(unittest.TestCase):
    def setUp(self):
        self.default_thresholds = {
            "efficientnet": 0.9000,
            "vit": 0.9000,
            "community_vit": 1.0000,
            "sbi": 0.9997,
            "hybrid_srm": 0.9000,
        }

    def _make_pred(self, prob_fake: float, loaded: bool = True) -> dict:
        return {
            "loaded": loaded,
            "label": "FAKE" if prob_fake >= 0.5 else "REAL",
            "score": prob_fake if prob_fake >= 0.5 else (1.0 - prob_fake),
            "probabilities": {
                "REAL": round(1.0 - prob_fake, 4),
                "FAKE": round(prob_fake, 4),
            },
        }

    def test_01_any_specialist_crossing_threshold_produces_fake(self):
        """1. Any specialist crossing threshold produces FAKE with primary trigger."""
        models = {
            "efficientnet": self._make_pred(0.20),
            "vit": self._make_pred(0.15),
            "community_vit": self._make_pred(1.00),  # Fires! (threshold 1.0000)
            "sbi": self._make_pred(0.05),
            "hybrid_srm": self._make_pred(0.30),
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "FAKE")
        self.assertEqual(res["reason"], "specialist_threshold_exceeded")
        self.assertEqual(res["primary_trigger"], "community_vit")
        self.assertIn("community_vit", res["triggering_models"])
        self.assertEqual(len(res["triggering_models"]), 1)

    def test_02_no_specialist_crossing_threshold_produces_inconclusive(self):
        """2. No specialist crossing threshold produces INCONCLUSIVE (never REAL)."""
        models = {
            "efficientnet": self._make_pred(0.50),
            "vit": self._make_pred(0.40),
            "community_vit": self._make_pred(0.80),
            "sbi": self._make_pred(0.60),
            "hybrid_srm": self._make_pred(0.70),
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "INCONCLUSIVE")
        self.assertEqual(res["reason"], "no_high_confidence_fake_signal")
        self.assertEqual(len(res["triggering_models"]), 0)
        self.assertIsNone(res["primary_trigger"])

    def test_03_no_usable_face_produces_inconclusive(self):
        """3. No usable face produces INCONCLUSIVE with reason 'no_usable_face'."""
        engine = DeepFakeInferenceEngine(auto_load=False, load_router=False)
        engine.active_engine = "v3"
        engine.allow_real = False
        engine.primary_model = MagicMock()
        engine.primary_checkpoint = Path("mock.pth")

        # Mock face detector returning zero faces
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = []
        engine.face_detector = mock_detector

        # Create dummy image
        import numpy as np
        import cv2
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy_img)

        result = engine.predict_image(buf.tobytes())
        self.assertEqual(result.final_prediction.label, "INCONCLUSIVE")
        self.assertEqual(result.final_prediction.reason, "no_usable_face")
        self.assertEqual(result.verdict_source, "specialist_any_fake_v3")

    def test_04_all_models_unavailable_produces_inconclusive(self):
        """4. All models unavailable produces INCONCLUSIVE with reason 'all_models_unavailable'."""
        models = {
            "efficientnet": {"loaded": False, "error": "not_loaded"},
            "vit": {"loaded": False, "error": "oom"},
            "community_vit": {"loaded": False},
            "sbi": None,
            "hybrid_srm": {"loaded": False},
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "INCONCLUSIVE")
        self.assertEqual(res["reason"], "all_models_unavailable")
        self.assertEqual(len(res["available_scores"]), 0)
        self.assertEqual(len(res["unavailable_models"]), 5)

    def test_05_one_model_unavailable_while_another_detects_fake_produces_fake(self):
        """5. One unavailable model must not block another specialist from firing FAKE."""
        models = {
            "efficientnet": {"loaded": False, "error": "model_file_missing"},
            "vit": self._make_pred(0.20),
            "community_vit": self._make_pred(0.30),
            "sbi": self._make_pred(0.10),
            "hybrid_srm": self._make_pred(0.95),  # Fires! (threshold 0.9000)
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "FAKE")
        self.assertEqual(res["primary_trigger"], "hybrid_srm")
        self.assertIn("efficientnet", res["unavailable_models"])
        self.assertIn("hybrid_srm", res["triggering_models"])

    def test_06_missing_model_scores_handled_safely(self):
        """6. Missing or malformed model predictions handled safely without exceptions."""
        malformed_models = {
            "efficientnet": {},
            "vit": {"loaded": True, "score": None},
            "community_vit": {"loaded": True, "probabilities": {}},
            "sbi": None,
            "hybrid_srm": {"loaded": True, "probabilities": {"FAKE": 0.40}},
        }
        res = evaluate_verdict_v3(malformed_models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "INCONCLUSIVE")
        self.assertIn("hybrid_srm", res["available_scores"])
        self.assertIn("efficientnet", res["unavailable_models"])
        self.assertIn("vit", res["unavailable_models"])
        self.assertIn("community_vit", res["unavailable_models"])
        self.assertIn("sbi", res["unavailable_models"])

    def test_07_invalid_thresholds_rejected(self):
        """7. Invalid thresholds (outside [0.0, 1.0]) raise ValueError."""
        invalid_thresh_negative = dict(self.default_thresholds, efficientnet=-0.1)
        invalid_thresh_over = dict(self.default_thresholds, vit=1.5)

        models = {"efficientnet": self._make_pred(0.95)}
        with self.assertRaises(ValueError):
            evaluate_verdict_v3(models, invalid_thresh_negative)

        with self.assertRaises(ValueError):
            evaluate_verdict_v3(models, invalid_thresh_over)

    def test_08_allow_real_false_prevents_real_verdict(self):
        """8. allow_real=false strictly prevents REAL verdicts even on 100% authentic scores."""
        models = {
            "efficientnet": self._make_pred(0.001),  # Strong real
            "vit": self._make_pred(0.002),           # Strong real
            "community_vit": self._make_pred(0.001), # Strong real
            "sbi": self._make_pred(0.0001),          # Strong real
            "hybrid_srm": self._make_pred(0.005),    # Strong real
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertNotEqual(res["verdict"], "REAL")
        self.assertEqual(res["verdict"], "INCONCLUSIVE")
        self.assertEqual(res["reason"], "no_high_confidence_fake_signal")

    def test_09_legacy_engine_remains_operational_under_env_var(self):
        """9. Legacy engine remains operational when VERDICT_ENGINE=legacy."""
        with patch.dict(os.environ, {"VERDICT_ENGINE": "legacy"}):
            engine = DeepFakeInferenceEngine(auto_load=False, load_router=False)
            self.assertEqual(engine.active_engine, "legacy")

    def test_10_api_response_schema_contains_required_debugging_fields(self):
        """10. API schemas and responses expose all required v3 debugging telemetry."""
        from src.api.schemas import PredictionResponseSchema

        sample_response = {
            "status": "success",
            "faces_detected": 1,
            "model_name": "EfficientNet-B0",
            "execution_device": "cpu",
            "uncertainty_threshold": 0.60,
            "message": "Analyzed with Verdict Engine v3.",
            "final_prediction": {
                "face_id": 0,
                "label": "FAKE",
                "score": 0.9997,
                "score_type": "uncalibrated_model_score",
                "reason": "specialist_threshold_exceeded",
                "verdict_source": "specialist_any_fake_v3",
                "verdict_engine": "v3",
                "primary_trigger": "sbi",
                "triggering_models": ["sbi"],
                "triggering_scores": {"sbi": 0.9997},
                "available_scores": {"sbi": 0.9997, "efficientnet": 0.50},
                "unavailable_models": [],
                "thresholds": self.default_thresholds,
            },
            "predictions": [],
            "verdict_source": "specialist_any_fake_v3",
            "active_verdict_engine": "v3",
            "verdict": "FAKE",
            "verdict_reason": "specialist_threshold_exceeded",
            "available_scores": {"sbi": 0.9997, "efficientnet": 0.50},
            "unavailable_models": [],
            "triggering_models": ["sbi"],
            "primary_trigger": "sbi",
            "thresholds": self.default_thresholds,
        }

        # Validates against Pydantic schema without error
        validated = PredictionResponseSchema(**sample_response)
        self.assertEqual(validated.verdict_source, "specialist_any_fake_v3")
        self.assertEqual(validated.active_verdict_engine, "v3")
        self.assertEqual(validated.primary_trigger, "sbi")
        self.assertEqual(validated.triggering_models, ["sbi"])

    def test_regression_effnet_and_srm_fire(self):
        """Regression test 1: EffNet 1.00, ViT 0.209, CF ViT 0.817, SBI 0.003, SRM 1.00 -> FAKE."""
        models = {
            "efficientnet": self._make_pred(1.00),
            "vit": self._make_pred(0.209),
            "community_vit": self._make_pred(0.817),
            "sbi": self._make_pred(0.003),
            "hybrid_srm": self._make_pred(1.00),
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "FAKE")
        self.assertIn("efficientnet", res["triggering_models"])
        self.assertIn("hybrid_srm", res["triggering_models"])
        self.assertEqual(len(res["triggering_models"]), 2)
        self.assertIn(res["primary_trigger"], ["efficientnet", "hybrid_srm"])

    def test_regression_all_five_models_low_inconclusive(self):
        """Regression test 2: All 5 models low -> INCONCLUSIVE (never REAL)."""
        models = {
            "efficientnet": self._make_pred(0.05),
            "vit": self._make_pred(0.08),
            "community_vit": self._make_pred(0.12),
            "sbi": self._make_pred(0.01),
            "hybrid_srm": self._make_pred(0.03),
        }
        res = evaluate_verdict_v3(models, self.default_thresholds, allow_real=False)
        self.assertEqual(res["verdict"], "INCONCLUSIVE")
        self.assertNotEqual(res["verdict"], "REAL")
        self.assertEqual(res["reason"], "no_high_confidence_fake_signal")


if __name__ == "__main__":
    unittest.main()
