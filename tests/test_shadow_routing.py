"""
Tests for DINOv2 Image-Type Router & Shadow-Mode Specialist Routing Integration.

Verifies:
1. Default specialist routing configuration mapping for all 7 categories:
   - REAL -> ['efficientnet', 'vit']
   - STYLEGAN_GAN -> ['efficientnet', 'hybrid_srm']
   - DIFFUSION -> ['community_vit']
   - FACE_SWAP -> ['sbi']
   - INPAINTING -> ['community_vit', 'sbi']
   - OTHER_SYNTHETIC -> ['community_vit']
   - UNKNOWN -> ['efficientnet', 'vit', 'community_vit', 'sbi', 'hybrid_srm']
2. Dynamic / custom specialist routing configuration updates.
3. Router UNKNOWN and low-confidence rejection behavior (< 0.45).
4. Shadow mode isolation: Production verdict remains 100% authoritative and unchanged
   even if shadow routed verdict differs.
5. Handling missing or failed specialist models within shadow routing consensus.
6. Graceful degradation when the router fails or throws an unexpected exception.
7. No-face and quality-rejection edge cases attach shadow routing telemetry.
8. API schema validation for shadow_routing, health, and models info endpoints.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.image_detection.inference import (
    DeepFakeInferenceEngine,
    FaceCropArtifact,
    FacePrediction,
    DEFAULT_SPECIALIST_ROUTING_MAP,
    DETECTOR_DISPLAY_NAMES,
    compute_combined_result,
)
from src.image_detection.image_type_router import RouterPrediction
from src.api.schemas import (
    PredictionResponseSchema,
    HealthResponseSchema,
    ModelsInfoResponseSchema,
    ShadowRoutingSchema,
)


class TestShadowRoutingStrategy(unittest.TestCase):
    """Test routing strategy configuration and mapping rules."""

    def test_default_routing_map_completeness(self):
        """Verify all 7 image-type categories are mapped in DEFAULT_SPECIALIST_ROUTING_MAP."""
        expected_categories = [
            "REAL",
            "STYLEGAN_GAN",
            "DIFFUSION",
            "FACE_SWAP",
            "INPAINTING",
            "OTHER_SYNTHETIC",
            "UNKNOWN",
        ]
        for cat in expected_categories:
            self.assertIn(cat, DEFAULT_SPECIALIST_ROUTING_MAP)
            specialists = DEFAULT_SPECIALIST_ROUTING_MAP[cat]
            self.assertIsInstance(specialists, list)
            self.assertGreater(len(specialists), 0)

    def test_specialist_mapping_definitions(self):
        """Verify exact specialist assignments per specification."""
        self.assertEqual(DEFAULT_SPECIALIST_ROUTING_MAP["REAL"], ["efficientnet", "vit"])
        self.assertEqual(DEFAULT_SPECIALIST_ROUTING_MAP["STYLEGAN_GAN"], ["efficientnet", "hybrid_srm"])
        self.assertEqual(DEFAULT_SPECIALIST_ROUTING_MAP["DIFFUSION"], ["community_vit"])
        self.assertEqual(DEFAULT_SPECIALIST_ROUTING_MAP["FACE_SWAP"], ["sbi"])
        self.assertEqual(DEFAULT_SPECIALIST_ROUTING_MAP["INPAINTING"], ["community_vit", "sbi"])
        self.assertEqual(DEFAULT_SPECIALIST_ROUTING_MAP["OTHER_SYNTHETIC"], ["community_vit"])
        self.assertEqual(
            DEFAULT_SPECIALIST_ROUTING_MAP["UNKNOWN"],
            ["efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"],
        )

    def test_display_names_mapping(self):
        """Verify display names for all specialist detector keys."""
        all_keys = {"efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"}
        for k in all_keys:
            self.assertIn(k, DETECTOR_DISPLAY_NAMES)
            self.assertIsInstance(DETECTOR_DISPLAY_NAMES[k], str)

    def test_configurable_routing_map(self):
        """Verify routing map can be dynamically inspected and updated on engine."""
        engine = DeepFakeInferenceEngine(auto_load=False, load_router=False)
        self.assertEqual(engine.get_specialist_routing_map(), DEFAULT_SPECIALIST_ROUTING_MAP)

        custom_map = {"REAL": ["efficientnet"], "UNKNOWN": ["efficientnet"]}
        engine.set_specialist_routing_map(custom_map)
        self.assertEqual(engine.get_specialist_routing_map(), custom_map)


class TestShadowModeIsolationAndExecution(unittest.TestCase):
    """Test shadow-mode isolation: production verdict never altered by router."""

    def setUp(self):
        self.engine = DeepFakeInferenceEngine(auto_load=False, load_router=False)
        self.engine.primary_model = MagicMock()
        self.engine.primary_checkpoint = Path("mock_checkpoint.pth")
        self.engine.shadow_mode = True

    def test_shadow_routing_data_attached_and_isolated_when_disagreeing(self):
        """
        Verify that when shadow routed verdict differs from production verdict:
        - Production verdict (FacePrediction.label) is strictly preserved.
        - Shadow routing records the divergence in verdicts_match=False.
        - shadow_mode flag is True.
        """
        # Mock router returning DIFFUSION
        mock_router = MagicMock()
        mock_router.route_image.return_value = RouterPrediction(
            predicted_class="DIFFUSION",
            raw_predicted_class="DIFFUSION",
            confidence=0.95,
            raw_confidence=0.95,
            probabilities={"DIFFUSION": 0.95, "REAL": 0.05},
            is_rejected=False,
            rejection_reason=None,
        )
        self.engine.image_type_router = mock_router

        # Setup mock face artifacts
        test_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        test_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_artifact = FaceCropArtifact(
            face_id=1,
            box=(10, 10, 100, 100),
            crop_bgr=test_bgr,
            crop_pil=test_img,
            padded_box=(5, 5, 110, 110),
            detection_score=0.99,
            is_usable=True,
        )

        with patch.object(self.engine, "validate_and_read_image", return_value=(test_bgr, test_img)), \
             patch.object(self.engine, "detect_and_crop_faces", return_value=[mock_artifact]), \
             patch("src.image_detection.inference.get_efficientnet_transform", return_value=lambda x: MagicMock()), \
             patch("torch.no_grad"), \
             patch("torch.nn.functional.softmax", return_value=[MagicMock(__getitem__=lambda s, idx: MagicMock(item=lambda: 0.99 if idx == 0 else 0.01))]):

            # Primary model predicts FAKE (logit for FAKE is high)
            mock_tensor = MagicMock()
            mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
            self.engine.primary_model.return_value = MagicMock()
            self.engine.primary_model.return_value.__getitem__.return_value = MagicMock()

            # Mock secondary detectors:
            # Let community_vit (selected for DIFFUSION) predict REAL (disagreeing with primary!)
            mock_community_vit = MagicMock()
            mock_community_vit.predict.return_value = {
                "label": "REAL",
                "score": 0.98,
                "probabilities": {"REAL": 0.98, "FAKE": 0.02},
                "loaded": True,
            }
            self.engine.community_vit_model = mock_community_vit

            # Original ViT agrees with primary on FAKE
            mock_vit = MagicMock()
            mock_vit.predict.return_value = {
                "label": "FAKE",
                "score": 0.95,
                "probabilities": {"FAKE": 0.95, "REAL": 0.05},
                "loaded": True,
            }
            self.engine.secondary_model = mock_vit

            # Run inference
            result = self.engine.predict_image(
                image_input=b"dummy_bytes",
                uncertainty_threshold=0.60,
                include_gradcam=False,
            )

            # 1. Production verdict MUST remain FAKE (from primary + conservative fusion)
            self.assertEqual(result.final_prediction.label, "FAKE")
            pred = result.predictions[0]
            self.assertEqual(pred.label, "FAKE")

            # 2. Shadow routing MUST be populated with telemetry
            shadow = pred.shadow_routing
            self.assertIsNotNone(shadow)
            self.assertEqual(shadow["image_type"], "DIFFUSION")
            self.assertEqual(shadow["router_confidence"], 0.95)
            self.assertFalse(shadow["is_rejected"])
            self.assertEqual(shadow["selected_specialist_keys"], ["community_vit"])
            self.assertEqual(shadow["selected_specialists"], ["Community Forensics ViT"])

            # 3. Shadow verdict reflects the specialist consensus (REAL), NOT the production verdict
            self.assertEqual(shadow["shadow_verdict"]["label"], "REAL")
            self.assertEqual(shadow["production_verdict"]["label"], "FAKE")
            self.assertFalse(shadow["verdicts_match"])
            self.assertTrue(shadow["shadow_mode"])

    def test_low_confidence_rejection_routes_to_unknown(self):
        """Verify that low-confidence inputs (<0.45) route to UNKNOWN with all 5 detectors."""
        mock_router = MagicMock()
        mock_router.route_image.return_value = RouterPrediction(
            predicted_class="UNKNOWN",
            raw_predicted_class="DIFFUSION",
            confidence=0.32,
            raw_confidence=0.32,
            probabilities={"DIFFUSION": 0.32, "REAL": 0.28, "STYLEGAN_GAN": 0.20},
            is_rejected=True,
            rejection_reason="low_confidence (0.3200 < 0.4500)",
        )
        self.engine.image_type_router = mock_router

        test_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        test_bgr = np.zeros((256, 256, 3), dtype=np.uint8)

        with patch.object(self.engine, "validate_and_read_image", return_value=(test_bgr, test_img)), \
             patch.object(self.engine, "detect_and_crop_faces", return_value=[]):
            # No face detected
            result = self.engine.predict_image(image_input=b"noface")
            shadow = result.shadow_routing
            self.assertIsNotNone(shadow)
            self.assertEqual(shadow["image_type"], "UNKNOWN")
            self.assertTrue(shadow["is_rejected"])
            self.assertIn("low_confidence", shadow["rejection_reason"])
            self.assertEqual(len(shadow["selected_specialist_keys"]), 5)
            self.assertEqual(
                shadow["selected_specialist_keys"],
                ["efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"],
            )

    def test_router_exception_graceful_fallback(self):
        """Verify engine gracefully falls back to UNKNOWN if router throws an unexpected error."""
        mock_router = MagicMock()
        mock_router.route_image.side_effect = RuntimeError("DINOv2 CUDA OOM")
        self.engine.image_type_router = mock_router

        test_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        test_bgr = np.zeros((256, 256, 3), dtype=np.uint8)

        with patch.object(self.engine, "validate_and_read_image", return_value=(test_bgr, test_img)), \
             patch.object(self.engine, "detect_and_crop_faces", return_value=[]):
            result = self.engine.predict_image(image_input=b"noface")
            shadow = result.shadow_routing
            self.assertIsNotNone(shadow)
            self.assertEqual(shadow["image_type"], "UNKNOWN")
            self.assertTrue(shadow["is_rejected"])
            self.assertEqual(shadow["rejection_reason"], "router_unavailable_or_failed")
            self.assertEqual(len(shadow["selected_specialist_keys"]), 5)

    def test_missing_or_failed_specialist_in_shadow_consensus(self):
        """Verify shadow verdict handles unavailable or failing routed specialists cleanly."""
        models_dict = {
            "community_vit": {
                "label": "INCONCLUSIVE",
                "score": None,
                "reason": "model_error_during_forward_pass",
                "loaded": False,
            }
        }
        shadow_combined = compute_combined_result(models_dict, threshold=0.60)
        self.assertEqual(shadow_combined["label"], "INCONCLUSIVE")
        self.assertIn("community_vit", shadow_combined["unavailable_models"])


class TestShadowRoutingSchemaCompatibility(unittest.TestCase):
    """Test Pydantic schemas correctly validate shadow routing telemetry."""

    def test_shadow_routing_schema_validation(self):
        """Verify ShadowRoutingSchema parses and validates properly."""
        payload = {
            "image_type": "DIFFUSION",
            "raw_image_type": "DIFFUSION",
            "router_confidence": 0.9423,
            "is_rejected": False,
            "rejection_reason": None,
            "router_probabilities": {"DIFFUSION": 0.9423, "REAL": 0.0577},
            "selected_specialists": ["Community Forensics ViT"],
            "selected_specialist_keys": ["community_vit"],
            "shadow_verdict": {
                "label": "FAKE",
                "consensus": "unanimous_fake",
                "votes": {"REAL": 0, "FAKE": 1},
                "total_models": 1,
            },
            "production_verdict": {
                "label": "FAKE",
                "fusion_label": "FAKE",
                "combined_consensus_label": "FAKE",
            },
            "verdicts_match": True,
            "shadow_mode": True,
        }
        validated = ShadowRoutingSchema(**payload)
        self.assertEqual(validated.image_type, "DIFFUSION")
        self.assertTrue(validated.shadow_mode)
        self.assertTrue(validated.verdicts_match)

    def test_health_schema_includes_router_fields(self):
        """Verify HealthResponseSchema includes router_loaded and router_mode."""
        data = {
            "status": "healthy",
            "primary_model_loaded": True,
            "secondary_model_loaded": True,
            "community_vit_loaded": True,
            "sbi_loaded": True,
            "hybrid_srm_loaded": True,
            "router_loaded": True,
            "router_mode": "shadow",
            "primary_model": "EfficientNet-B0",
            "face_detector": "SCRFD (ONNX)",
            "device": "cpu",
            "cuda_available": False,
        }
        validated = HealthResponseSchema(**data)
        self.assertTrue(validated.router_loaded)
        self.assertEqual(validated.router_mode, "shadow")

    def test_models_info_schema_includes_router_field(self):
        """Verify ModelsInfoResponseSchema includes router configuration dictionary."""
        data = {
            "primary_model": {
                "name": "EfficientNet-B0",
                "checkpoint": "models/efficientnet_b0_best.pth",
                "status": "loaded",
                "input_resolution": "224x224",
                "normalization": "ImageNet",
                "classes": ["FAKE", "REAL"],
                "score_type": "uncalibrated_model_score",
                "description": "Primary model",
            },
            "router": {
                "status": "loaded",
                "mode": "shadow",
                "backbone": "DINOv2 (facebook/dinov2-base, frozen 768-d)",
                "routing_strategy": DEFAULT_SPECIALIST_ROUTING_MAP,
            },
            "default_model": "efficientnet",
            "face_detector": "SCRFD",
            "device": "cpu",
            "cuda_available": False,
            "default_uncertainty_threshold": 0.60,
        }
        validated = ModelsInfoResponseSchema(**data)
        self.assertIsNotNone(validated.router)
        self.assertEqual(validated.router["mode"], "shadow")


if __name__ == "__main__":
    unittest.main()
