"""
Unit tests for the Safeguard Specialist Routing Strategy.

Tests:
1. Low confidence (< 0.85) -> Safeguard group ['efficientnet', 'vit', 'community_vit']
2. Predicted category REAL (even with conf >= 0.85) -> Safeguard group
3. STYLEGAN_GAN with conf >= 0.85 -> ['efficientnet', 'hybrid_srm']
4. DIFFUSION with conf >= 0.85 -> ['community_vit']
5. FACE_SWAP with conf >= 0.85 -> ['sbi', 'community_vit']
6. INPAINTING with conf >= 0.85 -> ['community_vit', 'sbi']
7. OTHER_SYNTHETIC with conf >= 0.85 -> ['community_vit']
8. UNKNOWN -> All five detectors
9. Configurable threshold behavior (e.g. threshold = 0.70 or 0.90)
10. Policy switching between 'safeguard', 'legacy', and 'all'
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.image_detection.inference import (
    DeepFakeInferenceEngine,
    resolve_safeguard_routing,
    SAFEGUARD_DETECTOR_GROUP,
    ALL_DETECTORS_GROUP,
    DEFAULT_SPECIALIST_ROUTING_MAP,
)


class TestSafeguardRoutingPolicy(unittest.TestCase):
    """Verify all 8 rules and configurability of the safeguard policy."""

    def test_low_confidence_routes_to_safeguard_group(self):
        """Any class with confidence < 0.85 must route to safeguard group."""
        classes = ["STYLEGAN_GAN", "DIFFUSION", "FACE_SWAP", "INPAINTING", "OTHER_SYNTHETIC"]
        for cls in classes:
            specialists, reason = resolve_safeguard_routing(cls, confidence=0.84, threshold=0.85)
            self.assertEqual(specialists, SAFEGUARD_DETECTOR_GROUP)
            self.assertIn("low_confidence_safeguard", reason)

    def test_real_category_always_safeguard(self):
        """REAL must route to safeguard group regardless of high confidence."""
        specialists, reason = resolve_safeguard_routing("REAL", confidence=0.99, threshold=0.85)
        self.assertEqual(specialists, SAFEGUARD_DETECTOR_GROUP)
        self.assertIn("real_category_safeguard", reason)

        # Low confidence REAL also routes to safeguard
        specialists_low, reason_low = resolve_safeguard_routing("REAL", confidence=0.50, threshold=0.85)
        self.assertEqual(specialists_low, SAFEGUARD_DETECTOR_GROUP)

    def test_high_confidence_stylegan(self):
        """STYLEGAN_GAN with conf >= 0.85 -> ['efficientnet', 'hybrid_srm']."""
        specialists, reason = resolve_safeguard_routing("STYLEGAN_GAN", confidence=0.85, threshold=0.85)
        self.assertEqual(specialists, ["efficientnet", "hybrid_srm"])
        self.assertIn("high_confidence_stylegan_gan", reason)

    def test_high_confidence_diffusion(self):
        """DIFFUSION with conf >= 0.85 -> ['community_vit']."""
        specialists, reason = resolve_safeguard_routing("DIFFUSION", confidence=0.92, threshold=0.85)
        self.assertEqual(specialists, ["community_vit"])
        self.assertIn("high_confidence_diffusion", reason)

    def test_high_confidence_face_swap(self):
        """FACE_SWAP with conf >= 0.85 -> ['sbi', 'community_vit']."""
        specialists, reason = resolve_safeguard_routing("FACE_SWAP", confidence=0.88, threshold=0.85)
        self.assertEqual(specialists, ["sbi", "community_vit"])
        self.assertIn("high_confidence_face_swap", reason)

    def test_high_confidence_inpainting(self):
        """INPAINTING with conf >= 0.85 -> ['community_vit', 'sbi']."""
        specialists, reason = resolve_safeguard_routing("INPAINTING", confidence=0.90, threshold=0.85)
        self.assertEqual(specialists, ["community_vit", "sbi"])
        self.assertIn("high_confidence_inpainting", reason)

    def test_high_confidence_other_synthetic(self):
        """OTHER_SYNTHETIC with conf >= 0.85 -> ['community_vit']."""
        specialists, reason = resolve_safeguard_routing("OTHER_SYNTHETIC", confidence=0.86, threshold=0.85)
        self.assertEqual(specialists, ["community_vit"])
        self.assertIn("high_confidence_other_synthetic", reason)

    def test_unknown_routes_to_all_models(self):
        """UNKNOWN must route to all 5 detectors."""
        specialists, reason = resolve_safeguard_routing("UNKNOWN", confidence=0.20, threshold=0.85)
        self.assertEqual(specialists, ALL_DETECTORS_GROUP)
        self.assertIn("unknown_typology_fallback", reason)

    def test_configurable_threshold(self):
        """Verify changing threshold adjusts gating boundary."""
        # At 0.85, confidence 0.80 is low confidence -> safeguard
        spec_85, _ = resolve_safeguard_routing("DIFFUSION", confidence=0.80, threshold=0.85)
        self.assertEqual(spec_85, SAFEGUARD_DETECTOR_GROUP)

        # At 0.75, confidence 0.80 is high confidence -> specialist
        spec_75, _ = resolve_safeguard_routing("DIFFUSION", confidence=0.80, threshold=0.75)
        self.assertEqual(spec_75, ["community_vit"])

    def test_engine_configurable_safeguard_policy(self):
        """Verify engine methods for updating threshold and switching policy."""
        engine = DeepFakeInferenceEngine(auto_load=False, load_router=False)
        self.assertEqual(engine.get_routing_policy(), "safeguard")
        self.assertEqual(engine.get_safeguard_confidence_threshold(), 0.85)

        engine.set_safeguard_confidence_threshold(0.78)
        self.assertEqual(engine.get_safeguard_confidence_threshold(), 0.78)

        engine.set_routing_policy("legacy")
        self.assertEqual(engine.get_routing_policy(), "legacy")

        with self.assertRaises(ValueError):
            engine.set_safeguard_confidence_threshold(1.5)

        with self.assertRaises(ValueError):
            engine.set_routing_policy("invalid_policy")


if __name__ == "__main__":
    unittest.main()
