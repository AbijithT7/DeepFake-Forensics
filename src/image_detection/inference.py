"""
Face-Only DeepFake Forensic Inference Engine (EfficientNet-B0).

CRITICAL CONSTRAINTS:
1. Focuses ONLY on EfficientNet-B0.
2. Analyzes ONLY detected face regions, never the full-scene photograph.
3. Uses robust deep-learning face detection (SCRFD) with controlled margin cropping.
4. Pre-classification face quality and visibility validation:
   - no_face_detected
   - low_face_detection_confidence
   - face_too_small
   - face_too_blurry
   - face_not_usable (severe truncation or illumination/block occlusion)
5. Never falls back to classifying the full original image.
6. Exact EfficientNet-B0 ImageNet normalization and 224x224 input sizing.
7. Uncalibrated decision thresholding with explicit INCONCLUSIVE states.
8. Exposes modular Grad-CAM inputs for downstream explainability integration.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VERDICT_CONFIG_PATH = Path(__file__).resolve().parent / "verdict_config.json"

from src.explainability.gradcam import GradCAM
from src.explainability.visualizer import GradCAMVisualizer
from src.image_detection.efficientnet_model import (
    CLASS_NAMES,
    DeepfakeEfficientNetB0,
    get_eval_transform as get_efficientnet_transform,
)
from src.image_detection.face_detector import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_CONTEXT_MARGIN,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    SCRFDFaceDetector,
    crop_face_with_controlled_margin,
    validate_face_quality,
)
from src.image_detection.community_forensics_vit import CommunityForensicsViTDetector
from src.image_detection.hybrid_srm_detector import HybridDeepfakeDetector
from src.image_detection.predict_efficientnet import load_efficientnet_model
from src.image_detection.sbi_detector import SBIDetector
from src.image_detection.vit_detector import ViTDetector

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_CHECKPOINT = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
DEFAULT_UNCERTAINTY_THRESHOLD = 0.60
SCORE_TYPE_UNSETTLED = "uncalibrated_model_score"

DEFAULT_SPECIALIST_ROUTING_MAP: Dict[str, List[str]] = {
    "REAL": ["efficientnet", "vit"],
    "STYLEGAN_GAN": ["efficientnet", "hybrid_srm"],
    "DIFFUSION": ["community_vit"],
    "FACE_SWAP": ["sbi"],
    "INPAINTING": ["community_vit", "sbi"],
    "OTHER_SYNTHETIC": ["community_vit"],
    "UNKNOWN": ["efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"],
}

DETECTOR_DISPLAY_NAMES: Dict[str, str] = {
    "efficientnet": "EfficientNet-B0",
    "vit": "Original ViT",
    "community_vit": "Community Forensics ViT",
    "sbi": "SBI EfficientNet-B4",
    "hybrid_srm": "Hybrid SRM",
}

DEFAULT_SAFEGUARD_CONFIDENCE_THRESHOLD: float = 0.85
SAFEGUARD_DETECTOR_GROUP: List[str] = ["efficientnet", "vit", "community_vit"]
ALL_DETECTORS_GROUP: List[str] = ["efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"]

SAFEGUARD_SPECIALIST_HIGH_CONF_MAP: Dict[str, List[str]] = {
    "STYLEGAN_GAN": ["efficientnet", "hybrid_srm"],
    "DIFFUSION": ["community_vit"],
    "FACE_SWAP": ["sbi", "community_vit"],
    "INPAINTING": ["community_vit", "sbi"],
    "OTHER_SYNTHETIC": ["community_vit"],
}


def resolve_safeguard_routing(
    predicted_type: str,
    confidence: float,
    threshold: float = DEFAULT_SAFEGUARD_CONFIDENCE_THRESHOLD,
    policy: str = "safeguard",
    legacy_routing_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], str]:
    """
    Resolve detector specialists according to the routing policy.

    Policies:
    - 'safeguard': Confidence gating (< threshold -> safeguard group),
                   REAL category -> safeguard group,
                   High-confidence specialists for STYLEGAN, DIFFUSION, FACE_SWAP, INPAINTING, OTHER_SYNTHETIC,
                   UNKNOWN -> all five detectors.
    - 'legacy': Direct lookup in specialist_routing_map.
    - 'all': All five specialists.

    Returns (selected_detector_keys, policy_reason).
    """
    if policy == "all":
        return list(ALL_DETECTORS_GROUP), "all_models_policy"

    if policy == "legacy":
        routing_map = legacy_routing_map or DEFAULT_SPECIALIST_ROUTING_MAP
        keys = routing_map.get(predicted_type, list(ALL_DETECTORS_GROUP))
        return list(keys), f"legacy_routing_{predicted_type}"

    # Safeguard policy:
    # 8. UNKNOWN: All five detectors
    if predicted_type == "UNKNOWN":
        return list(ALL_DETECTORS_GROUP), "unknown_typology_fallback_all_models"

    # 1. If router confidence < threshold:
    # Use safeguard group: EfficientNet-B0, Original ViT, Community Forensics ViT
    if confidence < threshold:
        return (
            list(SAFEGUARD_DETECTOR_GROUP),
            f"low_confidence_safeguard (conf={confidence:.4f} < {threshold:.2f})",
        )

    # 2. If predicted category is REAL:
    # Use the same safeguard group
    if predicted_type == "REAL":
        return (
            list(SAFEGUARD_DETECTOR_GROUP),
            f"real_category_safeguard (conf={confidence:.4f} >= {threshold:.2f})",
        )

    # High confidence synthetic categories (confidence >= threshold):
    # 3. STYLEGAN_GAN: EfficientNet-B0 + Hybrid SRM
    # 4. DIFFUSION: Community Forensics ViT
    # 5. FACE_SWAP: SBI EfficientNet-B4 + Community Forensics ViT
    # 6. INPAINTING: Community Forensics ViT + SBI
    # 7. OTHER_SYNTHETIC: Community Forensics ViT
    if predicted_type in SAFEGUARD_SPECIALIST_HIGH_CONF_MAP:
        models = SAFEGUARD_SPECIALIST_HIGH_CONF_MAP[predicted_type]
        return (
            list(models),
            f"high_confidence_{predicted_type.lower()} (conf={confidence:.4f} >= {threshold:.2f})",
        )

    return list(ALL_DETECTORS_GROUP), f"unmapped_category_{predicted_type}_fallback"


class InvalidImageError(ValueError):
    """Raised when an uploaded or specified image cannot be read, decoded, or validated."""
    pass


def load_verdict_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load verdict configuration JSON with fallback defaults."""
    cfg_file = Path(config_path or os.environ.get("VERDICT_CONFIG_PATH") or DEFAULT_VERDICT_CONFIG_PATH)
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.warning("Failed to parse verdict config %s: %s", cfg_file, err)
    return {
        "active_engine": "v3",
        "allow_real": False,
        "preprocessing": {
            "vit_uncropped": True,
            "community_vit_uncropped": True,
            "efficientnet_margin_ratio": 0.15,
            "sbi_margin_ratio": 0.15,
            "hybrid_srm_margin_ratio": 0.15,
        },
        "thresholds": {
            "efficientnet": 0.9000,
            "vit": 0.9000,
            "community_vit": 1.0000,
            "sbi": 0.9997,
            "hybrid_srm": 0.9000,
        },
        "legacy_config": {
            "uncertainty_threshold": 0.60,
            "fusion_models": ["efficientnet", "vit"],
        },
    }


def evaluate_verdict_v3(
    model_predictions: Dict[str, Optional[Dict[str, Any]]],
    thresholds: Dict[str, float],
    allow_real: bool = False,
) -> Dict[str, Any]:
    """
    Verdict Engine v3: Specialist-Any-FAKE decision rule.
    
    A. Input availability:
       - If all specialists unavailable -> INCONCLUSIVE, reason "all_models_unavailable".
       - Record each unavailable specialist.
       - One unavailable model does not block remaining available models.
    B. FAKE decision:
       - For each available specialist, fires if P(fake_i) >= T_i.
       - If at least one fires -> FAKE.
       - Returns triggering models, triggering scores, primary trigger (highest score), thresholds.
    C. INCONCLUSIVE decision:
       - If no available specialist crosses threshold -> INCONCLUSIVE, reason "no_high_confidence_fake_signal".
       - Never emit REAL in v3 (allow_real=False).
    D. Restrictions:
       - No weighted voting, no majority voting, no consensus logic, no router influence.
    """
    for m, t in thresholds.items():
        if not (0.0 <= float(t) <= 1.0):
            raise ValueError(f"Invalid threshold for {m}: {t}. Must be between 0.0 and 1.0.")

    available_scores: Dict[str, float] = {}
    unavailable_models: List[str] = []

    for m, t in thresholds.items():
        pred = model_predictions.get(m)
        if not pred or not pred.get("loaded", False) or pred.get("error"):
            unavailable_models.append(m)
            continue

        prob_fake: Optional[float] = None
        probs = pred.get("probabilities")
        if probs and "FAKE" in probs:
            prob_fake = float(probs["FAKE"])
        elif pred.get("label") is not None and pred.get("score") is not None:
            if pred.get("label") == "FAKE":
                prob_fake = float(pred.get("score"))
            elif pred.get("label") == "REAL":
                prob_fake = 1.0 - float(pred.get("score"))

        if prob_fake is None:
            unavailable_models.append(m)
            continue

        available_scores[m] = round(prob_fake, 4)

    if len(available_scores) == 0:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "all_models_unavailable",
            "available_scores": {},
            "unavailable_models": unavailable_models,
            "triggering_models": [],
            "triggering_scores": {},
            "primary_trigger": None,
            "thresholds": dict(thresholds),
            "verdict_engine": "v3",
            "verdict_source": "specialist_any_fake_v3",
            "allow_real": allow_real,
        }

    triggering_models: List[str] = []
    triggering_scores: Dict[str, float] = {}
    for m, score in available_scores.items():
        if score >= thresholds[m]:
            triggering_models.append(m)
            triggering_scores[m] = score

    if len(triggering_models) > 0:
        primary_trigger = max(triggering_models, key=lambda m: (triggering_scores[m], m))
        return {
            "verdict": "FAKE",
            "reason": "specialist_threshold_exceeded",
            "available_scores": available_scores,
            "unavailable_models": unavailable_models,
            "triggering_models": triggering_models,
            "triggering_scores": triggering_scores,
            "primary_trigger": primary_trigger,
            "thresholds": dict(thresholds),
            "verdict_engine": "v3",
            "verdict_source": "specialist_any_fake_v3",
            "allow_real": allow_real,
        }

    return {
        "verdict": "INCONCLUSIVE",
        "reason": "no_high_confidence_fake_signal",
        "available_scores": available_scores,
        "unavailable_models": unavailable_models,
        "triggering_models": [],
        "triggering_scores": {},
        "primary_trigger": None,
        "thresholds": dict(thresholds),
        "verdict_engine": "v3",
        "verdict_source": "specialist_any_fake_v3",
        "allow_real": allow_real,
    }


def fuse_model_predictions(
    efficientnet: Dict[str, Any],
    vit: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Apply the unchanged conservative fusion policy and expose its reason."""
    e_score = efficientnet.get("score")
    v_score = vit.get("score")
    if e_score is None or v_score is None:
        return {"label": "INCONCLUSIVE", "score": None, "reason": "model_unavailable_or_no_face"}
    score = round(min(float(e_score), float(v_score)), 4)
    if float(e_score) < threshold or float(v_score) < threshold:
        return {"label": "INCONCLUSIVE", "score": score, "reason": "low_confidence"}
    if efficientnet.get("label") != vit.get("label"):
        return {"label": "INCONCLUSIVE", "score": score, "reason": "model_disagreement"}
    return {"label": efficientnet["label"], "score": score, "reason": None}


def compute_combined_result(
    model_predictions: Dict[str, Optional[Dict[str, Any]]],
    threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Combine predictions across heterogeneous deepfake detectors without assuming score calibration.

    Scores are uncalibrated and non-comparable across disparate architectures:
    - EfficientNet-B0: FaceForensics++ video compression artifacts
    - ViT (HF): Vision Transformer token patterns
    - Community Forensics ViT: 4,800+ generative model / diffusion artifacts
    - SBI EfficientNet-B4: Self-Blended Image boundary inconsistencies
    - Hybrid SRM CNN: Spatial EfficientNet-B4 + Steganalysis noise residual Xception (140k faces)

    Instead of averaging raw uncalibrated probabilities, this computes:
    - Individual model votes (REAL / FAKE) when confidence meets threshold.
    - Disagreement breakdown.
    - Consensus level: 'unanimous_fake', 'unanimous_real', 'majority_fake', 'majority_real', 'split_decision', or 'inconclusive'.
    """
    valid_votes: Dict[str, str] = {}
    low_confidence_models: List[str] = []
    unavailable_models: List[str] = []

    for model_key, pred in model_predictions.items():
        if not pred or not pred.get("loaded", False):
            unavailable_models.append(model_key)
            continue
        lbl = pred.get("label")
        scr = pred.get("score")
        if scr is None or lbl not in ("REAL", "FAKE"):
            unavailable_models.append(model_key)
        elif scr < threshold:
            low_confidence_models.append(model_key)
        else:
            valid_votes[model_key] = lbl

    real_count = sum(1 for v in valid_votes.values() if v == "REAL")
    fake_count = sum(1 for v in valid_votes.values() if v == "FAKE")
    total_votes = len(valid_votes)

    if total_votes == 0:
        return {
            "label": "INCONCLUSIVE",
            "consensus": "inconclusive",
            "votes": {"REAL": 0, "FAKE": 0},
            "total_models": len(model_predictions),
            "reason": "no_models_exceeded_confidence_threshold",
            "disclaimer": "Individual model scores are uncalibrated and cannot be directly averaged.",
            "low_confidence_models": low_confidence_models,
            "unavailable_models": unavailable_models,
        }

    if fake_count == total_votes:
        consensus = "unanimous_fake"
        combined_label = "FAKE"
        reason = f"All {total_votes} qualifying models unanimously agreed on FAKE."
    elif real_count == total_votes:
        consensus = "unanimous_real"
        combined_label = "REAL"
        reason = f"All {total_votes} qualifying models unanimously agreed on REAL."
    elif fake_count > real_count:
        consensus = "majority_fake"
        combined_label = "FAKE"
        reason = f"Majority vote: {fake_count} model(s) detected FAKE, {real_count} detected REAL."
    elif real_count > fake_count:
        consensus = "majority_real"
        combined_label = "REAL"
        reason = f"Majority vote: {real_count} model(s) detected REAL, {fake_count} detected FAKE."
    else:
        consensus = "split_decision"
        combined_label = "INCONCLUSIVE"
        reason = f"Split decision: {real_count} model(s) voted REAL and {fake_count} voted FAKE."

    return {
        "label": combined_label,
        "consensus": consensus,
        "votes": {"REAL": real_count, "FAKE": fake_count},
        "qualifying_votes": valid_votes,
        "total_models_evaluated": len(model_predictions),
        "reason": reason,
        "disclaimer": (
            "Model scores are uncalibrated across architectures and training domains. "
            "Combined result represents consensus voting without raw probability averaging."
        ),
        "low_confidence_models": low_confidence_models,
        "unavailable_models": unavailable_models,
    }


@dataclass
class FaceCropArtifact:
    """Artifacts associated with an extracted face crop for inference, testing, and XAI."""
    face_id: int
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    crop_bgr: np.ndarray
    crop_pil: Image.Image
    padded_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_score: float
    is_usable: bool
    quality_reason: Optional[str] = None


def pil_to_base64_data_url(img: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL image to a base64 data URL string."""
    buf = io.BytesIO()
    if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=format, quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64_str}"


@dataclass
class FacePrediction:
    """Structured detection outcome for an individual detected face region."""
    face_id: int
    label: str  # REAL, FAKE, or INCONCLUSIVE
    score: Optional[float]  # Uncalibrated score for winning class, or None
    score_type: str = SCORE_TYPE_UNSETTLED
    box: Optional[List[int]] = None  # [x, y, w, h]
    probabilities: Optional[Dict[str, float]] = None  # e.g. {"FAKE": 0.05, "REAL": 0.95}
    raw_logits: Optional[Dict[str, float]] = None
    reason: Optional[str] = None  # no_face_detected, face_too_small, face_too_blurry, etc.
    face_crop_base64: Optional[str] = None
    gradcam_heatmap_base64: Optional[str] = None
    gradcam_overlay_base64: Optional[str] = None
    efficientnet: Optional[Dict[str, Any]] = None
    vit: Optional[Dict[str, Any]] = None
    community_vit: Optional[Dict[str, Any]] = None
    sbi: Optional[Dict[str, Any]] = None
    hybrid_srm: Optional[Dict[str, Any]] = None
    fusion: Optional[Dict[str, Any]] = None
    combined_result: Optional[Dict[str, Any]] = None
    shadow_routing: Optional[Dict[str, Any]] = None
    # Specialist-Any-FAKE v3 fields:
    verdict_source: Optional[str] = None
    verdict_engine: Optional[str] = None
    verdict_reason: Optional[str] = None
    available_scores: Optional[Dict[str, float]] = None
    unavailable_models: Optional[List[str]] = None
    triggering_models: Optional[List[str]] = None
    triggering_scores: Optional[Dict[str, float]] = None
    primary_trigger: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    v3_verdict: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_id": self.face_id,
            "label": self.label,
            "score": self.score,
            "score_type": self.score_type,
            "box": self.box,
            "probabilities": self.probabilities,
            "reason": self.reason,
            "face_crop_base64": self.face_crop_base64,
            "gradcam_heatmap_base64": self.gradcam_heatmap_base64,
            "gradcam_overlay_base64": self.gradcam_overlay_base64,
            "efficientnet": self.efficientnet,
            "vit": self.vit,
            "community_vit": self.community_vit,
            "sbi": self.sbi,
            "hybrid_srm": self.hybrid_srm,
            "fusion": self.fusion,
            "combined_result": self.combined_result,
            "shadow_routing": self.shadow_routing,
            "verdict_source": self.verdict_source,
            "verdict_engine": self.verdict_engine,
            "verdict_reason": self.verdict_reason,
            "available_scores": self.available_scores,
            "unavailable_models": self.unavailable_models,
            "triggering_models": self.triggering_models,
            "triggering_scores": self.triggering_scores,
            "primary_trigger": self.primary_trigger,
            "thresholds": self.thresholds,
            "v3_verdict": self.v3_verdict,
        }


@dataclass
class ImageInferenceResult:
    """Structured inference response for an entire image."""
    status: str  # "success" or "inconclusive"
    faces_detected: int
    model_name: str
    checkpoint_path: str
    execution_device: str
    uncertainty_threshold: float
    predictions: List[FacePrediction]
    message: Optional[str] = None
    shadow_routing: Optional[Dict[str, Any]] = None
    active_verdict_engine: str = "v3"
    verdict_source: str = "specialist_any_fake_v3"

    @property
    def final_prediction(self) -> FacePrediction:
        """One image verdict according to the active verdict engine."""
        if not self.predictions:
            return FacePrediction(
                face_id=0,
                label="INCONCLUSIVE",
                score=None,
                reason="no_usable_face",
                verdict_source=self.verdict_source,
                verdict_engine=self.active_verdict_engine,
                verdict_reason="no_usable_face",
                available_scores={},
                unavailable_models=[],
                triggering_models=[],
                triggering_scores={},
                primary_trigger=None,
                thresholds=None,
            )

        if self.active_verdict_engine == "v3":
            # Any face detected as FAKE makes the whole image FAKE
            fake_preds = [p for p in self.predictions if p.label == "FAKE"]
            if fake_preds:
                # Representative is the highest-scoring fake prediction
                winning_pred = max(fake_preds, key=lambda p: (p.score or 0.0))
                return FacePrediction(
                    face_id=0,
                    label="FAKE",
                    score=winning_pred.score,
                    reason=winning_pred.reason or "specialist_threshold_exceeded",
                    face_crop_base64=winning_pred.face_crop_base64,
                    gradcam_heatmap_base64=winning_pred.gradcam_heatmap_base64,
                    gradcam_overlay_base64=winning_pred.gradcam_overlay_base64,
                    efficientnet=winning_pred.efficientnet,
                    vit=winning_pred.vit,
                    community_vit=winning_pred.community_vit,
                    sbi=winning_pred.sbi,
                    hybrid_srm=winning_pred.hybrid_srm,
                    fusion=winning_pred.fusion,
                    combined_result=winning_pred.combined_result,
                    shadow_routing=winning_pred.shadow_routing,
                    verdict_source=winning_pred.verdict_source or self.verdict_source,
                    verdict_engine="v3",
                    verdict_reason=winning_pred.verdict_reason or winning_pred.reason or "specialist_threshold_exceeded",
                    available_scores=winning_pred.available_scores,
                    unavailable_models=winning_pred.unavailable_models,
                    triggering_models=winning_pred.triggering_models,
                    triggering_scores=winning_pred.triggering_scores,
                    primary_trigger=winning_pred.primary_trigger,
                    thresholds=winning_pred.thresholds,
                    v3_verdict=winning_pred.v3_verdict,
                )

            # All predictions are INCONCLUSIVE
            first_pred = self.predictions[0]
            # Check reasons across predictions
            reasons = [p.reason for p in self.predictions if p.reason]
            if any(r == "no_usable_face" for r in reasons) and all(r in ("no_usable_face", "no_face_detected") or "face" in str(r) for r in reasons):
                final_reason = "no_usable_face"
            elif any(r == "all_models_unavailable" for r in reasons):
                final_reason = "all_models_unavailable"
            else:
                final_reason = "no_high_confidence_fake_signal"

            return FacePrediction(
                face_id=0,
                label="INCONCLUSIVE",
                score=None,
                reason=final_reason,
                face_crop_base64=first_pred.face_crop_base64,
                gradcam_heatmap_base64=first_pred.gradcam_heatmap_base64,
                gradcam_overlay_base64=first_pred.gradcam_overlay_base64,
                efficientnet=first_pred.efficientnet,
                vit=first_pred.vit,
                community_vit=first_pred.community_vit,
                sbi=first_pred.sbi,
                hybrid_srm=first_pred.hybrid_srm,
                fusion=first_pred.fusion,
                combined_result=first_pred.combined_result,
                shadow_routing=first_pred.shadow_routing,
                verdict_source=first_pred.verdict_source or self.verdict_source,
                verdict_engine="v3",
                verdict_reason=final_reason,
                available_scores=first_pred.available_scores,
                unavailable_models=first_pred.unavailable_models,
                triggering_models=[],
                triggering_scores={},
                primary_trigger=None,
                thresholds=first_pred.thresholds,
                v3_verdict=first_pred.v3_verdict,
            )

        # Legacy 2-model fusion property logic
        conclusive = [p for p in self.predictions if p.label != "INCONCLUSIVE"]
        if not conclusive or len(conclusive) != len(self.predictions):
            base = self.predictions[0]
            return FacePrediction(
                0, "INCONCLUSIVE", base.score, reason=base.reason or "inconclusive_face_result",
                face_crop_base64=base.face_crop_base64,
                gradcam_heatmap_base64=base.gradcam_heatmap_base64,
                gradcam_overlay_base64=base.gradcam_overlay_base64,
                efficientnet=base.efficientnet,
                vit=base.vit,
                community_vit=base.community_vit,
                sbi=base.sbi,
                hybrid_srm=base.hybrid_srm,
                fusion=base.fusion,
                combined_result=base.combined_result,
                shadow_routing=base.shadow_routing,
                verdict_source=self.verdict_source,
                verdict_engine=self.active_verdict_engine,
            )
        labels = {p.label for p in conclusive}
        if len(labels) != 1:
            base = conclusive[0]
            return FacePrediction(
                0, "INCONCLUSIVE", None, reason="multiple_face_disagreement",
                community_vit=base.community_vit,
                sbi=base.sbi,
                hybrid_srm=base.hybrid_srm,
                fusion={"label": "INCONCLUSIVE", "score": None, "reason": "multiple_face_disagreement"},
                combined_result={"label": "INCONCLUSIVE", "consensus": "split_decision", "reason": "multiple_face_disagreement"},
                shadow_routing=base.shadow_routing,
                verdict_source=self.verdict_source,
                verdict_engine=self.active_verdict_engine,
            )
        score = round(min(p.score for p in conclusive if p.score is not None), 4)
        representative = conclusive[0]
        return FacePrediction(
            0, representative.label, score, reason=None,
            face_crop_base64=representative.face_crop_base64,
            gradcam_heatmap_base64=representative.gradcam_heatmap_base64,
            gradcam_overlay_base64=representative.gradcam_overlay_base64,
            efficientnet=representative.efficientnet,
            vit=representative.vit,
            community_vit=representative.community_vit,
            sbi=representative.sbi,
            hybrid_srm=representative.hybrid_srm,
            fusion=representative.fusion,
            combined_result=representative.combined_result,
            shadow_routing=representative.shadow_routing,
            verdict_source=self.verdict_source,
            verdict_engine=self.active_verdict_engine,
        )

    def to_dict(self) -> Dict[str, Any]:
        final = self.final_prediction
        return {
            "status": self.status,
            "faces_detected": self.faces_detected,
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "execution_device": self.execution_device,
            "uncertainty_threshold": self.uncertainty_threshold,
            "message": self.message,
            "final_prediction": final.to_dict(),
            "predictions": [p.to_dict() for p in self.predictions],
            "shadow_routing": self.shadow_routing,
            "verdict_source": getattr(self, "verdict_source", "specialist_any_fake_v3"),
            "active_verdict_engine": getattr(self, "active_verdict_engine", "v3"),
            "verdict": final.label,
            "verdict_reason": final.reason,
            "available_scores": getattr(final, "available_scores", None),
            "unavailable_models": getattr(final, "unavailable_models", None),
            "triggering_models": getattr(final, "triggering_models", None),
            "primary_trigger": getattr(final, "primary_trigger", None),
            "thresholds": getattr(final, "thresholds", None),
        }


def detect_execution_device() -> torch.device:
    """
    Detect and return the active compute device.
    Supports CPU and CUDA without misrepresenting hardware capability.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using NVIDIA CUDA acceleration: {device_name}")
        return torch.device("cuda")
    logger.info("Using CPU execution device")
    return torch.device("cpu")


class DeepFakeInferenceEngine:
    """
    Production-grade Inference Engine for DeepFake Media Forensics.

    Guarantees that classification is ONLY performed on detected, validated face crops
    using the EfficientNet-B0 architecture.
    """

    def __init__(
        self,
        primary_checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        auto_load: bool = True,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        context_margin: float = DEFAULT_CONTEXT_MARGIN,
        specialist_routing_map: Optional[Dict[str, List[str]]] = None,
        shadow_mode: bool = True,
        load_router: bool = True,
        routing_policy: str = "safeguard",
        safeguard_confidence_threshold: float = DEFAULT_SAFEGUARD_CONFIDENCE_THRESHOLD,
        verdict_config_path: Optional[Union[str, Path]] = None,
    ):
        self.primary_checkpoint = Path(primary_checkpoint or DEFAULT_PRIMARY_CHECKPOINT)
        self.device = device or detect_execution_device()
        self.min_detection_confidence = min_detection_confidence
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.context_margin = context_margin
        self.specialist_routing_map = (
            dict(specialist_routing_map) if specialist_routing_map is not None
            else dict(DEFAULT_SPECIALIST_ROUTING_MAP)
        )
        self.shadow_mode = shadow_mode
        self.load_router = load_router
        self.routing_policy = routing_policy
        self.safeguard_confidence_threshold = float(safeguard_confidence_threshold)

        # Verdict engine configuration (v3 Specialist-Any-FAKE vs legacy)
        self.verdict_config = load_verdict_config(verdict_config_path)
        self.active_engine = os.environ.get("VERDICT_ENGINE") or self.verdict_config.get("active_engine", "v3")
        self.allow_real = bool(self.verdict_config.get("allow_real", False))
        self.v3_thresholds = {k: float(v) for k, v in self.verdict_config.get("thresholds", {}).items()}
        self.preprocessing_config = self.verdict_config.get("preprocessing", {})
        self.vit_uncropped = bool(self.preprocessing_config.get("vit_uncropped", True))
        self.community_vit_uncropped = bool(self.preprocessing_config.get("community_vit_uncropped", True))

        self.primary_model: Optional[DeepfakeEfficientNetB0] = None
        self.secondary_model: Optional[ViTDetector] = None
        self.community_vit_model: Optional[CommunityForensicsViTDetector] = None
        self.sbi_model: Optional[SBIDetector] = None
        self.hybrid_srm_model: Optional[HybridDeepfakeDetector] = None
        self.image_type_router: Optional[Any] = None
        self.detector = SCRFDFaceDetector(conf_threshold=self.min_detection_confidence)
        self.crop_artifacts: Dict[int, FaceCropArtifact] = {}
        self.last_received_tensor_shape: Optional[Tuple[int, ...]] = None

        if auto_load:
            self.load_models()

    def set_safeguard_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for safeguard specialist routing."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"safeguard threshold must be between 0.0 and 1.0, got {threshold}")
        self.safeguard_confidence_threshold = float(threshold)

    def get_safeguard_confidence_threshold(self) -> float:
        """Return the active safeguard confidence threshold."""
        return float(self.safeguard_confidence_threshold)

    def set_routing_policy(self, policy: str) -> None:
        """Set active routing policy ('safeguard', 'legacy', or 'all')."""
        if policy not in ("safeguard", "legacy", "all"):
            raise ValueError(f"Unknown routing policy '{policy}'")
        self.routing_policy = policy

    def get_routing_policy(self) -> str:
        """Return the active routing policy."""
        return self.routing_policy

    def set_specialist_routing_map(self, routing_map: Dict[str, List[str]]) -> None:
        """Update the specialist routing mapping dynamically."""
        self.specialist_routing_map = dict(routing_map)

    def get_specialist_routing_map(self) -> Dict[str, List[str]]:
        """Return the current specialist routing mapping."""
        return dict(self.specialist_routing_map)

    def load_models(self) -> None:
        """
        Load EfficientNet-B0, ViT, Community Forensics ViT, SBI, Hybrid SRM, and ImageTypeRouter models into memory.
        """
        logger.info("Initializing primary model: EfficientNet-B0")
        self.primary_model, _ = load_efficientnet_model(
            checkpoint_path=self.primary_checkpoint,
            device=self.device,
        )
        self.primary_model.eval()
        try:
            self.secondary_model = ViTDetector(
                device=self.device,
                local_files_only=True,
            )
            logger.info("Loaded secondary model: ViT (%s)", self.secondary_model.id2label)
        except Exception:
            self.secondary_model = None
            logger.exception("Failed to load secondary ViT model")

        try:
            self.community_vit_model = CommunityForensicsViTDetector(device=self.device)
            logger.info("Loaded Community Forensics ViT detector")
        except Exception:
            self.community_vit_model = None
            logger.exception("Failed to load Community Forensics ViT detector")

        try:
            self.sbi_model = SBIDetector(device=self.device)
            logger.info("Loaded SBI EfficientNet-B4 detector")
        except Exception:
            self.sbi_model = None
            logger.exception("Failed to load SBI EfficientNet-B4 detector")

        try:
            self.hybrid_srm_model = HybridDeepfakeDetector(device=self.device)
            logger.info("Loaded Hybrid SRM (EfficientNet-B4 + Xception) detector")
        except Exception:
            self.hybrid_srm_model = None
            logger.exception("Failed to load Hybrid SRM detector")

        if self.load_router:
            try:
                from src.image_detection.image_type_router import ImageTypeRouter
                self.image_type_router = ImageTypeRouter(
                    device=self.device,
                    local_files_only=True,
                )
                logger.info("Loaded ImageTypeRouter (Shadow Mode active)")
            except Exception:
                self.image_type_router = None
                logger.exception("Failed to load ImageTypeRouter")

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return (
            self.primary_model is not None
            and self.secondary_model is not None
            and self.community_vit_model is not None
            and self.sbi_model is not None
            and self.hybrid_srm_model is not None
        )

    def validate_and_read_image(
        self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Validate and decode the input image into both OpenCV BGR and PIL RGB representations.

        Raises:
            InvalidImageError: If the input cannot be decoded or has invalid dimensions.
        """
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.is_file():
                    raise InvalidImageError(f"Image file does not exist: {img_path}")
                if img_path.stat().st_size == 0:
                    raise InvalidImageError(f"Image file is empty (0 bytes): {img_path}")

                cv_img = cv2.imread(str(img_path))
                if cv_img is None:
                    raise InvalidImageError(f"Failed to decode image from path: {img_path}")
                with Image.open(img_path) as pil_raw:
                    pil_img = pil_raw.convert("RGB")

            elif isinstance(image_input, (bytes, bytearray)):
                if len(image_input) == 0:
                    raise InvalidImageError("Uploaded image payload is empty (0 bytes)")
                np_buf = np.frombuffer(image_input, dtype=np.uint8)
                cv_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if cv_img is None:
                    raise InvalidImageError("Could not decode image from provided byte stream")
                with Image.open(io.BytesIO(image_input)) as pil_raw:
                    pil_img = pil_raw.convert("RGB")

            elif isinstance(image_input, io.BytesIO):
                image_input.seek(0)
                bytes_data = image_input.read()
                return self.validate_and_read_image(bytes_data)

            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    raise InvalidImageError("Provided numpy image array is empty")
                if len(image_input.shape) == 2:
                    cv_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                elif len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    cv_img = image_input.copy()
                elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
                    cv_img = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
                else:
                    raise InvalidImageError(f"Unsupported image array shape: {image_input.shape}")
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            else:
                raise InvalidImageError(f"Unsupported image input type: {type(image_input).__name__}")

            # Dimensions validation
            h, w = cv_img.shape[:2]
            if h < 10 or w < 10:
                raise InvalidImageError(
                    f"Image resolution too small ({w}x{h}). Minimum required is 10x10 pixels."
                )

            return cv_img, pil_img

        except InvalidImageError:
            raise
        except Exception as err:
            raise InvalidImageError(f"Corrupt or unreadable image file: {err}") from err

    def detect_and_crop_faces(
        self,
        image_bgr: np.ndarray,
        margin_ratio: Optional[float] = None,
    ) -> List[FaceCropArtifact]:
        """
        Detect faces using SCRFD and crop each independently with a controlled margin.
        Evaluates pre-classification quality and visibility.
        """
        margin = margin_ratio if margin_ratio is not None else self.context_margin
        detections = self.detector.detect(image_bgr, conf_threshold=self.min_detection_confidence)
        artifacts: List[FaceCropArtifact] = []

        for idx, det in enumerate(detections):
            box = det["box"]  # (x, y, w, h)
            det_score = det["score"]

            crop_bgr, padded_box = crop_face_with_controlled_margin(image_bgr, box, margin_ratio=margin)
            if crop_bgr.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            # Pre-classification quality validation
            is_usable, quality_reason = validate_face_quality(
                crop_bgr=crop_bgr,
                detection_score=det_score,
                original_image_shape=image_bgr.shape,
                box=box,
                truncation_ratio=det.get("truncation_ratio", 1.0),
                min_confidence=self.min_detection_confidence,
                min_size=self.min_face_size,
                blur_threshold=self.blur_threshold,
            )

            artifacts.append(
                FaceCropArtifact(
                    face_id=idx + 1,
                    box=box,
                    crop_bgr=crop_bgr,
                    crop_pil=crop_pil,
                    padded_box=padded_box,
                    detection_score=det_score,
                    is_usable=is_usable,
                    quality_reason=quality_reason,
                )
            )

        return artifacts

    def predict_image(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
        model_name: str = "efficientnet",
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        include_gradcam: bool = True,
    ) -> ImageInferenceResult:
        """
        Execute face-only deepfake detection on an image.

        Guarantees:
        - Only detected face crops are classified.
        - The full photograph is NEVER classified.
        - Low quality/truncated faces return INCONCLUSIVE with specific reasons.
        """
        if self.primary_model is None:
            raise RuntimeError("EfficientNet-B0 model is not loaded.")

        if not (0.50 <= uncertainty_threshold <= 1.0):
            raise ValueError(
                f"uncertainty_threshold must be between 0.50 and 1.00, got {uncertainty_threshold}"
            )

        # 1. Validate and decode image
        cv_img, pil_img = self.validate_and_read_image(image_input)

        # 1b. Shadow Mode: Route image typology via ImageTypeRouter
        router_pred = None
        if self.image_type_router is not None:
            try:
                router_pred = self.image_type_router.route_image(pil_img)
            except Exception as r_err:
                logger.warning(f"Router inference failed: {r_err}")
                router_pred = None

        if router_pred is not None:
            predicted_image_type = router_pred.predicted_class
            raw_image_type = router_pred.raw_predicted_class
            router_conf = round(float(router_pred.confidence), 4)
            is_rejected = router_pred.is_rejected
            rejection_reason = router_pred.rejection_reason
            router_probs = {k: round(float(v), 4) for k, v in router_pred.probabilities.items()}
        else:
            predicted_image_type = "UNKNOWN"
            raw_image_type = "UNKNOWN"
            router_conf = 0.0
            is_rejected = True
            rejection_reason = "router_unavailable_or_failed"
            router_probs = {}

        selected_keys, policy_reason = resolve_safeguard_routing(
            predicted_type=predicted_image_type,
            confidence=router_conf,
            threshold=self.safeguard_confidence_threshold,
            policy=self.routing_policy,
            legacy_routing_map=self.specialist_routing_map,
        )
        selected_display_names = [DETECTOR_DISPLAY_NAMES.get(k, k) for k in selected_keys]
        logger.info(
            "Routing policy '%s' selected specialists %s (reason: %s, predicted: %s, conf: %.4f)",
            self.routing_policy,
            selected_keys,
            policy_reason,
            predicted_image_type,
            router_conf,
        )

        # 2. Detect and crop faces using SCRFD
        face_artifacts = self.detect_and_crop_faces(cv_img)
        self.crop_artifacts = {art.face_id: art for art in face_artifacts}

        display_model_name = "EfficientNet-B0"
        checkpoint_str = str(self.primary_checkpoint)

        # 3. Case A: No faces detected in image -> INCONCLUSIVE
        if len(face_artifacts) == 0:
            logger.info("No face detected in image; returning INCONCLUSIVE.")
            shadow_routing_data = {
                "routing_policy": self.routing_policy,
                "safeguard_threshold": self.safeguard_confidence_threshold,
                "policy_reason": policy_reason,
                "image_type": predicted_image_type,
                "raw_image_type": raw_image_type,
                "router_confidence": router_conf,
                "is_rejected": is_rejected,
                "rejection_reason": rejection_reason,
                "router_probabilities": router_probs,
                "selected_specialists": selected_display_names,
                "selected_specialist_keys": selected_keys,
                "shadow_verdict": {
                    "label": "INCONCLUSIVE",
                    "consensus": "inconclusive",
                    "votes": {"REAL": 0, "FAKE": 0},
                    "total_models": len(selected_keys),
                    "reason": "no_usable_face" if self.active_engine == "v3" else "no_face_detected",
                },
                "production_verdict": {
                    "label": "INCONCLUSIVE",
                    "fusion_label": "INCONCLUSIVE",
                    "combined_consensus_label": "INCONCLUSIVE",
                    "combined_consensus": "inconclusive",
                },
                "verdicts_match": True,
                "shadow_mode": True,
            }
            if self.active_engine == "v3":
                v3_no_face = {
                    "verdict": "INCONCLUSIVE",
                    "reason": "no_usable_face",
                    "available_scores": {},
                    "unavailable_models": list(self.v3_thresholds.keys()),
                    "triggering_models": [],
                    "triggering_scores": {},
                    "primary_trigger": None,
                    "thresholds": dict(self.v3_thresholds),
                    "verdict_engine": "v3",
                    "verdict_source": "specialist_any_fake_v3",
                    "allow_real": self.allow_real,
                }
                return ImageInferenceResult(
                    status="inconclusive",
                    faces_detected=0,
                    model_name=display_model_name,
                    checkpoint_path=checkpoint_str,
                    execution_device=str(self.device),
                    uncertainty_threshold=uncertainty_threshold,
                    message="No human faces detected in the image.",
                    active_verdict_engine="v3",
                    verdict_source="specialist_any_fake_v3",
                    predictions=[
                        FacePrediction(
                            face_id=0,
                            label="INCONCLUSIVE",
                            score=None,
                            score_type=SCORE_TYPE_UNSETTLED,
                            box=None,
                            probabilities=None,
                            raw_logits=None,
                            reason="no_usable_face",
                            face_crop_base64=None,
                            gradcam_heatmap_base64=None,
                            gradcam_overlay_base64=None,
                            efficientnet=None,
                            vit=None,
                            community_vit=None,
                            sbi=None,
                            hybrid_srm=None,
                            fusion={"label": "INCONCLUSIVE", "score": None, "reason": "no_usable_face"},
                            combined_result=None,
                            shadow_routing=shadow_routing_data,
                            verdict_source="specialist_any_fake_v3",
                            verdict_engine="v3",
                            verdict_reason="no_usable_face",
                            available_scores={},
                            unavailable_models=list(self.v3_thresholds.keys()),
                            triggering_models=[],
                            triggering_scores={},
                            primary_trigger=None,
                            thresholds=dict(self.v3_thresholds),
                            v3_verdict=v3_no_face,
                        )
                    ],
                    shadow_routing=shadow_routing_data,
                )

            return ImageInferenceResult(
                status="inconclusive",
                faces_detected=0,
                model_name=display_model_name,
                checkpoint_path=checkpoint_str,
                execution_device=str(self.device),
                uncertainty_threshold=uncertainty_threshold,
                message="No human faces detected in the image.",
                active_verdict_engine="legacy",
                verdict_source="two_model_agreement_legacy",
                predictions=[
                    FacePrediction(
                        face_id=0,
                        label="INCONCLUSIVE",
                        score=None,
                        score_type=SCORE_TYPE_UNSETTLED,
                        box=None,
                        probabilities=None,
                        raw_logits=None,
                        reason="no_face_detected",
                        face_crop_base64=None,
                        gradcam_heatmap_base64=None,
                        gradcam_overlay_base64=None,
                        efficientnet=None,
                        vit=None,
                        community_vit=None,
                        sbi=None,
                        hybrid_srm=None,
                        fusion={"label": "INCONCLUSIVE", "score": None, "reason": "no_face_detected"},
                        combined_result={
                            "label": "INCONCLUSIVE",
                            "consensus": "inconclusive",
                            "votes": {"REAL": 0, "FAKE": 0},
                            "total_models": 5,
                            "reason": "no_face_detected",
                            "disclaimer": "Individual model scores are uncalibrated and cannot be directly averaged.",
                            "low_confidence_models": [],
                            "unavailable_models": [],
                        },
                        shadow_routing=shadow_routing_data,
                        verdict_source="two_model_agreement_legacy",
                        verdict_engine="legacy",
                    )
                ],
                shadow_routing=shadow_routing_data,
            )

        # 4. Case B: Faces detected -> process each face crop independently
        transform = get_efficientnet_transform()
        predictions: List[FacePrediction] = []
        self.primary_model.eval()

        for art in face_artifacts:
            # Quality and visibility check
            if not art.is_usable:
                logger.info(
                    f"Face ID {art.face_id} failed quality check: {art.quality_reason}"
                )
                face_crop_b64 = pil_to_base64_data_url(art.crop_pil) if art.crop_pil else None
                face_shadow_routing = {
                    "routing_policy": self.routing_policy,
                    "safeguard_threshold": self.safeguard_confidence_threshold,
                    "policy_reason": policy_reason,
                    "image_type": predicted_image_type,
                    "raw_image_type": raw_image_type,
                    "router_confidence": router_conf,
                    "is_rejected": is_rejected,
                    "rejection_reason": rejection_reason,
                    "router_probabilities": router_probs,
                    "selected_specialists": selected_display_names,
                    "selected_specialist_keys": selected_keys,
                    "shadow_verdict": {
                        "label": "INCONCLUSIVE",
                        "consensus": "inconclusive",
                        "votes": {"REAL": 0, "FAKE": 0},
                        "total_models": len(selected_keys),
                        "reason": "no_usable_face" if self.active_engine == "v3" else art.quality_reason,
                    },
                    "production_verdict": {
                        "label": "INCONCLUSIVE",
                        "fusion_label": "INCONCLUSIVE",
                        "combined_consensus_label": "INCONCLUSIVE",
                        "combined_consensus": "inconclusive",
                    },
                    "verdicts_match": True,
                    "shadow_mode": True,
                }
                if self.active_engine == "v3":
                    v3_unusable = {
                        "verdict": "INCONCLUSIVE",
                        "reason": "no_usable_face",
                        "available_scores": {},
                        "unavailable_models": list(self.v3_thresholds.keys()),
                        "triggering_models": [],
                        "triggering_scores": {},
                        "primary_trigger": None,
                        "thresholds": dict(self.v3_thresholds),
                        "verdict_engine": "v3",
                        "verdict_source": "specialist_any_fake_v3",
                        "allow_real": self.allow_real,
                    }
                    predictions.append(
                        FacePrediction(
                            face_id=art.face_id,
                            label="INCONCLUSIVE",
                            score=None,
                            score_type=SCORE_TYPE_UNSETTLED,
                            box=list(art.box),
                            probabilities=None,
                            raw_logits=None,
                            reason="no_usable_face",
                            face_crop_base64=face_crop_b64,
                            gradcam_heatmap_base64=None,
                            gradcam_overlay_base64=None,
                            efficientnet=None,
                            vit=None,
                            community_vit=None,
                            sbi=None,
                            hybrid_srm=None,
                            fusion={"label": "INCONCLUSIVE", "score": None, "reason": "no_usable_face"},
                            combined_result=None,
                            shadow_routing=face_shadow_routing,
                            verdict_source="specialist_any_fake_v3",
                            verdict_engine="v3",
                            verdict_reason="no_usable_face",
                            available_scores={},
                            unavailable_models=list(self.v3_thresholds.keys()),
                            triggering_models=[],
                            triggering_scores={},
                            primary_trigger=None,
                            thresholds=dict(self.v3_thresholds),
                            v3_verdict=v3_unusable,
                        )
                    )
                    continue

                predictions.append(
                    FacePrediction(
                        face_id=art.face_id,
                        label="INCONCLUSIVE",
                        score=None,
                        score_type=SCORE_TYPE_UNSETTLED,
                        box=list(art.box),
                        probabilities=None,
                        raw_logits=None,
                        reason=art.quality_reason,
                        face_crop_base64=face_crop_b64,
                        gradcam_heatmap_base64=None,
                        gradcam_overlay_base64=None,
                        efficientnet=None,
                        vit=None,
                        community_vit=None,
                        sbi=None,
                        hybrid_srm=None,
                        fusion={"label": "INCONCLUSIVE", "score": None, "reason": art.quality_reason},
                        combined_result={
                            "label": "INCONCLUSIVE",
                            "consensus": "inconclusive",
                            "votes": {"REAL": 0, "FAKE": 0},
                            "total_models": 5,
                            "reason": art.quality_reason,
                            "disclaimer": "Individual model scores are uncalibrated and cannot be directly averaged.",
                            "low_confidence_models": [],
                            "unavailable_models": [],
                        },
                        shadow_routing=face_shadow_routing,
                        verdict_source="two_model_agreement_legacy",
                        verdict_engine="legacy",
                    )
                )
                continue

            # Crop is strictly resized to 224x224 and normalized for EfficientNet-B0
            tensor = transform(art.crop_pil).unsqueeze(0).to(self.device)
            self.last_received_tensor_shape = tuple(tensor.shape)

            with torch.no_grad():
                logits = self.primary_model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                raw_logits_dict = {
                    "FAKE": round(float(logits[0][0].item()), 4),
                    "REAL": round(float(logits[0][1].item()), 4),
                }

            prob_fake = float(probs[0].item())
            prob_real = float(probs[1].item())

            efficientnet_label = "REAL" if prob_real >= prob_fake else "FAKE"
            efficientnet_score = max(prob_real, prob_fake)
            efficientnet_result = {
                "label": efficientnet_label,
                "score": round(efficientnet_score, 4),
                "probabilities": {"FAKE": round(prob_fake, 4), "REAL": round(prob_real, 4)},
                "raw_logits": raw_logits_dict,
                "loaded": True,
                "preprocessing": "224x224 RGB, ImageNet mean/std normalization",
            }

            vit_result: Optional[Dict[str, Any]] = None
            vit_input = (
                pil_img
                if (self.active_engine == "v3" and self.vit_uncropped)
                else art.crop_pil
            )
            vit_prep = (
                "ViTImageProcessor on full uncropped image"
                if (self.active_engine == "v3" and self.vit_uncropped)
                else "ViTImageProcessor on the same SCRFD face crop"
            )
            if self.secondary_model is not None:
                try:
                    vit_result = dict(self.secondary_model.predict(vit_input))
                    vit_result.update({
                        "loaded": True,
                        "preprocessing": vit_prep,
                    })
                except Exception as vit_err:
                    logger.warning(f"ViT inference failed for face ID {art.face_id}: {vit_err}")
                    vit_result = {
                        "label": "INCONCLUSIVE",
                        "score": None,
                        "reason": str(vit_err),
                        "loaded": False,
                    }
            else:
                vit_result = {
                    "label": "INCONCLUSIVE",
                    "score": None,
                    "reason": "secondary_model_unavailable",
                    "loaded": False,
                }

            fusion_result = fuse_model_predictions(
                efficientnet_result, vit_result, uncertainty_threshold
            )
            fusion_result.update({
                "policy": "both models must meet threshold and agree; score=min(model scores)",
            })
            label = fusion_result["label"]
            score = fusion_result["score"]
            reason = fusion_result["reason"]

            # Community Forensics ViT (CVPR 2025)
            community_vit_result: Optional[Dict[str, Any]] = None
            comm_vit_input = (
                pil_img
                if (self.active_engine == "v3" and self.community_vit_uncropped)
                else art.crop_pil
            )
            comm_vit_prep = (
                "CLIP processor on full uncropped image (shortest-edge 440 + center-crop 384)"
                if (self.active_engine == "v3" and self.community_vit_uncropped)
                else "CLIP processor on SCRFD face crop (shortest-edge 440 + center-crop 384)"
            )
            if self.community_vit_model is not None:
                try:
                    community_vit_result = dict(self.community_vit_model.predict(comm_vit_input))
                    community_vit_result.update({
                        "loaded": True,
                        "preprocessing": comm_vit_prep,
                    })
                except Exception as cv_err:
                    logger.warning(f"Community ViT inference failed for face ID {art.face_id}: {cv_err}")
                    community_vit_result = {
                        "label": "INCONCLUSIVE",
                        "score": None,
                        "reason": str(cv_err),
                        "loaded": False,
                    }
            else:
                community_vit_result = {
                    "label": "INCONCLUSIVE",
                    "score": None,
                    "reason": "community_vit_unavailable",
                    "loaded": False,
                }

            # SBI EfficientNet-B4 (CVPR 2022)
            sbi_result: Optional[Dict[str, Any]] = None
            if self.sbi_model is not None:
                try:
                    sbi_result = dict(self.sbi_model.predict(art.crop_pil))
                except Exception as sbi_err:
                    logger.warning(f"SBI inference failed for face ID {art.face_id}: {sbi_err}")
                    sbi_result = {
                        "label": "INCONCLUSIVE",
                        "score": None,
                        "reason": str(sbi_err),
                        "loaded": False,
                    }
            else:
                sbi_result = {
                    "label": "INCONCLUSIVE",
                    "score": None,
                    "reason": "sbi_model_unavailable",
                    "loaded": False,
                }

            # Hybrid SRM Detector (AdityaManojShinde/deepfake-detector)
            hybrid_srm_result: Optional[Dict[str, Any]] = None
            if self.hybrid_srm_model is not None:
                try:
                    hybrid_srm_result = dict(self.hybrid_srm_model.predict(art.crop_pil))
                except Exception as srm_err:
                    logger.warning(f"Hybrid SRM inference failed for face ID {art.face_id}: {srm_err}")
                    hybrid_srm_result = {
                        "label": "INCONCLUSIVE",
                        "score": None,
                        "reason": str(srm_err),
                        "loaded": False,
                    }
            else:
                hybrid_srm_result = {
                    "label": "INCONCLUSIVE",
                    "score": None,
                    "reason": "hybrid_srm_model_unavailable",
                    "loaded": False,
                }

            # Uncalibrated Multi-Model Combined Consensus
            all_models_dict = {
                "efficientnet": efficientnet_result,
                "vit": vit_result,
                "community_vit": community_vit_result,
                "sbi": sbi_result,
                "hybrid_srm": hybrid_srm_result,
            }
            combined_result = compute_combined_result(
                all_models_dict,
                threshold=uncertainty_threshold,
            )

            # Evaluate active verdict engine
            if self.active_engine == "v3":
                v3_res = evaluate_verdict_v3(
                    model_predictions=all_models_dict,
                    thresholds=self.v3_thresholds,
                    allow_real=self.allow_real,
                )
                label = v3_res["verdict"]
                reason = v3_res["reason"]
                score = v3_res["triggering_scores"].get(v3_res["primary_trigger"]) if label == "FAKE" else None
                verdict_source = "specialist_any_fake_v3"
            else:
                v3_res = None
                label = fusion_result["label"]
                score = fusion_result["score"]
                reason = fusion_result["reason"]
                verdict_source = "two_model_agreement_legacy"

            # Shadow Mode: Evaluate selected specialist consensus without altering production verdict
            routed_models_dict = {
                k: all_models_dict[k]
                for k in selected_keys
                if k in all_models_dict
            }
            shadow_combined_result = compute_combined_result(
                routed_models_dict,
                threshold=uncertainty_threshold,
            )

            face_shadow_routing = {
                "routing_policy": self.routing_policy,
                "safeguard_threshold": self.safeguard_confidence_threshold,
                "policy_reason": policy_reason,
                "image_type": predicted_image_type,
                "raw_image_type": raw_image_type,
                "router_confidence": router_conf,
                "is_rejected": is_rejected,
                "rejection_reason": rejection_reason,
                "router_probabilities": router_probs,
                "selected_specialists": selected_display_names,
                "selected_specialist_keys": selected_keys,
                "shadow_verdict": {
                    "label": shadow_combined_result.get("label", "INCONCLUSIVE"),
                    "consensus": shadow_combined_result.get("consensus", "inconclusive"),
                    "votes": shadow_combined_result.get("votes", {"REAL": 0, "FAKE": 0}),
                    "total_models": shadow_combined_result.get("total_models", len(selected_keys)),
                    "reason": shadow_combined_result.get("reason"),
                    "low_confidence_models": shadow_combined_result.get("low_confidence_models", []),
                    "unavailable_models": shadow_combined_result.get("unavailable_models", []),
                },
                "production_verdict": {
                    "label": label,
                    "fusion_label": fusion_result.get("label", "INCONCLUSIVE") if fusion_result else "INCONCLUSIVE",
                    "combined_consensus_label": combined_result.get("label", "INCONCLUSIVE"),
                    "combined_consensus": combined_result.get("consensus", "inconclusive"),
                },
                "verdicts_match": (shadow_combined_result.get("label") == label),
                "shadow_mode": True,
            }

            face_crop_b64 = pil_to_base64_data_url(art.crop_pil) if art.crop_pil else None
            gradcam_heatmap_b64 = None
            gradcam_overlay_b64 = None

            if include_gradcam:
                try:
                    target_layer = self.primary_model.get_target_layer_for_gradcam()
                    cam_engine = GradCAM(
                        model=self.primary_model,
                        target_layer=target_layer,
                        device=self.device,
                    )
                    cam_res = cam_engine.generate(
                        input_tensor=tensor,
                        target_class="predicted",
                        target_size=art.crop_pil.size,
                    )
                    vis = GradCAMVisualizer(default_colormap="turbo", default_alpha=0.5)
                    heatmap_rgb = vis.apply_colormap(cam_res.heatmap)
                    heatmap_pil = Image.fromarray(heatmap_rgb)
                    overlay_pil = vis.create_overlay(art.crop_pil, cam_res.heatmap)

                    gradcam_heatmap_b64 = pil_to_base64_data_url(heatmap_pil, format="JPEG")
                    gradcam_overlay_b64 = pil_to_base64_data_url(overlay_pil, format="JPEG")
                except Exception as cam_err:
                    logger.warning(
                        f"Grad-CAM generation failed for face ID {art.face_id}: {cam_err}"
                    )

            predictions.append(
                FacePrediction(
                    face_id=art.face_id,
                    label=label,
                    score=score,
                    score_type=SCORE_TYPE_UNSETTLED,
                    box=list(art.box),
                    probabilities={
                        "FAKE": round(prob_fake, 4),
                        "REAL": round(prob_real, 4),
                    },
                    raw_logits=raw_logits_dict,
                    reason=reason,
                    face_crop_base64=face_crop_b64,
                    gradcam_heatmap_base64=gradcam_heatmap_b64,
                    gradcam_overlay_base64=gradcam_overlay_b64,
                    efficientnet=efficientnet_result,
                    vit=vit_result,
                    community_vit=community_vit_result,
                    sbi=sbi_result,
                    hybrid_srm=hybrid_srm_result,
                    fusion=fusion_result,
                    combined_result=combined_result,
                    shadow_routing=face_shadow_routing,
                    verdict_source=verdict_source,
                    verdict_engine=self.active_engine,
                    verdict_reason=reason,
                    available_scores=v3_res["available_scores"] if v3_res else None,
                    unavailable_models=v3_res["unavailable_models"] if v3_res else None,
                    triggering_models=v3_res["triggering_models"] if v3_res else None,
                    triggering_scores=v3_res["triggering_scores"] if v3_res else None,
                    primary_trigger=v3_res["primary_trigger"] if v3_res else None,
                    thresholds=v3_res["thresholds"] if v3_res else None,
                    v3_verdict=v3_res,
                )
            )

        all_inconclusive = all(p.label == "INCONCLUSIVE" for p in predictions)
        overall_status = "inconclusive" if all_inconclusive else "success"

        return ImageInferenceResult(
            status=overall_status,
            faces_detected=len(face_artifacts),
            model_name=display_model_name,
            checkpoint_path=checkpoint_str,
            execution_device=str(self.device),
            uncertainty_threshold=uncertainty_threshold,
            message=f"Analyzed {len(face_artifacts)} detected face(s) independently.",
            predictions=predictions,
            shadow_routing=predictions[0].shadow_routing if predictions else None,
            active_verdict_engine=self.active_engine,
            verdict_source="specialist_any_fake_v3" if self.active_engine == "v3" else "two_model_agreement_legacy",
        )

    def get_gradcam_payload(
        self,
        face_id: int = 1,
        model_name: str = "efficientnet",
    ) -> Dict[str, Any]:
        """
        Exposes modular inputs required by downstream Explainability (Grad-CAM) modules:
        - face_crop_pil: The detected and cropped face as a PIL Image.
        - face_crop_bgr: The detected and cropped face in OpenCV BGR format.
        - model: The active EfficientNet-B0 model instance (eval mode).
        - transform: The model's evaluation preprocessing pipeline.
        - target_layer: The terminal convolutional layer suitable for feature attribution.
        - predicted_class: Winning model class ("FAKE" or "REAL").
        - predicted_class_idx: Winning index (0 for FAKE, 1 for REAL).
        """
        if face_id not in self.crop_artifacts:
            raise KeyError(
                f"Face ID {face_id} not found in recent inference crops. "
                f"Available IDs: {list(self.crop_artifacts.keys())}"
            )

        artifact = self.crop_artifacts[face_id]
        if self.primary_model is None:
            raise RuntimeError("Primary model is not loaded.")

        model = self.primary_model
        transform = get_efficientnet_transform()
        target_layer = self.primary_model.get_target_layer_for_gradcam()

        # Compute prediction index for target attribution
        tensor = transform(artifact.crop_pil).unsqueeze(0).to(self.device)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        return {
            "face_id": face_id,
            "face_crop_pil": artifact.crop_pil,
            "face_crop_bgr": artifact.crop_bgr,
            "box": artifact.box,
            "padded_box": artifact.padded_box,
            "model": model,
            "transform": transform,
            "target_layer": target_layer,
            "predicted_class": CLASS_NAMES[pred_idx],
            "predicted_class_idx": pred_idx,
            "device": self.device,
        }
