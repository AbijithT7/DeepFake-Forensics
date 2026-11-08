"""
Pydantic API Schemas for DeepFake Forensics Backend.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ShadowRoutingSchema(BaseModel):
    """Diagnostic telemetry for shadow-mode specialist routing."""
    routing_policy: Optional[str] = Field("safeguard", description="Active routing policy ('safeguard', 'legacy', or 'all')")
    safeguard_threshold: Optional[float] = Field(0.85, description="Confidence threshold applied for safeguard gating")
    policy_reason: Optional[str] = Field(None, description="Explanation for specialist group assignment under active policy")
    image_type: str = Field(..., description="Predicted image typology (or UNKNOWN if rejected)")
    raw_image_type: str = Field(..., description="Argmax predicted typology before rejection threshold")
    router_confidence: float = Field(..., description="Predicted class probability from router classifier")
    is_rejected: bool = Field(..., description="Whether input was routed to UNKNOWN due to low confidence")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection, if applicable")
    router_probabilities: Optional[Dict[str, float]] = Field(None, description="Softmax probabilities across all 7 categories")
    selected_specialists: List[str] = Field(..., description="Display names of specialists selected by routing policy")
    selected_specialist_keys: List[str] = Field(..., description="Internal identifiers of selected specialist detectors")
    shadow_verdict: Optional[Dict] = Field(None, description="Hypothetical routed consensus verdict")
    production_verdict: Optional[Dict] = Field(None, description="Actual production verdict executed for this image")
    verdicts_match: Optional[bool] = Field(None, description="Whether shadow routed verdict matches production verdict")
    shadow_mode: bool = Field(True, description="Strict shadow-mode indicator")


class FacePredictionSchema(BaseModel):
    """Prediction outcome for an individual detected face."""
    face_id: int = Field(..., description="1-indexed identifier for the detected face (0 for no face)")
    label: str = Field(..., description="Outcome classification: REAL, FAKE, or INCONCLUSIVE")
    score: Optional[float] = Field(None, description="Uncalibrated probability score for the prediction")
    score_type: str = Field("uncalibrated_model_score", description="Explicit note that score is uncalibrated")
    box: Optional[List[int]] = Field(None, description="Bounding box [x, y, width, height] in pixel coordinates")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Softmax probabilities for each class")
    reason: Optional[str] = Field(None, description="Explanation when label is INCONCLUSIVE (e.g. no_face_detected, low_confidence)")
    face_crop_base64: Optional[str] = Field(None, description="Base64 data URL for the detected face crop")
    gradcam_heatmap_base64: Optional[str] = Field(None, description="Base64 data URL for the Grad-CAM attribution heatmap")
    gradcam_overlay_base64: Optional[str] = Field(None, description="Base64 data URL for the Grad-CAM heatmap overlay on face crop")
    efficientnet: Optional[Dict] = Field(None, description="EfficientNet individual prediction and class scores")
    vit: Optional[Dict] = Field(None, description="ViT individual prediction and class scores")
    community_vit: Optional[Dict] = Field(None, description="Community Forensics ViT individual prediction and class scores")
    sbi: Optional[Dict] = Field(None, description="SBI EfficientNet-B4 individual prediction and class scores")
    hybrid_srm: Optional[Dict] = Field(None, description="Hybrid SRM (EfficientNet-B4 + Xception) individual prediction and class scores")
    deepguard: Optional[Dict] = Field(None, description="DeepGuard face manipulation detector scores")
    authenticity: Optional[Dict] = Field(None, description="Trained DINOv2 whole-image binary classifier")
    prediction_scope: str = Field("face", description="Image or face assessment")
    quality_reason: Optional[str] = Field(None, description="Face quality rejection, independent of the image verdict")
    verdict_details: Optional[Dict] = Field(None, description="Decision rule, applied thresholds and diagnostic signals")
    fusion: Optional[Dict] = Field(None, description="Fusion decision and reason")
    combined_result: Optional[Dict] = Field(None, description="Multi-model consensus assessment across all detectors")
    shadow_routing: Optional[ShadowRoutingSchema] = Field(None, description="Shadow-mode specialist routing diagnostics")
    verdict_source: Optional[str] = Field(None, description="Verdict engine source identifier")
    verdict_engine: Optional[str] = Field(None, description="Active verdict engine ID")
    verdict_reason: Optional[str] = Field(None, description="Detailed explanation for the verdict")
    available_scores: Optional[Dict[str, float]] = Field(None, description="Probabilities of fake for available specialists")
    unavailable_models: Optional[List[str]] = Field(None, description="Specialists that could not be evaluated")
    triggering_models: Optional[List[str]] = Field(None, description="Specialists whose score met or exceeded threshold")
    triggering_scores: Optional[Dict[str, float]] = Field(None, description="Scores of the triggering specialists")
    primary_trigger: Optional[str] = Field(None, description="Specialist with the highest score among triggers")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Per-model thresholds applied")
    v3_verdict: Optional[Dict] = Field(None, description="Full v3 verdict engine evaluation object")

    model_config = {
        "json_schema_extra": {
            "example": {
                "face_id": 1,
                "label": "FAKE",
                "score": 0.998,
                "score_type": "uncalibrated_model_score",
                "box": [120, 85, 210, 210],
                "probabilities": {"FAKE": 0.998, "REAL": 0.002},
                "reason": None,
                "face_crop_base64": "data:image/jpeg;base64,...",
                "gradcam_heatmap_base64": "data:image/jpeg;base64,...",
                "gradcam_overlay_base64": "data:image/jpeg;base64,...",
            }
        }
    }


class PredictionResponseSchema(BaseModel):
    """Structured response for /predict/image endpoint."""
    status: str = Field(..., description="Status of the analysis ('success' or 'inconclusive')")
    faces_detected: int = Field(..., description="Total count of human faces detected")
    model_name: str = Field(..., description="Name of the model utilized for inference")
    execution_device: str = Field(..., description="Active compute device (e.g., cpu, cuda)")
    uncertainty_threshold: float = Field(..., description="Threshold applied to determine conclusive labels")
    message: Optional[str] = Field(None, description="Informational message regarding the inference outcome")
    final_prediction: FacePredictionSchema = Field(..., description="Image verdict from available image and face specialists")
    predictions: List[FacePredictionSchema] = Field(..., description="Individual per-face prediction objects")
    shadow_routing: Optional[ShadowRoutingSchema] = Field(None, description="Shadow-mode specialist routing diagnostics")
    verdict_source: Optional[str] = Field("specialist_any_fake_v3", description="Active verdict engine source identifier")
    active_verdict_engine: Optional[str] = Field("v3", description="Active verdict engine identifier")
    verdict: Optional[str] = Field(None, description="Final image verdict: REAL, FAKE or INCONCLUSIVE")
    verdict_reason: Optional[str] = Field(None, description="Detailed reason for final verdict")
    available_scores: Optional[Dict[str, float]] = Field(None, description="All available specialist detector scores")
    unavailable_models: Optional[List[str]] = Field(None, description="Unavailable specialist detector names")
    triggering_models: Optional[List[str]] = Field(None, description="Names of specialists that triggered FAKE")
    primary_trigger: Optional[str] = Field(None, description="Primary triggering specialist detector")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Operating thresholds used for evaluation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "faces_detected": 1,
                "model_name": "EfficientNet-B0",
                "execution_device": "cpu",
                "uncertainty_threshold": 0.60,
                "message": "Analyzed 1 detected face(s) independently.",
                "predictions": [
                    {
                        "face_id": 1,
                        "label": "FAKE",
                        "score": 0.998,
                        "score_type": "uncalibrated_model_score",
                    }
                ],
            }
        }
    }


class HealthResponseSchema(BaseModel):
    """Health check response schema."""
    deepguard_loaded: bool = Field(False, description="DeepGuard checkpoint availability")
    authenticity_loaded: bool = Field(False, description="Trained authenticity classifier availability")
    all_specialists_loaded: bool = Field(False, description="Whether every configured specialist loaded")
    active_verdict_engine: str = Field("v3", description="Active verdict engine ID ('v3' or 'legacy')")
    status: str = Field(..., description="Overall backend health status ('healthy' or 'degraded')")
    primary_model_loaded: bool = Field(..., description="Whether the primary EfficientNet-B0 model is loaded")
    secondary_model_loaded: bool = Field(..., description="Whether the ViT secondary model is loaded")
    community_vit_loaded: bool = Field(False, description="Whether Community Forensics ViT model is loaded")
    sbi_loaded: bool = Field(False, description="Whether SBI EfficientNet-B4 model is loaded")
    hybrid_srm_loaded: bool = Field(False, description="Whether Hybrid SRM detector model is loaded")
    router_loaded: bool = Field(False, description="Whether ImageTypeRouter is loaded")
    router_mode: str = Field("shadow", description="Router operating mode ('shadow' or 'production')")
    routing_policy: Optional[str] = Field("safeguard", description="Configured shadow routing policy ('safeguard', 'legacy', or 'all')")
    safeguard_confidence_threshold: Optional[float] = Field(0.85, description="Configured safeguard confidence threshold")
    primary_model: str = Field("EfficientNet-B0", description="Primary model identifier")
    face_detector: str = Field("SCRFD (ONNX)", description="Face detector identifier")
    device: str = Field(..., description="Execution compute device")
    cuda_available: bool = Field(..., description="Hardware NVIDIA CUDA availability")


class ModelDetailSchema(BaseModel):
    """Metadata specifications for an available detector model."""
    name: str = Field(..., description="Model display name")
    checkpoint: str = Field(..., description="Path to checkpoint weights file")
    status: str = Field(..., description="Current status in memory")
    input_resolution: str = Field(..., description="Input resolution expected by network")
    normalization: str = Field(..., description="Image normalization scheme")
    classes: List[str] = Field(..., description="Output classes")
    score_type: str = Field("uncalibrated_model_score", description="Score calibration category")
    description: str = Field(..., description="Model architecture summary and training lineage")


class ModelsInfoResponseSchema(BaseModel):
    """Response schema for /models/info endpoint."""
    deepguard_model: Optional[ModelDetailSchema] = None
    authenticity_model: Optional[ModelDetailSchema] = None
    authenticity_thresholds: Optional[Dict[str, float]] = None
    decision_rule: Optional[Dict] = None
    active_verdict_engine: str = "v3"
    specialist_thresholds: Dict[str, float] = Field(default_factory=dict)
    primary_model: ModelDetailSchema
    secondary_model: Optional[ModelDetailSchema] = Field(None, description="Secondary ViT model details")
    community_vit_model: Optional[ModelDetailSchema] = Field(None, description="Community Forensics ViT model details")
    sbi_model: Optional[ModelDetailSchema] = Field(None, description="SBI EfficientNet-B4 model details")
    hybrid_srm_model: Optional[ModelDetailSchema] = Field(None, description="Hybrid SRM (EfficientNet-B4 + Xception) model details")
    router: Optional[Dict] = Field(None, description="Image-type router classifier details and routing strategy")
    default_model: str = Field("efficientnet", description="Default model used for inference")
    face_detector: str = Field("SCRFD", description="Active face detector")
    device: str = Field(..., description="Execution device")
    cuda_available: bool = Field(..., description="Whether CUDA is present and active")
    default_uncertainty_threshold: float = Field(..., description="Default uncertainty rejection threshold")


class ErrorResponseSchema(BaseModel):
    """Standardized error response schema."""
    detail: str = Field(..., description="Description of the error encountered")
