"""
Pydantic API Schemas for DeepFake Forensics Backend.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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
    fusion: Optional[Dict] = Field(None, description="Fusion decision and reason")

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
    final_prediction: FacePredictionSchema = Field(..., description="Single fused image-level verdict from both classifiers")
    predictions: List[FacePredictionSchema] = Field(..., description="Individual per-face prediction objects")

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
    status: str = Field(..., description="Overall backend health status ('healthy' or 'degraded')")
    primary_model_loaded: bool = Field(..., description="Whether the primary EfficientNet-B0 model is loaded")
    secondary_model_loaded: bool = Field(..., description="Whether the ViT secondary model is loaded")
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
    primary_model: ModelDetailSchema
    default_model: str = Field("efficientnet", description="Default model used for inference")
    face_detector: str = Field("SCRFD", description="Active face detector")
    device: str = Field(..., description="Execution device")
    cuda_available: bool = Field(..., description="Whether CUDA is present and active")
    default_uncertainty_threshold: float = Field(..., description="Default uncertainty rejection threshold")


class ErrorResponseSchema(BaseModel):
    """Standardized error response schema."""
    detail: str = Field(..., description="Description of the error encountered")
