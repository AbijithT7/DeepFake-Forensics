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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
from src.image_detection.predict_efficientnet import load_efficientnet_model
from src.image_detection.vit_detector import ViTDetector

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_CHECKPOINT = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
DEFAULT_UNCERTAINTY_THRESHOLD = 0.60
SCORE_TYPE_UNSETTLED = "uncalibrated_model_score"


class InvalidImageError(ValueError):
    """Raised when an uploaded or specified image cannot be read, decoded, or validated."""
    pass


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
    fusion: Optional[Dict[str, Any]] = None

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
            "fusion": self.fusion,
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

    @property
    def final_prediction(self) -> FacePrediction:
        """One image verdict: all usable faces must reach the same fused result."""
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
                fusion=base.fusion,
            )
        labels = {p.label for p in conclusive}
        if len(labels) != 1:
            return FacePrediction(0, "INCONCLUSIVE", None, reason="multiple_face_disagreement")
        score = round(min(p.score for p in conclusive if p.score is not None), 4)
        representative = conclusive[0]
        return FacePrediction(
            0, representative.label, score, reason=None,
            face_crop_base64=representative.face_crop_base64,
            gradcam_heatmap_base64=representative.gradcam_heatmap_base64,
            gradcam_overlay_base64=representative.gradcam_overlay_base64,
            efficientnet=representative.efficientnet,
            vit=representative.vit,
            fusion=representative.fusion,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "faces_detected": self.faces_detected,
            "model_name": self.model_name,
            "execution_device": self.execution_device,
            "uncertainty_threshold": self.uncertainty_threshold,
            "message": self.message,
            "final_prediction": self.final_prediction.to_dict(),
            "predictions": [p.to_dict() for p in self.predictions],
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
    ):
        self.primary_checkpoint = Path(primary_checkpoint or DEFAULT_PRIMARY_CHECKPOINT)
        self.device = device or detect_execution_device()
        self.min_detection_confidence = min_detection_confidence
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.context_margin = context_margin

        self.primary_model: Optional[DeepfakeEfficientNetB0] = None
        self.secondary_model: Optional[ViTDetector] = None
        self.detector = SCRFDFaceDetector(conf_threshold=self.min_detection_confidence)
        self.crop_artifacts: Dict[int, FaceCropArtifact] = {}
        self.last_received_tensor_shape: Optional[Tuple[int, ...]] = None

        if auto_load:
            self.load_models()

    def load_models(self) -> None:
        """
        Load EfficientNet-B0 classifier into memory once.
        Reuses existing project loader to avoid duplicate code.
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

    @property
    def is_ready(self) -> bool:
        """Check if primary model is loaded and ready for inference."""
        return self.primary_model is not None and self.secondary_model is not None

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
        cv_img, _ = self.validate_and_read_image(image_input)

        # 2. Detect and crop faces using SCRFD
        face_artifacts = self.detect_and_crop_faces(cv_img)
        self.crop_artifacts = {art.face_id: art for art in face_artifacts}

        display_model_name = "EfficientNet-B0"
        checkpoint_str = str(self.primary_checkpoint)

        # 3. Case A: No faces detected in image -> INCONCLUSIVE
        if len(face_artifacts) == 0:
            logger.info("No face detected in image; returning INCONCLUSIVE.")
            return ImageInferenceResult(
                status="inconclusive",
                faces_detected=0,
                model_name=display_model_name,
                checkpoint_path=checkpoint_str,
                execution_device=str(self.device),
                uncertainty_threshold=uncertainty_threshold,
                message="No human faces detected in the image.",
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
                    )
                ],
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
                        fusion={"label": "INCONCLUSIVE", "score": None, "reason": art.quality_reason},
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
            if self.secondary_model is not None:
                vit_result = dict(self.secondary_model.predict(art.crop_pil))
                vit_result.update({
                    "loaded": True,
                    "preprocessing": "ViTImageProcessor on the same SCRFD face crop",
                })
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
                    fusion=fusion_result,
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
