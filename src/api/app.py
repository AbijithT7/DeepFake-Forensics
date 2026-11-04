"""
FastAPI Backend Application for DeepFake Forensics.

Serves the unified, single-model architecture:
- Model: EfficientNet-B0 (strictly face-only)
- Face Detector: SCRFD (500M ONNX)
- Explainability: Standard Grad-CAM
- Frontend: Single-page UI served at / and /static

Endpoints:
- GET /health: Operational status, compute device, and checkpoint status.
- GET /models/info: Metadata for EfficientNet-B0 and SCRFD.
- POST /predict/image: Image upload endpoint returning per-face detection predictions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    ErrorResponseSchema,
    HealthResponseSchema,
    ModelDetailSchema,
    ModelsInfoResponseSchema,
    PredictionResponseSchema,
)
from src.image_detection.inference import (
    DEFAULT_UNCERTAINTY_THRESHOLD,
    DeepFakeInferenceEngine,
    InvalidImageError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_api")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
    "application/octet-stream",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: loads EfficientNet-B0, ViT, and SCRFD once when server starts.
    """
    logger.info("Initializing DeepFake Forensic Inference Engine (EfficientNet-B0 + ViT + SCRFD)...")
    engine = DeepFakeInferenceEngine(auto_load=True)
    app.state.engine = engine
    logger.info(
        f"Inference Engine ready. Compute device: {engine.device} "
        f"(CUDA Available: {torch.cuda.is_available()})"
    )
    yield
    logger.info("Shutting down DeepFake Forensic API backend.")


app = FastAPI(
    title="DeepFake Forensics API",
    version="1.0.0",
    description=(
        "Production API for Image-Level DeepFake Authenticity Detection. "
        "Evaluates faces using SCRFD localization, quality validation, "
        "and EfficientNet-B0 classification with standard Grad-CAM visual explainability."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return JSONResponse(
        {
            "status": "healthy",
            "service": "DeepFake Forensics API",
            "docs": "/docs",
        }
    )


def get_engine(request: Request) -> DeepFakeInferenceEngine:
    """Dependency that returns the preloaded singleton inference engine."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not yet initialized.",
        )
    return engine


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Backend Health & Model Loading Status",
    tags=["System"],
)
async def health_check(engine: DeepFakeInferenceEngine = Depends(get_engine)):
    """
    Returns real-time operational status, active compute device (CPU/CUDA),
    and EfficientNet-B0 checkpoint loading status.
    """
    is_primary_ready = engine.primary_model is not None
    overall_status = "healthy" if engine.is_ready else "degraded"

    return HealthResponseSchema(
        status=overall_status,
        primary_model_loaded=is_primary_ready,
        secondary_model_loaded=engine.secondary_model is not None,
        primary_model="EfficientNet-B0",
        face_detector="SCRFD (ONNX)",
        device=str(engine.device),
        cuda_available=torch.cuda.is_available(),
    )


@app.get(
    "/models/info",
    response_model=ModelsInfoResponseSchema,
    summary="Model Registry & Architecture Details",
    tags=["Models"],
)
async def models_info(engine: DeepFakeInferenceEngine = Depends(get_engine)):
    """
    Returns architecture metadata, checkpoint path, resolution requirements,
    and output classification schema for the EfficientNet-B0 classifier.
    """
    return ModelsInfoResponseSchema(
        primary_model=ModelDetailSchema(
            name="EfficientNet-B0",
            checkpoint=str(engine.primary_checkpoint),
            status="loaded" if engine.primary_model is not None else "unavailable",
            input_resolution="224x224",
            normalization="ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            classes=["FAKE", "REAL"],
            score_type="uncalibrated_model_score",
            description=(
                "Primary classifier based on pretrained EfficientNet-B0 "
                "with Global Average Pooling and binary classification head. "
                "Evaluates strictly isolated, validated face crops."
            ),
        ),
        default_model="efficientnet",
        face_detector="SCRFD",
        device=str(engine.device),
        cuda_available=torch.cuda.is_available(),
        default_uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD,
    )


@app.post(
    "/predict/image",
    response_model=PredictionResponseSchema,
    responses={
        400: {"model": ErrorResponseSchema, "description": "Invalid image or unreadable payload"},
        503: {"model": ErrorResponseSchema, "description": "Model engine unready"},
    },
    summary="DeepFake Authenticity Detection for Image Uploads",
    tags=["Inference"],
)
async def predict_image_upload(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WEBP, BMP)"),
    uncertainty_threshold: float = Query(
        DEFAULT_UNCERTAINTY_THRESHOLD,
        ge=0.50,
        le=1.00,
        description="Minimum confidence score to assign REAL or FAKE. Predictions below this threshold are labeled INCONCLUSIVE.",
    ),
    include_gradcam: bool = Query(
        True,
        description="Whether to generate and include Grad-CAM spatial attribution heatmaps and overlays.",
    ),
    engine: DeepFakeInferenceEngine = Depends(get_engine),
):
    """
    Analyzes an uploaded image for deepfake or AI manipulation:
    1. Validates the uploaded file format and decodes image bytes.
    2. Detects human faces with SCRFD.
    3. Handles no-face (INCONCLUSIVE) and multiple-face cases independently.
    4. Crops face regions with contextual margins (20%) and validates quality.
    5. Normalizes and runs inference using EfficientNet-B0.
    6. Computes standard Grad-CAM visual explainability.
    7. Returns structured predictions with uncertainty thresholding.
    """
    # 1. Validate file extension if filename is present
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Supported image extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # 2. Validate MIME type if provided and not generic
    if file.content_type and file.content_type not in SUPPORTED_MIME_TYPES:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file MIME type '{file.content_type}' is not a recognized image format.",
            )

    # 3. Read image bytes
    try:
        image_bytes = await file.read()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read upload payload: {err}",
        )

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 4. Perform inference
    try:
        result = engine.predict_image(
            image_input=image_bytes,
            model_name="efficientnet",
            uncertainty_threshold=uncertainty_threshold,
            include_gradcam=include_gradcam,
        )
    except InvalidImageError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unsupported image file: {err}",
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error during image inference:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {err}",
        )

    return result.to_dict()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
