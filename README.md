# DeepFake Forensics: AI-Powered Media Authenticity Detection

A unified, production-grade deepfake media forensics application focusing strictly on detected facial regions using **EfficientNet-B0**, **ViT**, **SCRFD Face Detection**, and **Grad-CAM Model Explainability**.

---

## 1. System Architecture

The application operates as a single, consistent pipeline:

```text
Frontend (HTML, CSS, JavaScript)
        │
        ▼
FastAPI Backend (src/api/app.py)
        │
        ▼
SCRFD Face Detection (models/scrfd_500m.onnx)
        │
        ▼
Face Cropping & Quality Validation (blur, size, truncation)
        │
        ▼
EfficientNet-B0 + ViT Classification (same SCRFD face crop)
        │
        ├───► Verdict: REAL / FAKE / INCONCLUSIVE
        │
        └───► Standard Grad-CAM (Target Layer: features[8])
        │
        ▼
Analysis Results & Attributions returned to Frontend
```

### Core Architectural Guarantees:
1. **Dual classifier fusion**: EfficientNet-B0 and the cached ViT detector both receive the same validated SCRFD face crop. A result is conclusive only when both scores meet the configured threshold and both labels agree.
2. **Single Face Detector**: Deep-learning **SCRFD (500M ONNX)** is the only active detector. The old heuristic skin detector has been completely removed.
3. **No Scene Fallback**: The full photograph is **never** passed to the classifier. If no face is detected or if face quality is insufficient (too blurry, too small, or occluded), the engine returns an explicit `INCONCLUSIVE` verdict.
4. **Standard Grad-CAM Formulation**: Attributions are computed directly on the terminal convolutional layer (`backbone.features[8]`) using global average pooling, ReLU rectification, min-max normalization, and bilinear upsampling to the face crop dimensions.
5. **Transparent Inactive Heatmaps**: Inactive attribution regions remain 100% transparent on the visual overlay rather than receiving an artificial purple colormap tint.
6. **Model Attention Disclaimer**: Grad-CAM is presented accurately as convolutional model attention/attribution, not standalone mathematical proof of manipulation.

---

## 2. Project Directory Structure

```text
DeepFake-Forensics/
├── datasets/                 # Dataset documentation and pointers
│   └── README.md
├── frontend/                 # Single-page Web Interface (HTML5, CSS3, ES6 JS)
│   ├── index.html            # User interface layout
│   ├── app.js                # Frontend client logic & API interaction
│   ├── styles.css            # Modern styling & responsive layout
│   └── samples/              # Quick-test sample portraits (Real & Fake)
│       ├── real_sample.jpg
│       └── fake_sample.jpg
├── models/                   # Neural network weights
│   ├── efficientnet_b0_best.pth  # Trained EfficientNet-B0 classifier (Untouched)
│   ├── scrfd_500m.onnx       # SCRFD face detector ONNX model
│   └── README.md
├── outputs/                  # Exported test figures and crops
│   ├── efficientnet_confusion_matrix.png
│   ├── efficientnet_test_metrics.json
│   └── gradcam/              # Visual attribution artifacts
├── src/                      # Production application source code
│   ├── api/                  # FastAPI backend service
│   │   ├── app.py            # Endpoints: /health, /models/info, /predict/image
│   │   ├── schemas.py        # Pydantic request/response schemas
│   │   └── __init__.py
│   ├── evaluation/           # Benchmark evaluation on EfficientNet-B0
│   │   ├── evaluate_model.py
│   │   └── __init__.py
│   ├── explainability/       # Grad-CAM explainability module
│   │   ├── gradcam.py        # Standard Grad-CAM engine
│   │   ├── visualizer.py     # Colormap & intensity-modulated overlays
│   │   ├── pipeline.py       # Explainability CLI pipeline
│   │   ├── README.md
│   │   └── __init__.py
│   └── image_detection/      # Face detection & inference engine
│       ├── efficientnet_model.py     # DeepfakeEfficientNetB0 architecture
│       ├── face_detector.py          # SCRFD ONNX detector & quality gates
│       ├── inference.py              # Unified DeepFakeInferenceEngine
│       ├── predict_efficientnet.py   # Standalone prediction CLI
│       ├── train_efficientnet.py     # EfficientNet training reference
│       └── __init__.py
├── tests/                    # Automated test suite
│   ├── fixtures/             # Evaluation fixtures (blurry, small, occluded)
│   ├── sample_image_real.jpg # Genuine portrait sample
│   ├── sample_image_fake.jpg # AI-generated portrait sample (StyleGAN)
│   ├── sample_landscape_noface.jpg # No-face test image
│   ├── sample_multiple_faces.jpg   # Two-face test image
│   ├── test_api.py           # Backend endpoints & static asset tests
│   ├── test_face_pipeline.py # SCRFD cropping & quality validation tests
│   ├── test_gradcam.py       # Grad-CAM math, hooks, and overlay tests
│   ├── test_inference.py     # Unified inference engine unit tests
│   └── README.md
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
└── README.md                 # Project documentation
```

---

## 3. Quick Start & Application Startup

### 1. Environment Setup

Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/Abijith282005/DeepFake-Forensics.git
cd DeepFake-Forensics

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Once running:
- **Web Interface**: Open your browser at `http://127.0.0.1:8000`
- **Interactive API Docs (Swagger UI)**: Navigate to `http://127.0.0.1:8000/docs`

---

## 4. API Specification

### `GET /health`
Returns backend readiness, compute device, and checkpoint status.

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "primary_model_loaded": true,
  "primary_model": "EfficientNet-B0",
  "face_detector": "SCRFD (ONNX)",
  "device": "cpu",
  "cuda_available": false
}
```

### `GET /models/info`
Returns architecture metadata for EfficientNet-B0 and SCRFD.

```bash
curl -X GET http://127.0.0.1:8000/models/info
```

### `POST /predict/image`
Analyzes an uploaded image, isolates face regions with SCRFD, validates quality, runs EfficientNet-B0 and ViT classification, and computes EfficientNet Grad-CAM heatmaps. The response includes both model predictions, class scores, preprocessing metadata, and the fusion reason.

```bash
curl -X POST "http://127.0.0.1:8000/predict/image?include_gradcam=true" \
  -F "file=@tests/sample_image_real.jpg"
```

**Response:**
```json
{
  "status": "success",
  "faces_detected": 1,
  "model_name": "EfficientNet-B0",
  "execution_device": "cpu",
  "uncertainty_threshold": 0.6,
  "message": "Analyzed 1 detected face(s) independently.",
  "predictions": [
    {
      "face_id": 1,
      "label": "REAL",
      "score": 0.9939,
      "score_type": "uncalibrated_model_score",
      "box": [45, 55, 150, 204],
      "probabilities": {
        "FAKE": 0.0061,
        "REAL": 0.9939
      },
      "reason": null,
      "face_crop_base64": "data:image/jpeg;base64,...",
      "gradcam_heatmap_base64": "data:image/jpeg;base64,...",
      "gradcam_overlay_base64": "data:image/jpeg;base64,..."
    }
  ]
}
```

---

## 5. Command-Line Interface (CLI)

Run inference and generate visual Grad-CAM overlays directly from the terminal:

```bash
python -m src.explainability.pipeline --image tests/sample_image_real.jpg
```

Generated artifacts (face crop, Grad-CAM heatmap, overlay, side-by-side) will be exported to `outputs/gradcam/<image_name>/`.

---

## 6. Running Tests

Run the full automated test suite covering all components:

```bash
python -m unittest discover tests
```

Test coverage includes:
- **`test_inference.py`**: Model loading, device detection, REAL/FAKE predictions, no-face handling, multi-face isolation, uncertainty thresholds.
- **`test_face_pipeline.py`**: 7 critical face scenarios (full scene, portrait, multi-face, no face, small face, blurry face, occluded face).
- **`test_api.py`**: FastAPI endpoints, static frontend serving, upload validation, HTTP error codes.
- **`test_gradcam.py`**: Mathematical formulation, hook cleanup, memory leak absence, resolution invariance, transparent overlay verification.
