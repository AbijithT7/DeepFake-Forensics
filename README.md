# DeepFake Forensics: AI-Assisted Facial Authenticity Detection

A unified, face-first deepfake forensic analysis platform combining deep learning face localization, dual-model classification (**EfficientNet-B0** and **Vision Transformer (ViT)**), quality validation gates, and convolutional spatial explainability via **Grad-CAM**.

---

## 1. Overview

**DeepFake-Forensics** performs automated forensic inspection of portrait images to assist investigators and analysts in identifying synthetic face generations and manipulation artifacts. Rather than making ungrounded predictions on whole-scene photographs, the system enforces a strict **face-first pipeline**: it localizes facial regions, screens each face for quality and usability, and evaluates them with a dual-model ensemble.

```text
Uploaded Image
      │
      ▼
SCRFD Deep Face Detector (500M ONNX via OpenCV DNN)
      │
      ▼
Facial Quality & Usability Gate
 (Laplacian blur check, min size 40px, bounding box truncation)
      │
      ├── [Quality Failed / No Face] ──► Verdict: INCONCLUSIVE
      │
      ▼ [Passed Quality Gate]
Face Region Extraction with Bounded Context Margins (15%)
      │
      ├──► Primary Classifier: EfficientNet-B0 (PyTorch)
      │       └── Standard Grad-CAM Attribution (features[8])
      │
      ├──► Secondary Classifier: Vision Transformer (ViT)
      │
      ▼
Decision Fusion Logic
 ├── Both agree with high confidence ──► REAL / FAKE
 └── Disagreement or low confidence  ──► INCONCLUSIVE (disagreement logged)
      │
      ▼
Interactive Web Interface & REST API Output
```

---

## 2. Key Features

- **Face-First Forensics**: Eliminates background bias and full-scene shortcuts by isolating face crops. The classifier never operates on arbitrary non-facial backgrounds.
- **Pre-Classification Quality Validation**: Detects and flags low-confidence face detections, severe boundary truncation, small face sizes (<40px), and blurry faces (Laplacian variance thresholding), gracefully outputting `INCONCLUSIVE` instead of guessing.
- **Dual-Model Fusion**: Combines a fine-tuned convolutional network (**EfficientNet-B0**) with an attention-based **Vision Transformer (ViT)**. Predictions are marked `INCONCLUSIVE` if model outputs conflict or fall below uncertainty thresholds.
- **Spatial Explainability with Grad-CAM**: Generates standard Grad-CAM attributions directly from the terminal convolutional feature map (`backbone.features[8]`), rendered as transparent heatmaps over the original face crop.
- **Lightweight, Zero-Build Frontend**: Clean, responsive single-page application built with standard HTML5, CSS3, and JavaScript, served directly by FastAPI without external bundlers or node dependencies.
- **Comprehensive REST API**: Fully documented endpoints with OpenAPI/Swagger schemas for programmatic integration.

---

## 3. Technology Stack

- **Deep Learning Framework**: PyTorch, Torchvision
- **Transformers & Attention Models**: Hugging Face `transformers`, `huggingface-hub`
- **Computer Vision & Inference**: OpenCV (`cv2.dnn`), Pillow, NumPy, Scikit-learn
- **Backend Web Framework**: FastAPI, Uvicorn, Pydantic
- **Explainability**: Standard Grad-CAM with Matplotlib color mapping
- **Frontend**: Modern Vanilla JavaScript (ES6+), HTML5, CSS3

---

## 4. Repository Structure

```text
DeepFake-Forensics/
├── datasets/                 # Dataset documentation and directory scaffolds
│   └── README.md
├── frontend/                 # Single-page web user interface
│   ├── index.html            # Main dashboard layout
│   ├── app.js                # Frontend client logic & REST API bindings
│   ├── styles.css            # Responsive styles and dark theme
│   └── samples/              # Test sample portraits (Real & Fake)
├── models/                   # Production model weights and configurations
│   ├── efficientnet_b0_best.pth  # Production EfficientNet-B0 weights (~16MB)
│   ├── scrfd_500m.onnx       # SCRFD deep-learning face detector ONNX (~2.6MB)
│   └── README.md
├── outputs/                  # Baseline metrics, evaluation plots, and reports
│   ├── efficientnet_confusion_matrix.png
│   ├── efficientnet_test_metrics.json
│   ├── efficientnet_training_curves.png
│   ├── efficientnet_training_history.json
│   └── README.md
├── src/                      # Core application source code
│   ├── api/                  # FastAPI backend service
│   │   ├── app.py            # API routes (/health, /models/info, /predict/image)
│   │   └── schemas.py        # Pydantic request & response schemas
│   ├── evaluation/           # Evaluation tools & metrics calculation
│   │   ├── evaluate_model.py
│   │   └── evaluate_external_dataset.py
│   ├── explainability/       # Grad-CAM attribution pipeline
│   │   ├── gradcam.py        # Mathematical Grad-CAM implementation
│   │   ├── visualizer.py     # Alpha-blended heatmap overlays
│   │   └── pipeline.py       # Standalone CLI explainability tool
│   └── image_detection/      # Face detection and classification engines
│       ├── efficientnet_model.py # EfficientNet-B0 architecture definition
│       ├── face_detector.py      # SCRFD ONNX face detector & quality checks
│       ├── inference.py          # Unified DeepFakeInferenceEngine & fusion
│       ├── predict_efficientnet.py
│       ├── train_efficientnet.py
│       └── vit_detector.py       # Vision Transformer (ViT) wrapper
├── tests/                    # Automated regression and integration tests
│   ├── test_api.py           # Backend endpoints & static asset tests
│   ├── test_face_pipeline.py # SCRFD cropping & quality validation gates
│   ├── test_gradcam.py       # Grad-CAM attribution correctness tests
│   ├── test_inference.py     # Unified inference engine & fusion tests
│   ├── test_quality_validation.py # Blur, size, occlusion, and truncation tests
│   ├── test_regressions.py   # Pipeline regression tests
│   └── test_vit_detector.py  # ViT detector integration tests
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AbijithT7/DeepFake-Forensics.git
cd DeepFake-Forensics
```

### 2. Set Up Virtual Environment
```bash
# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Model Weights & Downloads

The project uses three models:

| Model | File / Identifier | Size | Source / Setup |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | `models/efficientnet_b0_best.pth` | ~16.3 MB | Included in Git repository |
| **SCRFD Face Detector** | `models/scrfd_500m.onnx` | ~2.6 MB | Included in Git repository |
| **Vision Transformer (ViT)** | `prithivMLmods/Deep-Fake-Detector-v2-Model` | ~343 MB | Automatically downloaded from Hugging Face on first run into `models/huggingface/` |

> **Note on ViT Weights**: On the initial run of the server or test suite, Hugging Face `transformers` will automatically download the ViT model weights and cache them in `models/huggingface/`. The directory `models/huggingface/` is excluded from git tracking to prevent large binaries from bloating the repository.

---

## 7. Running the Application

### Start the FastAPI Backend
```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the System
- **Web Interface**: Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger)**: Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 8. Command-Line Usage

### Forensic Attribution with Grad-CAM
You can inspect any image directly from the terminal without launching the web server:

```bash
python -m src.explainability.pipeline --image tests/sample_image_real.jpg
```

Generated outputs will be saved to `outputs/gradcam/<image_stem>/`:
- `face_crop.png`: Extracted face region with context padding
- `heatmap.png`: Raw Grad-CAM attribution intensity map
- `overlay.png`: Alpha-blended visual overlay
- `side_by_side.png`: Full forensic comparison panel

---

## 9. Running Tests

Run the full automated test suite using Python's standard `unittest` framework:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 66 automated tests verify:
- REST API responses, error codes, and static file serving
- SCRFD face localization and bounding box calculation
- Quality gates (blur, minimum resolution, out-of-bounds truncation)
- Dual-model inference, fusion logic, and disagreement handling
- Mathematical correctness of Grad-CAM activations and memory management

---

## 10. Limitations & Forensic Guidance

- **No Universal Detection Guarantee**: Deepfake generation methods (diffusion models, GANs, latent face-swaps) evolve rapidly. No automated system can reliably detect 100% of synthetic or manipulated imagery, especially novel architectures or unseen pipelines.
- **Face-Restricted Inspection**: The classifier examines facial regions only. Manipulations confined to image backgrounds, audio tracks, clothing, or metadata are out of scope.
- **Image Quality Sensitivity**: Extreme compression artifacts, heavy motion blur, or severe pixelation will trigger an `INCONCLUSIVE` verdict rather than forcing an unreliable prediction.
- **Grad-CAM Attribution Interpretation**: Grad-CAM visualizes the convolutional filter activations that most strongly influenced the network's classification decision. It highlights areas of model focus; it is **not** standalone mathematical proof of forgery or tampering.
