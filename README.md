# DeepFake Forensics: Intelligent Multi-Specialist Facial Authenticity Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Git LFS](https://img.shields.io/badge/Git-LFS-orange.svg)](https://git-lfs.github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A unified, face-first deepfake forensic inspection platform combining **SCRFD deep face localization**, a **DINOv2 image-type router**, **five specialized deepfake detectors**, an intelligent **Safeguard Routing Policy** with non-intrusive **Shadow Mode**, and **Grad-CAM visual explainability**.

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Forensic Architecture & Workflow](#2-forensic-architecture--workflow)
3. [The Five Specialist Detectors](#3-the-five-specialist-detectors)
4. [DINOv2 Typology Router & Safeguard Strategy](#4-dinov2-typology-router--safeguard-strategy)
5. [Evaluation Benchmarks & Empirical Findings](#5-evaluation-benchmarks--empirical-findings)
6. [Repository Structure](#6-repository-structure)
7. [Installation & Setup](#7-installation--setup)
8. [Automated Model Weight Setup](#8-automated-model-weight-setup)
9. [Running the Application](#9-running-the-application)
10. [REST API Documentation](#10-rest-api-documentation)
11. [Limitations & Responsible Use](#11-limitations--responsible-use)

---

## 1. Overview & Philosophy

Modern face manipulation tools span vastly different paradigms—from GAN-generated faces (StyleGAN) and diffusion synthesis (Stable Diffusion, Midjourney) to blended face swaps (DeepFaceLab, SimSwap) and generative inpainting. Generalist models that attempt to detect all manipulations simultaneously frequently suffer from cross-manipulation blind spots and calibration degradation.

**DeepFake Forensics** solves this through a modular, multi-specialist approach:
- **Face-First Forensics**: Eliminates background bias and spurious scene correlations by isolating facial regions via SCRFD with controlled contextual margins.
- **Pre-Classification Quality Gating**: Detects small faces (<40px), boundary truncation, severe blur (Laplacian variance), and illumination/occlusion issues, returning structured `INCONCLUSIVE` states rather than guessing.
- **Five Diverse Specialists**: Combines complementary convolutional, transformer, noise-residual, and boundary-blending detectors.
- **DINOv2 Router in Shadow Mode**: Predicts manipulation typology (`STYLEGAN_GAN`, `DIFFUSION`, `FACE_SWAP`, `INPAINTING`, `REAL`, `OTHER_SYNTHETIC`) and evaluates intelligent routing safely in shadow mode without altering production verdicts.
- **Safeguard Routing Policy**: Applies confidence gating ($\tau = 0.85$) and routes authentic images to a multi-model safeguard group to protect genuine portraits against misclassification.

---

## 2. Forensic Architecture & Workflow

```text
Uploaded Image / Frame
         │
         ▼
SCRFD Deep Face Localization (500M ONNX)
         │
         ▼
Facial Quality Validation Gate
 (Laplacian blur check, size >= 40px, truncation check)
         │
         ├── [Quality Failed / No Face] ──► Verdict: INCONCLUSIVE
         │
         ▼ [Passed Quality Gate]
Face Region Extraction (15% context margin)
         │
         ├────────────────────────────────────────┐
         │ (Shadow Mode Evaluation)               │ (Production Consensus Pipeline)
         ▼                                        ▼
DINOv2 Typology Router                   Exhaustive Specialist Ensemble:
 (Extract 768-d cls token)               ├─ 1. EfficientNet-B0 (RVF10K)
         │                               ├─ 2. Original ViT (Token Attention)
         ▼                               ├─ 3. Community Forensics ViT (Diffusion)
Predicted Category & Confidence          ├─ 4. SBI EfficientNet-B4 (Blended Edges)
         │                               └─ 5. Hybrid SRM (Spatial + Steganalysis)
         ▼                                        │
Safeguard Routing Policy                          ▼
 (tau = 0.85 confidence gating)          Uncalibrated Consensus Decision Engine
         │                                (Disagreement tracking & majority voting)
         ▼                                        │
Hypothetical Routed Verdict                       ▼
         │                               Final Production Forensic Verdict
         └───────────────────┬────────────────────┘
                             ▼
               FastAPI JSON / Web Dashboard
               (Attribution Heatmaps via Grad-CAM)
```

---

## 3. The Five Specialist Detectors

| # | Detector Name | Architecture | Backbone / Source | Primary Forensic Specialization |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Primary EfficientNet-B0** | CNN (PyTorch) | Pretrained EfficientNet-B0 fine-tuned on RVF10K | Video compression artifacts, FaceForensics++, spatial artifacts. Supports Grad-CAM. |
| **2** | **Original ViT** | Vision Transformer | `prithivMLmods/Deep-Fake-Detector-v2-Model` | Global patch-token attention anomalies, structural facial symmetry inconsistencies. |
| **3** | **Community Forensics ViT** | Vision Transformer | `buildborderless/CommunityForensics-DeepfakeDet-ViT` | State-of-the-art diffusion artifacts (4,800+ generators), text-to-image synthesis, inpainting. |
| **4** | **SBI EfficientNet-B4** | CNN (Self-Blended) | Pretrained on Self-Blended Images (`FFraw.tar`) | Face swap boundary discrepancies, false facial edge blending, facial cut-and-paste seams. |
| **5** | **Hybrid SRM Detector** | Dual-Stream CNN | EfficientNet-B4 (RGB) + Steganalysis Residual Xception | Steganalysis noise residuals, high-frequency frequency-domain sensor noise inconsistencies. |

---

## 4. DINOv2 Typology Router & Safeguard Strategy

### The DINOv2 Router
- **Encoder**: Pretrained `facebook/dinov2-base` (768-dimensional CLS token embeddings).
- **Classifier**: Calibrated multi-class classification head trained across held-out datasets.
- **Classes**: `REAL`, `STYLEGAN_GAN`, `DIFFUSION`, `FACE_SWAP`, `INPAINTING`, `OTHER_SYNTHETIC`, `UNKNOWN`.

### Safeguard Routing Policy ($\tau = 0.85$)
In empirical testing, unconstrained routing suffered from a critical failure mode: misclassifying authentic portraits as synthetic and routing them to single specialist detectors, causing authentic face accuracy to collapse to 72%. 

The **Safeguard Routing Policy** solves this:
1. **Low Confidence Gating**: If DINOv2 confidence $< 0.85$, route to the **Safeguard Group**: `[EfficientNet-B0, Original ViT, Community Forensics ViT]`.
2. **Authentic Image Safeguard**: If predicted category is `REAL`, route to the **Safeguard Group**.
3. **High-Confidence Synthetic Specialists ($\ge 0.85$)**:
   - `STYLEGAN_GAN` $\rightarrow$ `[EfficientNet-B0, Hybrid SRM]`
   - `DIFFUSION` $\rightarrow$ `[Community Forensics ViT]`
   - `FACE_SWAP` $\rightarrow$ `[SBI EfficientNet-B4, Community Forensics ViT]`
   - `INPAINTING` $\rightarrow$ `[Community Forensics ViT, SBI EfficientNet-B4]`
   - `OTHER_SYNTHETIC` $\rightarrow$ `[Community Forensics ViT]`
4. **Fallback**: `UNKNOWN` $\rightarrow$ Run all five detectors.

### Shadow Mode
Shadow mode is enabled by default (`shadow_mode=True`). The DINOv2 router and safeguard policy run alongside the production pipeline, recording hypothetical decisions, telemetry, and compute metrics into `shadow_routing` without altering the final production verdict.

---

## 5. Evaluation Benchmarks & Empirical Findings

Evaluated on the preserved **frozen 250-image out-of-distribution benchmark** (50 samples $\times$ 5 manipulation categories: Real, StyleGAN, Diffusion Text2Img, FaceSwap, and Inpainting):

### Three-Way Benchmark Comparison ($N=250$)

| Metric | Pipeline 1 (All-5 Models) | Pipeline 2 (Legacy DINOv2 Routing) | Pipeline 3 (Safeguard Routing, $\tau=0.85$) | Delta (P3 vs P1) | Delta (P3 vs P2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strict Accuracy** | 26.80% (67/250) | 34.40% (86/250) | **30.00% (75/250)** | **+3.20%** | -4.40% |
| **Conclusive Accuracy** | 27.80% | 41.35% | **31.78%** | **+3.98%** | -9.57% |
| **Coverage (Decisive Rate)** | 96.40% | 83.20% | **94.40%** | -2.00% | **+11.20%** |
| **Fake Recall (Sensitivity)** | 8.50% (17/200) | 25.00% (50/200) | **13.00% (26/200)** | **+4.50% (+53% rel.)** | -12.00% |
| **Fake Precision** | 100.00% | 98.04% | **100.00%** | 0.00% | +1.96% |
| **Real Recall (Authentic Acc)** | 100.00% (50/50) | 72.00% (36/50) | **98.00% (49/50)** | -2.00% | **+26.00% (Restored)** |
| **False REAL Predictions** | 174 (87.0%) | 121 (60.5%) | **161 (80.5%)** | **-13 (Bypasses reduced)** | +40 |
| **Mean Forward Passes / Face** | **5.00** | **2.60** | **3.44** | **-1.56 (31.3% Saved)** | +0.84 |
| **Mean Latency per Face** | 847.7 ms | 708.7 ms | 842.1 ms | -5.5 ms | +133.4 ms |

### Per-Category Performance Breakdown ($N=50$ each)

| Category | Ground Truth | Pipeline 1 (All-5) | Pipeline 2 (Legacy) | Pipeline 3 (Safeguard) | Impact of Safeguard |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **REAL** | REAL | 100.0% | 72.0% | **98.0%** | **+26.0%**: Authentic face accuracy restored |
| **STYLEGAN_GAN** | FAKE | 28.0% | 54.0% | **36.0%** | **+8.0%**: Outperforms all-5 consensus |
| **DIFFUSION** | FAKE | 2.0% | 14.0% | **6.0%** | **+4.0%**: Triples all-5 detection rate |
| **FACE_SWAP** | FAKE | 2.0% | 12.0% | **4.0%** | **+2.0%**: Doubles all-5 detection rate |
| **INPAINTING** | FAKE | 2.0% | 20.0% | **6.0%** | **+4.0%**: Triples all-5 detection rate |

Detailed reports, metrics JSONs, and evaluation plots are saved in [`outputs/evaluation/safeguard_routing/`](outputs/evaluation/safeguard_routing/).

---

## 6. Repository Structure

```text
DeepFake-Forensics/
├── .gitattributes             # Git LFS tracking for >100MB checkpoints
├── .gitignore                # Excludes venv, dataset images, and download caches
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── frontend/                 # Zero-build single-page web application
│   ├── index.html            # Dashboard layout
│   ├── app.js                # Frontend controller & API bindings
│   └── styles.css            # Responsive dark/light theme styles
│
├── models/                   # Forensic model checkpoints & configs
│   ├── efficientnet_b0_best.pth  # Primary classifier (46 MB, Git tracked)
│   ├── scrfd_500m.onnx           # SCRFD face detector (2.2 MB, Git tracked)
│   ├── hybrid_detector/          # Hybrid SRM checkpoint (151 MB, Git LFS)
│   ├── sbi/                      # SBI detector weights (135 MB, Git LFS)
│   ├── deepguard/                # DeepGuard model architecture & weights
│   └── image_type_router/        # Trained DINOv2 router model & configs
│
├── datasets/                 # Dataset manifests & documentation
│   └── image_type_router/
│       └── manifest.csv      # Ground-truth manifest (1,750 images, 5 splits)
│
├── outputs/                  # Evaluation deliverables & diagnostic plots
│   ├── evaluation/           # Legacy vs All-5 evaluation reports & plots
│   └── evaluation/safeguard_routing/ # Three-way benchmark & validation tuning
│
├── scripts/                  # Standalone tools & reproduction scripts
│   ├── download_pretrained.py           # Automated model weight downloader
│   ├── evaluate_safeguard_routing.py    # 3-way evaluation & validation tuning
│   ├── evaluate_routing_comparison.py   # Legacy routing evaluation
│   ├── train_and_evaluate_router.py     # DINOv2 router training script
│   ├── assemble_router_dataset.py       # Dataset curation utility
│   └── run_preprocessing_comparison.py  # RVF10K preprocessing benchmark
│
├── src/                      # Application source code
│   ├── api/                  # FastAPI web service
│   │   ├── app.py            # Endpoints: /predict/image, /health, /models/info
│   │   └── schemas.py        # Pydantic schemas for request & telemetry responses
│   ├── explainability/       # Grad-CAM spatial heatmap generation
│   │   ├── gradcam.py        # Mathematical attribution engine
│   │   └── visualizer.py     # Alpha-blended heatmap overlays
│   └── image_detection/      # Detection engines & inference logic
│       ├── inference.py      # Unified DeepFakeInferenceEngine & safeguard routing
│       ├── face_detector.py  # SCRFD face detection & quality validation
│       ├── image_type_router.py      # DINOv2 manipulation typology router
│       ├── community_forensics_vit.py # Community Forensics ViT detector
│       ├── hybrid_srm_detector.py    # Hybrid SRM dual-stream detector
│       ├── sbi_detector.py           # SBI EfficientNet-B4 detector
│       └── vit_detector.py           # Secondary ViT detector
│
└── tests/                    # Comprehensive unit & integration test suite (133 tests)
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.14)
- Git & Git LFS ([Install Git LFS](https://git-lfs.github.com/))
- Recommended: CUDA-capable GPU (CPU is fully supported)

### Step 1: Clone the Repository
```bash
git clone https://github.com/AbijithT7/DeepFake-Forensics.git
cd DeepFake-Forensics
git lfs pull
```

### Step 2: Create a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Automated Model Weight Setup

If you cloned without Git LFS or need to download all Hugging Face hub checkpoints, run the automated downloader:

```bash
# Download and verify all required model weights:
python scripts/download_pretrained.py --all
```

Alternatively, download specific models:
```bash
python scripts/download_pretrained.py --model sbi            # SBI EfficientNet-B4
python scripts/download_pretrained.py --model community_vit  # Community Forensics ViT
python scripts/download_pretrained.py --model hybrid_srm     # Hybrid SRM Phase 2 weights
python scripts/download_pretrained.py --model vit            # Original ViT
python scripts/download_pretrained.py --model dinov2         # DINOv2 Base backbone
```

---

## 9. Running the Application

### Start the FastAPI Server
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The application automatically serves:
- **Interactive Web Interface**: `http://localhost:8000/`
- **Interactive OpenAPI Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative API Documentation (ReDoc)**: `http://localhost:8000/redoc`

---

## 10. REST API Documentation

### 1. `POST /predict/image`
Analyzes an uploaded image, localizes faces, validates quality, runs the specialist ensemble, and records shadow routing telemetry.

**Parameters (multipart/form-data)**:
- `file`: Image file (`image/jpeg`, `image/png`, `image/webp`).
- `include_gradcam` (optional, bool, default: `true`): Include base64-encoded Grad-CAM overlay.

**Example Response**:
```json
{
  "filename": "portrait.jpg",
  "prediction": "FAKE",
  "confidence": 0.9984,
  "verdict_source": "consensus_voting",
  "faces_detected": 1,
  "face_usable": true,
  "face_bbox": [124, 82, 310, 315],
  "facial_quality": {
    "is_usable": true,
    "quality_reason": null,
    "laplacian_variance": 420.5,
    "face_size": 233
  },
  "consensus_details": {
    "consensus": "majority_fake",
    "votes": { "REAL": 0, "FAKE": 4 },
    "total_models": 5
  },
  "model_predictions": {
    "primary_efficientnet": { "label": "FAKE", "score": 0.9984 },
    "secondary_vit": { "label": "FAKE", "score": 0.9851 },
    "community_forensics_vit": { "label": "FAKE", "score": 0.9992 },
    "sbi": { "label": "FAKE", "score": 0.8842 },
    "hybrid_srm": { "label": "FAKE", "score": 0.9412 }
  },
  "shadow_routing": {
    "predicted_image_type": "DIFFUSION",
    "confidence": 0.9521,
    "routing_policy": "safeguard",
    "safeguard_threshold": 0.85,
    "policy_reason": "high_confidence_diffusion (conf=0.9521 >= 0.85)",
    "selected_specialists": ["community_forensics_vit"],
    "shadow_verdict": "FAKE",
    "shadow_consensus": "unanimous_fake",
    "matches_production": true
  },
  "gradcam_base64": "data:image/jpeg;base64,..."
}
```

### 2. `GET /health`
Returns API service status, compute device, active specialist models, and active routing policy:
```json
{
  "status": "healthy",
  "device": "cpu",
  "cuda_available": false,
  "models": {
    "face_detector": "SCRFD (500M)",
    "primary": "EfficientNet-B0",
    "secondary": "ViT (HF)",
    "community_forensics_vit": "Community Forensics ViT",
    "sbi": "SBI EfficientNet-B4",
    "hybrid_srm": "Hybrid SRM"
  },
  "image_type_router": {
    "status": "loaded",
    "mode": "shadow",
    "routing_policy": "safeguard",
    "safeguard_confidence_threshold": 0.85
  }
}
```

### 3. `GET /models/info`
Returns architectural metadata, checkpoint sources, and detection specializations for all models.

---

## 11. Limitations & Responsible Use

1. **Facial Boundary Focus**: Detectors evaluate isolated face crops. Whole-body deepfakes or full-scene background manipulations without a visible face are flagged as `INCONCLUSIVE`.
2. **Adversarial Perturbations**: High-frequency noise residuals (Hybrid SRM) can be influenced by heavy JPEG re-compression or adversarial noise.
3. **Decision Support Only**: This platform is designed as an assistive tool for investigative and forensic workflows. Forensic verdicts should always be considered alongside provenance metadata and contextual evidence.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
