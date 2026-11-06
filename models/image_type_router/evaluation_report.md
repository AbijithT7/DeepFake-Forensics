# Image-Type Router Classifier - Evaluation Report

## 1. Executive Summary
- **Selected Champion Model**: `LogisticRegression_norm_C=10.0`
- **Feature Preprocessing**: `norm` (L2 feature normalization in DINOv2 embedding space)
- **Backbone Feature Extractor**: Frozen `facebook/dinov2-base` (768-dimensional CLS embeddings, zero fine-tuning)
- **Test Overall Accuracy**: **90.60%** (with rejection mechanism)
- **Test Macro F1-Score**: **90.65%**
- **Confidence Rejection Threshold**: `0.45` (inputs with max probability < threshold are safely routed to `UNKNOWN`)
- **Test Rejection Rate**: **3.01%**

> [!IMPORTANT]
> **Forensic Typology Scope**: The Image-Type Router classifies manipulation and synthesis typology (e.g. StyleGAN, Diffusion, Face-Swap, Inpainting, or Real/Unknown). It is strictly decoupled from binary authenticity verdicts (REAL vs FAKE) and does NOT dictate authenticity independently.

## 2. Model Selection on Validation Split
The following candidate models were trained strictly on the 1,225 training embeddings and evaluated on the 259 validation embeddings:

| Model | Feature Space | Validation Accuracy | Validation Macro F1 |
| :--- | :--- | :--- | :--- |
| `LogisticRegression_norm_C=10.0` | `norm` | 94.98% | 95.08% |
| `LogisticRegression_scaled_C=1.0` | `scaled` | 94.98% | 95.08% |
| `LogisticRegression_scaled_C=0.1` | `scaled` | 94.59% | 94.72% |
| `LinearSVM_calibrated_C=0.1` | `scaled` | 94.59% | 94.67% |
| `LinearSVM_calibrated_C=1.0` | `scaled` | 94.59% | 94.67% |
| `LogisticRegression_scaled_C=0.01` | `scaled` | 94.21% | 94.31% |
| `LogisticRegression_scaled_C=10.0` | `scaled` | 94.21% | 94.29% |
| `RBF_SVM_scaled_C=10.0` | `scaled` | 93.05% | 93.16% |
| `LogisticRegression_norm_C=1.0` | `norm` | 91.12% | 91.20% |
| `RBF_SVM_scaled_C=1.0` | `scaled` | 90.35% | 90.53% |
| `XGBoost_raw` | `raw` | 89.58% | 89.72% |
| `LogisticRegression_norm_C=0.1` | `norm` | 83.40% | 83.08% |
| `RandomForest_raw` | `raw` | 81.47% | 81.33% |

## 3. Rejection & UNKNOWN Routing Calibration
To prevent forcing out-of-distribution or ambiguous inputs into incorrect categories, confidence thresholding was calibrated on the validation set:

| Threshold ($\tau$) | Val Accuracy | Val Macro F1 | Rejection Rate |
| :--- | :--- | :--- | :--- |
| `0.35` | 94.98% | 95.08% | 0.00% |
| `0.40` | 94.98% | 95.03% | 0.77% |
| `0.45` | 94.59% | 94.64% | 1.54% |
| `0.50` | 91.89% | 92.02% | 5.02% |
| `0.55` | 88.03% | 88.67% | 9.65% |
| `0.60` | 86.10% | 86.95% | 11.58% |
| `0.65` | 83.40% | 84.64% | 15.06% |
| `0.70` | 81.08% | 82.91% | 18.53% |

## 4. Test Split Performance (Unseen Evaluation Set)
- **Raw Test Accuracy (no rejection)**: 92.48%
- **Raw Test Macro F1 (no rejection)**: 92.56%
- **Routed Test Accuracy (with rejection)**: **90.60%**
- **Routed Test Macro F1 (with rejection)**: **90.65%**

### Per-Class Breakdown (266 Test Samples)
| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`DIFFUSION`** | 100.00% | 94.74% | 97.30% | 38.0 |
| **`FACE_SWAP`** | 97.14% | 89.47% | 93.15% | 38.0 |
| **`INPAINTING`** | 100.00% | 94.74% | 97.30% | 38.0 |
| **`OTHER_SYNTHETIC`** | 100.00% | 100.00% | 100.00% | 38.0 |
| **`REAL`** | 81.08% | 78.95% | 80.00% | 38.0 |
| **`STYLEGAN_GAN`** | 76.32% | 76.32% | 76.32% | 38.0 |
| **`UNKNOWN`** | 82.61% | 100.00% | 90.48% | 38.0 |

## 5. Confusion Matrix
### Normalized Confusion Matrix (Rows = Ground Truth, Columns = Predicted):

| True \ Pred | `DIFFUSION` | `FACE_SWAP` | `INPAINTING` | `OTHER_SYNTHETIC` | `REAL` | `STYLEGAN_GAN` | `UNKNOWN` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DIFFUSION`** | 94.7% | 0.0% | 0.0% | 0.0% | 2.6% | 0.0% | 2.6% |
| **`FACE_SWAP`** | 0.0% | 89.5% | 0.0% | 0.0% | 0.0% | 5.3% | 5.3% |
| **`INPAINTING`** | 0.0% | 0.0% | 94.7% | 0.0% | 2.6% | 0.0% | 2.6% |
| **`OTHER_SYNTHETIC`** | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **`REAL`** | 0.0% | 0.0% | 0.0% | 0.0% | 78.9% | 18.4% | 2.6% |
| **`STYLEGAN_GAN`** | 0.0% | 2.6% | 0.0% | 0.0% | 13.2% | 76.3% | 7.9% |
| **`UNKNOWN`** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |

### Raw Count Confusion Matrix:

| True \ Pred | `DIFFUSION` | `FACE_SWAP` | `INPAINTING` | `OTHER_SYNTHETIC` | `REAL` | `STYLEGAN_GAN` | `UNKNOWN` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DIFFUSION`** | 36 | 0 | 0 | 0 | 1 | 0 | 1 |
| **`FACE_SWAP`** | 0 | 34 | 0 | 0 | 0 | 2 | 2 |
| **`INPAINTING`** | 0 | 0 | 36 | 0 | 1 | 0 | 1 |
| **`OTHER_SYNTHETIC`** | 0 | 0 | 0 | 38 | 0 | 0 | 0 |
| **`REAL`** | 0 | 0 | 0 | 0 | 30 | 7 | 1 |
| **`STYLEGAN_GAN`** | 0 | 1 | 0 | 0 | 5 | 29 | 3 |
| **`UNKNOWN`** | 0 | 0 | 0 | 0 | 0 | 0 | 38 |

## 6. Dataset Provenance and Integrity
- **REAL**: FFHQ Authentic Human Portraits (CC BY-NC-SA 4.0)
- **STYLEGAN_GAN**: StyleGAN High-Resolution Synthetic Portraits (CC BY-NC-SA 4.0)
- **DIFFUSION**: RunwayML Stable Diffusion v1.5 text2img Portraits (CreativeML OpenRAIL-M)
- **FACE_SWAP**: BitMind FaceSwap InsightFace CelebA-HQ Swaps (MIT)
- **INPAINTING**: CVPR NTIRE 2022 FFHQ Inpainting Benchmark (Academic Research License)
- **OTHER_SYNTHETIC**: HugGAN Anime GAN Synthetic Faces (MIT)
- **UNKNOWN**: CIFAR-10 Natural Objects + Project Non-Face Fixtures (MIT)
- **Blocklist Verification**: All 1,750 images were strictly deduplicated and verified against `experiments/v1/blocklist.csv` (zero SHA-256 and zero pHash overlap).

## 7. Operational Deployment Specification
- Checkpoint saved to: `models/image_type_router/router_model.joblib`
- Preprocessing metadata: `models/image_type_router/router_config.json`
- Class index mapping: `models/image_type_router/label_mapping.json`