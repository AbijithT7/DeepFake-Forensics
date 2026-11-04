# EfficientNet-B0 Preprocessing Validation Report

This report provides an empirical evaluation comparing **Original Training Preprocessing** against **Current SCRFD Face Detection & Cropping Preprocessing** on the official RVF10K test set.

---

## 1. Executive Summary

| Evaluation Setup | Total Evaluated | Correct | Accuracy | Misclassified / Inconclusive | Performance vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Approach 1: Original Training Preprocessing** | 1,000 | 993 | **99.30%** | 7 misclassified (1 FAKE, 6 REAL) | **Baseline** |
| **Approach 2: Current SCRFD Pipeline (with Quality Gates)** | 1,000 | 897 | **89.70%** (overall)<br>**97.29%** (conclusive) | 25 misclassified (5 FAKE, 20 REAL)<br>78 inconclusive (`face_not_usable`) | **-9.60%** (overall)<br>**-2.01%** (conclusive) |
| **Approach 3: Direct SCRFD Face Crop (without Quality Rejections)** | 1,000 | 973 | **97.30%** | 27 misclassified (6 FAKE, 21 REAL) | **-2.00%** |

### Key Takeaway
> **Does SCRFD preprocessing cause a performance drop?**
> - **Pure spatial face cropping** causes a small **2.00% drop** (from 99.30% to 97.30%).
> - **The deployed SCRFD pipeline** exhibits an apparent **9.60% drop** (down to 89.70%), but **78 of the 103 non-correct outcomes are INCONCLUSIVE rejections**, not model misclassifications.
> - Specifically, **74 out of 78 inconclusive samples** are triggered by an overly sensitive threshold in `validate_face_quality` (`uniform_black > 0.15`), which falsely flags natural dark hair or dark studio backgrounds as solid block occlusion.
> - When those 78 rejected face crops are passed to EfficientNet-B0, the model classifies **76 out of 78 (97.4%) correctly**.

---

## 2. Experimental Setup

- **Dataset**: RVF10K Test Set (`8020/test`)
  - Total: **1,000 images** (500 FAKE, 500 REAL)
  - Dimensions: $256 \times 256$ pixels (RGB)
- **Model Checkpoint**: `models/efficientnet_b0_best.pth` (untouched, loaded in `eval()` mode)
- **Face Detector**: `models/scrfd_500m.onnx` (100% detection rate on RVF10K test set)
- **Hardware**: CPU execution

```
Approach 1 (Training): Full 256x256 Image ──> Resize(224, 224) ──> Normalize ──> EfficientNet-B0 (99.30%)
Approach 2 (Pipeline): Full 256x256 Image ──> SCRFD Detect ──> Quality Gates ──> Crop(15% margin) ──> Resize(224, 224) ──> EfficientNet-B0 (89.70%)
Approach 3 (Crop Only): Full 256x256 Image ──> SCRFD Detect ──> Crop(15% margin) ──> Resize(224, 224) ──> Normalize ──> EfficientNet-B0 (97.30%)
```

---

## 3. Detailed Results & Confusion Matrices

### Approach 1: Original Training Preprocessing

Direct resizing of the uncropped $256 \times 256$ image to $224 \times 224$ followed by standard ImageNet normalization.

- **Accuracy**: **99.30%** (993 / 1000)
- **Confusion Matrix**:
  | | Predicted FAKE | Predicted REAL |
  | :--- | :---: | :---: |
  | **Actual FAKE** | **499** (TN) | 1 (FP) |
  | **Actual REAL** | 6 (FN) | **494** (TP) |
- **Total Misclassified Samples (7)**:
  1. `75NEMJOW2U.jpg` (FAKE $\rightarrow$ REAL, score: 0.6123)
  2. `06055.jpg` (REAL $\rightarrow$ FAKE, score: 0.7905)
  3. `09541.jpg` (REAL $\rightarrow$ FAKE, score: 0.5536)
  4. `15026.jpg` (REAL $\rightarrow$ FAKE, score: 0.5619)
  5. `19345.jpg` (REAL $\rightarrow$ FAKE, score: 0.7254)
  6. `20500.jpg` (REAL $\rightarrow$ FAKE, score: 0.6883)
  7. `23050.jpg` (REAL $\rightarrow$ FAKE, score: 0.8828)

---

### Approach 2: Current SCRFD Inference Pipeline

Full inference pipeline in `src/image_detection/inference.py`: SCRFD face detection $\rightarrow$ `validate_face_quality` $\rightarrow$ controlled margin cropping (15%) $\rightarrow$ $224 \times 224$ resize $\rightarrow$ EfficientNet-B0.

- **Overall Accuracy**: **89.70%** (897 / 1000)
- **Conclusive Subset Accuracy**: **97.29%** (897 / 922)
- **Inconclusive Outcomes**: **78 samples** (all marked `face_not_usable`)
- **Active Misclassifications (on Conclusive Samples)**: **25 samples**
  - FAKE misclassified as REAL: **5** (`0WO53Z9WD6.jpg`, `5USK9A4R48.jpg`, `71I1PREXJI.jpg`, `75NEMJOW2U.jpg`, `MOI5OFJ4Y5.jpg`)
  - REAL misclassified as FAKE: **20** (`03006.jpg`, `03151.jpg`, `03695.jpg`, `06055.jpg`, `06269.jpg`, `08820.jpg`, `09491.jpg`, `12183.jpg`, `13172.jpg`, `13177.jpg`, `13918.jpg`, `16228.jpg`, `16479.jpg`, `17324.jpg`, `17445.jpg`, `19801.jpg`, `19815.jpg`, `19873.jpg`, `20500.jpg`, `23050.jpg`)

---

### Approach 3: Direct SCRFD Face Crop (No Quality Rejection)

To isolate the pure spatial effect of face cropping from heuristic quality checks, SCRFD face crops were evaluated directly by EfficientNet-B0 without rejection.

- **Accuracy**: **97.30%** (973 / 1000)
- **Confusion Matrix**:
  | | Predicted FAKE | Predicted REAL |
  | :--- | :---: | :---: |
  | **Actual FAKE** | **494** (TN) | 6 (FP) |
  | **Actual REAL** | 21 (FN) | **479** (TP) |
- **Total Misclassified Samples (27)**:
  - **FAKE $\rightarrow$ REAL (6)**: `0WO53Z9WD6.jpg`, `0YJG9QBYQZ.jpg`, `5USK9A4R48.jpg`, `71I1PREXJI.jpg`, `75NEMJOW2U.jpg`, `MOI5OFJ4Y5.jpg`
  - **REAL $\rightarrow$ FAKE (21)**: `03006.jpg`, `03151.jpg`, `03695.jpg`, `06055.jpg`, `06269.jpg`, `08820.jpg`, `09491.jpg`, `12183.jpg`, `13172.jpg`, `13177.jpg`, `13918.jpg`, `16228.jpg`, `16479.jpg`, `17324.jpg`, `17445.jpg`, `19801.jpg`, `19815.jpg`, `19873.jpg`, `20500.jpg`, `21554.jpg`, `23050.jpg`

---

## 4. Root Cause Analysis

### Cause 1: Domain Shift from Spatial Cropping (-2.00%)
1. **Training Preprocessing**: EfficientNet-B0 was trained on the raw RVF10K images ($256 \times 256$), which are already centered portraits showing the full head, hair volume, neckline, and background. The training pipeline simply applied `transforms.Resize((224, 224))` to the entire image.
2. **Inference Preprocessing**: SCRFD detects the tight face bounding box and crops it with a 15% context margin. When this crop is resized to $224 \times 224$, the face appears significantly larger (zoomed-in) relative to the training distribution, altering spatial frequency and removing background cues.
3. Despite this domain shift, EfficientNet-B0 generalizes remarkably well, retaining **97.30% accuracy** on the crops alone.

### Cause 2: Overly Sensitive Block-Occlusion Heuristic (-7.60%)
1. The quality validation routine `validate_face_quality` contains:
   ```python
   uniform_black = float(np.sum(gray < 15)) / float(gray.size)
   if uniform_black > 0.15:
       return False, "face_not_usable"
   ```
2. In dark-haired subjects or portraits taken against dark studio backdrops, pixels with intensity $< 15$ frequently exceed 15% of the cropped area.
3. This single rule triggered on **74 out of the 78 rejected samples**.
4. When those 78 images were evaluated directly by EfficientNet-B0 on the exact same SCRFD crop, **76 out of 78 (97.4%) were classified correctly**.

---

## 5. Artifacts and Data Files

- Benchmark script: `outputs/run_preprocessing_comparison.py`
- Full per-sample JSON export: `outputs/preprocessing_comparison.json`
