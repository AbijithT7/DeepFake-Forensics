# Grad-CAM Explainability & Forensic Diagnostic Module (`src.explainability`)

This module provides **Gradient-weighted Class Activation Mapping (Grad-CAM)** and **Shortcut Learning Diagnostics** for the **DeepFake-Forensics** framework. It enables forensic analysts and investigators to inspect *which spatial regions and facial features* influenced a deep learning model's prediction when classifying an image as **REAL** or **FAKE**.

---

## 1. What is Grad-CAM?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is an established Explainable Artificial Intelligence (XAI) technique developed by Selvaraju et al. (2017). 

While traditional deep neural networks act as "black boxes" that output only class probabilities, Grad-CAM produces a coarse 2D heatmap showing which spatial regions in an input image contributed most strongly to a specific class score.

In plain terms:
- Standard prediction answers: *"What does the model think this image is?"*
- Grad-CAM answers: *"Where was the model looking when it made that decision?"*

---

## 2. Why Does the Custom CNN Focus on Backgrounds and Corners?

### The Forensic Finding:
When evaluating images, the Custom CNN often produces heatmaps highlighting corners, background walls, or hair edges rather than the face. Meanwhile, EfficientNet-B0 concentrates directly on the face.

### The Root Cause:
The Grad-CAM implementation itself is mathematically sound. Rather, **Grad-CAM is accurately revealing a severe "shortcut learning" (Clever Hans) problem in the Custom CNN model:**

1. **Format Asymmetry in Training Data:** The Custom CNN was trained on the Kaggle `deepfake-vs-real-60k` subset, where Fake images were stored as `.jpg` and Real images were stored as `.png`. The model learned compression artifacts and high-frequency boundary patterns instead of facial anatomy.
2. **Tiny Sample Size Without Augmentations:** The Custom CNN was trained on only **800 images** without data augmentations (no flips, rotations, or color jitter).
3. **Absence of Global Average Pooling:** The Custom CNN architecture flattens feature maps directly (`Linear(128 * 16 * 16, 128)`). Unlike EfficientNet-B0's `AdaptiveAvgPool2d(1)`, the Custom CNN lacks spatial translation invariance, memorizing specific pixel coordinates (e.g., corners and borders).
4. **Zero-Padding Boundary Effects:** Convolutions with `padding=1` on images without Batch Normalization create unnatural gradient discontinuities at the outer borders.

### Empirical Proof via Diagnostics:
On `sample_image.jpg`:
- **Original Image:** Custom CNN predicts FAKE: **98.94%** | EfficientNet predicts FAKE: **26.27%**
- **Face-Only (Background Neutralized):** Custom CNN predicts FAKE: **96.82%** | EfficientNet predicts FAKE: **9.61%**
- **Background-Only (Face Completely Masked):** Custom CNN predicts FAKE: **93.16%** | EfficientNet predicts FAKE: **0.09%**

> [!WARNING]
> When the face is completely removed from the image, the **Custom CNN still classifies the background as FAKE with 93.16% confidence** (a 94.2% Background Dependency Ratio). EfficientNet-B0's FAKE score drops to **0.09%**, demonstrating that EfficientNet relies on facial features while Custom CNN relies primarily on background shortcuts.

---

## 3. How Grad-CAM Works Mathematically

1. **Forward Pass:** The input image $X$ is propagated through the model to obtain logits $Y = [Y_{\text{FAKE}}, Y_{\text{REAL}}]$ and terminal convolutional activations $A \in \mathbb{R}^{K \times H' \times W'}$.
2. **Target Score Selection:** The target class $c \in \{\text{FAKE}, \text{REAL}\}$ is selected. We take the unnormalized logit score $Y^c$ (before softmax, avoiding cross-class probability saturation).
3. **Gradient Calculation:** We compute the gradient of the target class score with respect to the feature map activations:
   $$\frac{\partial Y^c}{\partial A^k_{i, j}}$$
4. **Global Average Pooling of Gradients:** We compute neuron importance weights $\alpha_k^c$:
   $$\alpha_k^c = \frac{1}{H' \times W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial Y^c}{\partial A_{i, j}^k}$$
5. **Weighted Linear Combination:**
   $$L^c = \sum_{k=1}^K \alpha_k^c A^k$$
6. **Rectified Linear Unit (ReLU):**
   $$L_{\text{Grad-CAM}}^c = \max(0, L^c)$$
7. **Min-Max Normalization & Upsampling:** The normalized 2D heatmap is interpolated back to original image dimensions $(W, H)$ and blended with the input image.

---

## 4. Architecture Support & Target Layer Selection

| Model | Checkpoint File | Final Conv Target Layer | Activation Shape | Spatial Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | `models/deepfake_cnn.pth` | `features[6]` (`Conv2d(64, 128)`) | `(1, 128, 32, 32)` | 32×32 px (from 128×128 input) |
| **EfficientNet-B0** | `models/efficientnet_b0_best.pth` | `backbone.features[8]` (`Conv2dNormActivation`) | `(1, 1280, 7, 7)` | 7×7 px (from 224×224 input) |

---

## 5. Diagnostic Testing Capabilities

The module includes 4 specialized testing modes to isolate and diagnose background bias:

1. **Face-Only Testing (`--face-only`):** An anatomical elliptical mask neutralizes all background pixels to neutral gray `(128, 128, 128)`, leaving only the face.
2. **Background-Only Testing (`--background-only`):** Neutralizes the facial region with gray, leaving only background, hair edges, and clothing. Reveals whether the model classifies the image without seeing the face.
3. **Face-Crop Testing (`--face-crop`):** Crops directly to the detected face bounding box, eliminating background padding.
4. **Fair FAKE Comparison (`--fair-compare` or `--target fake`):** Forces both models to explain the exact same class (`FAKE`), enabling direct visual comparison of what features each model attributes to manipulation.
5. **Occlusion Sensitivity Analysis (`--occlusion`):** Slides a neutral gray patch across the image and records the drop in target class probability:
   $$\text{Drop}(x, y) = \max(0, P_{\text{baseline}} - P_{\text{occluded}(x, y)})$$
   Because this test does not use gradients or hooks, it provides an independent validation of spatial importance.

---

## 6. Command-Line Interface (CLI) Usage

Execute the pipeline using the `-m` module syntax:

### Basic Inference (Both Models)
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg
```

### Face-Only Test (Background Neutralized)
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg --face-only
```

### Background-Only Test (Face Neutralized)
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg --background-only
```

### Face-Crop Test (Cropped Bounding Box)
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg --face-crop
```

### Fair Comparison (Both Models Target FAKE)
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg --fair-compare
```

### Run Occlusion Sensitivity Analysis
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg --occlusion
```

### Automated Multi-Face Detection
```bash
python -m src.explainability.pipeline --image tests/sample_image.jpg --detect-faces
```

### Run Automated Tests
```bash
python tests/test_gradcam.py
```

---

## 7. Output Directory Structure

Generated visual artifacts are saved under `outputs/gradcam/<image_stem>/`:

```text
outputs/
└── gradcam/
    └── sample_image/
        ├── original.png                   # Original input image
        ├── face_only.png                  # Face-only diagnostic input
        ├── background_only.png            # Background-only diagnostic input
        ├── cnn_heatmap.png                # Custom CNN raw Grad-CAM heatmap
        ├── cnn_overlay.png                # Custom CNN alpha-blended overlay
        ├── cnn_side_by_side.png           # 3-panel figure: [Original | Heatmap | Overlay]
        ├── efficientnet_heatmap.png       # EfficientNet-B0 raw Grad-CAM heatmap
        ├── efficientnet_overlay.png       # EfficientNet-B0 alpha-blended overlay
        ├── efficientnet_side_by_side.png  # 3-panel diagnostic figure for EfficientNet-B0
        ├── comparison.png                 # Dual-model comparison [Original | CNN | EffNet]
        ├── fair_fake_comparison.png       # Fair FAKE attribution comparison
        ├── occlusion_overlay.png          # Sliding-window occlusion sensitivity map
        └── diagnostic_dashboard.png       # 6-panel comprehensive diagnostic summary
```

---

## 8. Simple Recommendations to Improve Custom CNN Training

To prevent the Custom CNN from learning background shortcuts, apply the following simple, beginner-friendly improvements during future retraining:

1. **Pre-crop and Align Faces Before Training:**
   Never train on full scenes. Run face detection (e.g. OpenCV Haar Cascade or MTCNN) as a preprocessing step so that 100% of the training image consists of facial features.
2. **Replace `Flatten()` with `AdaptiveAvgPool2d(1)`:**
   Instead of flattening $128 \times 16 \times 16 = 32,768$ features directly into a Linear layer, add `nn.AdaptiveAvgPool2d((1, 1))` followed by `nn.Linear(128, 2)`. This provides spatial translation invariance and eliminates fixed corner/border biases.
3. **Add Batch Normalization:**
   Add `nn.BatchNorm2d(num_channels)` after each `Conv2d` layer. This stabilizes activation scales and prevents zero-padding edge spikes.
4. **Data Augmentation Pipeline:**
   Apply standard augmentations during training:
   ```python
   transforms.Compose([
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(degrees=10),
       transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
   ])
   ```
5. **Use the Clean RVF10K Dataset:**
   Retrain using the audited `Downloads/Real vs Fake Faces/8020` dataset, which has balanced JPEG formatting for both Real and Fake classes, eliminating format-based shortcut leakage.

---

## 9. Critical Limitations & Forensic Boundaries

> [!CAUTION]
> **Grad-CAM highlights regions of influence, NOT mathematical proof of manipulation.**

1. **Influence $\neq$ Manipulation:** Grad-CAM shows which pixels moved the model's score. It does not certify that those pixels are manipulated.
2. **Not an Independent Classifier:** If a model predicts incorrectly, Grad-CAM simply explains *why the model made that incorrect prediction*.
3. **Background Artifact Sensitivity:** As proven by the Background-Only test, high-scoring heatmaps over background edges reflect model shortcut vulnerabilities, not deepfake tampering.
4. **Spatial Resolution Bounds:** Heatmaps are derived from coarse feature maps ($32 \times 32$ for CNN, $7 \times 7$ for EfficientNet) and upsampled bilinearly.
