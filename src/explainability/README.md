# Grad-CAM Explainability Module (`src.explainability`)

This module provides **Gradient-weighted Class Activation Mapping (Grad-CAM)** for **EfficientNet-B0** within the **DeepFake-Forensics** framework. It enables investigators and users to inspect which spatial regions in an isolated face crop influenced the convolutional neural network's classification as **REAL** or **FAKE**.

---

## 1. What is Grad-CAM?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is an established Explainable Artificial Intelligence (XAI) technique developed by Selvaraju et al. (2017). 

While deep neural networks act as black boxes that output class probabilities, Grad-CAM produces a 2D spatial attribution heatmap indicating which feature map regions contributed most strongly to a specific class score.

> [!IMPORTANT]
> **Grad-CAM highlights model attention, not forensic proof of manipulation.**
> Grad-CAM visualizes where the convolutional layers exhibited positive activations toward the predicted class. It is a visual explanation of model behavior, not standalone proof of forgery.

---

## 2. Standard Mathematical Formulation

The implementation strictly adheres to the standard formulation:

1. **Forward Pass:** The preprocessed face crop tensor $X \in \mathbb{R}^{1 \times 3 \times 224 \times 224}$ is evaluated through EfficientNet-B0 to produce logits $y = [y_{\text{FAKE}}, y_{\text{REAL}}]$ and terminal convolutional activations $A \in \mathbb{R}^{1280 \times 7 \times 7}$ from `backbone.features[8]`.
2. **Target Score Selection:** Unnormalized logit score $y^c$ is selected for the predicted class $c$ (preventing softmax cross-class saturation).
3. **Gradient Calculation:** Gradients of $y^c$ with respect to activation maps $A^k$ are calculated via backward hooks:
   $$\frac{\partial y^c}{\partial A^k_{i, j}}$$
4. **Global Average Pooling:** Channel importance weights $\alpha_k^c$ are computed across spatial dimensions:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
5. **Weighted Combination & Positive Rectification (ReLU):**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   ReLU ensures that only features positively contributing to the target class are retained.
6. **Min-Max Normalization:**
   $$L_{\text{norm}}^c = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + 10^{-8}}$$
7. **Bilinear Interpolation:** The $7 \times 7$ activation map is bilinearly upsampled to the exact dimensions $(W, H)$ of the original face crop.
8. **Intensity-Modulated Alpha Overlay:**
   $$\alpha(x, y) = \text{clip}(L_{\text{norm}}(x, y) \cdot \alpha_{\text{max}}, 0, \alpha_{\text{max}})$$
   $$\text{Overlay}(x, y) = (1 - \alpha(x, y)) \cdot \text{Face}(x, y) + \alpha(x, y) \cdot \text{Colormap}(x, y)$$
   **Inactive regions ($L = 0.0$) receive $\alpha = 0.0$, remaining completely transparent without any purple colormap tint.**

---

## 3. CLI Usage

Run Grad-CAM on any image using the unified inference pipeline:

```bash
python -m src.explainability.pipeline --image tests/sample_image_real.jpg
```

Arguments:
- `--image`, `-i`: Path to the input image file.
- `--device`, `-d`: Compute device (`auto`, `cpu`, `cuda`).
- `--output-dir`, `-o`: Directory to save visual artifacts (default: `outputs/gradcam/`).
- `--alpha`: Maximum overlay opacity (default: `0.5`).
- `--threshold`: Uncertainty rejection threshold (default: `0.60`).
