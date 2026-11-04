"""
Grad-CAM Visualization Module for DeepFake Forensics.

Provides rendering and export of:
1. Original input image and annotated face bounding boxes.
2. Cropped face images used for model inference.
3. Raw Grad-CAM heatmaps (Turbo colormap).
4. Alpha-blended heatmap overlays directly aligned to the face crop (intensity-modulated transparency).
5. 3-panel diagnostic figures: [Face Crop | Heatmap | Overlay].
6. 3-panel face detection overview: [Original Image with Box | Face Crop | Overlay].
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.explainability.gradcam import GradCAMResult


class GradCAMVisualizer:
    """
    Renders and exports forensic attribution visualizations.
    """

    def __init__(self, default_colormap: str = "turbo", default_alpha: float = 0.5):
        """
        Args:
            default_colormap: Matplotlib colormap name ('turbo', 'jet', etc.).
            default_alpha: Maximum opacity factor for peak activations (0.0 - 1.0).
        """
        self.default_colormap = default_colormap
        self.default_alpha = default_alpha

    def apply_colormap(
        self,
        heatmap: np.ndarray,
        colormap: Optional[str] = None,
    ) -> np.ndarray:
        """
        Transform a 2D normalized float32 heatmap [0.0, 1.0] to an RGB uint8 array.
        """
        cmap_name = colormap or self.default_colormap
        try:
            cmap = matplotlib.colormaps[cmap_name]
        except (AttributeError, KeyError):
            cmap = plt.get_cmap(cmap_name)

        clamped = np.clip(heatmap, 0.0, 1.0)
        rgba = cmap(clamped)
        rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
        return rgb

    def create_overlay(
        self,
        image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an alpha-blended overlay of the heatmap on top of the original/cropped image.

        Inactive regions (heatmap near 0.0) remain transparent rather than receiving a purple tint.
        Alpha is modulated by activation intensity:
        - heatmap == 0.0: alpha is 0.0 (completely transparent natural face crop)
        - heatmap > 0.0: alpha scales smoothly up to alpha_val for peak model attributions
        """
        alpha_val = self.default_alpha if alpha is None else alpha

        if isinstance(image, Image.Image):
            rgb_img = np.array(image.convert("RGB"))
        else:
            rgb_img = image

        h_img, w_img = rgb_img.shape[:2]
        h_hm, w_hm = heatmap.shape[:2]

        if (h_hm, w_hm) != (h_img, w_img):
            pil_hm = Image.fromarray(heatmap)
            pil_hm_resized = pil_hm.resize((w_img, h_img), resample=Image.Resampling.BILINEAR)
            heatmap = np.array(pil_hm_resized)

        heatmap_rgb = self.apply_colormap(heatmap, colormap=colormap)

        # Intensity-modulated alpha transparency:
        # Inactive regions (heatmap == 0.0) receive alpha = 0.0 (clean natural face crop, no purple tint).
        alpha_mask = np.clip(heatmap[..., None] * alpha_val, 0.0, alpha_val)
        blended = (1.0 - alpha_mask) * rgb_img.astype(np.float32) + alpha_mask * heatmap_rgb.astype(np.float32)
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended_uint8)

    def create_side_by_side(
        self,
        image: Image.Image,
        heatmap_img: Image.Image,
        overlay_img: Image.Image,
        result: GradCAMResult,
        title: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a 3-panel figure: [Input/Crop | Grad-CAM Heatmap | Overlay].
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150)
        plt.subplots_adjust(top=0.82, wspace=0.1)

        pred_color = "#d9534f" if result.predicted_class == "FAKE" else "#5cb85c"

        main_title = title or f"{result.model_name} Grad-CAM Model Attention"
        fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98)

        subtitle = (
            f"Prediction: {result.predicted_class} ({result.confidence:.2f}%)  |  "
            f"Softmax: FAKE: {result.probabilities.get('FAKE', 0.0):.1f}%, "
            f"REAL: {result.probabilities.get('REAL', 0.0):.1f}%  |  "
            f"Target: {result.target_class} (logit: {result.target_score:+.2f})  |  "
            f"Layer: {result.layer_name}"
        )
        fig.text(0.5, 0.88, subtitle, ha="center", fontsize=10, color="#333333")

        axes[0].imshow(image)
        axes[0].set_title("1. Evaluated Face Crop", fontsize=12, fontweight="semibold")
        axes[0].axis("off")

        axes[1].imshow(heatmap_img)
        axes[1].set_title(f"2. Grad-CAM ({result.target_class} Attention)", fontsize=12, fontweight="semibold")
        axes[1].axis("off")

        axes[2].imshow(overlay_img)
        axes[2].set_title(f"3. Attention Overlay ({result.predicted_class})", fontsize=12, fontweight="semibold", color=pred_color)
        axes[2].axis("off")

        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rendered = Image.fromarray(rgba_buffer[:, :, :3])
        plt.close(fig)
        return rendered

    def create_face_side_by_side(
        self,
        annotated_original: Image.Image,
        face_crop: Image.Image,
        overlay: Image.Image,
        primary_result: GradCAMResult,
        title: str = "DeepFake Forensics: Face Detection & Grad-CAM Model Attention",
    ) -> Image.Image:
        """
        Generate the 3-panel face detection overview:
        Original Image (with Bounding Box) | Detected Face Crop | Face Grad-CAM Overlay
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
        plt.subplots_adjust(top=0.82, wspace=0.1)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        subtitle = f"{primary_result.model_name}: {primary_result.predicted_class} ({primary_result.confidence:.1f}%)"
        fig.text(0.5, 0.88, subtitle, ha="center", fontsize=11, color="#333333")

        # 1. Original with face box
        axes[0].imshow(annotated_original)
        axes[0].set_title("1. Original Image (Face Detected)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 2. Detected face crop
        axes[1].imshow(face_crop)
        axes[1].set_title("2. Extracted Face Crop", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(f"Crop Size: {face_crop.size[0]}x{face_crop.size[1]} px", fontsize=9)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # 3. Face Grad-CAM overlay
        pred_color = "#d9534f" if primary_result.predicted_class == "FAKE" else "#5cb85c"
        axes[2].imshow(overlay)
        axes[2].set_title(f"3. Facial Grad-CAM Overlay ({primary_result.model_name})", fontsize=12, fontweight="bold", color=pred_color)
        axes[2].set_xlabel("Model attention focused strictly on facial regions", fontsize=9)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rendered = Image.fromarray(rgba_buffer[:, :, :3])
        plt.close(fig)
        return rendered

    def save_visualizations(
        self,
        output_dir: Union[str, Path],
        original_image: Image.Image,
        evaluated_image: Image.Image,
        original_with_faces: Optional[Image.Image] = None,
        face_crop_img: Optional[Image.Image] = None,
        face_side_by_side: Optional[Image.Image] = None,
        effnet_result: Optional[GradCAMResult] = None,
        effnet_overlay: Optional[Image.Image] = None,
        effnet_heatmap_img: Optional[Image.Image] = None,
    ) -> Dict[str, Path]:
        """
        Save individual and combined visualizations adhering to the project directory structure.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}

        # 1. Original image
        orig_path = out_path / "original.png"
        original_image.save(orig_path, format="PNG")
        saved_files["original"] = orig_path

        # 2. Original with annotated face boxes
        if original_with_faces is not None:
            ann_path = out_path / "original_with_faces.png"
            original_with_faces.save(ann_path, format="PNG")
            saved_files["original_with_faces"] = ann_path

        # 3. Face Crop image
        if face_crop_img is not None:
            fc_path = out_path / "face_crop.png"
            face_crop_img.save(fc_path, format="PNG")
            saved_files["face_crop"] = fc_path

        # 4. Face Side-by-Side (Original with Box | Face Crop | Grad-CAM Overlay)
        if face_side_by_side is not None:
            fsbs_path = out_path / "face_side_by_side.png"
            face_side_by_side.save(fsbs_path, format="PNG")
            saved_files["face_side_by_side"] = fsbs_path

        # 5. EfficientNet visualizations (on evaluated face crop)
        if effnet_result is not None and effnet_heatmap_img is not None and effnet_overlay is not None:
            eff_hm_path = out_path / "efficientnet_heatmap.png"
            effnet_heatmap_img.save(eff_hm_path, format="PNG")
            saved_files["efficientnet_heatmap"] = eff_hm_path

            eff_ov_path = out_path / "efficientnet_overlay.png"
            effnet_overlay.save(eff_ov_path, format="PNG")
            saved_files["efficientnet_overlay"] = eff_ov_path

            eff_sbs = self.create_side_by_side(
                evaluated_image, effnet_heatmap_img, effnet_overlay, effnet_result
            )
            eff_sbs_path = out_path / "efficientnet_side_by_side.png"
            eff_sbs.save(eff_sbs_path, format="PNG")
            saved_files["efficientnet_side_by_side"] = eff_sbs_path

        return saved_files
