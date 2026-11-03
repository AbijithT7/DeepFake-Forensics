"""
Grad-CAM Visualization Module for DeepFake Forensics.

Provides rendering and export of:
1. Original input image and annotated face bounding boxes.
2. Cropped face images used for model inference.
3. 3-panel face detection overview [Original with Box | Detected Face Crop | Face Grad-CAM Overlay].
4. Raw Grad-CAM heatmaps (TURBO or JET colormap).
5. Alpha-blended heatmap overlays directly aligned to the face crop.
6. Per-model diagnostic figures [Face Crop | Heatmap | Overlay].
7. Combined cross-model comparison [Face Crop | Custom CNN | EfficientNet-B0].
8. Fair-target comparison (both models explaining class FAKE).
9. Occlusion sensitivity overlays.
10. Forensic diagnostic dashboard.
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
            default_alpha: Opacity factor for the heatmap overlay (0.0 - 1.0).
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

        blended = (1.0 - alpha_val) * rgb_img.astype(np.float32) + alpha_val * heatmap_rgb.astype(np.float32)
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

        model_display = "Custom CNN" if "CNN" in result.model_name else "EfficientNet-B0"
        pred_color = "#d9534f" if result.predicted_class == "FAKE" else "#5cb85c"

        main_title = title or f"{model_display} Grad-CAM Attribution"
        fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98)

        subtitle = (
            f"Prediction: {result.predicted_class} ({result.confidence:.2f}%)  |  "
            f"Probabilities: FAKE: {result.probabilities.get('FAKE', 0.0):.1f}%, "
            f"REAL: {result.probabilities.get('REAL', 0.0):.1f}%  |  "
            f"Target: {result.target_class} (logit: {result.target_score:+.2f})  |  "
            f"Layer: {result.layer_name}"
        )
        fig.text(0.5, 0.88, subtitle, ha="center", fontsize=10, color="#333333")

        axes[0].imshow(image)
        axes[0].set_title("1. Evaluated Face Crop", fontsize=12, fontweight="semibold")
        axes[0].axis("off")

        axes[1].imshow(heatmap_img)
        axes[1].set_title(f"2. Grad-CAM ({result.target_class} Evidence)", fontsize=12, fontweight="semibold")
        axes[1].axis("off")

        axes[2].imshow(overlay_img)
        axes[2].set_title(f"3. Forensic Overlay ({result.predicted_class})", fontsize=12, fontweight="semibold", color=pred_color)
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
        secondary_result: Optional[GradCAMResult] = None,
        title: str = "DeepFake Forensics: Face-Based Detection & Grad-CAM Analysis",
    ) -> Image.Image:
        """
        Generate the 3-panel face detection overview:
        Original Image (with Bounding Box) | Detected Face Crop | Face Grad-CAM Overlay
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
        plt.subplots_adjust(top=0.82, wspace=0.1)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        sub_parts = [
            f"{primary_result.model_name}: {primary_result.predicted_class} ({primary_result.confidence:.1f}%)"
        ]
        if secondary_result is not None:
            sub_parts.append(
                f"{secondary_result.model_name}: {secondary_result.predicted_class} ({secondary_result.confidence:.1f}%)"
            )
        subtitle = "  |  ".join(sub_parts)
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
        axes[2].set_xlabel("Attributions focused strictly on facial regions", fontsize=9)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rendered = Image.fromarray(rgba_buffer[:, :, :3])
        plt.close(fig)
        return rendered

    def create_model_comparison(
        self,
        face_crop: Image.Image,
        cnn_overlay: Image.Image,
        effnet_overlay: Image.Image,
        cnn_result: GradCAMResult,
        effnet_result: GradCAMResult,
        title: str = "DeepFake Forensics: Cross-Model Face Attribution Comparison",
    ) -> Image.Image:
        """
        Generate 3-panel comparative visualization:
        Detected Face Crop | Custom CNN Face Grad-CAM | EfficientNet-B0 Face Grad-CAM
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
        plt.subplots_adjust(top=0.82, wspace=0.1)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        # Panel 1: Face Crop
        axes[0].imshow(face_crop)
        axes[0].set_title("Evaluated Face Crop", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(f"Dimensions: {face_crop.size[0]}x{face_crop.size[1]} px", fontsize=9)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Panel 2: Custom CNN
        cnn_color = "#d9534f" if cnn_result.predicted_class == "FAKE" else "#5cb85c"
        axes[1].imshow(cnn_overlay)
        axes[1].set_title(
            f"Custom CNN ({cnn_result.predicted_class} {cnn_result.confidence:.1f}%)",
            fontsize=12,
            fontweight="bold",
            color=cnn_color,
        )
        axes[1].set_xlabel(
            f"Target: {cnn_result.target_class} | FAKE: {cnn_result.probabilities['FAKE']:.1f}% | REAL: {cnn_result.probabilities['REAL']:.1f}%\n"
            f"Layer: {cnn_result.layer_name}",
            fontsize=9,
        )
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Panel 3: EfficientNet-B0
        eff_color = "#d9534f" if effnet_result.predicted_class == "FAKE" else "#5cb85c"
        axes[2].imshow(effnet_overlay)
        axes[2].set_title(
            f"EfficientNet-B0 ({effnet_result.predicted_class} {effnet_result.confidence:.1f}%)",
            fontsize=12,
            fontweight="bold",
            color=eff_color,
        )
        axes[2].set_xlabel(
            f"Target: {effnet_result.target_class} | FAKE: {effnet_result.probabilities['FAKE']:.1f}% | REAL: {effnet_result.probabilities['REAL']:.1f}%\n"
            f"Layer: {effnet_result.layer_name}",
            fontsize=9,
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rendered = Image.fromarray(rgba_buffer[:, :, :3])
        plt.close(fig)
        return rendered

    def create_fair_fake_comparison(
        self,
        face_crop: Image.Image,
        cnn_overlay: Image.Image,
        effnet_overlay: Image.Image,
        cnn_fake_prob: float,
        effnet_fake_prob: float,
        title: str = "Fair Comparison: FAKE Facial Evidence Attribution",
    ) -> Image.Image:
        """
        Generate a side-by-side comparison where BOTH models explain the exact same target class: FAKE.
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
        plt.subplots_adjust(top=0.82, wspace=0.1)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        subtitle = (
            "Both models evaluated on the cropped face, explaining target class: FAKE. "
            "Warm regions highlight where each model found manipulation evidence."
        )
        fig.text(0.5, 0.88, subtitle, ha="center", fontsize=10, color="#555555")

        axes[0].imshow(face_crop)
        axes[0].set_title("Cropped Face Input", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cnn_overlay)
        axes[1].set_title(f"Custom CNN (FAKE Prob: {cnn_fake_prob:.1f}%)", fontsize=12, fontweight="bold", color="#d9534f")
        axes[1].set_xlabel("Target: FAKE | Final Conv: features.6", fontsize=9)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].imshow(effnet_overlay)
        axes[2].set_title(f"EfficientNet-B0 (FAKE Prob: {effnet_fake_prob:.1f}%)", fontsize=12, fontweight="bold", color="#337ab7")
        axes[2].set_xlabel("Target: FAKE | Final Conv: features.8", fontsize=9)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rendered = Image.fromarray(rgba_buffer[:, :, :3])
        plt.close(fig)
        return rendered

    def create_diagnostic_summary_dashboard(
        self,
        original_image: Image.Image,
        face_crop_image: Image.Image,
        background_only_image: Image.Image,
        cnn_overlay: Image.Image,
        effnet_overlay: Image.Image,
        occlusion_overlay: Optional[Image.Image],
        diag_metrics: Dict[str, Any],
    ) -> Image.Image:
        """
        Create a 6-panel comprehensive forensic diagnostic dashboard:
        Row 1: [ Original Image | Cropped Face Input | Background-Only (Face Masked) ]
        Row 2: [ Custom CNN Face Grad-CAM | EfficientNet-B0 Face Grad-CAM | Occlusion Map ]
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
        plt.subplots_adjust(top=0.88, hspace=0.25, wspace=0.12)

        fig.suptitle(
            "DeepFake Forensics: Face-Based Detection & Attribution Dashboard",
            fontsize=17,
            fontweight="bold",
            y=0.97,
        )

        cnn_bg_dep = diag_metrics.get("cnn_bg_dependency", 0.0)
        eff_bg_dep = diag_metrics.get("eff_bg_dependency", 0.0)
        banner = (
            f"Custom CNN Background Bias: {cnn_bg_dep:.1f}% "
            f"(Original FAKE: {diag_metrics.get('cnn_orig_fake', 0.0):.1f}% -> "
            f"BG-Only FAKE: {diag_metrics.get('cnn_bg_fake', 0.0):.1f}%)   |   "
            f"EfficientNet-B0 Background Bias: {eff_bg_dep:.1f}% "
            f"(Original FAKE: {diag_metrics.get('eff_orig_fake', 0.0):.1f}% -> "
            f"BG-Only FAKE: {diag_metrics.get('eff_bg_fake', 0.0):.1f}%)"
        )
        fig.text(0.5, 0.915, banner, ha="center", fontsize=11, fontweight="semibold", color="#222222")

        # Row 1, Col 1: Original
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("1. Original Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # Row 1, Col 2: Face Crop
        axes[0, 1].imshow(face_crop_image)
        axes[0, 1].set_title("2. Extracted Face Crop (Model Input)", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # Row 1, Col 3: Background-Only
        axes[0, 2].imshow(background_only_image)
        axes[0, 2].set_title("3. Background-Only (Face Masked)", fontsize=12, fontweight="bold")
        axes[0, 2].axis("off")

        # Row 2, Col 1: CNN Grad-CAM
        axes[1, 0].imshow(cnn_overlay)
        axes[1, 0].set_title("4. Custom CNN Facial Grad-CAM", fontsize=12, fontweight="bold", color="#d9534f")
        axes[1, 0].set_xlabel("Target: FAKE | Evaluated on Face Crop", fontsize=9)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

        # Row 2, Col 2: EfficientNet Grad-CAM
        axes[1, 1].imshow(effnet_overlay)
        axes[1, 1].set_title("5. EfficientNet-B0 Facial Grad-CAM", fontsize=12, fontweight="bold", color="#337ab7")
        axes[1, 1].set_xlabel("Target: FAKE | Evaluated on Face Crop", fontsize=9)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        # Row 2, Col 3: Occlusion Sensitivity Map
        if occlusion_overlay is not None:
            axes[1, 2].imshow(occlusion_overlay)
            axes[1, 2].set_title("6. Facial Occlusion Sensitivity", fontsize=12, fontweight="bold", color="#5cb85c")
            axes[1, 2].set_xlabel("Warm regions indicate features critical to classification", fontsize=9)
        else:
            axes[1, 2].text(0.5, 0.5, "Occlusion Analysis Not Run", ha="center", va="center", fontsize=12)
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])

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
        cnn_result: Optional[GradCAMResult] = None,
        cnn_overlay: Optional[Image.Image] = None,
        cnn_heatmap_img: Optional[Image.Image] = None,
        effnet_result: Optional[GradCAMResult] = None,
        effnet_overlay: Optional[Image.Image] = None,
        effnet_heatmap_img: Optional[Image.Image] = None,
        comparison_img: Optional[Image.Image] = None,
        fair_fake_img: Optional[Image.Image] = None,
        background_only_img: Optional[Image.Image] = None,
        occlusion_overlay: Optional[Image.Image] = None,
        diagnostic_dashboard: Optional[Image.Image] = None,
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

        # 5. Background only image
        if background_only_img is not None:
            bo_path = out_path / "background_only.png"
            background_only_img.save(bo_path, format="PNG")
            saved_files["background_only"] = bo_path

        # 6. Custom CNN visualizations (on evaluated face crop)
        if cnn_result is not None and cnn_heatmap_img is not None and cnn_overlay is not None:
            cnn_hm_path = out_path / "cnn_heatmap.png"
            cnn_heatmap_img.save(cnn_hm_path, format="PNG")
            saved_files["cnn_heatmap"] = cnn_hm_path

            cnn_ov_path = out_path / "cnn_overlay.png"
            cnn_overlay.save(cnn_ov_path, format="PNG")
            saved_files["cnn_overlay"] = cnn_ov_path

            cnn_sbs = self.create_side_by_side(
                evaluated_image, cnn_heatmap_img, cnn_overlay, cnn_result
            )
            cnn_sbs_path = out_path / "cnn_side_by_side.png"
            cnn_sbs.save(cnn_sbs_path, format="PNG")
            saved_files["cnn_side_by_side"] = cnn_sbs_path

        # 7. EfficientNet visualizations (on evaluated face crop)
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

        # 8. Comparative visualizations
        if comparison_img is not None:
            comp_path = out_path / "comparison.png"
            comparison_img.save(comp_path, format="PNG")
            saved_files["comparison"] = comp_path

        if fair_fake_img is not None:
            ff_path = out_path / "fair_fake_comparison.png"
            fair_fake_img.save(ff_path, format="PNG")
            saved_files["fair_fake_comparison"] = ff_path

        if occlusion_overlay is not None:
            occ_path = out_path / "occlusion_overlay.png"
            occlusion_overlay.save(occ_path, format="PNG")
            saved_files["occlusion_overlay"] = occ_path

        if diagnostic_dashboard is not None:
            dd_path = out_path / "diagnostic_dashboard.png"
            diagnostic_dashboard.save(dd_path, format="PNG")
            saved_files["diagnostic_dashboard"] = dd_path

        return saved_files
