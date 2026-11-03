"""
DeepFake Forensics: Face-Based Explainability Pipeline and Command-Line Interface.

Architecture:
1. Automatic Face Detection & Anatomical Bounding Box Cropping (using FaceDetector).
2. Model Input Scaling & Normalization:
   - Custom CNN: 128x128 RGB, [-1, 1] normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
   - EfficientNet-B0: 224x224 RGB, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. Multi-Model Inference: Evaluates both Custom CNN and EfficientNet-B0 on the face crop.
4. Facial Grad-CAM Attribution: Generates heatmaps focused strictly on facial anatomy.
5. Multi-Face Handling: Individually processes and saves per-face forensics if multiple faces are present.
6. No-Face Integrity Guard: Safely halts predictions when no face is found to avoid background shortcut bias.
7. Diagnostic Dashboard: Compares original scene, face crop, background-only, and occlusion sensitivity.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import GradCAM, GradCAMResult, find_target_layer
from src.explainability.occlusion import OcclusionResult, OcclusionSensitivity
from src.explainability.visualizer import GradCAMVisualizer
from src.forensics.facial_manipulation import (
    FacialManipulationForensics,
    ForensicAnalysisResult,
)
from src.utils.face_detector import FaceDetector

# Alias for backwards compatibility
FaceDetectorHelper = FaceDetector
from src.image_detection.predict import (
    CLASS_NAMES as CNN_CLASSES,
    DEFAULT_CHECKPOINT_PATH as CNN_CHECKPOINT,
    get_device as cnn_get_device,
    get_transform as get_cnn_transform,
    load_model as load_cnn_model,
)
from src.image_detection.predict_efficientnet import (
    CLASS_NAMES as EFF_CLASSES,
    DEFAULT_CHECKPOINT_PATH as EFF_CHECKPOINT,
    get_eval_transform as get_eff_transform,
    load_efficientnet_model,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"


def resolve_device(device_arg: Optional[str] = None) -> torch.device:
    """
    Determine the computation device based on user preference and hardware availability.
    """
    if device_arg is None or device_arg.lower() == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"[INFO] Execution Device: CUDA ({torch.cuda.get_device_name(0)})")
            return dev
        print("[INFO] Execution Device: CPU")
        return torch.device("cpu")

    dev_str = device_arg.lower().strip()
    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA was requested but is not available. Falling back to CPU.")
            return torch.device("cpu")
        print(f"[INFO] Execution Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device(dev_str)

    print("[INFO] Execution Device: CPU")
    return torch.device("cpu")


def create_background_only_mask(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    fill_color: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    Neutralize face region with gray, leaving background for bias diagnostic testing.
    """
    w_img, h_img = image.size
    x, y, w, h = bbox
    xc, yc = x + w // 2, y + h // 2
    rx, ry = int(w * 0.55), int(h * 0.60)

    mask = Image.new("L", (w_img, h_img), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((xc - rx, yc - ry, xc + rx, yc + ry), fill=255)

    fill_img = Image.new("RGB", (w_img, h_img), fill_color)
    return Image.composite(fill_img, image, mask)


class ExplainabilityPipeline:
    """
    End-to-end Face-Centric Grad-CAM Explainability & Diagnostic Pipeline.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "auto",
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        colormap: str = "turbo",
        alpha: float = 0.5,
    ):
        self.device = resolve_device(str(device) if isinstance(device, (str, type(None))) else None)
        self.output_dir = Path(output_dir)
        self.colormap = colormap
        self.alpha = alpha

        self.visualizer = GradCAMVisualizer(default_colormap=self.colormap, default_alpha=self.alpha)
        self.face_detector = FaceDetector()
        self.forensics = FacialManipulationForensics()

        # Model containers (lazy loaded)
        self.cnn_model: Optional[nn.Module] = None
        self.cnn_target_layer: Optional[nn.Module] = None
        self.cnn_layer_name: Optional[str] = None
        self.cnn_transform = get_cnn_transform()

        self.effnet_model: Optional[nn.Module] = None
        self.effnet_target_layer: Optional[nn.Module] = None
        self.effnet_layer_name: Optional[str] = None
        self.effnet_transform = get_eff_transform()

    def get_or_load_cnn(self, checkpoint_path: Optional[Union[str, Path]] = None) -> nn.Module:
        """Load and cache Custom CNN model and locate its terminal conv layer."""
        if self.cnn_model is None:
            self.cnn_model, _ = load_cnn_model(checkpoint_path=checkpoint_path, device=self.device)
            self.cnn_target_layer, self.cnn_layer_name = find_target_layer(self.cnn_model)
            print(f"[INFO] Custom CNN Target Layer for Grad-CAM: {self.cnn_layer_name}")
        return self.cnn_model

    def get_or_load_effnet(self, checkpoint_path: Optional[Union[str, Path]] = None) -> nn.Module:
        """Load and cache EfficientNet-B0 model and locate its terminal conv layer."""
        if self.effnet_model is None:
            self.effnet_model, _ = load_efficientnet_model(checkpoint_path=checkpoint_path, device=self.device)
            self.effnet_target_layer, self.effnet_layer_name = find_target_layer(self.effnet_model)
            print(f"[INFO] EfficientNet-B0 Target Layer for Grad-CAM: {self.effnet_layer_name}")
        return self.effnet_model

    def run_image(
        self,
        image_input: Union[str, Path, Image.Image],
        model_choice: str = "efficientnet",
        target_class: str = "predicted",
        output_subdir: Optional[Union[str, Path]] = None,
        face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        full_image: bool = False,
        face_index: Optional[int] = None,
        run_occlusion: bool = False,
        fair_compare: bool = False,
        diagnostics: bool = True,
        run_forensics: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Face-Based Detection & Explainability pipeline on an image.

        By default, detects the face, crops it automatically, and passes the face crop
        to EfficientNet-B0 (the primary model).

        Args:
            image_input: Image path or PIL Image.
            model_choice: 'efficientnet' (primary), or 'both'/'cnn' (legacy/backup).
            target_class: 'predicted', 'real', or 'fake'.
            output_subdir: Destination directory for artifacts.
            face_boxes: Optional user-supplied (x, y, w, h) bounding boxes.
            full_image: If True, bypasses face cropping and evaluates full image.
            face_index: If specified, evaluates only this face index (0, 1, ...).
            run_occlusion: Whether to run occlusion sensitivity test.
            fair_compare: Force models to explain target class 'fake'.
            diagnostics: Whether to generate diagnostic dashboard.
            run_forensics: Whether to execute facial manipulation forensics (ELA, symmetry, etc.).

        Returns:
            Dictionary containing model results, paths to saved artifacts, and metadata.
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found: {img_path}")
            stem = img_path.stem
            with Image.open(img_path) as opened:
                original_img = opened.convert("RGB")
        elif isinstance(image_input, Image.Image):
            stem = "sample_image"
            original_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        target_dir = Path(output_subdir) if output_subdir else self.output_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Face Detection Step
        if face_boxes:
            detected_boxes = list(face_boxes)
        else:
            detected_boxes = self.face_detector.detect_faces(original_img)

        # 2. Case: No Face Detected
        if len(detected_boxes) == 0:
            if not full_image:
                print("\n" + "=" * 70)
                print(" [WARNING] NO FACE DETECTED IN IMAGE")
                print("=" * 70)
                print(f" Image:       {stem} ({original_img.size[0]}x{original_img.size[1]} px)")
                print(" Status:      PREDICTION HALTED (Integrity Guard)")
                print(" Reason:      DeepFake Forensics requires a detectable face to make reliable")
                print("              forensic predictions. Background-only classification is susceptible")
                print("              to severe shortcut learning bias.")
                print(" Workaround:  To force evaluation on the full uncropped scene anyway, run with:")
                print("              --full-image")
                print("=" * 70 + "\n")

                orig_path = target_dir / "original.png"
                original_img.save(orig_path, format="PNG")
                return {
                    "status": "NO_FACE_DETECTED",
                    "faces_detected": 0,
                    "image": stem,
                    "saved_files": {"original": orig_path},
                    "cnn_result": None,
                    "effnet_result": None,
                }
            else:
                print("[WARNING] Running in --full-image mode without detected face. Results may exhibit background bias.")
                eval_image = original_img
                primary_box = (0, 0, original_img.size[0], original_img.size[1])
                original_annotated = original_img
                face_crop_img = None
        else:
            # Faces detected: select target face
            if face_index is not None and 0 <= face_index < len(detected_boxes):
                primary_box = detected_boxes[face_index]
                print(f"[INFO] Evaluating selected face {face_index + 1}/{len(detected_boxes)}: box={primary_box}")
            else:
                primary_box = detected_boxes[0]
                if len(detected_boxes) > 1:
                    print(f"[INFO] Multiple faces detected ({len(detected_boxes)}). Evaluating primary face and sub-faces.")
                else:
                    print(f"[INFO] Face detected: box={primary_box}")

            face_crop_img, actual_box = self.face_detector.crop_face(original_img, primary_box)
            original_annotated = self.face_detector.draw_face_boxes(original_img, detected_boxes)
            eval_image = face_crop_img

        # Effective target class
        effective_target = "fake" if fair_compare else target_class

        # Run model inference & Grad-CAM on the evaluated image (face crop)
        full_res = self._run_single_image(
            image=eval_image,
            model_choice=model_choice,
            target_class=effective_target,
            save_dir=target_dir,
            original_image=original_img,
            original_with_faces=original_annotated,
            face_crop_img=face_crop_img,
        )
        full_res["status"] = "SUCCESS"
        full_res["faces_detected"] = len(detected_boxes)
        full_res["primary_face_box"] = primary_box

        # 3. Facial Manipulation Forensics Analysis
        forensic_res: Optional[ForensicAnalysisResult] = None
        if run_forensics and eval_image is not None:
            print("[INFO] Executing Facial Manipulation Forensics...")
            forensic_res = self.forensics.analyze(
                face_image=eval_image,
                full_image=original_img,
                face_bbox=primary_box if len(detected_boxes) > 0 else None,
                save_dir=target_dir,
            )
            full_res["forensic_analysis"] = forensic_res
            for art_k, art_v in forensic_res.artifact_files.items():
                full_res["saved_files"][art_k] = Path(art_v)

        # 4. Fair Comparison (Both target class FAKE on the face crop)
        fair_fake_img = None
        if model_choice == "both" and (fair_compare or diagnostics):
            fair_fake_img = self._generate_fair_fake_comparison(eval_image)
            if fair_fake_img is not None:
                ff_path = target_dir / "fair_fake_comparison.png"
                fair_fake_img.save(ff_path, format="PNG")
                full_res["saved_files"]["fair_fake_comparison"] = ff_path
                full_res["fair_fake_img"] = fair_fake_img

        # 4. Occlusion Sensitivity Analysis on the face crop
        occlusion_overlay = None
        if run_occlusion:
            print("[INFO] Running Facial Occlusion Sensitivity Analysis...")
            occlusion_overlay = self._run_occlusion_test(eval_image, target_dir, model_choice)
            if occlusion_overlay is not None:
                occ_path = target_dir / "occlusion_overlay.png"
                occlusion_overlay.save(occ_path, format="PNG")
                full_res["saved_files"]["occlusion_overlay"] = occ_path
                full_res["occlusion_overlay"] = occlusion_overlay

        # 5. Diagnostic Dashboard (Comparing original vs face crop vs background-only)
        diag_metrics = {}
        if diagnostics and len(detected_boxes) > 0 and model_choice == "both" and face_crop_img is not None:
            bg_only_img = create_background_only_mask(original_img, primary_box)
            diag_metrics = self._compute_diagnostic_metrics(original_img, face_crop_img, bg_only_img)
            full_res["diagnostic_metrics"] = diag_metrics

            dashboard = self.visualizer.create_diagnostic_summary_dashboard(
                original_image=original_img,
                face_crop_image=face_crop_img,
                background_only_image=bg_only_img,
                cnn_overlay=full_res["cnn_overlay"],
                effnet_overlay=full_res["effnet_overlay"],
                occlusion_overlay=occlusion_overlay,
                diag_metrics=diag_metrics,
            )
            dash_path = target_dir / "diagnostic_dashboard.png"
            dashboard.save(dash_path, format="PNG")
            full_res["saved_files"]["diagnostic_dashboard"] = dash_path
            full_res["diagnostic_dashboard"] = dashboard

        # 6. Multi-Face Processing: If multiple faces found, analyze each separately
        face_runs = []
        if len(detected_boxes) > 1 and face_index is None:
            for idx, box in enumerate(detected_boxes):
                cropped, actual_b = self.face_detector.crop_face(original_img, box)
                face_dir = target_dir / f"face_{idx}"
                print(f"[INFO] Processing detected Face {idx + 1}/{len(detected_boxes)}: box={actual_b}")
                f_res = self._run_single_image(
                    image=cropped,
                    model_choice=model_choice,
                    target_class=effective_target,
                    save_dir=face_dir,
                    original_image=original_img,
                    original_with_faces=original_annotated,
                    face_crop_img=cropped,
                )
                f_res["face_box"] = actual_b
                f_res["face_index"] = idx
                face_runs.append(f_res)
        full_res["faces"] = face_runs

        self._print_summary_table(full_res, stem, eval_image.size, effective_target, diag_metrics, len(detected_boxes))
        return full_res

    def _run_single_image(
        self,
        image: Image.Image,
        model_choice: str,
        target_class: str,
        save_dir: Path,
        original_image: Image.Image,
        original_with_faces: Optional[Image.Image] = None,
        face_crop_img: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Internal helper to execute Grad-CAM on a single image and save artifacts."""
        save_dir.mkdir(parents=True, exist_ok=True)

        cnn_result: Optional[GradCAMResult] = None
        cnn_overlay: Optional[Image.Image] = None
        cnn_heatmap_img: Optional[Image.Image] = None

        effnet_result: Optional[GradCAMResult] = None
        effnet_overlay: Optional[Image.Image] = None
        effnet_heatmap_img: Optional[Image.Image] = None

        comparison_img: Optional[Image.Image] = None
        face_side_by_side: Optional[Image.Image] = None

        # 1. Custom CNN on evaluated image (Face Crop)
        if model_choice in ("cnn", "both"):
            self.get_or_load_cnn()
            tensor_cnn = self.cnn_transform(image).unsqueeze(0).to(self.device)
            cam_engine_cnn = GradCAM(
                model=self.cnn_model,
                target_layer=self.cnn_target_layer,
                device=self.device,
            )
            cnn_result = cam_engine_cnn.generate(
                input_tensor=tensor_cnn,
                target_class=target_class,
                target_size=image.size,
            )
            cnn_hm_rgb = self.visualizer.apply_colormap(cnn_result.heatmap)
            cnn_heatmap_img = Image.fromarray(cnn_hm_rgb)
            cnn_overlay = self.visualizer.create_overlay(image, cnn_result.heatmap)

        # 2. EfficientNet-B0 on evaluated image (Face Crop)
        if model_choice in ("efficientnet", "both"):
            self.get_or_load_effnet()
            tensor_eff = self.effnet_transform(image).unsqueeze(0).to(self.device)
            cam_engine_eff = GradCAM(
                model=self.effnet_model,
                target_layer=self.effnet_target_layer,
                device=self.device,
            )
            effnet_result = cam_engine_eff.generate(
                input_tensor=tensor_eff,
                target_class=target_class,
                target_size=image.size,
            )
            eff_hm_rgb = self.visualizer.apply_colormap(effnet_result.heatmap)
            effnet_heatmap_img = Image.fromarray(eff_hm_rgb)
            effnet_overlay = self.visualizer.create_overlay(image, effnet_result.heatmap)

        # 3. Combined comparison [Face Crop | Custom CNN | EfficientNet]
        if model_choice == "both" and cnn_overlay is not None and effnet_overlay is not None:
            comparison_img = self.visualizer.create_model_comparison(
                face_crop=image,
                cnn_overlay=cnn_overlay,
                effnet_overlay=effnet_overlay,
                cnn_result=cnn_result,
                effnet_result=effnet_result,
            )

        # 4. Face Side-by-Side [Original (Box) | Face Crop | Face Grad-CAM Overlay]
        if face_crop_img is not None and original_with_faces is not None:
            primary_res = effnet_result if effnet_result is not None else cnn_result
            primary_ov = effnet_overlay if effnet_overlay is not None else cnn_overlay
            secondary_res = cnn_result if (cnn_result is not None and effnet_result is not None) else None
            if primary_res is not None and primary_ov is not None:
                face_side_by_side = self.visualizer.create_face_side_by_side(
                    annotated_original=original_with_faces,
                    face_crop=face_crop_img,
                    overlay=primary_ov,
                    primary_result=primary_res,
                    secondary_result=secondary_res,
                )

        # 5. Save artifacts
        saved = self.visualizer.save_visualizations(
            output_dir=save_dir,
            original_image=original_image,
            evaluated_image=image,
            original_with_faces=original_with_faces,
            face_crop_img=face_crop_img,
            face_side_by_side=face_side_by_side,
            cnn_result=cnn_result,
            cnn_overlay=cnn_overlay,
            cnn_heatmap_img=cnn_heatmap_img,
            effnet_result=effnet_result,
            effnet_overlay=effnet_overlay,
            effnet_heatmap_img=effnet_heatmap_img,
            comparison_img=comparison_img,
        )

        return {
            "cnn_result": cnn_result,
            "cnn_overlay": cnn_overlay,
            "cnn_heatmap_img": cnn_heatmap_img,
            "effnet_result": effnet_result,
            "effnet_overlay": effnet_overlay,
            "effnet_heatmap_img": effnet_heatmap_img,
            "comparison_img": comparison_img,
            "face_side_by_side": face_side_by_side,
            "saved_files": saved,
            "save_dir": save_dir,
        }

    def _generate_fair_fake_comparison(self, image: Image.Image) -> Optional[Image.Image]:
        """Generate side-by-side comparison where both models explain class FAKE."""
        self.get_or_load_cnn()
        self.get_or_load_effnet()

        cam_cnn = GradCAM(self.cnn_model, self.cnn_target_layer, device=self.device)
        t_cnn = self.cnn_transform(image).unsqueeze(0).to(self.device)
        res_cnn = cam_cnn.generate(t_cnn, target_class="fake", target_size=image.size)
        cnn_overlay = self.visualizer.create_overlay(image, res_cnn.heatmap)

        cam_eff = GradCAM(self.effnet_model, self.effnet_target_layer, device=self.device)
        t_eff = self.effnet_transform(image).unsqueeze(0).to(self.device)
        res_eff = cam_eff.generate(t_eff, target_class="fake", target_size=image.size)
        eff_overlay = self.visualizer.create_overlay(image, res_eff.heatmap)

        return self.visualizer.create_fair_fake_comparison(
            face_crop=image,
            cnn_overlay=cnn_overlay,
            effnet_overlay=eff_overlay,
            cnn_fake_prob=res_cnn.probabilities.get("FAKE", 0.0),
            effnet_fake_prob=res_eff.probabilities.get("FAKE", 0.0),
        )

    def _run_occlusion_test(
        self,
        image: Image.Image,
        save_dir: Path,
        model_choice: str = "both",
    ) -> Optional[Image.Image]:
        """Execute occlusion sensitivity test and save overlays."""
        overlays = []
        if model_choice in ("cnn", "both"):
            self.get_or_load_cnn()
            occ_cnn = OcclusionSensitivity(self.cnn_model, self.cnn_transform, self.device, patch_size=32, stride=16)
            res_cnn = occ_cnn.generate(image, target_class="fake", target_size=image.size)
            ov_cnn = self.visualizer.create_overlay(image, res_cnn.heatmap)
            ov_cnn.save(save_dir / "occlusion_cnn.png")
            overlays.append(ov_cnn)

        if model_choice in ("efficientnet", "both"):
            self.get_or_load_effnet()
            occ_eff = OcclusionSensitivity(self.effnet_model, self.effnet_transform, self.device, patch_size=32, stride=16)
            res_eff = occ_eff.generate(image, target_class="fake", target_size=image.size)
            ov_eff = self.visualizer.create_overlay(image, res_eff.heatmap)
            ov_eff.save(save_dir / "occlusion_efficientnet.png")
            overlays.append(ov_eff)

        return overlays[0] if overlays else None

    def _compute_diagnostic_metrics(
        self,
        orig_img: Image.Image,
        face_crop_img: Image.Image,
        bg_only_img: Image.Image,
    ) -> Dict[str, Any]:
        """Compute comparative probabilities across original, face crop, and background-only."""
        self.get_or_load_cnn()
        self.get_or_load_effnet()

        def predict_both(pil_im):
            t1 = self.cnn_transform(pil_im).unsqueeze(0).to(self.device)
            t2 = self.effnet_transform(pil_im).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p1 = torch.softmax(self.cnn_model(t1), dim=1)[0]
                p2 = torch.softmax(self.effnet_model(t2), dim=1)[0]
            return {
                "FAKE": p1[0].item() * 100.0,
                "REAL": p1[1].item() * 100.0,
            }, {
                "FAKE": p2[0].item() * 100.0,
                "REAL": p2[1].item() * 100.0,
            }

        cnn_orig, eff_orig = predict_both(orig_img)
        cnn_crop, eff_crop = predict_both(face_crop_img)
        cnn_bg, eff_bg = predict_both(bg_only_img)

        cnn_bg_dep = (cnn_bg["FAKE"] / max(cnn_orig["FAKE"], 1e-2)) * 100.0
        eff_bg_dep = (eff_bg["FAKE"] / max(eff_orig["FAKE"], 1e-2)) * 100.0

        return {
            "cnn_orig_fake": cnn_orig["FAKE"],
            "cnn_orig_real": cnn_orig["REAL"],
            "eff_orig_fake": eff_orig["FAKE"],
            "eff_orig_real": eff_orig["REAL"],
            "cnn_crop_fake": cnn_crop["FAKE"],
            "cnn_crop_real": cnn_crop["REAL"],
            "eff_crop_fake": eff_crop["FAKE"],
            "eff_crop_real": eff_crop["REAL"],
            "cnn_bg_fake": cnn_bg["FAKE"],
            "eff_bg_fake": eff_bg["FAKE"],
            "cnn_bg_dependency": min(cnn_bg_dep, 100.0),
            "eff_bg_dependency": min(eff_bg_dep, 100.0),
        }

    def _print_summary_table(
        self,
        results: Dict[str, Any],
        name: str,
        crop_dimensions: Tuple[int, int],
        target_class: str,
        diag: Optional[Dict[str, Any]] = None,
        num_faces: int = 1,
    ) -> None:
        """Print a structured forensic report table in the console."""
        cw, ch = crop_dimensions
        print("\n" + "=" * 72)
        print(" DEEPFAKE FORENSICS: FACE-BASED DETECTION & EXPLAINABILITY REPORT")
        print("=" * 72)
        print(f" Target Image:      {name}")
        print(f" Detected Faces:    {num_faces} face(s)")
        print(f" Evaluated Region:  Cropped Face ({cw}x{ch} px)")
        print(f" Execution Device:  {self.device.type.upper()}")
        print(f" Target Class Mode: {target_class}")
        print("-" * 72)

        if results.get("cnn_result") is not None:
            c = results["cnn_result"]
            print(f" [Custom CNN - Face Crop Prediction]")
            print(f"   Prediction:      {c.predicted_class} ({c.confidence:.2f}%)")
            print(f"   Probabilities:   FAKE: {c.probabilities.get('FAKE', 0.0):.2f}% | REAL: {c.probabilities.get('REAL', 0.0):.2f}%")
            print(f"   Target Class:    {c.target_class} (logit: {c.target_score:+.3f})")
            print(f"   Target Layer:    {c.layer_name}")
            print("-" * 72)

        if results.get("effnet_result") is not None:
            e = results["effnet_result"]
            print(f" [EfficientNet-B0 - Face Crop Prediction]")
            print(f"   Prediction:      {e.predicted_class} ({e.confidence:.2f}%)")
            print(f"   Probabilities:   FAKE: {e.probabilities.get('FAKE', 0.0):.2f}% | REAL: {e.probabilities.get('REAL', 0.0):.2f}%")
            print(f"   Target Class:    {e.target_class} (logit: {e.target_score:+.3f})")
            print(f"   Target Layer:    {e.layer_name}")
            print("-" * 72)

        if diag:
            print(" Face vs Full-Scene Diagnostic Comparison:")
            print(" Condition            | Custom CNN (FAKE %) | EfficientNet-B0 (FAKE %)")
            print(" ---------------------+---------------------+-------------------------")
            print(f" 1. Face-Crop (INPUT) | {diag['cnn_crop_fake']:17.2f}%  | {diag['eff_crop_fake']:21.2f}%")
            print(f" 2. Full Scene (Orig) | {diag['cnn_orig_fake']:17.2f}%  | {diag['eff_orig_fake']:21.2f}%")
            print(f" 3. Background-Only   | {diag['cnn_bg_fake']:17.2f}%  | {diag['eff_bg_fake']:21.2f}%")
            print(" ---------------------+---------------------+-------------------------")
            print(f" Background Bias %    | {diag['cnn_bg_dependency']:17.1f}%  | {diag['eff_bg_dependency']:21.1f}%")
            print("-" * 72)

        print(" Saved Forensic Artifacts (Outputs):")
        for key, path in results["saved_files"].items():
            print(f"   - {key:<22}: {path}")

        if results.get("faces"):
            print(f"\n Multi-Face Analysis Summary ({len(results['faces'])} faces analyzed):")
            for f in results["faces"]:
                cnn_p = f["cnn_result"].predicted_class if f.get("cnn_result") else "N/A"
                eff_p = f["effnet_result"].predicted_class if f.get("effnet_result") else "N/A"
                print(f"   - Face {f['face_index'] + 1} (box {f['face_box']}): CNN -> {cnn_p} | EffNet -> {eff_p} (artifacts: {f['save_dir']})")

        print("=" * 72 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepFake-Forensics: Face-Based Detection & Grad-CAM Module.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m src.explainability.pipeline --image tests/sample_image.jpg
  python -m src.explainability.pipeline --image tests/sample_multiface.jpg
  python -m src.explainability.pipeline --image tests/sample_image.jpg --model cnn
  python -m src.explainability.pipeline --image tests/sample_image.jpg --model efficientnet
  python -m src.explainability.pipeline --image tests/sample_image.jpg --fair-compare
  python -m src.explainability.pipeline --image tests/sample_image.jpg --occlusion
  python -m src.explainability.pipeline --image tests/sample_image.jpg --full-image
""",
    )
    parser.add_argument("image_pos", nargs="?", default=None, help="Input image path.")
    parser.add_argument("--image", "-i", dest="image_flag", default=None, help="Input image path.")
    parser.add_argument("--model", "-m", dest="model", choices=["cnn", "efficientnet", "both"], default="efficientnet", help="Primary classification model (default: efficientnet).")
    parser.add_argument("--target", "-t", dest="target", choices=["predicted", "real", "fake"], default="predicted")
    parser.add_argument("--device", "-d", dest="device", default="auto")
    parser.add_argument("--output-dir", "-o", dest="output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--colormap", dest="colormap", default="turbo")
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.5)
    parser.add_argument("--full-image", dest="full_image", action="store_true", help="Bypass face cropping and evaluate entire scene.")
    parser.add_argument("--face-index", dest="face_index", type=int, default=None, help="Evaluate a specific detected face index (0, 1, ...).")
    parser.add_argument("--occlusion", dest="occlusion", action="store_true", help="Run occlusion sensitivity test on the face crop.")
    parser.add_argument("--fair-compare", dest="fair_compare", action="store_true", help="Force model(s) to explain target class FAKE.")
    parser.add_argument("--no-forensics", dest="no_forensics", action="store_true", help="Disable facial manipulation forensics analysis.")
    return parser.parse_args()


def main():
    args = parse_args()
    target_image = args.image_flag or args.image_pos

    if not target_image:
        print("[ERROR] Please provide an image to analyze. Example: --image tests/sample_image.jpg")
        sys.exit(1)

    pipeline = ExplainabilityPipeline(
        device=args.device,
        output_dir=args.output_dir,
        colormap=args.colormap,
        alpha=args.alpha,
    )

    pipeline.run_image(
        image_input=target_image,
        model_choice=args.model,
        target_class=args.target,
        full_image=args.full_image,
        face_index=args.face_index,
        run_occlusion=args.occlusion,
        fair_compare=args.fair_compare,
        diagnostics=True,
        run_forensics=not args.no_forensics,
    )


if __name__ == "__main__":
    main()
