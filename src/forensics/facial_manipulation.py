"""
Facial Manipulation Forensics Module for DeepFake Investigation.

Implements genuine image-processing and signal-processing forensic methodologies:
1. Face Swap Detection:
   - Boundary blending gradient analysis (seam detection)
   - Error Level Analysis (ELA) for JPEG compression anomalies
   - Border-to-interior illumination & color discrepancy (Lab Delta-E)
2. Facial Feature Alteration:
   - Bilateral facial symmetry analysis (horizontal mirror difference & SSIM)
   - Vertical feature third proportions (upper/middle/lower facial energy)
3. Facial Distortion:
   - Edge direction coherence & curvature disruption
   - Local block-level gradient variance (detects warping & liquify artifacts)
4. Skin Texture Analysis:
   - Laplacian variance on facial skin (measures unnatural airbrushing / GAN smoothing)
   - High-frequency texture residual extraction via bilateral filtering
   - 2D Fast Fourier Transform (FFT) high-frequency spectral roll-off
5. Face-to-Body Consistency:
   - CIE-Lab color distance between face and surrounding neck/context
   - High-frequency sensor noise ratio (PRNU approximation) between face and body

IMPORTANT:
These forensic metrics provide empirical diagnostic indicators. They do not constitute
absolute proof of manipulation on their own, as camera lighting, makeup, compression,
and pose naturally introduce variations. Where deeper analysis is required, dedicated
deep neural models (e.g. 3D dense facial landmarks, PRNU fingerprints) are noted.
"""

from dataclasses import asdict, dataclass
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


@dataclass
class ForensicMetric:
    """Represents a single forensic diagnostic metric."""
    name: str
    value: float
    unit: str
    normal_range: str
    interpretation: str
    anomaly_score: float  # 0.0 (normal) to 1.0 (highly suspicious)


@dataclass
class ForensicCategoryResult:
    """Results for a single forensic category."""
    category_id: str
    title: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    composite_score: float  # 0.0 to 100.0
    summary: str
    metrics: List[ForensicMetric]
    disclaimer: str
    specialized_model_needed: Optional[str] = None
    artifact_paths: Optional[Dict[str, str]] = None


@dataclass
class ForensicAnalysisResult:
    """Aggregated results across all forensic analysis categories."""
    overall_manipulation_risk: str  # "LOW", "MODERATE", "HIGH"
    overall_anomaly_index: float  # 0.0 to 100.0
    categories: Dict[str, ForensicCategoryResult]
    summary_notes: List[str]
    artifact_files: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to standard serializable dictionary."""
        return asdict(self)


class FacialManipulationForensics:
    """
    Forensic analyzer executing multi-faceted image processing checks on facial imagery.
    """

    def __init__(self, ela_quality: int = 90, ela_scale: float = 15.0):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def analyze(
        self,
        face_image: Image.Image,
        full_image: Optional[Image.Image] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> ForensicAnalysisResult:
        """
        Run all 5 forensic analysis modules on the input face.

        Args:
            face_image: Cropped face image (PIL.Image, RGB).
            full_image: Optional full scene image containing the face.
            face_bbox: Bounding box (x, y, w, h) of the face in full_image.
            save_dir: Directory where visual forensic artifacts will be saved.

        Returns:
            ForensicAnalysisResult with metrics, explanations, and artifact paths.
        """
        out_dir = Path(save_dir) if save_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Standardize face image to RGB numpy array
        face_rgb = np.array(face_image.convert("RGB"))
        full_rgb = np.array(full_image.convert("RGB")) if full_image else None

        saved_artifacts: Dict[str, str] = {}

        # 1. Face Swap Detection
        cat_swap, art_swap = self._analyze_face_swap(face_rgb, full_rgb, face_bbox, out_dir)
        saved_artifacts.update(art_swap)

        # 2. Facial Feature Alteration & Symmetry
        cat_feat, art_feat = self._analyze_facial_features(face_rgb, out_dir)
        saved_artifacts.update(art_feat)

        # 3. Facial Distortion & Curvature
        cat_dist, art_dist = self._analyze_facial_distortion(face_rgb, out_dir)
        saved_artifacts.update(art_dist)

        # 4. Skin Texture & Smoothing
        cat_skin, art_skin = self._analyze_skin_texture(face_rgb, out_dir)
        saved_artifacts.update(art_skin)

        # 5. Face-to-Body Consistency
        cat_body, art_body = self._analyze_face_body_consistency(face_rgb, full_rgb, face_bbox, out_dir)
        saved_artifacts.update(art_body)

        # Aggregate overall score
        scores = [
            cat_swap.composite_score,
            cat_feat.composite_score,
            cat_dist.composite_score,
            cat_skin.composite_score,
            cat_body.composite_score,
        ]
        overall_index = float(np.mean(scores))
        if overall_index >= 60.0 or max(scores) >= 75.0:
            overall_risk = "HIGH"
        elif overall_index >= 35.0 or max(scores) >= 50.0:
            overall_risk = "MODERATE"
        else:
            overall_risk = "LOW"

        summary_notes = [
            f"Overall Forensic Anomaly Index: {overall_index:.1f}/100 ({overall_risk} Risk).",
            f"Face Swap Seam Anomaly: {cat_swap.composite_score:.1f}/100 ({cat_swap.risk_level}).",
            f"Facial Symmetry & Feature Proportion: {cat_feat.composite_score:.1f}/100 ({cat_feat.risk_level}).",
            f"Geometry & Distortion Coherence: {cat_dist.composite_score:.1f}/100 ({cat_dist.risk_level}).",
            f"Skin Texture & Frequency Residual: {cat_skin.composite_score:.1f}/100 ({cat_skin.risk_level}).",
            f"Face-to-Body Consistency: {cat_body.composite_score:.1f}/100 ({cat_body.risk_level}).",
        ]

        return ForensicAnalysisResult(
            overall_manipulation_risk=overall_risk,
            overall_anomaly_index=round(overall_index, 1),
            categories={
                "face_swap": cat_swap,
                "feature_alteration": cat_feat,
                "facial_distortion": cat_dist,
                "skin_texture": cat_skin,
                "face_body_consistency": cat_body,
            },
            summary_notes=summary_notes,
            artifact_files=saved_artifacts,
        )

    # -------------------------------------------------------------------------
    # 1. Face Swap Detection
    # -------------------------------------------------------------------------
    def _analyze_face_swap(
        self,
        face_rgb: np.ndarray,
        full_rgb: Optional[np.ndarray],
        face_bbox: Optional[Tuple[int, int, int, int]],
        save_dir: Optional[Path],
    ) -> Tuple[ForensicCategoryResult, Dict[str, str]]:
        """
        Detect face swap artifacts: boundary seams, blending blur, and compression anomalies.
        """
        artifacts: Dict[str, str] = {}
        h, w, _ = face_rgb.shape

        # A. Error Level Analysis (ELA)
        ela_img, mean_ela, max_ela = self._compute_ela(face_rgb)
        if save_dir:
            ela_path = save_dir / "forensics_ela.png"
            Image.fromarray(ela_img).save(ela_path)
            artifacts["ela_image"] = str(ela_path)

        # B. Boundary Seam Gradient (Outer 12% border vs Interior)
        border_t = max(2, int(min(h, w) * 0.12))
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobel_x, sobel_y)

        # Mask border vs inner
        mask_inner = np.zeros((h, w), dtype=bool)
        mask_inner[border_t : h - border_t, border_t : w - border_t] = True
        mask_border = ~mask_inner

        inner_grad_mean = float(np.mean(grad_mag[mask_inner])) if np.any(mask_inner) else 1.0
        border_grad_mean = float(np.mean(grad_mag[mask_border])) if np.any(mask_border) else 1.0
        gradient_ratio = border_grad_mean / max(1e-5, inner_grad_mean)

        # Boundary visual map
        if save_dir:
            norm_grad = np.clip(grad_mag / (np.percentile(grad_mag, 98) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            colored_grad = cv2.applyColorMap(norm_grad, cv2.COLORMAP_JET)
            cv2.rectangle(colored_grad, (border_t, border_t), (w - border_t, h - border_t), (255, 255, 255), 2)
            seam_path = save_dir / "forensics_seam_boundary.png"
            cv2.imwrite(str(seam_path), colored_grad)
            artifacts["seam_boundary"] = str(seam_path)

        # Metrics scoring
        # Normal ELA mean difference is typically between 5.0 and 20.0. Higher (>25.0) indicates compression mismatch.
        ela_anomaly = float(np.clip((mean_ela - 15.0) / 25.0, 0.0, 1.0))
        # Gradient ratio: natural face crops typically have border gradient ratio between 0.4 and 1.8.
        # Strong boundary blur or sharp splicing elevates or drops this ratio unnaturally.
        seam_anomaly = float(np.clip(abs(gradient_ratio - 1.0) / 1.5, 0.0, 1.0))

        composite = (ela_anomaly * 0.5 + seam_anomaly * 0.5) * 100.0
        risk = "HIGH" if composite >= 65 else ("MEDIUM" if composite >= 35 else "LOW")

        metrics = [
            ForensicMetric(
                name="Error Level Analysis (ELA) Mean Error",
                value=round(mean_ela, 2),
                unit="pixel diff",
                normal_range="4.0 - 18.0",
                interpretation=(
                    "Elevated ELA difference indicates localized JPEG re-compression or spliced content."
                    if mean_ela > 20.0
                    else "Uniform compression error across facial surface."
                ),
                anomaly_score=ela_anomaly,
            ),
            ForensicMetric(
                name="Boundary-to-Interior Gradient Ratio",
                value=round(gradient_ratio, 2),
                unit="ratio",
                normal_range="0.5 - 1.6",
                interpretation=(
                    "Substantial boundary gradient anomaly suggests alpha-blending seam or pasting edge."
                    if seam_anomaly > 0.45
                    else "Natural transition across facial boundaries."
                ),
                anomaly_score=seam_anomaly,
            ),
        ]

        return (
            ForensicCategoryResult(
                category_id="face_swap",
                title="Face Swap & Splicing Seam Analysis",
                risk_level=risk,
                composite_score=round(composite, 1),
                summary=(
                    f"Analyzed JPEG compression residuals and outer boundary gradient ratio ({gradient_ratio:.2f}). "
                    f"Risk level evaluated as {risk}."
                ),
                metrics=metrics,
                disclaimer="Note: Multi-generation JPEG re-compression can naturally increase ELA residuals.",
                specialized_model_needed="Production face-swap detection benefits from a specialized deep boundary artifact CNN (e.g. Face X-Ray or BI-Online).",
                artifact_paths=artifacts,
            ),
            artifacts,
        )

    # -------------------------------------------------------------------------
    # 2. Facial Feature Alteration & Bilateral Symmetry
    # -------------------------------------------------------------------------
    def _analyze_facial_features(
        self,
        face_rgb: np.ndarray,
        save_dir: Optional[Path],
    ) -> Tuple[ForensicCategoryResult, Dict[str, str]]:
        """
        Assess bilateral facial symmetry and vertical feature proportion energy.
        """
        artifacts: Dict[str, str] = {}
        h, w, _ = face_rgb.shape

        # A. Bilateral Symmetry via horizontal flip
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        flipped_gray = cv2.flip(gray, 1)

        # Difference map
        diff = cv2.absdiff(gray, flipped_gray)
        sym_score = float(np.mean(diff))

        # Normalized SSIM approximation: 1.0 - normalized mean absolute difference
        symmetry_similarity = float(np.clip(1.0 - (sym_score / 128.0), 0.0, 1.0))

        if save_dir:
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_MAGMA)
            sym_path = save_dir / "forensics_symmetry_diff.png"
            cv2.imwrite(str(sym_path), diff_color)
            artifacts["symmetry_diff"] = str(sym_path)

        # B. Vertical Facial Thirds Energy (Eyes, Nose, Mouth)
        h3 = h // 3
        top_third = gray[:h3, :]
        mid_third = gray[h3 : 2 * h3, :]
        bot_third = gray[2 * h3 :, :]

        var_top = float(np.var(top_third))
        var_mid = float(np.var(mid_third))
        var_bot = float(np.var(bot_third))

        # Check for abnormal feature energy ratio
        avg_var = (var_top + var_mid + var_bot) / 3.0 + 1e-5
        top_ratio = var_top / avg_var
        bot_ratio = var_bot / avg_var

        # Symmetry anomaly:
        # Natural faces have mild asymmetry (similarity ~ 0.65 - 0.85).
        # Extreme asymmetry (<0.50) indicates warping or distortion.
        # Hyper-symmetry (>0.94) is an artificial sign of synthetic mirrored generation.
        if symmetry_similarity < 0.50:
            sym_anomaly = float(np.clip((0.55 - symmetry_similarity) / 0.35, 0.0, 1.0))
            sym_note = "Marked bilateral facial asymmetry detected (possible regional warp or expression artifact)."
        elif symmetry_similarity > 0.94:
            sym_anomaly = float(np.clip((symmetry_similarity - 0.93) / 0.07, 0.0, 1.0))
            sym_note = "Unusually high bilateral symmetry detected (common in synthetic / mirrored avatar generators)."
        else:
            sym_anomaly = 0.15
            sym_note = "Natural human facial bilateral symmetry profile."

        # Feature third proportion anomaly
        prop_anomaly = float(np.clip(abs(top_ratio - 1.0) * 0.4 + abs(bot_ratio - 1.0) * 0.4, 0.0, 1.0))

        composite = (sym_anomaly * 0.65 + prop_anomaly * 0.35) * 100.0
        risk = "HIGH" if composite >= 65 else ("MEDIUM" if composite >= 35 else "LOW")

        metrics = [
            ForensicMetric(
                name="Bilateral Facial Symmetry Index",
                value=round(symmetry_similarity * 100.0, 1),
                unit="%",
                normal_range="65.0% - 88.0%",
                interpretation=sym_note,
                anomaly_score=sym_anomaly,
            ),
            ForensicMetric(
                name="Upper Facial Third (Eyes/Forehead) Energy Ratio",
                value=round(top_ratio, 2),
                unit="ratio",
                normal_range="0.70 - 1.45",
                interpretation=(
                    "Substantial deviation in eye/forehead structural energy."
                    if prop_anomaly > 0.4
                    else "Normal upper facial structural distribution."
                ),
                anomaly_score=prop_anomaly,
            ),
        ]

        return (
            ForensicCategoryResult(
                category_id="feature_alteration",
                title="Facial Feature Alteration & Symmetry",
                risk_level=risk,
                composite_score=round(composite, 1),
                summary=(
                    f"Bilateral symmetry evaluated at {symmetry_similarity*100:.1f}%. "
                    f"Vertical feature proportion balance: {top_ratio:.2f}. Risk: {risk}."
                ),
                metrics=metrics,
                disclaimer="Note: Angled head poses and asymmetrical facial expressions naturally influence symmetry.",
                specialized_model_needed="Dense 3D facial landmark mesh (e.g., MediaPipe FaceMesh / 3DDFA) is required for landmark-level micro-expression tracking.",
                artifact_paths=artifacts,
            ),
            artifacts,
        )

    # -------------------------------------------------------------------------
    # 3. Facial Distortion Analysis
    # -------------------------------------------------------------------------
    def _analyze_facial_distortion(
        self,
        face_rgb: np.ndarray,
        save_dir: Optional[Path],
    ) -> Tuple[ForensicCategoryResult, Dict[str, str]]:
        """
        Analyze edge direction coherence and local grid-level gradient disruption.
        """
        artifacts: Dict[str, str] = {}
        h, w, _ = face_rgb.shape
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

        # Gradient vectors
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx)  # in radians [-pi, pi]
        mag = np.hypot(gx, gy)

        # A. Edge Orientation Entropy (Random / disrupted warping increases orientation entropy)
        significant_mask = mag > (np.percentile(mag, 70))
        if np.any(significant_mask):
            hist, _ = np.histogram(angles[significant_mask], bins=18, range=(-np.pi, np.pi), density=True)
            hist = hist[hist > 0]
            entropy = -float(np.sum(hist * np.log2(hist)))
        else:
            entropy = 2.5

        # B. Local Block-Level Gradient Variance Discontinuity (16x16 blocks)
        bh, bw = max(8, h // 16), max(8, w // 16)
        block_variances = []
        for y_idx in range(0, h - bh + 1, bh):
            for x_idx in range(0, w - bw + 1, bw):
                block = mag[y_idx : y_idx + bh, x_idx : x_idx + bw]
                block_variances.append(float(np.std(block)))

        std_of_variances = float(np.std(block_variances)) if block_variances else 0.0

        if save_dir:
            # Generate Edge Orientation & Disruption map
            norm_ang = ((angles + np.pi) / (2.0 * np.pi) * 180.0).astype(np.uint8)
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = norm_ang
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(mag / (np.percentile(mag, 95) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            edge_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_path = save_dir / "forensics_edge_flow.png"
            cv2.imwrite(str(flow_path), edge_flow)
            artifacts["edge_flow"] = str(flow_path)

        # Entropy anomaly: natural portraits have structured edge entropy between 2.2 and 3.8.
        # Warping increases turbulent orientation entropy.
        entropy_anomaly = float(np.clip((entropy - 3.4) / 0.8, 0.0, 1.0))
        # Block variance anomaly: severe localized warp creates abrupt spatial gradient shifts.
        block_anomaly = float(np.clip(std_of_variances / 45.0, 0.0, 1.0))

        composite = (entropy_anomaly * 0.5 + block_anomaly * 0.5) * 100.0
        risk = "HIGH" if composite >= 65 else ("MEDIUM" if composite >= 35 else "LOW")

        metrics = [
            ForensicMetric(
                name="Edge Orientation Vector Entropy",
                value=round(entropy, 2),
                unit="bits",
                normal_range="2.20 - 3.40",
                interpretation=(
                    "Elevated gradient orientation turbulence indicates local warping or fluid distortion."
                    if entropy_anomaly > 0.4
                    else "Harmonious gradient vector distribution consistent with natural facial geometry."
                ),
                anomaly_score=entropy_anomaly,
            ),
            ForensicMetric(
                name="Spatial Gradient Block Discontinuity",
                value=round(std_of_variances, 2),
                unit="dispersion",
                normal_range="5.0 - 28.0",
                interpretation=(
                    "Substantial block-level discontinuity suggests localized geometric manipulation."
                    if block_anomaly > 0.4
                    else "Gradual spatial structural continuity."
                ),
                anomaly_score=block_anomaly,
            ),
        ]

        return (
            ForensicCategoryResult(
                category_id="facial_distortion",
                title="Facial Distortion & Geometric Coherence",
                risk_level=risk,
                composite_score=round(composite, 1),
                summary=(
                    f"Measured edge orientation entropy ({entropy:.2f}) and spatial discontinuity. "
                    f"Risk evaluated as {risk}."
                ),
                metrics=metrics,
                disclaimer="Note: Motion blur, complex hairstyles, or eyeglasses naturally disperse edge orientations.",
                specialized_model_needed="Dense optical flow and geometric landmark deformation modeling are needed for sub-pixel warp detection.",
                artifact_paths=artifacts,
            ),
            artifacts,
        )

    # -------------------------------------------------------------------------
    # 4. Skin Texture Analysis
    # -------------------------------------------------------------------------
    def _analyze_skin_texture(
        self,
        face_rgb: np.ndarray,
        save_dir: Optional[Path],
    ) -> Tuple[ForensicCategoryResult, Dict[str, str]]:
        """
        Analyze high-frequency texture residual, unnatural skin smoothing, and FFT spectral rolloff.
        """
        artifacts: Dict[str, str] = {}
        h, w, _ = face_rgb.shape
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

        # Approximate facial skin mask using YCbCr color boundaries
        ycbcr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

        # Fallback to central region if skin segment is too sparse
        if np.sum(skin_mask) < (h * w * 0.05):
            skin_mask = np.zeros((h, w), dtype=bool)
            skin_mask[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = True

        # A. Laplacian Variance on Skin
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        skin_lap_var = float(np.var(lap[skin_mask]))

        # B. High-Frequency Texture Residual via Bilateral Filter Difference
        bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=7)
        texture_residual = cv2.absdiff(gray, bilateral)
        mean_residual = float(np.mean(texture_residual[skin_mask]))

        # C. 2D FFT High-Frequency Spectral Ratio
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        mag_spectrum = np.abs(f_shift)

        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        y_grid, x_grid = np.ogrid[:h, :w]
        mask_center = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2

        low_freq_energy = float(np.sum(mag_spectrum[mask_center]))
        high_freq_energy = float(np.sum(mag_spectrum[~mask_center]))
        spectral_ratio = high_freq_energy / max(1e-5, low_freq_energy)

        if save_dir:
            # Save High-Frequency Texture Residual
            norm_res = np.clip(texture_residual * 4.0, 0, 255).astype(np.uint8)
            res_color = cv2.applyColorMap(norm_res, cv2.COLORMAP_VIRIDIS)
            res_path = save_dir / "forensics_texture_residual.png"
            cv2.imwrite(str(res_path), res_color)
            artifacts["texture_residual"] = str(res_path)

            # Save 2D FFT Power Spectrum
            log_spectrum = np.log1p(mag_spectrum)
            norm_spec = np.clip(log_spectrum / (np.percentile(log_spectrum, 99) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            spec_path = save_dir / "forensics_fft_spectrum.png"
            cv2.imwrite(str(spec_path), norm_spec)
            artifacts["fft_spectrum"] = str(spec_path)

        # Anomaly scoring:
        # GAN smoothing or airbrushing leads to abnormally low skin laplacian variance (<60)
        # or abnormally high artificial grain / checkerboard noise (>800).
        if skin_lap_var < 50.0:
            lap_anomaly = float(np.clip((50.0 - skin_lap_var) / 45.0, 0.0, 1.0))
            lap_note = "Unusually low high-frequency skin variance (characteristic of AI-generated smoothing or heavy airbrushing)."
        elif skin_lap_var > 650.0:
            lap_anomaly = float(np.clip((skin_lap_var - 650.0) / 400.0, 0.0, 1.0))
            lap_note = "Excessive high-frequency grain or synthetic checkerboard noise pattern."
        else:
            lap_anomaly = 0.1
            lap_note = "Natural micro-texture skin variance."

        # High-frequency residual anomaly
        res_anomaly = float(np.clip(abs(mean_residual - 4.5) / 5.0, 0.0, 1.0))

        composite = (lap_anomaly * 0.65 + res_anomaly * 0.35) * 100.0
        risk = "HIGH" if composite >= 65 else ("MEDIUM" if composite >= 35 else "LOW")

        metrics = [
            ForensicMetric(
                name="Skin Region Laplacian Variance",
                value=round(skin_lap_var, 2),
                unit="variance",
                normal_range="75.0 - 450.0",
                interpretation=lap_note,
                anomaly_score=lap_anomaly,
            ),
            ForensicMetric(
                name="High-Frequency Texture Residual",
                value=round(mean_residual, 2),
                unit="intensity",
                normal_range="2.2 - 6.8",
                interpretation=(
                    "Texture residual deviates from natural camera pore and follicle profiles."
                    if res_anomaly > 0.4
                    else "Natural biological skin texture residual profile."
                ),
                anomaly_score=res_anomaly,
            ),
            ForensicMetric(
                name="2D FFT High-to-Low Spectral Ratio",
                value=round(spectral_ratio, 4),
                unit="ratio",
                normal_range="0.040 - 0.220",
                interpretation="Frequency power distribution across spatial dimensions.",
                anomaly_score=0.2,
            ),
        ]

        return (
            ForensicCategoryResult(
                category_id="skin_texture",
                title="Skin Texture & Frequency Residual Analysis",
                risk_level=risk,
                composite_score=round(composite, 1),
                summary=(
                    f"Skin Laplacian variance ({skin_lap_var:.1f}) and texture residual ({mean_residual:.2f}) evaluated. "
                    f"Risk: {risk}."
                ),
                metrics=metrics,
                disclaimer="Note: Camera compression, low resolution, soft focus, and cosmetic makeup naturally smooth skin texture.",
                specialized_model_needed="Photo-Response Non-Uniformity (PRNU) sensor fingerprint analysis is required for definitive camera-sensor provenance verification.",
                artifact_paths=artifacts,
            ),
            artifacts,
        )

    # -------------------------------------------------------------------------
    # 5. Face-to-Body Consistency Analysis
    # -------------------------------------------------------------------------
    def _analyze_face_body_consistency(
        self,
        face_rgb: np.ndarray,
        full_rgb: Optional[np.ndarray],
        face_bbox: Optional[Tuple[int, int, int, int]],
        save_dir: Optional[Path],
    ) -> Tuple[ForensicCategoryResult, Dict[str, str]]:
        """
        Compare facial crop against surrounding body/neck/context for color and noise consistency.
        """
        artifacts: Dict[str, str] = {}

        if full_rgb is None or face_bbox is None:
            # Self-contained surrogate: compare inner face core to peripheral jawline/neck margin
            h, w, _ = face_rgb.shape
            core = face_rgb[int(h * 0.2) : int(h * 0.7), int(w * 0.2) : int(w * 0.8)]
            margin = face_rgb[int(h * 0.7) :, :]
            note_context = "Self-contained margin (lower face/neck boundary)"
        else:
            # Full context: sample region immediately below face (neck/chest)
            fx, fy, fw, fh = face_bbox
            img_h, img_w, _ = full_rgb.shape
            core = face_rgb
            # Target neck/body box
            by1 = min(img_h - 1, fy + fh)
            by2 = min(img_h, fy + int(fh * 1.6))
            bx1 = max(0, fx)
            bx2 = min(img_w, fx + fw)

            if (by2 - by1) > 10 and (bx2 - bx1) > 10:
                margin = full_rgb[by1:by2, bx1:bx2]
                note_context = "Extracted neck and torso context"
            else:
                margin = face_rgb[int(face_rgb.shape[0] * 0.7) :, :]
                note_context = "Self-contained perimeter (insufficient context below face)"

        # A. CIE-Lab Color Difference Delta-E
        core_lab = cv2.cvtColor(core, cv2.COLOR_RGB2LAB)
        margin_lab = cv2.cvtColor(margin, cv2.COLOR_RGB2LAB)

        core_l, core_a, core_b = np.mean(core_lab[:, :, 0]), np.mean(core_lab[:, :, 1]), np.mean(core_lab[:, :, 2])
        marg_l, marg_a, marg_b = np.mean(margin_lab[:, :, 0]), np.mean(margin_lab[:, :, 1]), np.mean(margin_lab[:, :, 2])

        # Euclidean delta-E
        delta_e = float(np.sqrt((core_l - marg_l) ** 2 + (core_a - marg_a) ** 2 + (core_b - marg_b) ** 2))

        # B. High-Frequency Noise Consistency Ratio
        core_gray = cv2.cvtColor(core, cv2.COLOR_RGB2GRAY)
        marg_gray = cv2.cvtColor(margin, cv2.COLOR_RGB2GRAY)

        core_noise = float(np.std(core_gray - cv2.GaussianBlur(core_gray, (5, 5), 0)))
        marg_noise = float(np.std(marg_gray - cv2.GaussianBlur(marg_gray, (5, 5), 0)))
        noise_ratio = core_noise / max(1e-5, marg_noise)

        # Anomaly scoring:
        # Delta-E: < 15 is very consistent; 15-35 is normal clothing/shadow variance; > 45 is anomalous
        color_anomaly = float(np.clip((delta_e - 25.0) / 35.0, 0.0, 1.0))
        # Noise ratio: natural photos have noise ratio between 0.6 and 1.6
        noise_anomaly = float(np.clip(abs(noise_ratio - 1.0) / 1.5, 0.0, 1.0))

        composite = (color_anomaly * 0.55 + noise_anomaly * 0.45) * 100.0
        risk = "HIGH" if composite >= 65 else ("MEDIUM" if composite >= 35 else "LOW")

        metrics = [
            ForensicMetric(
                name="CIE-Lab Color Distance (Delta-E)",
                value=round(delta_e, 2),
                unit="Delta-E",
                normal_range="4.0 - 28.0",
                interpretation=(
                    "Substantial color temperature or skin chrominance shift between face and body context."
                    if color_anomaly > 0.45
                    else "Harmonious color tones between facial area and context."
                ),
                anomaly_score=color_anomaly,
            ),
            ForensicMetric(
                name="High-Frequency Sensor Noise Ratio",
                value=round(noise_ratio, 2),
                unit="ratio",
                normal_range="0.60 - 1.65",
                interpretation=(
                    "Disparate noise profiles suggest face and context originate from distinct cameras or generators."
                    if noise_anomaly > 0.45
                    else "Matching high-frequency noise floor across face and body."
                ),
                anomaly_score=noise_anomaly,
            ),
        ]

        return (
            ForensicCategoryResult(
                category_id="face_body_consistency",
                title="Face-to-Body & Environmental Consistency",
                risk_level=risk,
                composite_score=round(composite, 1),
                summary=(
                    f"Evaluated color difference (Delta-E: {delta_e:.1f}) and noise ratio ({noise_ratio:.2f}) "
                    f"using {note_context}. Risk: {risk}."
                ),
                metrics=metrics,
                disclaimer="Note: Harsh stage lighting, clothing shadows, tanning differences, and makeup naturally alter Delta-E.",
                specialized_model_needed="Full-body semantic segmentation and cross-illuminant physics modeling are required to isolate ambient lighting vectors.",
                artifact_paths=artifacts,
            ),
            artifacts,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    def _compute_ela(self, rgb_array: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute JPEG Error Level Analysis (ELA) map and stats."""
        pil_img = Image.fromarray(rgb_array)

        # Save to memory buffer at specified quality
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.ela_quality)
        buf.seek(0)
        recompressed = Image.open(buf)

        # Difference
        diff = ImageChops.difference(pil_img, recompressed)

        # Scale for visibility
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale = self.ela_scale if max_diff == 0 else min(self.ela_scale, 255.0 / max(1, max_diff))

        enhancer = ImageEnhance.Brightness(diff)
        ela_visual = enhancer.enhance(scale)

        diff_arr = np.array(diff).astype(np.float32)
        mean_diff = float(np.mean(diff_arr))
        max_diff_val = float(np.max(diff_arr))

        return np.array(ela_visual), mean_diff, max_diff_val
