"""
Deep Learning Face Detection and Quality Validation Module.

Implements:
1. Robust SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
   face detector using OpenCV DNN for CPU-accelerated deep learning inference.
2. Controlled face region cropping with bounded contextual margins.
3. Pre-classification face quality, usability, and visibility verification:
   - Detection confidence validation (low_face_detection_confidence)
   - Minimum resolution checks (face_too_small)
   - Laplacian variance blur detection (face_too_blurry)
   - Severe boundary truncation & artificial block occlusion checks (face_not_usable)
4. Graceful fallback to Haar Cascade if SCRFD weights are unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCRFD_WEIGHTS = PROJECT_ROOT / "models" / "scrfd_500m.onnx"

# Quality validation default parameters
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.50
DEFAULT_MIN_FACE_SIZE = 40  # pixels in width or height
DEFAULT_BLUR_THRESHOLD = 50.0  # Laplacian variance threshold
DEFAULT_CONTEXT_MARGIN = 0.15  # 15% controlled padding


class SCRFDFaceDetector:
    """
    SCRFD deep-learning face detector using OpenCV DNN.
    Executes efficiently on CPU with pre-allocated anchor grids and Non-Maximum Suppression.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        nms_threshold: float = 0.40,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.model_path = Path(model_path or DEFAULT_SCRFD_WEIGHTS)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net: Optional[cv2.dnn.Net] = None
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.anchors_cache: Dict[int, np.ndarray] = {}

        self._initialize_detector()
        self._precompute_anchors()

    def _initialize_detector(self) -> None:
        """Load the ONNX model into OpenCV DNN runtime."""
        if self.model_path.is_file():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info(f"Loaded SCRFD face detector from {self.model_path}")
                return
            except Exception as err:
                logger.warning(f"Failed to initialize SCRFD from {self.model_path}: {err}")
        else:
            logger.warning(f"SCRFD weights not found at {self.model_path}.")
        self.net = None

    def _precompute_anchors(self) -> None:
        """Precompute anchor coordinates for 640x640 input grid to optimize inference speed."""
        in_w, in_h = self.input_size
        for stride in self.strides:
            feat_h = in_h // stride
            feat_w = in_w // stride
            grid_y, grid_x = np.mgrid[:feat_h, :feat_w]
            anchor_centers = np.stack((grid_x, grid_y), axis=-1).astype(np.float32) * stride
            # Repeat for 2 anchors per spatial location
            anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=1).reshape(-1, 2)
            self.anchors_cache[stride] = anchor_centers

    @property
    def is_available(self) -> bool:
        return self.net is not None

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect human faces in an image using SCRFD.

        Returns:
            List of detected face dicts sorted by area descending:
            [{"box": (x, y, w, h), "score": float, "detector": "scrfd_500m", "truncation_ratio": float}]
        """
        if self.net is None:
            return self._detect_fallback_haar(image_bgr)

        conf_thresh = conf_threshold or self.conf_threshold
        h0, w0 = image_bgr.shape[:2]
        in_w, in_h = self.input_size

        # Aspect-ratio preserving resize and padding
        scale = min(in_w / float(w0), in_h / float(h0))
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # SCRFD standard normalization: 1/128, mean=127.5, swapRB=True (RGB)
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 128.0,
            size=(in_w, in_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        self.net.setInput(blob)
        out_names = self.net.getUnconnectedOutLayersNames()
        outputs = dict(zip(out_names, self.net.forward(out_names)))

        boxes: List[List[int]] = []
        scores: List[float] = []
        truncation_ratios: List[float] = []

        for stride in self.strides:
            score_mat = outputs[f"score_{stride}"][0]
            bbox_mat = outputs[f"bbox_{stride}"][0]
            anchors = self.anchors_cache[stride]

            pos_indices = np.where(score_mat[:, 0] >= conf_thresh)[0]
            for idx in pos_indices:
                score = float(score_mat[idx, 0])
                cx, cy = anchors[idx]
                l, t, r, b = bbox_mat[idx] * stride

                # Unclipped estimated coordinates
                raw_x1 = (cx - l) / scale
                raw_y1 = (cy - t) / scale
                raw_x2 = (cx + r) / scale
                raw_y2 = (cy + b) / scale

                # Bounded coordinates
                x1 = max(0.0, min(float(w0), raw_x1))
                y1 = max(0.0, min(float(h0), raw_y1))
                x2 = max(0.0, min(float(w0), raw_x2))
                y2 = max(0.0, min(float(h0), raw_y2))

                box_w = x2 - x1
                box_h = y2 - y1

                # Calculate boundary truncation ratio (ratio of bounded area to raw predicted area)
                expected_area = max(1.0, (raw_x2 - raw_x1) * (raw_y2 - raw_y1))
                visible_area = max(0.0, box_w * box_h)
                trunc_ratio = round(visible_area / expected_area, 4)

                if box_w > 0 and box_h > 0:
                    boxes.append([int(x1), int(y1), int(box_w), int(box_h)])
                    scores.append(score)
                    truncation_ratios.append(trunc_ratio)

        if not boxes:
            return []

        # Non-Maximum Suppression to remove overlapping detections
        nms_indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, self.nms_threshold)
        detections: List[Dict[str, Any]] = []

        if len(nms_indices) > 0:
            flat_indices = [
                i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in nms_indices
            ]
            for idx in flat_indices:
                detections.append(
                    {
                        "box": tuple(boxes[idx]),
                        "score": round(scores[idx], 4),
                        "truncation_ratio": truncation_ratios[idx],
                        "detector": "scrfd_500m",
                    }
                )

        # Sort by bounding box area descending
        detections.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)
        return detections

    def _detect_fallback_haar(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback face detector using OpenCV Haar Cascade if SCRFD weights are missing."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))
        detections = []
        for x, y, w, h in boxes:
            detections.append(
                {
                    "box": (int(x), int(y), int(w), int(h)),
                    "score": 0.80,  # Estimated score for Haar fallback
                    "truncation_ratio": 1.0,
                    "detector": "haar_cascade_fallback",
                }
            )
        detections.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)
        return detections


def validate_face_quality(
    crop_bgr: np.ndarray,
    detection_score: float,
    original_image_shape: Tuple[int, ...],
    box: Tuple[int, int, int, int],
    truncation_ratio: float = 1.0,
    min_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    min_size: int = DEFAULT_MIN_FACE_SIZE,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> Tuple[bool, Optional[str]]:
    """
    Validate whether a detected face region is sufficiently usable for deepfake classification.

    Checks:
    1. Detection confidence (low_face_detection_confidence).
    2. Minimum dimensions (face_too_small).
    3. Laplacian variance blur (face_too_blurry).
    4. Boundary truncation & degenerate illumination / block occlusion (face_not_usable).

    Returns:
        (is_usable: bool, reason_code: Optional[str])
    """
    # 1. Detection confidence check
    if detection_score < min_confidence:
        return False, "low_face_detection_confidence"

    # 2. Dimensions check
    x, y, w, h = box
    if w < min_size or h < min_size:
        return False, "face_too_small"

    if crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return False, "face_too_small"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Degenerate illumination / flat illumination check (pitch black, blown out, or flat blank image)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    if mean_val < 15.0 or mean_val > 240.0 or std_val < 12.0:
        return False, "face_not_usable"

    # 4. Truncation check
    if truncation_ratio < 0.70:
        # More than 30% of the face region is truncated outside image boundary
        return False, "face_not_usable"

    # Additional boundary truncation check against original image dimensions
    img_h, img_w = original_image_shape[:2]
    intersect_w = max(0, min(x + w, img_w) - max(x, 0))
    intersect_h = max(0, min(y + h, img_h) - max(y, 0))
    visible_ratio = (intersect_w * intersect_h) / float(w * h)
    if visible_ratio < 0.70:
        return False, "face_not_usable"

    # 5. Excessive blur check using Laplacian variance
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < blur_threshold:
        return False, "face_too_blurry"

    # Severe artificial block occlusion check.
    # Evaluates the central facial features (eyes, nose, mouth) rather than the outer margin,
    # preventing false rejections caused by natural dark hair, dark skin, collars, or studio backdrops.
    pad_x = int(w * DEFAULT_CONTEXT_MARGIN)
    pad_y = int(h * DEFAULT_CONTEXT_MARGIN)
    fx1 = max(0, x - max(0, x - pad_x))
    fy1 = max(0, y - max(0, y - pad_y))
    fx2 = min(crop_bgr.shape[1], fx1 + w)
    fy2 = min(crop_bgr.shape[0], fy1 + h)
    face_region = crop_bgr[fy1:fy2, fx1:fx2]

    if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
        eval_gray = gray
    else:
        eval_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    fh, fw = eval_gray.shape
    cy1, cy2 = int(0.20 * fh), int(0.80 * fh)
    cx1, cx2 = int(0.10 * fw), int(0.90 * fw)
    central = eval_gray[cy1:cy2, cx1:cx2]

    if central.size > 0:
        c_size = float(central.size)
        # Check for large connected solid black censor patch (>25% of central face)
        mask_black = (central < 10).astype(np.uint8)
        n_b, _, stats_b, _ = cv2.connectedComponentsWithStats(mask_black, connectivity=8)
        if n_b > 1 and (max(stats_b[1:, cv2.CC_STAT_AREA]) / c_size) >= 0.25:
            return False, "face_not_usable"

        # Check for large connected solid white censor patch (>25% of central face)
        mask_white = (central > 245).astype(np.uint8)
        n_w, _, stats_w, _ = cv2.connectedComponentsWithStats(mask_white, connectivity=8)
        if n_w > 1 and (max(stats_w[1:, cv2.CC_STAT_AREA]) / c_size) >= 0.25:
            return False, "face_not_usable"

        # Check for large uniform flat block of arbitrary color (zero local texture)
        k = 16
        ch, cw = central.shape
        if ch >= k and cw >= k:
            flat_blocks = 0
            total_blocks = 0
            for r in range(0, ch - k + 1, k // 2):
                for c in range(0, cw - k + 1, k // 2):
                    block = central[r:r + k, c:c + k]
                    total_blocks += 1
                    if float(np.std(block)) < 2.0:
                        flat_blocks += 1
            if total_blocks > 0 and (flat_blocks / total_blocks) >= 0.25:
                return False, "face_not_usable"

    return True, None


def crop_face_with_controlled_margin(
    image_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    margin_ratio: float = DEFAULT_CONTEXT_MARGIN,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop face with a controlled contextual margin around the face contours,
    avoiding excessive background capture while preserving facial periphery.

    Returns:
        (cropped_bgr, padded_coordinates (x1, y1, x2, y2))
    """
    x, y, w, h = box
    img_h, img_w = image_bgr.shape[:2]

    pad_x = int(w * margin_ratio)
    pad_y = int(h * margin_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    crop = image_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)
