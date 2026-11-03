"""
Face Detection and Cropping Module for DeepFake Forensics.

Provides automatic face detection, bounding box extraction, and margin-padded cropping
to ensure forensic models evaluate facial features rather than background shortcuts.

Works offline with zero external model downloads, combining chrominance segmentation,
morphological geometry analysis, and facial gradient verification.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FaceDetector:
    """
    Robust, lightweight face detector and cropper.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.03,
        aspect_ratio_range: Tuple[float, float] = (0.65, 2.2),
        min_gradient_magnitude: float = 14.0,
        default_margin: float = 0.15,
    ):
        """
        Args:
            min_area_ratio: Minimum face area as fraction of downscaled image area.
            aspect_ratio_range: Valid (min_aspect, max_aspect) ratio (height / width).
            min_gradient_magnitude: Minimum Sobel edge gradient to distinguish faces from flat walls.
            default_margin: Proportional boundary padding added around detected faces.
        """
        self.min_area_ratio = min_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.min_gradient_magnitude = min_gradient_magnitude
        self.default_margin = default_margin

    def detect_faces(
        self,
        image: Union[Image.Image, np.ndarray],
        margin: Optional[float] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in the image.

        Args:
            image: Input image as PIL Image or uint8 RGB numpy array.
            margin: Optional margin expansion factor (defaults to self.default_margin).

        Returns:
            List of bounding boxes [(x, y, w, h), ...] sorted by area (largest face first).
            Returns an empty list if no face is detected.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        w_orig, h_orig = pil_img.size
        pad_margin = self.default_margin if margin is None else margin

        # Scale down for fast, scale-invariant processing
        target_dim = 320.0
        scale = target_dim / max(w_orig, h_orig) if max(w_orig, h_orig) > target_dim else 1.0
        sw, sh = max(int(w_orig * scale), 32), max(int(h_orig * scale), 32)
        small = pil_img.resize((sw, sh), resample=Image.Resampling.BILINEAR)

        rgb = np.array(small, dtype=np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # 1. YCbCr Chrominance Clustering Rule
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        skin_ycbcr = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)

        # 2. HSV Saturation & Value Rule
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        s = np.zeros_like(maxc)
        pos = maxc > 0
        s[pos] = (maxc[pos] - minc[pos]) / maxc[pos] * 255.0
        skin_hsv = (s >= 20) & (s <= 210) & (v >= 35)

        skin_mask = (skin_ycbcr & skin_hsv).astype(np.uint8) * 255

        # 3. Morphological Elliptical Smoothing
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # 4. Connected Components
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(skin_mask)
        min_area = (sw * sh) * self.min_area_ratio
        candidates = []

        min_aspect, max_aspect = self.aspect_ratio_range

        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                sx = stats[i, cv2.CC_STAT_LEFT]
                sy = stats[i, cv2.CC_STAT_TOP]
                swidth = stats[i, cv2.CC_STAT_WIDTH]
                sheight = stats[i, cv2.CC_STAT_HEIGHT]
                aspect = sheight / max(swidth, 1)

                if min_aspect <= aspect <= max_aspect:
                    # 5. Facial Feature & Texture Gradient Verification
                    patch_gray = (
                        0.299 * r[sy : sy + sheight, sx : sx + swidth]
                        + 0.587 * g[sy : sy + sheight, sx : sx + swidth]
                        + 0.114 * b[sy : sy + sheight, sx : sx + swidth]
                    )
                    sobelx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
                    grad_mag = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

                    # Real facial features (eyes, nose, mouth) have distinct edge gradients
                    if grad_mag >= self.min_gradient_magnitude:
                        ox = int(sx / scale)
                        oy = int(sy / scale)
                        ow = int(swidth / scale)
                        oh = int(sheight / scale)

                        # Margin padding
                        mx = int(ow * pad_margin)
                        my = int(oh * pad_margin)
                        x1 = max(0, ox - mx)
                        y1 = max(0, oy - my)
                        x2 = min(w_orig, ox + ow + mx)
                        y2 = min(h_orig, oy + oh + my)

                        candidates.append((x1, y1, x2 - x1, y2 - y1, area))

        # Sort largest area first
        candidates.sort(key=lambda item: item[4], reverse=True)
        return [(c[0], c[1], c[2], c[3]) for c in candidates]

    def crop_face(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        margin: Optional[float] = None,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Crop face region with safety bounds.

        Args:
            image: Source PIL Image.
            bbox: (x, y, w, h) bounding box.
            margin: Extra margin if not already padded.

        Returns:
            Tuple of (cropped PIL Image, actual (x, y, w, h)).
        """
        w_img, h_img = image.size
        x, y, w, h = bbox

        if margin is not None and margin > 0:
            mx = int(w * margin)
            my = int(h * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w_img, x + w + mx)
            y2 = min(h_img, y + h + my)
        else:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

        actual_w = max(x2 - x1, 1)
        actual_h = max(y2 - y1, 1)
        cropped = image.crop((x1, y1, x1 + actual_w, y1 + actual_h))
        return cropped, (x1, y1, actual_w, actual_h)

    def create_face_only_image(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        fill_color: Tuple[int, int, int] = (128, 128, 128),
    ) -> Image.Image:
        """
        Retain ONLY the facial region by masking the background with neutral gray.
        """
        w_img, h_img = image.size
        x, y, w, h = bbox
        xc, yc = x + w // 2, y + h // 2
        rx, ry = int(w * 0.55), int(h * 0.60)

        mask = Image.new("L", (w_img, h_img), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((xc - rx, yc - ry, xc + rx, yc + ry), fill=255)

        fill_img = Image.new("RGB", (w_img, h_img), fill_color)
        return Image.composite(image, fill_img, mask)

    def create_background_only_image(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        fill_color: Tuple[int, int, int] = (128, 128, 128),
    ) -> Image.Image:
        """
        Mask out the facial region with neutral gray, retaining background for bias tests.
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

    def crop_all_faces(
        self,
        image: Image.Image,
        margin: Optional[float] = None,
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Detect and crop all faces found in an image.

        Returns:
            List of (cropped_image, bounding_box) tuples.
        """
        boxes = self.detect_faces(image, margin=margin)
        crops = []
        for b in boxes:
            cropped, actual_b = self.crop_face(image, b, margin=0.0)
            crops.append((cropped, actual_b))
        return crops

    def draw_face_boxes(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
        outline_color: str = "#00ffcc",
        width: int = 3,
    ) -> Image.Image:
        """
        Draw detected face bounding boxes and indexes onto the original image.
        """
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        for idx, (x, y, w, h) in enumerate(boxes):
            draw.rectangle([(x, y), (x + w, y + h)], outline=outline_color, width=width)
            label = f"Face {idx + 1}"
            draw.rectangle([(x, max(0, y - 24)), (x + 70, y)], fill=outline_color)
            draw.text((x + 4, max(0, y - 20)), label, fill="#000000")

        return annotated
