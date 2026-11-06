"""Frozen DINOv2 ViT-B/14 features for a future image-type router.

This module does not import, construct, or call any detection model. Loading is
explicit: importing it never downloads weights or creates a model. The official
Meta/Facebook checkpoint and its preprocessing config are pinned together.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import BitImageProcessor, Dinov2Model

MODEL_ID = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_WEIGHTS_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "image_type_encoder" / ".cache"

ImageInput = Image.Image | np.ndarray | str | Path


class ImageTypeEncoder:
    """Extract global, layer-normalized CLS embeddings, with no classifier.

    Args:
        device: ``None``/``"auto"`` selects CUDA when available, otherwise CPU.
            Explicit unavailable CUDA devices raise instead of silently falling
            back to CPU.
        cache_dir: Dedicated checkpoint cache; existing detector caches are unused.
        local_files_only: Require cached files and never retry online when True.

    The backbone stays in float32, evaluation mode, with all parameters frozen.
    Features are ordinary detached CPU tensors of shape ``(N, 768)`` so they can
    be cached or used as inputs to a future trainable head. They are features,
    not class probabilities. No classification head is constructed here.
    """

    embedding_dim = 768

    def __init__(
        self,
        device: str | torch.device | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        local_files_only: bool = False,
    ) -> None:
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir)
        loading_options = {
            "revision": MODEL_REVISION,
            "cache_dir": str(self.cache_dir),
            "local_files_only": local_files_only,
        }
        # The checkpoint's published evaluation transform: RGB, shortest edge
        # 256 (bicubic), center crop 224, /255, ImageNet mean/std normalization.
        self.processor = BitImageProcessor.from_pretrained(MODEL_ID, **loading_options)
        self.model, loading_info = Dinov2Model.from_pretrained(
            MODEL_ID,
            use_safetensors=True,
            output_loading_info=True,
            **loading_options,
        )
        problems = {
            name: loading_info[name]
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(name)
        }
        if problems:
            raise RuntimeError(f"DINOv2 pretrained checkpoint did not load exactly: {problems}")
        config = self.model.config
        if (config.hidden_size, config.patch_size, config.num_hidden_layers, config.num_attention_heads) != (
            768, 14, 12, 12
        ):
            raise RuntimeError("Expected the official DINOv2 ViT-B/14 backbone.")
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.requires_grad_(False)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None or str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("ImageTypeEncoder supports CPU and CUDA devices.")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; use device='cpu' or CUDA-enabled PyTorch.")
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {resolved.index} is unavailable.")
        return resolved

    @staticmethod
    def _as_batch(images: ImageInput | Sequence[ImageInput]) -> list[ImageInput]:
        if isinstance(images, (Image.Image, np.ndarray, str, Path)):
            return [images]
        if not isinstance(images, Sequence):
            raise TypeError("Expected an image or a sequence of images.")
        batch = list(images)
        if not batch:
            raise ValueError("At least one image is required.")
        return batch

    @staticmethod
    def _as_rgb(image: ImageInput, color_order: str) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        if isinstance(image, Image.Image):
            if min(image.size) == 0:
                raise ValueError("Input image is empty.")
            return ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty.")
            if image.dtype != np.uint8:
                raise TypeError("NumPy images must be uint8 with values in [0, 255].")
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("NumPy images must have shape (H, W) or (H, W, 3).")
            if color_order == "BGR":
                image = image[:, :, ::-1].copy()
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        color_order: str = "RGB",
    ) -> torch.Tensor:
        """Return float32 CPU pixels shaped ``(N, 3, 224, 224)``.

        Accepts file paths, PIL images (including grayscale/RGBA), and uint8
        NumPy images. NumPy arrays default to RGB; pass ``color_order="BGR"``
        explicitly for OpenCV arrays. Paths and PIL images use their own modes.
        EXIF orientation is applied to PIL images and files. No face detector,
        random augmentation, or quality gate runs here.
        """
        if color_order not in {"RGB", "BGR"}:
            raise ValueError("color_order must be 'RGB' or 'BGR'.")
        batch = [self._as_rgb(image, color_order) for image in self._as_batch(images)]
        return self.processor(images=batch, return_tensors="pt")["pixel_values"].to(dtype=torch.float32)

    @torch.no_grad()
    def extract_features(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        batch_size: int = 16,
        color_order: str = "RGB",
        normalize: bool = False,
    ) -> torch.Tensor:
        """Return detached float32 CPU embeddings shaped ``(N, 768)``.

        Output order matches input order. Single images keep the batch dimension.
        ``batch_size`` bounds model/preprocessing memory; ``normalize=True``
        optionally applies L2 normalization. The default retains DINOv2's raw
        layer-normalized CLS representation. Gradients never reach the backbone.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch = self._as_batch(images)
        # Restore the frozen evaluation contract even if a caller toggled the
        # exposed backbone's mode or gradient flags between extractions.
        self.model.requires_grad_(False)
        self.model.eval()
        features = []
        for start in range(0, len(batch), batch_size):
            pixels = self.preprocess(batch[start : start + batch_size], color_order=color_order)
            outputs = self.model(pixel_values=pixels.to(self.device), return_dict=True)
            cls_features = outputs.last_hidden_state[:, 0, :]
            if normalize:
                cls_features = F.normalize(cls_features, p=2, dim=-1)
            features.append(cls_features.detach().to(device="cpu", dtype=torch.float32))
        # no_grad (rather than inference_mode) leaves these tensors usable as
        # inputs to a future head whose weights require gradients.
        return torch.cat(features, dim=0)
