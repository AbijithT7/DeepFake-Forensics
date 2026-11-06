"""Offline contract tests; no pretrained weights or datasets are downloaded.

Run the companion verification script for real pretrained CPU/CUDA inference.
Load this file's subject directly to avoid the detection package's eager imports.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from transformers import BitImageProcessor

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "image_detection" / "image_type_encoder.py"
SPEC = importlib.util.spec_from_file_location("isolated_image_type_encoder", MODULE_PATH)
encoder_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(encoder_module)
ImageTypeEncoder = encoder_module.ImageTypeEncoder


class _Backbone(torch.nn.Module):
    """Small test double that exposes mode/gradient/device mistakes."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-1, 1, 768))
        self.config = SimpleNamespace(hidden_size=768, patch_size=14, num_hidden_layers=12, num_attention_heads=12)
        self.observations = []

    def forward(self, pixel_values, return_dict=True):
        self.observations.append((self.training, torch.is_grad_enabled(), pixel_values.device.type))
        values = pixel_values.mean(dim=(1, 2, 3))[:, None] + self.weight[None, :]
        return SimpleNamespace(last_hidden_state=torch.stack((values, values + 10), dim=1))


class TestImageTypeEncoder(unittest.TestCase):
    def setUp(self):
        self.backbone = _Backbone()
        processor = BitImageProcessor(
            size={"shortest_edge": 256}, crop_size={"height": 224, "width": 224},
            resample=Image.Resampling.BICUBIC,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        )
        processor_patch = patch.object(encoder_module.BitImageProcessor, "from_pretrained", return_value=processor)
        model_patch = patch.object(encoder_module.Dinov2Model, "from_pretrained", return_value=(self.backbone, {}))
        self.processor_loader = processor_patch.start()
        self.model_loader = model_patch.start()
        self.addCleanup(processor_patch.stop)
        self.addCleanup(model_patch.stop)
        self.encoder = ImageTypeEncoder(device="cpu", local_files_only=True)
        self.red = Image.new("RGB", (400, 200), (255, 0, 0))

    def test_pinned_official_backbone_and_offline_loading(self):
        for loader in (self.processor_loader, self.model_loader):
            self.assertEqual(loader.call_args.args[0], "facebook/dinov2-base")
            self.assertEqual(loader.call_args.kwargs["revision"], encoder_module.MODEL_REVISION)
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertTrue(self.model_loader.call_args.kwargs["use_safetensors"])
        self.assertFalse(self.backbone.training)
        self.assertFalse(any(p.requires_grad for p in self.backbone.parameters()))

    def test_partial_checkpoint_is_rejected(self):
        self.model_loader.return_value = (self.backbone, {"missing_keys": ["weight"]})
        with self.assertRaisesRegex(RuntimeError, "did not load exactly"):
            ImageTypeEncoder(device="cpu", local_files_only=True)

    def test_offline_failure_never_retries_online(self):
        self.model_loader.reset_mock()
        self.model_loader.side_effect = OSError("not cached")
        with self.assertRaisesRegex(OSError, "not cached"):
            ImageTypeEncoder(device="cpu", local_files_only=True)
        self.assertEqual(self.model_loader.call_count, 1)
        self.assertTrue(self.model_loader.call_args.kwargs["local_files_only"])

    def test_imagenet_preprocessing(self):
        pixels = self.encoder.preprocess(self.red)
        self.assertEqual(pixels.shape, (1, 3, 224, 224))
        self.assertEqual(pixels.dtype, torch.float32)
        expected = torch.tensor([(1 - .485) / .229, -.456 / .224, -.406 / .225])
        torch.testing.assert_close(pixels[0, :, 100, 100], expected)

    def test_rgb_bgr_and_paths_agree(self):
        rgb = np.asarray(self.red)
        expected = self.encoder.preprocess(self.red)
        torch.testing.assert_close(self.encoder.preprocess(rgb), expected)
        torch.testing.assert_close(self.encoder.preprocess(rgb[:, :, ::-1], color_order="BGR"), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "face.png"
            self.red.save(path)
            torch.testing.assert_close(self.encoder.preprocess(path), expected)

    def test_grayscale_alpha_and_exif_orientation(self):
        for image in (self.red.convert("L"), self.red.convert("RGBA"), np.zeros((90, 80), dtype=np.uint8)):
            self.assertEqual(self.encoder.preprocess(image).shape, (1, 3, 224, 224))
        oriented = Image.fromarray(np.arange(240 * 400 * 3, dtype=np.uint8).reshape(240, 400, 3))
        oriented.getexif()[274] = 6
        expected = oriented.transpose(Image.Transpose.ROTATE_270)
        expected.getexif().clear()
        torch.testing.assert_close(self.encoder.preprocess(oriented), self.encoder.preprocess(expected))

    def test_embeddings_preserve_order_and_batch_consistency(self):
        images = [self.red, Image.new("RGB", (250, 400), (0, 0, 255)), self.red]
        features = self.encoder.extract_features(images, batch_size=2)
        singles = torch.cat([self.encoder.extract_features(image) for image in images])
        self.assertEqual(features.shape, (3, 768))
        torch.testing.assert_close(features, singles)
        self.assertFalse(torch.equal(features[0], features[1]))
        self.assertEqual(features.device.type, "cpu")
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())
        unit_features = self.encoder.extract_features(images, normalize=True)
        torch.testing.assert_close(unit_features.norm(dim=1), torch.ones(3))

    def test_frozen_eval_and_future_autograd_compatibility(self):
        self.backbone.train().requires_grad_(True)
        features = self.encoder.extract_features(self.red)
        self.assertFalse(features.requires_grad)
        self.assertFalse(torch.is_inference(features))
        self.assertEqual(self.backbone.observations, [(False, False, "cpu")])
        self.assertFalse(any(p.requires_grad for p in self.backbone.parameters()))
        # Check tensor compatibility only: no classification head or optimizer.
        scale = torch.tensor(1.0, requires_grad=True)
        (features * scale).sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))

    def test_invalid_inputs(self):
        for images, exception in (
            ([], ValueError), (np.array([], dtype=np.uint8), ValueError),
            (np.zeros((20, 20, 3), dtype=np.float32), TypeError),
            (np.zeros((20, 20, 4), dtype=np.uint8), ValueError), (42, TypeError),
        ):
            with self.subTest(images=type(images).__name__, exception=exception):
                with self.assertRaises(exception):
                    self.encoder.extract_features(images)
        for batch_size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.encoder.extract_features(self.red, batch_size=batch_size)
        with self.assertRaises(ValueError):
            self.encoder.preprocess(self.red, color_order="guess")
        with self.assertRaises(FileNotFoundError):
            self.encoder.preprocess(Path("missing-dinov2-test-image.png"))

    def test_cpu_fallback_and_explicit_cuda_errors(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(ImageTypeEncoder._resolve_device(None), torch.device("cpu"))
            self.assertEqual(ImageTypeEncoder._resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                ImageTypeEncoder(device="cuda")
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=1):
            self.assertEqual(ImageTypeEncoder._resolve_device(None), torch.device("cuda"))
            self.assertEqual(ImageTypeEncoder._resolve_device("cuda:0"), torch.device("cuda:0"))
            with self.assertRaises(ValueError):
                ImageTypeEncoder._resolve_device("cuda:1")
        with self.assertRaises(ValueError):
            ImageTypeEncoder._resolve_device("meta")


if __name__ == "__main__":
    unittest.main()
