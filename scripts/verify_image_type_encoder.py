"""Run real DINOv2 inference on existing local images; never train or fetch data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Bypass image_detection/__init__.py's existing eager detector imports. This
# smoke test must work with only the independent encoder's dependencies.
spec = importlib.util.spec_from_file_location(
    "standalone_image_type_encoder", PROJECT_ROOT / "src" / "image_detection" / "image_type_encoder.py"
)
encoder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder_module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="*", help="Existing local images only.")
    parser.add_argument("--download", action="store_true", help="Allow the official model/config download if missing.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless real CUDA inference is possible.")
    parser.add_argument("--cache-dir", type=Path, default=encoder_module.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "diagnostics" / "dinov2")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda requires a CUDA-enabled PyTorch installation and an available GPU.")
    paths = args.images or [PROJECT_ROOT / "tests" / name for name in (
        "sample_image_real.jpg", "sample_image_fake.jpg", "real.jpg", "face-merge-simple-3.jpeg"
    )]
    for path in paths:
        if not path.is_file():
            parser.error(f"Local sample image does not exist: {path}")
    devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
    report = {
        "model_id": encoder_module.MODEL_ID,
        "revision": encoder_module.MODEL_REVISION,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "samples": [str(path.resolve()) for path in paths],
        "devices": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_features = None
    for device in devices:
        print(f"Loading frozen official DINOv2 on {device}...", flush=True)
        encoder = encoder_module.ImageTypeEncoder(
            device=device, cache_dir=args.cache_dir, local_files_only=not args.download
        )
        start = time.perf_counter()
        features = encoder.extract_features(paths, batch_size=2)
        elapsed = time.perf_counter() - start
        repeated = encoder.extract_features(paths, batch_size=1)
        torch.testing.assert_close(features, repeated, rtol=1e-4, atol=1e-4)
        assert features.shape == (len(paths), 768)
        assert features.dtype == torch.float32 and features.device.type == "cpu"
        assert torch.isfinite(features).all() and not features.requires_grad
        assert not torch.is_inference(features)
        assert not encoder.model.training
        assert all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
        unit_features = encoder.extract_features(paths, batch_size=2, normalize=True)
        torch.testing.assert_close(unit_features.norm(dim=1), torch.ones(len(paths)))
        if device == "cpu":
            cpu_features = features
        else:
            torch.testing.assert_close(features, cpu_features, rtol=1e-3, atol=1e-3)
            report["cpu_cuda_max_absolute_difference"] = (features - cpu_features).abs().max().item()
            report["cpu_cuda_min_cosine_similarity"] = torch.nn.functional.cosine_similarity(features, cpu_features).min().item()
        np.save(args.output_dir / f"embeddings_{device.replace(':', '_')}.npy", features.numpy())
        report["devices"][device] = {
            "shape": list(features.shape), "dtype": str(features.dtype),
            "backbone_device": str(next(encoder.model.parameters()).device),
            "parameters": sum(p.numel() for p in encoder.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in encoder.model.parameters() if p.requires_grad),
            "first_batch_seconds": elapsed,
            "repeat_batch_max_absolute_difference": (features - repeated).abs().max().item(),
            "all_checks_passed": True,
        }
        del encoder
    checkpoint = Path(hf_hub_download(
        repo_id=encoder_module.MODEL_ID, filename="model.safetensors", revision=encoder_module.MODEL_REVISION,
        cache_dir=str(args.cache_dir), local_files_only=True,
    ))
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == encoder_module.MODEL_WEIGHTS_SHA256, "Checkpoint SHA-256 does not match the official revision."
    report["checkpoint"] = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": digest}
    assert not any(name.startswith("src.image_detection") for name in sys.modules), "Detector package was imported."
    report["detection_models_imported"] = False
    report_path = args.output_dir / "verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Verification passed; report: {report_path}")


if __name__ == "__main__":
    main()
