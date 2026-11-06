"""
Automated Pretrained Model Downloader for DeepFake Forensics.

Downloads and validates all required model weights:
1. SBI EfficientNet-B4 (Google Drive FFraw.tar -> models/sbi/)
2. Community Forensics ViT (Hugging Face: buildborderless/CommunityForensics-DeepfakeDet-ViT)
3. Hybrid SRM Detector (Hugging Face: AdityaManojShinde/deepfake-detector -> deepfake_detector_phase2.pth)
4. Original ViT (Hugging Face: prithivMLmods/Deep-Fake-Detector-v2-Model)
5. DINOv2 Base Typology Encoder (Hugging Face: facebook/dinov2-base)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")


def download_sbi() -> Path:
    """Download and extract Self-Blended Images (SBI) EfficientNet-B4 checkpoint."""
    weights_dir = PROJECT_ROOT / "models" / "sbi"
    weights_dir.mkdir(parents=True, exist_ok=True)
    tar_path = weights_dir / "FFraw.tar"
    extracted_tar = weights_dir / "best.tar"

    if extracted_tar.exists() and extracted_tar.stat().st_size > 10_000_000:
        logger.info(f"SBI best.tar already available ({extracted_tar.stat().st_size / (1024*1024):.1f} MB)")
        return extracted_tar

    if not tar_path.exists() or tar_path.stat().st_size < 100_000_000:
        logger.info("Downloading SBI FFraw.tar from Google Drive...")
        file_id = "12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV"
        client = httpx.Client(follow_redirects=True, timeout=180.0)

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = client.get(url)
        form_match = re.search(r'<form id="download-form" action="([^"]+)" method="get">', resp.text)
        inputs = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)">', resp.text))

        if not form_match or not inputs:
            raise RuntimeError("Could not parse Google Drive download confirmation form.")

        action = form_match.group(1)
        with client.stream("GET", action, params=inputs) as stream_resp:
            total = int(stream_resp.headers.get("content-length", 0))
            logger.info(f"Total size: {total / (1024*1024):.2f} MB")
            downloaded = 0
            with open(tar_path, "wb") as f:
                for chunk in stream_resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (20 * 1024 * 1024) < len(chunk):
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB...")
        logger.info(f"SBI tar archive downloaded: {tar_path} ({tar_path.stat().st_size / (1024*1024):.1f} MB)")

    logger.info("Extracting SBI archive...")
    try:
        with tarfile.open(tar_path, "r") as t:
            t.extractall(path=weights_dir)
        logger.info("SBI weights extracted successfully.")
    except Exception as err:
        logger.warning(f"Tar extraction note: {err}.")
    return extracted_tar


def download_community_vit() -> None:
    """Download Community Forensics ViT from Hugging Face Hub."""
    logger.info("Downloading Community Forensics ViT (buildborderless/CommunityForensics-DeepfakeDet-ViT)...")
    cache_dir = PROJECT_ROOT / "models" / "community_forensics"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor, ViTForImageClassification
    repo_id = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    AutoImageProcessor.from_pretrained(repo_id, cache_dir=str(cache_dir))
    model = ViTForImageClassification.from_pretrained(repo_id, cache_dir=str(cache_dir))
    logger.info(f"Community Forensics ViT ready (num_labels={model.config.num_labels})")


def download_hybrid_srm() -> Path:
    """Download Hybrid SRM detector (Phase 2 weights) from Hugging Face Hub."""
    target_dir = PROJECT_ROOT / "models" / "hybrid_detector"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "deepfake_detector_phase2.pth"

    if target_file.exists() and target_file.stat().st_size > 100_000_000:
        logger.info(f"Hybrid SRM weights already exist ({target_file.stat().st_size / (1024*1024):.1f} MB)")
        return target_file

    logger.info("Downloading Hybrid SRM detector weights from Hugging Face Hub...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="AdityaManojShinde/deepfake-detector",
            filename="deepfake_detector_phase2.pth",
            local_dir=str(target_dir),
        )
        logger.info(f"Hybrid SRM downloaded successfully to: {downloaded}")
        return Path(downloaded)
    except Exception as exc:
        logger.error(f"Failed to download Hybrid SRM via huggingface_hub: {exc}")
        raise


def download_original_vit() -> None:
    """Download secondary ViT detector weights."""
    logger.info("Downloading Original ViT (prithivMLmods/Deep-Fake-Detector-v2-Model)...")
    cache_dir = PROJECT_ROOT / "models" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor, ViTForImageClassification
    repo_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    AutoImageProcessor.from_pretrained(repo_id, cache_dir=str(cache_dir))
    model = ViTForImageClassification.from_pretrained(repo_id, cache_dir=str(cache_dir))
    logger.info(f"Original ViT ready (num_labels={model.config.num_labels})")


def download_dinov2() -> None:
    """Download DINOv2 Base backbone weights for typology router feature extraction."""
    logger.info("Downloading DINOv2 Base backbone (facebook/dinov2-base)...")
    cache_dir = PROJECT_ROOT / "models" / "image_type_encoder" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor, AutoModel
    repo_id = "facebook/dinov2-base"
    AutoImageProcessor.from_pretrained(repo_id, cache_dir=str(cache_dir))
    AutoModel.from_pretrained(repo_id, cache_dir=str(cache_dir))
    logger.info("DINOv2 Base backbone ready.")


def download_all():
    logger.info("=" * 70)
    logger.info("DOWNLOADING AND VALIDATING ALL PRETRAINED FORENSIC CHECKPOINTS")
    logger.info("=" * 70)
    download_sbi()
    download_community_vit()
    download_hybrid_srm()
    download_original_vit()
    download_dinov2()
    logger.info("All specialist models and router backbone downloaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake Forensics Pretrained Weights Downloader")
    parser.add_argument("--all", action="store_true", help="Download all models (default)")
    parser.add_argument("--model", choices=["sbi", "community_vit", "hybrid_srm", "vit", "dinov2"], help="Download a specific model")
    args = parser.parse_args()

    if args.model == "sbi":
        download_sbi()
    elif args.model == "community_vit":
        download_community_vit()
    elif args.model == "hybrid_srm":
        download_hybrid_srm()
    elif args.model == "vit":
        download_original_vit()
    elif args.model == "dinov2":
        download_dinov2()
    else:
        download_all()
