"""Assemble and curate a balanced, deduplicated dataset for the Image Type Router.

Target classes:
1. REAL: Authentic human portraits from FFHQ (excluding benchmark blocklist).
2. STYLEGAN_GAN: StyleGAN synthetic portraits (excluding benchmark blocklist).
3. DIFFUSION: Stable Diffusion v1.5 text2img portraits (RunwayML / Sasha).
4. FACE_SWAP: InsightFace identity swaps (BitMind FaceSwap).
5. INPAINTING: CVPR NTIRE 2022 Inpainting Benchmark on FFHQ (afromero).
6. OTHER_SYNTHETIC: Non-photorealistic / anime GAN faces (HugGAN).
7. UNKNOWN: Out-of-distribution natural scenes & objects (CIFAR-10 & project fixtures).

Strict verification:
- Checks SHA-256 and pHash against experiments/v1/blocklist.csv.
- Deduplicates within the curated dataset.
- Saves clean manifest to datasets/image_type_router/manifest.csv.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST_PATH = REPO_ROOT / "experiments" / "v1" / "blocklist.csv"
DATASET_DIR = REPO_ROOT / "datasets" / "image_type_router"
IMAGES_DIR = DATASET_DIR / "images"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

# Local source directories (configurable via environment variables)
REAL_LOCAL_DIR = Path(os.environ.get("REAL_SOURCE_DIR", "datasets/raw/real"))
STYLEGAN_LOCAL_DIR = Path(os.environ.get("STYLEGAN_SOURCE_DIR", "datasets/raw/fake"))

SAMPLES_PER_CLASS = 250
RANDOM_SEED = 42

CLASSES = [
    "REAL",
    "STYLEGAN_GAN",
    "DIFFUSION",
    "FACE_SWAP",
    "INPAINTING",
    "OTHER_SYNTHETIC",
    "UNKNOWN",
]


def compute_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def compute_phash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit perceptual hash (difference hash / dHash)."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return f"{decimal_val:016x}"


def load_blocklist() -> tuple[set[str], set[str]]:
    """Load blocked sha256 hashes and phashes from benchmark blocklist."""
    sha256_set = set()
    phash_set = set()
    if not BLOCKLIST_PATH.exists():
        print(f"[WARN] Blocklist file {BLOCKLIST_PATH} not found!")
        return sha256_set, phash_set

    with open(BLOCKLIST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "sha256" in row and row["sha256"]:
                sha256_set.add(row["sha256"].strip().lower())
            if "phash" in row and row["phash"]:
                phash_set.add(row["phash"].strip().lower())
    print(f"[INFO] Loaded blocklist: {len(sha256_set)} SHA256s, {len(phash_set)} pHashes.")
    return sha256_set, phash_set


def process_image(
    image: Image.Image,
    target_path: Path,
    blocked_sha256: set[str],
    blocked_phash: set[str],
    seen_sha256: set[str],
    seen_phash: set[str],
) -> tuple[bool, str, str, int, int, int]:
    """Validate, deduplicate, save image, and return metadata."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()

    sha = compute_sha256(img_bytes)
    phash = compute_phash(image)

    # Check blocklist
    if sha in blocked_sha256 or phash in blocked_phash:
        return False, sha, phash, 0, 0, 0

    # Check internal duplicate
    if sha in seen_sha256 or phash in seen_phash:
        return False, sha, phash, 0, 0, 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(img_bytes)

    seen_sha256.add(sha)
    seen_phash.add(phash)
    w, h = image.size
    return True, sha, phash, w, h, len(img_bytes)


def collect_real(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting REAL samples from FFHQ...")
    files = sorted(REAL_LOCAL_DIR.glob("*.jpg"))
    records = []
    class_name = "REAL"
    out_dir = IMAGES_DIR / class_name

    for fpath in files:
        if len(records) >= SAMPLES_PER_CLASS:
            break
        try:
            with Image.open(fpath) as img:
                out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
                ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
                if ok:
                    records.append({
                        "filename": out_path.name,
                        "class_label": class_name,
                        "source_dataset": "FFHQ (Real vs Fake Faces 8020)",
                        "license": "CC BY-NC-SA 4.0",
                        "sha256": sha,
                        "phash": phash,
                        "width": w,
                        "height": h,
                        "file_bytes": size,
                    })
        except Exception as e:
            print(f"[WARN] Failed to process {fpath}: {e}")

    print(f"[INFO] Collected {len(records)} REAL samples.")
    return records


def collect_stylegan(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting STYLEGAN_GAN samples...")
    files = sorted(STYLEGAN_LOCAL_DIR.glob("*.jpg"))
    records = []
    class_name = "STYLEGAN_GAN"
    out_dir = IMAGES_DIR / class_name

    for fpath in files:
        if len(records) >= SAMPLES_PER_CLASS:
            break
        try:
            with Image.open(fpath) as img:
                out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
                ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
                if ok:
                    records.append({
                        "filename": out_path.name,
                        "class_label": class_name,
                        "source_dataset": "StyleGAN (Real vs Fake Faces 8020)",
                        "license": "CC BY-NC-SA 4.0",
                        "sha256": sha,
                        "phash": phash,
                        "width": w,
                        "height": h,
                        "file_bytes": size,
                    })
        except Exception as e:
            print(f"[WARN] Failed to process {fpath}: {e}")

    print(f"[INFO] Collected {len(records)} STYLEGAN_GAN samples.")
    return records


def collect_diffusion(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting DIFFUSION samples (Stable Diffusion v1.5)...")
    class_name = "DIFFUSION"
    out_dir = IMAGES_DIR / class_name
    records = []

    parquet_files = [
        "data/artist-00000-of-00001-4d7b40d251941349.parquet",
        "data/architect-00000-of-00001-d2b1ee4d0b671e5e.parquet",
        "data/author-00000-of-00001-faffa682cbffc0e6.parquet",
    ]

    for pq_name in parquet_files:
        if len(records) >= SAMPLES_PER_CLASS:
            break
        local_pq = hf_hub_download(
            repo_id="sasha/prof_images_blip__runwayml-stable-diffusion-v1-5",
            filename=pq_name,
            repo_type="dataset",
        )
        table = pq.read_table(local_pq)
        img_col = table["images"]
        for i in range(len(img_col)):
            if len(records) >= SAMPLES_PER_CLASS:
                break
            try:
                entry = img_col[i].as_py()
                raw_bytes = entry["bytes"] if isinstance(entry, dict) else entry
                img = Image.open(io.BytesIO(raw_bytes))
                out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
                ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
                if ok:
                    records.append({
                        "filename": out_path.name,
                        "class_label": class_name,
                        "source_dataset": "RunwayML Stable Diffusion v1.5 (Sasha)",
                        "license": "CreativeML OpenRAIL-M",
                        "sha256": sha,
                        "phash": phash,
                        "width": w,
                        "height": h,
                        "file_bytes": size,
                    })
            except Exception as e:
                print(f"[WARN] Error decoding diffusion image {i}: {e}")

    print(f"[INFO] Collected {len(records)} DIFFUSION samples.")
    return records


def collect_face_swap(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting FACE_SWAP samples (BitMind FaceSwap)...")
    class_name = "FACE_SWAP"
    out_dir = IMAGES_DIR / class_name
    records = []

    local_pq = hf_hub_download(
        repo_id="bitmind/face-swap",
        filename="data/face_validation.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(local_pq)
    img_col = table["image"]

    for i in range(len(img_col)):
        if len(records) >= SAMPLES_PER_CLASS:
            break
        try:
            entry = img_col[i].as_py()
            raw_bytes = entry["bytes"] if isinstance(entry, dict) else entry
            img = Image.open(io.BytesIO(raw_bytes))
            out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
            ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
            if ok:
                records.append({
                    "filename": out_path.name,
                    "class_label": class_name,
                    "source_dataset": "BitMind FaceSwap (InsightFace CelebA-HQ)",
                    "license": "MIT",
                    "sha256": sha,
                    "phash": phash,
                    "width": w,
                    "height": h,
                    "file_bytes": size,
                })
        except Exception as e:
            print(f"[WARN] Error decoding face-swap image {i}: {e}")

    print(f"[INFO] Collected {len(records)} FACE_SWAP samples.")
    return records


def collect_inpainting(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting INPAINTING samples (NTIRE 2022 FFHQ Benchmark)...")
    class_name = "INPAINTING"
    out_dir = IMAGES_DIR / class_name
    records = []

    def download_and_read(idx: int):
        try:
            p = hf_hub_download(
                repo_id="afromero/NTIRE22_Inpainting",
                filename=f"FFHQ/test/Completion/{idx:06d}.png",
                repo_type="dataset",
            )
            with Image.open(p) as img:
                return idx, img.copy()
        except Exception as e:
            return idx, None

    indices = list(range(SAMPLES_PER_CLASS + 50))
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, img in executor.map(lambda i: download_and_read(i), indices):
            if len(records) >= SAMPLES_PER_CLASS:
                break
            if img is None:
                continue
            out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
            ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
            if ok:
                records.append({
                    "filename": out_path.name,
                    "class_label": class_name,
                    "source_dataset": "CVPR NTIRE 2022 FFHQ Inpainting Benchmark",
                    "license": "Academic Research License",
                    "sha256": sha,
                    "phash": phash,
                    "width": w,
                    "height": h,
                    "file_bytes": size,
                })

    print(f"[INFO] Collected {len(records)} INPAINTING samples.")
    return records


def collect_other_synthetic(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting OTHER_SYNTHETIC samples (HugGAN Synthetic Faces)...")
    class_name = "OTHER_SYNTHETIC"
    out_dir = IMAGES_DIR / class_name
    records = []

    def download_and_read(idx: int):
        try:
            p = hf_hub_download(
                repo_id="huggan/anime-faces",
                filename=f"images/{idx}.png",
                repo_type="dataset",
            )
            with Image.open(p) as img:
                return idx, img.copy()
        except Exception:
            return idx, None

    indices = list(range(1, SAMPLES_PER_CLASS + 60))
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, img in executor.map(lambda i: download_and_read(i), indices):
            if len(records) >= SAMPLES_PER_CLASS:
                break
            if img is None:
                continue
            out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
            ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
            if ok:
                records.append({
                    "filename": out_path.name,
                    "class_label": class_name,
                    "source_dataset": "HugGAN Anime GAN Synthetic Faces",
                    "license": "MIT",
                    "sha256": sha,
                    "phash": phash,
                    "width": w,
                    "height": h,
                    "file_bytes": size,
                })

    print(f"[INFO] Collected {len(records)} OTHER_SYNTHETIC samples.")
    return records


def collect_unknown(blocked_sha256: set[str], blocked_phash: set[str], seen_sha256: set[str], seen_phash: set[str]) -> list[dict]:
    print("[INFO] Collecting UNKNOWN samples (CIFAR-10 objects & natural fixtures)...")
    class_name = "UNKNOWN"
    out_dir = IMAGES_DIR / class_name
    records = []

    # First add project non-face fixtures
    test_dir = REPO_ROOT / "tests"
    fixture_files = [
        test_dir / "sample_landscape_noface.jpg",
        test_dir / "sample_noface.jpg",
        test_dir / "sample_scene.jpg",
    ]
    for p in fixture_files:
        if p.exists():
            with Image.open(p) as img:
                out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
                ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
                if ok:
                    records.append({
                        "filename": out_path.name,
                        "class_label": class_name,
                        "source_dataset": "Project Non-Face Test Fixtures",
                        "license": "MIT",
                        "sha256": sha,
                        "phash": phash,
                        "width": w,
                        "height": h,
                        "file_bytes": size,
                    })

    # Next fill the rest from CIFAR-10
    local_pq = hf_hub_download(
        repo_id="uoft-cs/cifar10",
        filename="plain_text/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(local_pq)
    img_col = table["img"]

    for i in range(len(img_col)):
        if len(records) >= SAMPLES_PER_CLASS:
            break
        try:
            entry = img_col[i].as_py()
            raw_bytes = entry["bytes"] if isinstance(entry, dict) else entry
            img = Image.open(io.BytesIO(raw_bytes))
            out_path = out_dir / f"{class_name.lower()}_{len(records):04d}.jpg"
            ok, sha, phash, w, h, size = process_image(img, out_path, blocked_sha256, blocked_phash, seen_sha256, seen_phash)
            if ok:
                records.append({
                    "filename": out_path.name,
                    "class_label": class_name,
                    "source_dataset": "CIFAR-10 Natural Objects (uoft-cs)",
                    "license": "MIT",
                    "sha256": sha,
                    "phash": phash,
                    "width": w,
                    "height": h,
                    "file_bytes": size,
                })
        except Exception as e:
            print(f"[WARN] Error decoding CIFAR-10 image {i}: {e}")

    print(f"[INFO] Collected {len(records)} UNKNOWN samples.")
    return records


def main():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    blocked_sha256, blocked_phash = load_blocklist()

    seen_sha256 = set()
    seen_phash = set()

    all_records = []
    all_records.extend(collect_real(blocked_sha256, blocked_phash, seen_sha256, seen_phash))
    all_records.extend(collect_stylegan(blocked_sha256, blocked_phash, seen_sha256, seen_phash))
    all_records.extend(collect_diffusion(blocked_sha256, blocked_phash, seen_sha256, seen_phash))
    all_records.extend(collect_face_swap(blocked_sha256, blocked_phash, seen_sha256, seen_phash))
    all_records.extend(collect_inpainting(blocked_sha256, blocked_phash, seen_sha256, seen_phash))
    all_records.extend(collect_other_synthetic(blocked_sha256, blocked_phash, seen_sha256, seen_phash))
    all_records.extend(collect_unknown(blocked_sha256, blocked_phash, seen_sha256, seen_phash))

    print(f"\n[INFO] Total collected images across all classes: {len(all_records)}")

    # Assign stratified Train (70%), Val (15%), Test (15%) splits
    np.random.seed(RANDOM_SEED)
    class_groups: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for r in all_records:
        class_groups[r["class_label"]].append(r)

    final_manifest = []
    for c, items in class_groups.items():
        np.random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        for i, item in enumerate(items):
            if i < n_train:
                item["split"] = "train"
            elif i < n_train + n_val:
                item["split"] = "val"
            else:
                item["split"] = "test"
            item["image_id"] = f"{item['class_label']}_{i:04d}"
            item["relative_path"] = f"images/{item['class_label']}/{item['filename']}"
            final_manifest.append(item)

    # Save manifest.csv
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "relative_path",
        "filename",
        "class_label",
        "split",
        "source_dataset",
        "license",
        "sha256",
        "phash",
        "width",
        "height",
        "file_bytes",
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in final_manifest:
            writer.writerow(r)

    print(f"[SUCCESS] Wrote manifest to {MANIFEST_PATH} with {len(final_manifest)} entries.")

    # Print summary table
    split_summary = {}
    for r in final_manifest:
        key = (r["class_label"], r["split"])
        split_summary[key] = split_summary.get(key, 0) + 1

    print("\nDataset Split Summary:")
    print(f"{'Class':<18} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 52)
    for c in CLASSES:
        tr = split_summary.get((c, "train"), 0)
        va = split_summary.get((c, "val"), 0)
        te = split_summary.get((c, "test"), 0)
        tot = tr + va + te
        print(f"{c:<18} {tr:<8} {va:<8} {te:<8} {tot:<8}")


if __name__ == "__main__":
    main()
