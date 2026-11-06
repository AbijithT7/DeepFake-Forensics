"""Extract frozen 768-dimensional DINOv2 features for router training & evaluation.

Uses the existing ImageTypeEncoder without modifying or retraining DINOv2.
Saves extracted embeddings to models/image_type_router/extracted_embeddings.npz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.image_detection.image_type_encoder import ImageTypeEncoder
DATASET_DIR = REPO_ROOT / "datasets" / "image_type_router"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
OUTPUT_DIR = REPO_ROOT / "models" / "image_type_router"
EMBEDDINGS_PATH = OUTPUT_DIR / "extracted_embeddings.npz"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Total images in manifest: {len(manifest)}")

    # Distinct class labels and integer mapping
    classes = sorted(manifest["class_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[INFO] Classes ({len(classes)}): {class_to_idx}")

    # Build image absolute paths
    image_paths = [str(DATASET_DIR / rel_p) for rel_p in manifest["relative_path"]]
    labels = np.array([class_to_idx[c] for c in manifest["class_label"]], dtype=np.int64)
    splits = manifest["split"].values
    image_ids = manifest["image_id"].values

    # Initialize frozen DINOv2 encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing ImageTypeEncoder on device: {device}...")
    encoder = ImageTypeEncoder(device=device, local_files_only=True)

    batch_size = 32
    print(f"[INFO] Extracting DINOv2 768-d features for {len(image_paths)} images (batch size {batch_size})...")
    start_time = time.time()

    # Extract raw features
    features = encoder.extract_features(
        image_paths,
        batch_size=batch_size,
        normalize=False,
    ).numpy()

    elapsed = time.time() - start_time
    print(f"[INFO] Feature extraction completed in {elapsed:.2f}s ({len(image_paths)/elapsed:.1f} img/s).")
    print(f"[INFO] Features matrix shape: {features.shape}, dtype: {features.dtype}")

    # Separate by split
    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    X_train, y_train = features[train_mask], labels[train_mask]
    X_val, y_val = features[val_mask], labels[val_mask]
    X_test, y_test = features[test_mask], labels[test_mask]

    train_ids = image_ids[train_mask]
    val_ids = image_ids[val_mask]
    test_ids = image_ids[test_mask]

    print(f"[INFO] Train split: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Val split:   X={X_val.shape}, y={y_val.shape}")
    print(f"[INFO] Test split:  X={X_test.shape}, y={y_test.shape}")

    # Save to npz
    np.savez_compressed(
        EMBEDDINGS_PATH,
        X_train=X_train,
        y_train=y_train,
        train_ids=train_ids,
        X_val=X_val,
        y_val=y_val,
        val_ids=val_ids,
        X_test=X_test,
        y_test=y_test,
        test_ids=test_ids,
        class_names=np.array(classes),
        features_all=features,
        labels_all=labels,
        splits_all=splits,
        image_ids_all=image_ids,
    )
    print(f"[SUCCESS] Embeddings saved to {EMBEDDINGS_PATH} ({EMBEDDINGS_PATH.stat().st_size / 1024 / 1024:.2f} MB).")


if __name__ == "__main__":
    main()
