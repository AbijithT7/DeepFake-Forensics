# Datasets Directory

This directory manages all datasets used for training, evaluating, and benchmarking the **DeepFake Forensics** detection models.

## Directory Structure

```text
datasets/
├── raw/          # Original, unmodified datasets (e.g., FaceForensics++, Celeb-DF, DFDC)
├── processed/    # Preprocessed faces, extracted frames, metadata CSVs, and normalized data
└── README.md     # Dataset overview and setup guidelines
```

## Guidelines

- **Raw Datasets (`raw/`)**: Place original archives and uncompressed raw video/image collections here. Never edit or alter raw files directly.
- **Processed Datasets (`processed/`)**: Contains extracted facial crops, pre-filtered frames, augmented samples, and pre-computed train/val/test splits.
- **Storage & Git Policy**: Large data files (`.mp4`, `.zip`, `.tar`, `.jpg`, `.png`, etc.) are ignored by `.gitignore` and must **not** be committed to version control.
