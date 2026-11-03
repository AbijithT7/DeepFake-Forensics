# Local Dataset Audit: Real vs Fake Faces (10K / 40K Multi-Split)

**Audit Date**: November 3, 2026  
**Dataset Location**: `Downloads/Real vs Fake Faces/Real vs Fake Faces/`  
**Dataset Provenance**: Sourced from the **RVF10K (Real vs Fake Faces - 10K)** benchmark repository, restructured into four standardized train/valid/test partition splits.  
**Auditor**: Antigravity AI Forensic Audit Suite  
**Status**: Completed — Verified 100% Uncorrupted, Balanced, and Cryptographically Audited.

---

## Executive Summary

A comprehensive diagnostic audit was conducted on the 40,000-file **Real vs Fake Faces** dataset located in the local `Downloads` directory. 

### Key Findings
1. **Underlying Sample Size**: The dataset does **not** contain 40,000 distinct images. It contains exactly **10,000 unique images** (5,000 Real and 5,000 Fake) duplicated across four alternative experimental split configurations: `6040`, `7030`, `7525`, and `8020`.
2. **File Integrity**: Zero corrupted or unreadable files were detected across all 40,000 image checks (`corrupted_count = 0`). All images are valid JPEG files in RGB color mode with uniform $256 \times 256$ dimensions.
3. **No Shortcut Bias (Real vs Fake)**: Unlike the previous Kaggle dataset (`prithivsakthiur/deepfake-vs-real-60k`) which had severe format asymmetry (Real=PNG vs Fake=JPEG), both Real and Fake classes in this dataset share identical formats (100% JPEG), identical dimensions ($256 \times 256$), and virtually identical file size distributions (Real mean: 28.89 KB; Fake mean: 28.36 KB).
4. **Intra-Split Cleanliness**: Within each individual configuration, there is **zero data leakage** between `train`, `valid`, and `test` splits, and zero overlap between `real` and `fake` classes.
5. **Severe Cross-Configuration Leakage Danger**: The splits across the four configurations are nested and overlapping. Training on one configuration (e.g., `8020/train`) and testing on another (e.g., `6040/test`) leaks up to 1,000 test images (50% of the test set) into the training data.
6. **Recommended Configuration**: **`8020` is the safest and most effective configuration** for training. It maximizes training volume (8,000 images, 33% more than 6040) while providing statistically robust, balanced validation (1,000) and test (1,000) sets.

---

## 1. Exact Image Counts Breakdown

Every configuration contains exactly 10,000 images with a strict 50:50 class balance (5,000 Real and 5,000 Fake) distributed across `train`, `valid`, and `test` subfolders:

| Configuration | Split | Real Images | Fake Images | Split Total | Split % | Config Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **6040** | `train` | 3,000 | 3,000 | 6,000 | 60.0% | |
| | `valid` | 1,000 | 1,000 | 2,000 | 20.0% | |
| | `test` | 1,000 | 1,000 | 2,000 | 20.0% | **10,000** |
| **7030** | `train` | 3,500 | 3,500 | 7,000 | 70.0% | |
| | `valid` | 750 | 750 | 1,500 | 15.0% | |
| | `test` | 750 | 750 | 1,500 | 15.0% | **10,000** |
| **7525** | `train` | 3,750 | 3,750 | 7,500 | 75.0% | |
| | `valid` | 625 | 625 | 1,250 | 12.5% | |
| | `test` | 625 | 625 | 1,250 | 12.5% | **10,000** |
| **8020** | `train` | 4,000 | 4,000 | 8,000 | 80.0% | |
| | `valid` | 500 | 500 | 1,000 | 10.0% | |
| | `test` | 500 | 500 | 1,000 | 10.0% | **10,000** |
| **Grand Total**| | **20,000** | **20,000** | **40,000** | | **40,000** |

*Verification*: The naming convention `6040`, `7030`, `7525`, and `8020` corresponds to the ratio of **Training Data : (Validation + Test) Data**. For example, in `6040`, 60% of images are allocated to `train` (6,000), while the remaining 40% are evenly split between `valid` (2,000) and `test` (2,000).

---

## 2. Cryptographic Hash Analysis & Redundancy

A full SHA-256 cryptographic hash was computed for each of the 40,000 files:

- **Total files scanned**: 40,000
- **Total unique SHA-256 hashes**: **10,000**
- **Unique hashes per configuration**:
  - `6040`: 10,000 unique hashes
  - `7030`: 10,000 unique hashes
  - `7525`: 10,000 unique hashes
  - `8020`: 10,000 unique hashes

### Pairwise Configuration Hash Intersection Matrix
$$\begin{array}{c|cccc}
\cap & \mathbf{6040} & \mathbf{7030} & \mathbf{7525} & \mathbf{8020} \\
\hline
\mathbf{6040} & 10,000 & 10,000 & 10,000 & 10,000 \\
\mathbf{7030} & 10,000 & 10,000 & 10,000 & 10,000 \\
\mathbf{7525} & 10,000 & 10,000 & 10,000 & 10,000 \\
\mathbf{8020} & 10,000 & 10,000 & 10,000 & 10,000 \\
\end{array}$$

### Filename-to-Content Mapping
- Number of filenames mapping to multiple hashes: **0**
- Number of hashes mapping to multiple filenames: **0**
- Every image has a globally unique, consistent filename across all configurations:
  - Real images use 5-digit zero-padded numbers: `00000.jpg`, `00003.jpg`, `00009.jpg`, ...
  - Fake images use 10-character alphanumeric strings: `002KDWZBHU.jpg`, `009X14ED19.jpg`, ...

**Conclusion**: The four folders `6040`, `7030`, `7525`, and `8020` represent **4 alternative partition schemes of the exact same 10,000 underlying images**.

---

## 3. Split Nesting & Data Leakage Assessment

### 3.1 Intra-Configuration Leakage (Within Splits)
Evaluating file hashes within each individual configuration revealed:
- `train` $\cap$ `valid` overlap: **0 images**
- `train` $\cap$ `test` overlap: **0 images**
- `valid` $\cap$ `test` overlap: **0 images**
- `real` $\cap$ `fake` overlap: **0 images**

Each individual split configuration (`6040`, `7030`, `7525`, or `8020`) is internally clean and isolated.

### 3.2 Inter-Configuration Split Relationships (Subset Analysis)
Empirical set operations on file paths demonstrated that the training sets are strictly nested subsets:

$$\text{Train}_{6040} \subset \text{Train}_{7030} \subset \text{Train}_{7525} \subset \text{Train}_{8020}$$

- `Train_6040` (6,000 images) is completely contained within `Train_7030` (7,000 images).
- `Train_7030` (7,000 images) is completely contained within `Train_7525` (7,500 images).
- `Train_7525` (7,500 images) is completely contained within `Train_8020` (8,000 images).

Conversely, the test sets are partially nested:
$$\text{Test}_{8020} \subset \text{Test}_{7525} \subset \text{Test}_{7030}$$
- `Test_8020` (1,000 images) is a strict subset of `Test_7525` (1,250 images).
- `Test_7525` (1,250 images) is a strict subset of `Test_7030` (1,500 images).

### 3.3 Critical Hazard: Cross-Configuration Data Leakage
Because images were reassigned across splits to vary partition ratios, **combining or cross-evaluating across configurations introduces catastrophic data leakage**:

```
[CRITICAL LEAKAGE RISKS IF CONFIGURATIONS ARE MIXED]
- Training on 8020 & Testing on 6040 -> 1,000 test images leaked into train! (50.0% of test set)
- Training on 8020 & Testing on 7030 -> 1,000 valid images leaked into train!
- Training on 7525 & Testing on 6040 -> 1,000 test images leaked into train!
- Training on 7030 & Testing on 6040 -> 1,000 test images leaked into train!
```

> [!CAUTION]
> **NEVER MERGE FOLDERS ACROSS CONFIGURATIONS**: Merging `6040`, `7030`, `7525`, and `8020` into a single folder or combining their training sets will pollute train, validation, and test splits with identical images, completely invalidating model evaluation.

---

## 4. Image Quality, Corruptions & Format Diagnostics

Every image was inspected using Pillow (`PIL.Image.open` and `verify()`):

- **Corrupted / Unreadable Files**: **0** (100% readable)
- **Unsupported Formats**: **0** (100% JPEG)
- **Color Mode**: **100% RGB** (no grayscale `L` or alpha-channel `RGBA` anomalies)

### Class Distribution & Statistical Comparison

| Metric | Real Class (20,000 files / 5,000 unique) | Fake Class (20,000 files / 5,000 unique) | Analysis |
| :--- | :--- | :--- | :--- |
| **Image Resolution** | **$256 \times 256$** (Std: 0.0) | **$256 \times 256$** (Std: 0.0) | Identical resolution across all images |
| **Image Format** | 100% JPEG | 100% JPEG | No format shortcut |
| **Color Channels** | 3-Channel RGB | 3-Channel RGB | Identical |
| **File Size (Min)** | 16,492 bytes (~16.1 KB) | 16,887 bytes (~16.5 KB) | Comparable |
| **File Size (Max)** | 46,804 bytes (~45.7 KB) | 46,068 bytes (~45.0 KB) | Comparable |
| **File Size (Mean)** | 28,889 bytes (~28.2 KB) | 28,360 bytes (~27.7 KB) | $\Delta = 1.8\%$ difference |
| **File Size (Std Dev)**| 3,932 bytes | 3,912 bytes | Nearly identical compression variance |
| **File Size (Median)**| 28,630.5 bytes | 27,971.5 bytes | Identical median footprint |

---

## 5. Provenance & Labeling Methodology

Based on Zenodo and Kaggle benchmark metadata (`sachchitkunichetty/rvf10k`):
1. **Real Images**: Sourced from NVIDIA's **Flickr-Faces-HQ (FFHQ)** dataset. Faces were crawled from Flickr, aligned and cropped using facial landmarks (eyes, nose, mouth), and downsampled uniformly to $256 \times 256$.
2. **Fake Images**: Sourced from generative models (**StyleGAN**) via Bojan Tunguz’s *1 Million Fake Faces* / *ThisPersonDoesNotExist*. Faces share the exact same alignment and centering geometry as FFHQ because StyleGAN was trained on FFHQ.
3. **Restructuring**: The author created the 4 configuration folders to enable researchers to test model sensitivity across different split ratios without having to manually partition the data.

### Comparison with Existing Kaggle Dataset
```
+--------------------------------+-----------------------------------+-----------------------------------+
| Feature                        | Previous Kaggle Dataset (60k)     | New Real vs Fake Faces (RVF10K)   |
+--------------------------------+-----------------------------------+-----------------------------------+
| Total Unique Images            | 57,071                            | 10,000                            |
| Real Image Formats             | PNG (Lossless)                    | JPEG (Lossy)                      |
| Fake Image Formats             | JPEG (Lossy)                      | JPEG (Lossy)                      |
| Real Image Dimensions          | 512x512                           | 256x256                           |
| Fake Image Dimensions          | Arbitrary (960x1490, 1125x1585)   | 256x256                           |
| Face Alignment Quality         | Mixed (Real aligned, Fake unalign)| Both Real & Fake aligned to FFHQ  |
| Compression Bias Risk          | EXTREME (Format shortcut learned) | MINIMAL (Both classes 256x256 JPG)|
+--------------------------------+-----------------------------------+-----------------------------------+
```

---

## 6. Recommendations & Safest Configuration Selection

### 6.1 Recommended Configuration: `8020`
**We strongly recommend selecting the `8020` configuration for training DeepFake-Forensics.**

#### Why `8020` is the Safest and Best Choice:
1. **Maximum Training Volume (8,000 Images)**:
   - Deep neural networks require maximum sample diversity to learn generalizable facial forgery features rather than memorizing individual faces.
   - `8020/train` provides 8,000 images (4,000 Real, 4,000 Fake) — **33.3% more training data** than `6040` (6,000 images).
2. **Sufficiently Sized, High-Powered Validation & Test Sets (1,000 Each)**:
   - `8020/valid` contains 1,000 images (500 Real, 500 Fake).
   - `8020/test` contains 1,000 images (500 Real, 500 Fake).
   - With $N = 1,000$, the 95% confidence interval margin of error at 95% accuracy is:
     $$\text{Margin of Error} = \pm 1.96 \times \sqrt{\frac{0.95 \times 0.05}{1000}} \approx \pm 1.35\%$$
   - A $\pm 1.35\%$ margin of error provides more than adequate statistical power for benchmark reporting while preserving the maximum possible data for model training.
3. **Class Parity**:
   - Every split in `8020` is balanced at exactly 50.0% Real and 50.0% Fake, ensuring loss functions (e.g., CrossEntropyLoss, BCEWithLogitsLoss) operate without class frequency skew.

---

## 7. Action Plan for Project Integration

When the team is ready to integrate this dataset into `DeepFake-Forensics`:

1. **Select and Isolate `8020`**:
   - Copy only the `8020` directory into `datasets/processed/rvf10k_8020/` (or create a symbolic link).
   - Archive or leave the remaining folders (`6040`, `7030`, `7525`) in `Downloads/` without merging.
2. **Do Not Overwrite Existing Datasets**:
   - Keep existing datasets in `datasets/raw/` intact.
   - Keep existing checkpoint `models/deepfake_cnn.pth` intact as a baseline benchmark.
3. **Data Augmentation Strategy for Training**:
   - Since both Real and Fake images are $256 \times 256$ JPEGs, incorporate Albumentations / Torchvision transforms during training:
     - Random horizontal flips ($p=0.5$)
     - JPEG compression simulation ($Q \in [60, 95]$)
     - Slight affine rotation / scaling ($\pm 8^\circ$)
     - Color jitter / Gaussian blur
4. **Model Architecture Requirement**:
   - Utilize Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`) to ensure translation invariance across the $256 \times 256$ input space.
