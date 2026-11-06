# Comparative Evaluation Report: All-Models vs. DINOv2 Selective Routing

> **Evaluation Objective**: Determine empirically whether intelligent DINOv2 selective routing maintains forensic detection efficacy while reducing computational cost compared to exhaustive all-specialist consensus.  
> **Evaluation Scope**: Identical 250-image frozen out-of-distribution benchmark (`evaluation_dataset`).  
> **Execution Mode**: Production consensus logic (`compute_combined_result()`, threshold=0.60), zero benchmark modification, zero test-time training or tuning.

---

## 1. Executive Summary & Verdict

| Metric | Pipeline A (All 5 Specialists) | Pipeline B (DINOv2 Selective Routing) | Delta (B - A) | Relative Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Strict Accuracy (All N=250)** | **26.80%** | **34.40%** | **+7.60%** | Improvement |
| **Conclusive Accuracy** | 27.80% | 41.35% | +13.55% | Evaluated on decisive outputs |
| **Coverage (Decisive Rate)** | 96.40% | 83.20% | -13.20% | Abstention rate: 16.8% |
| **Fake Precision** | 100.00% | 98.04% | -1.96% | Positive predictive value |
| **Fake Recall (Sensitivity)** | 8.50% | 25.00% | +16.50% | Detected out of 200 fakes |
| **Fake F1-Score** | 15.67% | 39.84% | +24.17% | Harmonic mean on fakes |
| **Real Precision** | 22.32% | 22.93% | +0.61% | Specificity accuracy |
| **Real Recall (Specificity)** | 100.00% | 72.00% | -28.00% | Detected out of 50 reals |
| **Macro F1-Score** | **26.08%** | **37.31%** | **+11.23%** | Unweighted class mean |
| **Balanced Accuracy** | 54.25% | 48.50% | -5.75% | Mean of Real & Fake recall |
| **False REAL Predictions** | **174 (87.0%)** | **121 (60.5%)** | **-53** | Critical security hazard |
| **Mean Latency per Image** | **1218.1 ms** | **1018.7 ms** | **-199.4 ms** | **16.4% Faster** |
| **Forward Passes per Image** | **5.00** | **2.60** | **-2.40** | **48.0% Compute Saved** |

### Bottom-Line Forensic Assessment:
**Is intelligent routing useful?**
- **Computational Efficiency**: DINOv2 selective routing achieves a **48.0% reduction in detector forward passes** (from 5.0 to 2.60 per face) and cuts average inference time by **16.4%** (1018.7 ms vs 1218.1 ms).
- **Detection Performance**: Strict accuracy changed by **+7.60%** (34.40% vs 26.80%), while Macro F1 changed by **+11.23%**.
- **Security Posture (False Reals)**: Pipeline B produced 121 false REAL predictions vs Pipeline A's 174 (-53).

---

## 2. Verdict Distribution & Confusion Matrices

### REAL / FAKE / INCONCLUSIVE Distribution

| Verdict | Pipeline A (All Models) | Pipeline A (%) | Pipeline B (DINOv2 Routed) | Pipeline B (%) |
| :--- | :---: | :---: | :---: | :---: |
| **REAL** | 224 | 89.6% | 157 | 62.8% |
| **FAKE** | 17 | 6.8% | 51 | 20.4% |
| **INCONCLUSIVE** | 9 | 3.6% | 42 | 16.8% |
| **Total** | 250 | 100.0% | 250 | 100.0% |

### Detailed 3x2 Confusion Matrices (with Inconclusive Abstentions)

#### Pipeline A (All 5 Specialists)
```text
                  Predicted REAL    Predicted FAKE    Predicted INCONCLUSIVE    Total
Actual REAL             50                 0                 0                         50
Actual FAKE             174                17                9                         200
```

#### Pipeline B (DINOv2 Selective Routing)
```text
                  Predicted REAL    Predicted FAKE    Predicted INCONCLUSIVE    Total
Actual REAL             36                 1                 13                        50
Actual FAKE             121                50                29                        200
```

---

## 3. Per-Manipulation-Category Performance Breakdown

| Category | Actual Label | Samples | Router Typology Acc | Pipeline A Strict Acc | Pipeline B Strict Acc | Accuracy Delta | Pipeline A (R / F / Inc) | Pipeline B (R / F / Inc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REAL** | REAL | 50 | 52.0% (26/50) | 100.0% | 72.0% | **-28.0%** | 50 / 0 / 0 | 36 / 1 / 13 |
| **STYLEGAN_GAN** | FAKE | 50 | 78.0% (39/50) | 28.0% | 54.0% | **+26.0%** | 29 / 14 / 7 | 9 / 27 / 14 |
| **DIFFUSION** | FAKE | 50 | 36.0% (18/50) | 2.0% | 14.0% | **+12.0%** | 48 / 1 / 1 | 39 / 7 / 4 |
| **FACE_SWAP** | FAKE | 50 | 8.0% (4/50) | 2.0% | 12.0% | **+10.0%** | 49 / 1 / 0 | 41 / 6 / 3 |
| **INPAINTING** | FAKE | 50 | 0.0% (0/50) | 2.0% | 20.0% | **+18.0%** | 48 / 1 / 1 | 32 / 10 / 8 |

---

## 4. Routing Failure & Divergence Analysis

### Categories Where Routing Changes Detection Efficacy
- **Categories where Routing Improves Efficacy**: STYLEGAN_GAN, DIFFUSION, FACE_SWAP, INPAINTING
- **Categories where Routing Degrades Efficacy**: REAL
- **Neutral Categories**: None

### Router Typology Classification
- **Router Overall Accuracy on Benchmark**: **34.80%**
- **Total Router Typology Misclassifications**: 163 out of 250 (65.2%)

#### Exemplar Router Misclassifications:
| Image Index | Filename | Ground Truth Category | Router Prediction | Confidence | Assigned Specialists | Pipeline B Verdict |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| 1 | `2650449_1954-02-26_2012.jpg` | DIFFUSION | **FACE_SWAP** | 0.9367 | `sbi` | REAL |
| 2 | `44823_1902-09-20_1952.jpg` | DIFFUSION | **REAL** | 0.8443 | `efficientnet;vit` | REAL |
| 3 | `27131814_1992-06-09_2012.jpg` | DIFFUSION | **FACE_SWAP** | 0.6045 | `sbi` | REAL |
| 6 | `5522094_1952-10-13_2011.jpg` | DIFFUSION | **UNKNOWN** | 0.0061 | `efficientnet;vit;community_vit;sbi;hybrid_srm` | REAL |
| 7 | `32069321_1977-03-30_2012.jpg` | DIFFUSION | **UNKNOWN** | 0.0043 | `efficientnet;vit;community_vit;sbi;hybrid_srm` | REAL |
| 8 | `14643878_1931-03-09_1963.jpg` | DIFFUSION | **UNKNOWN** | 0.0363 | `efficientnet;vit;community_vit;sbi;hybrid_srm` | REAL |
| 9 | `20913246_1984-11-22_2012.jpg` | DIFFUSION | **UNKNOWN** | 0.0021 | `efficientnet;vit;community_vit;sbi;hybrid_srm` | FAKE |
| 11 | `36298298_1992-07-02_2011.jpg` | DIFFUSION | **FACE_SWAP** | 0.9030 | `sbi` | FAKE |

### Specialist Misses (False REALs Under Selective Routing)
Total fake images where the selected specialist models mistakenly voted REAL: **121 images**.
When an image is misrouted or when the designated specialist has an out-of-domain false-real blind spot, the absence of the other 4 models leaves no dissenting votes to force an INCONCLUSIVE split.

#### Exemplar Specialist Misses:
| Image Index | Filename | Manipulation Category | Router Typology | Selected Specialists | Pipeline B Verdict | Pipeline A Verdict |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| 1 | `2650449_1954-02-26_2012.jpg` | DIFFUSION | FACE_SWAP | `sbi` | **unanimous_real** | REAL |
| 2 | `44823_1902-09-20_1952.jpg` | DIFFUSION | REAL | `efficientnet;vit` | **unanimous_real** | REAL |
| 3 | `27131814_1992-06-09_2012.jpg` | DIFFUSION | FACE_SWAP | `sbi` | **unanimous_real** | REAL |
| 4 | `43478489_1987-06-06_2013.jpg` | DIFFUSION | DIFFUSION | `community_vit` | **unanimous_real** | REAL |
| 5 | `3454142_1984-10-16_2006.jpg` | DIFFUSION | DIFFUSION | `community_vit` | **unanimous_real** | REAL |
| 6 | `5522094_1952-10-13_2011.jpg` | DIFFUSION | UNKNOWN | `efficientnet;vit;community_vit;sbi;hybrid_srm` | **majority_real** | REAL |
| 7 | `32069321_1977-03-30_2012.jpg` | DIFFUSION | UNKNOWN | `efficientnet;vit;community_vit;sbi;hybrid_srm` | **majority_real** | REAL |
| 8 | `14643878_1931-03-09_1963.jpg` | DIFFUSION | UNKNOWN | `efficientnet;vit;community_vit;sbi;hybrid_srm` | **majority_real** | REAL |

### Pipeline Divergence Cases
- **Pipeline A Succeeds, Pipeline B Fails**: **16 cases** (All-models consensus was correct, but selective routing failed)
- **Pipeline B Succeeds, Pipeline A Fails**: **35 cases** (Selective routing successfully removed distracting noisy models)

---

## 5. Artifacts and Visualization Assets

- **Full Evaluation Metrics (JSON)**: [`routing_comparison.json`](file:///C:/Users/abijith/Desktop/PROJECTS/DeepFake-Forensics/outputs/evaluation/routing_comparison.json)
- **Per-Image Predictions Log (CSV)**: [`routing_comparison.csv`](file:///C:/Users/abijith/Desktop/PROJECTS/DeepFake-Forensics/outputs/evaluation/routing_comparison.csv)
- **Confusion Matrix Plot**: [`confusion_matrix_comparison.png`](file:///C:/Users/abijith/Desktop/PROJECTS/DeepFake-Forensics/outputs/evaluation/confusion_matrix_comparison.png)
- **Per-Category Performance Plot**: [`per_category_comparison.png`](file:///C:/Users/abijith/Desktop/PROJECTS/DeepFake-Forensics/outputs/evaluation/per_category_comparison.png)
- **Computational Efficiency Plot**: [`efficiency_comparison.png`](file:///C:/Users/abijith/Desktop/PROJECTS/DeepFake-Forensics/outputs/evaluation/efficiency_comparison.png)
