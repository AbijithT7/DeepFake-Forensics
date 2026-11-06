# DeepFake Detector: Dataset and Experiment Spec

Version 1. Written before any training so the pass/fail criteria cannot be adjusted after seeing results.

**Status of facts.** Dataset names, sizes and licences below come from memory and must be verified before download (marked **VERIFY**). Numeric targets are planning values, not measurements. Nothing here claims a model will work; the experiment exists to find out.

## 1. Goal and rules

Build one detector that generalizes beyond a single generator, and prove it on data it never saw.

- The frozen 250-image benchmark is never used for training, tuning, threshold choice or model selection. It is evaluated once, at the end, on the final model only. Its results never decide between checkpoints or architecture changes.
- The original checkpoint `models/efficientnet_b0_best.pth` is never overwritten. Record its SHA-256 and set it read-only.
- Production inference, thresholds and fusion are not changed until section 9 passes.
- All split manifests are written and committed before the first training run. No re-splitting afterwards.
- Train and serve preprocessing must be identical (same SCRFD crop, same resize).

## 2. Data sources

| Role | Source | Contents | Notes |
| --- | --- | --- | --- |
| Real (primary) | FFHQ via RVF10K (already local) | Clean aligned faces | Source of StyleGAN reals |
| Real (second source) | CelebA-HQ | Celebrity faces | **VERIFY** access and licence. Prevents "real = one source" shortcut |
| Real (in-the-wild) | DeepFakeFace reals (IMDB-WIKI) | Old, compressed photos | Use only in train-group split, never the 250's source photos |
| GAN fakes | StyleGAN from RVF10K | StyleGAN faces | Small replay subset to avoid forgetting |
| Diffusion text2img | DeepFakeFace | Stable Diffusion celebrity faces | **VERIFY** |
| Face swap | DeepFakeFace | InsightFace swaps | **VERIFY** |
| Inpainting | DeepFakeFace | SD inpainting | **Held out from training** (see section 4) |
| Extra generators | Community Forensics, GenImage, or DF40 | Many generators | **VERIFY** which contain faces. Use one family as the external held-out set |

If an extra-generator dataset has no face content, crop faces with SCRFD and keep only images with a valid detection, or drop it.

## 3. Target composition (planning values)

Training set, roughly balanced real vs fake:

| Class | Source | Images |
| --- | --- | --: |
| Real | FFHQ | 4,000 |
| Real | CelebA-HQ | 4,000 |
| Real | IMDB-WIKI (DeepFakeFace) | 4,000 |
| Fake | StyleGAN (replay) | 3,000 |
| Fake | SD text2img | 4,500 |
| Fake | Face swap | 4,500 |

Total about 24,000. Validation: about 3,000, drawn from validation groups with the same mix.

The point is that no single real source or fake source exceeds roughly one third of its class.

## 4. Held-out generators (the real test)

Two test sets that contain generators never seen in training:

1. **Held-out manipulation:** SD **inpainting** from DeepFakeFace, taken only from test-split groups.
2. **External generator:** one family from the extra-generator source (prefer a different or newer diffusion generator), with reals from a source not used in training if one exists.

If no suitable external reals exist, use CelebA-HQ reals from test-split identities with identical processing, and state in the report that the real and fake sources differ.

**Matched reals.** If fakes and reals come from different datasets, a model can score well by recognising the dataset instead of the manipulation. Held-out inpainting has paired reals in DeepFakeFace (the original photograph), so use them. For the external generator, true pairs are often unavailable. Then process both classes with the same crop, resize and JPEG pipeline, take reals of similar resolution and framing, and require the metadata probe to pass on the external set too (section 7). If it fails, report the external results with a warning and do not treat them as evidence of generalization.

**Limit of the inpainting test.** It shares the DeepFakeFace pipeline, reals and processing with the training fakes, so it tests an unseen manipulation type, not an unseen data source. The external generator set is the main generalization test.

## 5. Splitting and leakage prevention

1. Assign a **group ID** to every image. A group is one source photo plus all its derivatives (the real photo, its text2img, inpainting and swap versions). For identity-level grouping, use the celebrity or identity label when the dataset provides it.
2. For datasets without identity labels (FFHQ, CelebA-HQ), cluster InsightFace face embeddings and merge clusters above a similarity threshold into one group. Pick the threshold by inspecting a sample of merged pairs, and record it.
3. Build the **blocklist**: every source photo, identity and group that appears in the frozen 250 is excluded from train, validation and all test sets.
4. Split by group, 70/10/20 (train/validation/test), with a fixed seed. Never split by file.
5. Deduplicate across all sources with a perceptual hash (pHash). Drop near-duplicates, keeping the copy in the earlier split.
6. Write manifests (`train.csv`, `val.csv`, `test.csv`, `external.csv`) with columns: `path, label, source, generator, group_id, split, sha256, phash, width, height, file_bytes, jpeg_quality`. Commit them and compute a SHA-256 of each manifest.
7. A script must assert: no group\_id in two splits, no blocklist group anywhere, no sha256 collision across splits.

## 6. Preprocessing

- Detect with the existing SCRFD, crop with the production margin, resize to 224x224, ImageNet normalization. Cache crops to disk once.
- Do **not** use the original direct 256 resize. Training must match what the website runs.
- Reject images with no face, multiple faces, or very small crops, using the same gate as production. Record rejects.

## 7. Shortcut control

The below-chance results from ViT and DeepGuard suggest the real class differs from the fake class in quality, not just content. So:

1. **Metadata probe (before training):** train a small classifier (logistic regression or gradient boosting) on `width, height, file_bytes, jpeg_quality` and crop size only. Train it on the train manifest and score it on the validation set, then on the test and external sets. Required: AUC below 0.65 on validation and on the external set. If higher, rebalance sources or resample, log each round (what changed, probe AUC) in the report, and allow at most three rounds. If it still fails, stop and report.
2. **Matched processing:** apply the same random pipeline to both classes at training time: JPEG quality 60-95, Gaussian blur (p=0.2), down-then-up resize (p=0.3), horizontal flip. Never apply it to one class only.
3. **Quality-matched re-evaluation of the baselines (optional, cheap):** re-encode the frozen 250 identically (same size and JPEG quality) and re-run the existing models. If their AUC rises toward or above 0.5, the quality confound is confirmed.

## 8. Experiment: two arms, same data

Both arms use the same train/val manifests and are judged on the same test sets. The arms do not cleanly separate backbone from data. Original B0 vs fine-tuned B0 shows the effect of the training data. Fine-tuned B0 vs the CLIP probe compares a fine-tuned CNN with frozen foundation-model features.

**Arm A: EfficientNet-B0 fine-tune**

- Start from a copy of the existing checkpoint under a new filename.
- Stage 1: freeze backbone, train the classifier head, 1-2 epochs, LR 1e-3.
- Stage 2: unfreeze all, AdamW, LR 1e-4 for later blocks and 1e-5 for early blocks, cosine schedule, 5-8 epochs, label smoothing 0.1, batch 32-64.
- Select the checkpoint by validation AUC. Never use the 250 or the test sets for selection.
- Seed 42, plus two more seeds if compute allows, to report variance.

**Arm B: frozen CLIP linear probe**

- Extract features once with CLIP ViT-L/14 (OpenAI weights via `open_clip` or the UnivFD repository, **VERIFY** licence and weights) from the same cached crops.
- Train logistic regression on the features with the same train split. Choose regularization on validation.
- Training takes seconds after extraction. Extraction on CPU is the main cost.

**Compute.** Use a free Kaggle or Colab GPU for Arm A, since the laptop has no CUDA. Upload only public-dataset crops. Estimated under 1-2 hours per run on a T4 (estimate, benchmark first). Arm B extraction can run on CPU in a few hours, or on the GPU in minutes.

## 9. Evaluation and pass/fail criteria

Report for every model on every test set: ROC-AUC, TPR at 95% specificity, accuracy, precision, recall, F1 (use one definition of each metric and state it), balanced accuracy, ECE, and 95% bootstrap confidence intervals (1,000 resamples, resampling by group).

Evaluate on:

- Internal test split (sanity check only).
- Held-out inpainting set.
- External generator set.
- Frozen 250 (primary 200 and StyleGAN separately). Run once, at the end, on the final model only: no checkpoint selection, no threshold tuning, no architecture changes based on it. It has already been seen during baseline work and shares sources with DeepFakeFace, so it is a necessary check, not proof.
- Robustness: JPEG Q70 and 0.5x downscale versions of the external set.

**A model is promoted only if all of these hold, judged on the lower bound of the 95% CI:**

| Check | Criterion |
| --- | --- |
| Metadata probe | Probe trained on train; AUC < 0.65 on validation and on the external set |
| No regression | On the original RVF10K test split (production crop pipeline, excluded from all fine-tuning data), scoring original and new model on identical files: StyleGAN recall >= 90% and FFHQ real specificity within 2 points of the original |
| Held-out inpainting | AUC lower bound >= 0.75 |
| External generator | AUC lower bound >= 0.75, balanced accuracy >= 0.65 |
| Frozen 250 (primary 200) | AUC >= 0.85 |
| Robustness | AUC drop <= 0.10 under JPEG Q70 and 0.5x downscale |
| Calibration | ECE <= 0.10 on the external set (after temperature scaling fitted on validation only) |

The thresholds above are my starting values. Adjust them **now**, before training, if you disagree. They must not change afterwards.

**Outcomes:**

- All pass: promote the model, then redesign fusion and the REAL/FAKE/INCONCLUSIVE wording.
- Held-out sets fail but internal and frozen-250 pass: you built a dataset-specific detector. Do not promote. Document it honestly.
- Arm B beats Arm A on held-out sets: frozen foundation-model features generalize better than a fine-tuned CNN on this data. Consider CLIP features as the next base, after checking the result holds across seeds.
- Both fail: the limit is data diversity or the problem needs a different approach. Stop and report, rather than looping.

## 10. Deliverables

```
experiments/v1/
  protocol.md          this spec, frozen
  manifests/           train.csv val.csv test.csv external.csv + hashes
  blocklist.csv        groups from the frozen 250
  scripts/             build_manifests.py, check_leakage.py, metadata_probe.py,
                       cache_crops.py, train_effnet.py, extract_clip.py,
                       train_probe.py, evaluate.py
  checkpoints/         new files only; originals untouched
  reports/             results.md, per_generator.csv, plots
```

## 11. Order of work

1. Verify dataset availability, licences and actual face-image counts (one afternoon). A source enters the manifests only after all three are confirmed. Pick the external generator family and check whether matched reals exist for it.
2. Build manifests, blocklist and leakage assertions. Download only the verified sources, since the probe needs image metadata. Run the metadata probe (train on train, score on validation, test and external). Rebalance with each round logged, up to three rounds. Then stop and report before caching crops or training.
3. Cache crops.
4. Run Arm B (cheap) and Arm A.
5. Evaluate with section 9 and write `results.md`.
6. Decide using the outcomes table. No extra diagnostics unless a criterion is ambiguous.

## 12. Brief for the repo agent (Codex)

> Implement section 10 of this spec. Do not modify production inference, thresholds, fusion, the original checkpoint or the frozen 250 dataset. Do not download anything until I confirm the dataset list from section 2. Start with `build_manifests.py` and `check_leakage.py`, and stop after the leakage assertions pass and the metadata probe result is reported. Report any discrepancy between this spec and the repository before proceeding.
