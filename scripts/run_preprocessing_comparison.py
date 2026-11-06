"""
Comparison of EfficientNet-B0 Preprocessing:
1. Original training preprocessing (direct 224x224 resize of full 256x256 image)
2. Current SCRFD face detection, cropping, and inference preprocessing (pipeline)
3. Direct SCRFD face crop classification (isolating spatial crop effect from quality gate)
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.image_detection.efficientnet_model import DeepfakeEfficientNetB0, get_eval_transform
from src.image_detection.inference import DeepFakeInferenceEngine

import os

DATASET_TEST_DIR = Path(os.environ.get("RVF10K_TEST_DIR", "datasets/rvf10k/test"))
CHECKPOINT_PATH = Path("models/efficientnet_b0_best.pth")
OUTPUT_JSON = Path("outputs/preprocessing_comparison.json")


def run_comparison():
    print("=" * 75)
    print("EFFICIENTNET-B0 PREPROCESSING VALIDATION: RVF10K TEST SET (1,000 SAMPLES)")
    print("=" * 75)

    device = torch.device("cpu")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = DeepfakeEfficientNetB0(pretrained=False)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    eval_transform = get_eval_transform()
    engine = DeepFakeInferenceEngine(primary_checkpoint=CHECKPOINT_PATH, device=device, auto_load=True)

    fake_files = sorted(list((DATASET_TEST_DIR / "fake").glob("*.jpg")))
    real_files = sorted(list((DATASET_TEST_DIR / "real").glob("*.jpg")))
    print(f"Found {len(fake_files)} FAKE test images, {len(real_files)} REAL test images (Total: {len(fake_files) + len(real_files)})")

    # List of (filepath, ground_truth_idx, ground_truth_label)
    # class_to_idx: {'fake': 0, 'real': 1}
    dataset_items = [(p, 0, "FAKE") for p in fake_files] + [(p, 1, "REAL") for p in real_files]

    # Containers for results
    results_orig = []
    results_scrfd_pipeline = []
    results_scrfd_crop_only = []

    print("\n[1/3] Evaluating with Original Training Preprocessing (Full 256x256 direct resize)...")
    t0 = time.time()
    with torch.no_grad():
        for i, (path, gt_idx, gt_label) in enumerate(dataset_items):
            im = Image.open(path).convert("RGB")
            tensor = eval_transform(im).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_label = "FAKE" if pred_idx == 0 else "REAL"
            conf = float(probs[pred_idx].item())

            results_orig.append({
                "filename": path.name,
                "ground_truth": gt_label,
                "ground_truth_idx": gt_idx,
                "predicted_label": pred_label,
                "predicted_idx": pred_idx,
                "confidence": round(conf, 4),
                "prob_fake": round(float(probs[0].item()), 4),
                "prob_real": round(float(probs[1].item()), 4),
                "is_correct": bool(pred_idx == gt_idx),
            })
            if (i + 1) % 250 == 0:
                print(f"  Processed {i + 1}/{len(dataset_items)} samples...")

    t1 = time.time()
    print(f"Original preprocessing evaluation complete in {t1 - t0:.2f}s")

    print("\n[2/3] Evaluating with Current SCRFD Pipeline (Detection + Quality Gates + Crop)...")
    t2 = time.time()
    for i, (path, gt_idx, gt_label) in enumerate(dataset_items):
        # Use default uncertainty threshold (0.60) or 0.50
        res = engine.predict_image(path, uncertainty_threshold=0.50, include_gradcam=False)
        pred = res.predictions[0]

        faces_detected = res.faces_detected
        pred_label = pred.label
        is_inconclusive = (pred_label == "INCONCLUSIVE")

        if pred_label == "FAKE":
            pred_idx = 0
            is_correct = (gt_idx == 0)
        elif pred_label == "REAL":
            pred_idx = 1
            is_correct = (gt_idx == 1)
        else:
            pred_idx = -1
            is_correct = False

        prob_fake = pred.probabilities.get("FAKE", None) if pred.probabilities else None
        prob_real = pred.probabilities.get("REAL", None) if pred.probabilities else None

        results_scrfd_pipeline.append({
            "filename": path.name,
            "ground_truth": gt_label,
            "ground_truth_idx": gt_idx,
            "predicted_label": pred_label,
            "predicted_idx": pred_idx,
            "confidence": pred.score,
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "faces_detected": faces_detected,
            "reason": pred.reason,
            "is_inconclusive": is_inconclusive,
            "is_correct": is_correct,
            "box": pred.box,
        })
        if (i + 1) % 250 == 0:
            print(f"  Processed {i + 1}/{len(dataset_items)} samples...")

    t3 = time.time()
    print(f"SCRFD pipeline evaluation complete in {t3 - t2:.2f}s")

    print("\n[3/3] Evaluating Direct SCRFD Face Crops (Isolated Face Spatial Cropping)...")
    t4 = time.time()
    with torch.no_grad():
        for i, (path, gt_idx, gt_label) in enumerate(dataset_items):
            cv_img, _ = engine.validate_and_read_image(path)
            artifacts = engine.detect_and_crop_faces(cv_img)
            if len(artifacts) == 0:
                results_scrfd_crop_only.append({
                    "filename": path.name,
                    "ground_truth": gt_label,
                    "ground_truth_idx": gt_idx,
                    "predicted_label": "NO_FACE",
                    "predicted_idx": -1,
                    "confidence": None,
                    "prob_fake": None,
                    "prob_real": None,
                    "is_correct": False,
                    "quality_reason": "no_face_detected",
                })
            else:
                art = artifacts[0]
                crop_pil = art.crop_pil
                tensor = eval_transform(crop_pil).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                pred_label = "FAKE" if pred_idx == 0 else "REAL"
                conf = float(probs[pred_idx].item())

                results_scrfd_crop_only.append({
                    "filename": path.name,
                    "ground_truth": gt_label,
                    "ground_truth_idx": gt_idx,
                    "predicted_label": pred_label,
                    "predicted_idx": pred_idx,
                    "confidence": round(conf, 4),
                    "prob_fake": round(float(probs[0].item()), 4),
                    "prob_real": round(float(probs[1].item()), 4),
                    "is_correct": bool(pred_idx == gt_idx),
                    "is_usable": art.is_usable,
                    "quality_reason": art.quality_reason,
                })
            if (i + 1) % 250 == 0:
                print(f"  Processed {i + 1}/{len(dataset_items)} samples...")

    t5 = time.time()
    print(f"Direct SCRFD face crop evaluation complete in {t5 - t4:.2f}s")

    # Metrics computation
    y_true = [r["ground_truth_idx"] for r in results_orig]

    # 1. Original
    y_pred_orig = [r["predicted_idx"] for r in results_orig]
    acc_orig = accuracy_score(y_true, y_pred_orig)
    cm_orig = confusion_matrix(y_true, y_pred_orig).tolist()
    misclassified_orig = [r for r in results_orig if not r["is_correct"]]

    # 2. SCRFD Pipeline
    conclusive_pipe = [r for r in results_scrfd_pipeline if not r["is_inconclusive"]]
    inconclusive_pipe = [r for r in results_scrfd_pipeline if r["is_inconclusive"]]
    acc_pipe_overall = sum(1 for r in results_scrfd_pipeline if r["is_correct"]) / len(results_scrfd_pipeline)
    acc_pipe_conclusive = (
        sum(1 for r in conclusive_pipe if r["is_correct"]) / len(conclusive_pipe)
        if len(conclusive_pipe) > 0 else 0.0
    )
    misclassified_pipe = [r for r in results_scrfd_pipeline if not r["is_correct"]]

    # 3. Direct SCRFD Face Crop (no gate)
    y_pred_crop = [r["predicted_idx"] for r in results_scrfd_crop_only]
    acc_crop = sum(1 for r in results_scrfd_crop_only if r["is_correct"]) / len(results_scrfd_crop_only)
    cm_crop = confusion_matrix(y_true, y_pred_crop).tolist()
    misclassified_crop = [r for r in results_scrfd_crop_only if not r["is_correct"]]

    # Breakdown of reasons for inconclusive in pipeline
    inconclusive_reasons = {}
    for r in inconclusive_pipe:
        reas = r["reason"] or "unknown"
        inconclusive_reasons[reas] = inconclusive_reasons.get(reas, 0) + 1

    # Discrepancies between Original and SCRFD Pipeline
    orig_correct_pipe_wrong = [
        {
            "filename": o["filename"],
            "gt": o["ground_truth"],
            "orig_pred": o["predicted_label"],
            "orig_conf": o["confidence"],
            "pipe_pred": s["predicted_label"],
            "pipe_conf": s["confidence"],
            "pipe_reason": s["reason"],
            "prob_fake": s["prob_fake"],
            "prob_real": s["prob_real"],
        }
        for o, s in zip(results_orig, results_scrfd_pipeline)
        if o["is_correct"] and not s["is_correct"]
    ]

    pipe_correct_orig_wrong = [
        {
            "filename": o["filename"],
            "gt": o["ground_truth"],
            "orig_pred": o["predicted_label"],
            "orig_conf": o["confidence"],
            "pipe_pred": s["predicted_label"],
            "pipe_conf": s["confidence"],
        }
        for o, s in zip(results_orig, results_scrfd_pipeline)
        if s["is_correct"] and not o["is_correct"]
    ]

    # Discrepancies between Original and Direct Crop
    orig_correct_crop_wrong = [
        {
            "filename": o["filename"],
            "gt": o["ground_truth"],
            "orig_pred": o["predicted_label"],
            "orig_conf": o["confidence"],
            "crop_pred": c["predicted_label"],
            "crop_conf": c["confidence"],
        }
        for o, c in zip(results_orig, results_scrfd_crop_only)
        if o["is_correct"] and not c["is_correct"]
    ]

    crop_correct_orig_wrong = [
        {
            "filename": o["filename"],
            "gt": o["ground_truth"],
            "orig_pred": o["predicted_label"],
            "orig_conf": o["confidence"],
            "crop_pred": c["predicted_label"],
            "crop_conf": c["confidence"],
        }
        for o, c in zip(results_orig, results_scrfd_crop_only)
        if c["is_correct"] and not o["is_correct"]
    ]

    print("\n" + "=" * 75)
    print("COMPARATIVE EVALUATION SUMMARY (RVF10K TEST SET)")
    print("=" * 75)
    print(f"Total Test Samples:                           {len(dataset_items)}")
    print(f"  - FAKE samples:                             {len(fake_files)}")
    print(f"  - REAL samples:                             {len(real_files)}")

    print(f"\n1. Original Training Preprocessing (Full 256x256 image direct resize):")
    print(f"   - Accuracy:                                {acc_orig * 100:.2f}% ({sum(r['is_correct'] for r in results_orig)} / {len(dataset_items)})")
    print(f"   - Confusion Matrix (TN,FP / FN,TP):        {cm_orig}")
    print(f"   - Total Misclassified:                     {len(misclassified_orig)}")
    print(f"     * FAKE misclassified as REAL:            {sum(1 for r in misclassified_orig if r['ground_truth'] == 'FAKE')}")
    print(f"     * REAL misclassified as FAKE:            {sum(1 for r in misclassified_orig if r['ground_truth'] == 'REAL')}")

    print(f"\n2. Current SCRFD Pipeline (Detection + Quality Gates + Crop):")
    print(f"   - Overall Accuracy:                        {acc_pipe_overall * 100:.2f}% ({sum(r['is_correct'] for r in results_scrfd_pipeline)} / {len(dataset_items)})")
    print(f"   - Conclusive Subset Accuracy:              {acc_pipe_conclusive * 100:.2f}% ({sum(r['is_correct'] for r in conclusive_pipe)} / {len(conclusive_pipe)})")
    print(f"   - Inconclusive Decisions:                  {len(inconclusive_pipe)} samples")
    for r_name, r_cnt in inconclusive_reasons.items():
        print(f"     * Reason '{r_name}': {r_cnt}")
    print(f"   - Total Misclassified / Failed:            {len(misclassified_pipe)}")
    print(f"     * FAKE misclassified as REAL:            {sum(1 for r in misclassified_pipe if r['ground_truth'] == 'FAKE' and not r['is_inconclusive'])}")
    print(f"     * REAL misclassified as FAKE:            {sum(1 for r in misclassified_pipe if r['ground_truth'] == 'REAL' and not r['is_inconclusive'])}")
    print(f"     * Inconclusive:                          {len(inconclusive_pipe)}")

    print(f"\n3. Direct SCRFD Face Crop (Without Quality Gate Rejections):")
    print(f"   - Accuracy:                                {acc_crop * 100:.2f}% ({sum(r['is_correct'] for r in results_scrfd_crop_only)} / {len(dataset_items)})")
    print(f"   - Confusion Matrix (TN,FP / FN,TP):        {cm_crop}")
    print(f"   - Total Misclassified:                     {len(misclassified_crop)}")
    print(f"     * FAKE misclassified as REAL:            {sum(1 for r in misclassified_crop if r['ground_truth'] == 'FAKE')}")
    print(f"     * REAL misclassified as FAKE:            {sum(1 for r in misclassified_crop if r['ground_truth'] == 'REAL')}")

    print(f"\n4. Performance Delta & Discrepancies:")
    print(f"   - Delta (SCRFD Pipeline vs Original):      {(acc_pipe_overall - acc_orig) * 100:+.2f}%")
    print(f"   - Delta (SCRFD Direct Crop vs Original):   {(acc_crop - acc_orig) * 100:+.2f}%")
    print(f"   - Correct in Original, Failed in Pipeline: {len(orig_correct_pipe_wrong)}")
    print(f"   - Correct in Pipeline, Wrong in Original:  {len(pipe_correct_orig_wrong)}")
    print(f"   - Correct in Original, Wrong in Crop:      {len(orig_correct_crop_wrong)}")
    print(f"   - Correct in Crop, Wrong in Original:      {len(crop_correct_orig_wrong)}")
    print("=" * 75)

    # Save detailed JSON report
    report_data = {
        "test_dataset": str(DATASET_TEST_DIR),
        "total_samples": len(dataset_items),
        "fake_samples": len(fake_files),
        "real_samples": len(real_files),
        "original_preprocessing": {
            "accuracy_pct": round(acc_orig * 100, 2),
            "correct_count": sum(r["is_correct"] for r in results_orig),
            "confusion_matrix": cm_orig,
            "misclassified_count": len(misclassified_orig),
            "misclassified_samples": misclassified_orig,
        },
        "current_scrfd_pipeline": {
            "overall_accuracy_pct": round(acc_pipe_overall * 100, 2),
            "conclusive_accuracy_pct": round(acc_pipe_conclusive * 100, 2),
            "correct_count": sum(r["is_correct"] for r in results_scrfd_pipeline),
            "inconclusive_count": len(inconclusive_pipe),
            "inconclusive_reasons": inconclusive_reasons,
            "misclassified_count": len(misclassified_pipe),
            "misclassified_samples": misclassified_pipe,
        },
        "direct_scrfd_crop_only": {
            "accuracy_pct": round(acc_crop * 100, 2),
            "correct_count": sum(r["is_correct"] for r in results_scrfd_crop_only),
            "confusion_matrix": cm_crop,
            "misclassified_count": len(misclassified_crop),
            "misclassified_samples": misclassified_crop,
        },
        "comparison": {
            "pipeline_vs_original_delta_pct": round((acc_pipe_overall - acc_orig) * 100, 2),
            "crop_vs_original_delta_pct": round((acc_crop - acc_orig) * 100, 2),
            "correct_in_original_only_pipeline": orig_correct_pipe_wrong,
            "correct_in_original_only_crop": orig_correct_crop_wrong,
            "correct_in_crop_only": crop_correct_orig_wrong,
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nDetailed JSON report exported to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_comparison()
