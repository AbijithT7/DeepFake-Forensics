"""
Comparative Evaluation: All-Models Consensus (Pipeline A) vs. DINOv2 Selective Routing (Pipeline B).

Evaluates:
- Pipeline A: Runs all 5 specialist detectors, combines with compute_combined_result().
- Pipeline B: Predicts typology with DINOv2 ImageTypeRouter, routes to specialists via
              DEFAULT_SPECIALIST_ROUTING_MAP, combines with compute_combined_result().

Operates on the frozen 250-image out-of-distribution benchmark:
- 50 REAL (real_photograph)
- 50 STYLEGAN_GAN (gan / StyleGAN)
- 50 DIFFUSION (diffusion_text2img / SD v1.5)
- 50 FACE_SWAP (face_swap / InsightFace)
- 50 INPAINTING (diffusion_inpainting / SD inpainting)

Calculates:
- Strict & Conclusive Accuracy
- Precision, Recall, F1 for REAL and FAKE
- Macro F1 & Balanced Accuracy
- 2x2 and 3x3 Confusion Matrices
- REAL / FAKE / INCONCLUSIVE distribution
- False REAL predictions (critical bypass rate)
- Per-manipulation-category performance breakdown
- Average inference latency per image (ms)
- Number of detector forward passes & percentage savings
- Deep failure analysis (router misclassifications, specialist misses, pipeline divergences)

Saves:
- outputs/evaluation/routing_comparison.json
- outputs/evaluation/routing_comparison.csv
- outputs/evaluation/routing_comparison_report.md
- outputs/evaluation/confusion_matrix_comparison.png
- outputs/evaluation/per_category_comparison.png
- outputs/evaluation/efficiency_comparison.png
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.image_detection.inference import (
    DeepFakeInferenceEngine,
    DEFAULT_SPECIALIST_ROUTING_MAP,
    DETECTOR_DISPLAY_NAMES,
    compute_combined_result,
    get_efficientnet_transform,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_routing")

BENCHMARK_DIR = Path(os.environ.get("BENCHMARK_DIR", "evaluation_dataset"))
METADATA_PATH = BENCHMARK_DIR / "metadata.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluation"

CATEGORY_MAPPING = {
    "real_photograph": ("REAL", "REAL"),
    "gan": ("STYLEGAN_GAN", "FAKE"),
    "diffusion_text2img": ("DIFFUSION", "FAKE"),
    "face_swap": ("FACE_SWAP", "FAKE"),
    "diffusion_inpainting": ("INPAINTING", "FAKE"),
}

CATEGORY_SUBFOLDERS = {
    "real_photograph": "real",
    "gan": os.path.join("fake", "gan", "stylegan"),
    "diffusion_text2img": os.path.join("fake", "diffusion", "text2img"),
    "face_swap": os.path.join("fake", "faceswap"),
    "diffusion_inpainting": os.path.join("fake", "diffusion", "inpainting"),
}


def load_and_verify_benchmark() -> pd.DataFrame:
    """Load metadata.csv and resolve exact verified image paths and labels."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Benchmark metadata not found at {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    logger.info(f"Loaded benchmark metadata with {len(df)} rows.")

    resolved_paths: List[str] = []
    standard_categories: List[str] = []
    ground_truth_labels: List[str] = []

    for idx, row in df.iterrows():
        cat = row["category"]
        fn = row["filename"]
        if cat not in CATEGORY_MAPPING:
            raise ValueError(f"Unknown category '{cat}' at row {idx}")

        std_cat, gt_lbl = CATEGORY_MAPPING[cat]
        sub = CATEGORY_SUBFOLDERS[cat]
        img_path = BENCHMARK_DIR / sub / fn

        if not img_path.exists():
            raise FileNotFoundError(f"Missing benchmark file at row {idx}: {img_path}")

        resolved_paths.append(str(img_path))
        standard_categories.append(std_cat)
        ground_truth_labels.append(gt_lbl)

    df["resolved_path"] = resolved_paths
    df["standard_category"] = standard_categories
    df["ground_truth_label"] = ground_truth_labels

    # Verify duplicates
    unique_hashes = set()
    for p in resolved_paths:
        with open(p, "rb") as f:
            h = hashlib_sha256(f.read())
            unique_hashes.add(h)

    logger.info(f"Verified {len(resolved_paths)} images. Unique SHA-256 hashes: {len(unique_hashes)}.")
    if len(unique_hashes) != len(resolved_paths):
        logger.warning(f"Duplicate hash detected! {len(resolved_paths) - len(unique_hashes)} duplicates found.")

    return df


def hashlib_sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def compute_binary_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute binary classification metrics treating INCONCLUSIVE strictly as non-match.
    y_true: ['REAL', 'FAKE', ...]
    y_pred: ['REAL', 'FAKE', 'INCONCLUSIVE', ...]
    """
    n = len(y_true)
    if n == 0:
        return {}

    # Strict accuracy: only exact match is correct
    correct_strict = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    strict_acc = correct_strict / n

    # Conclusive vs Inconclusive
    conclusive_mask = [yp in ("REAL", "FAKE") for yp in y_pred]
    conclusive_count = sum(conclusive_mask)
    coverage = conclusive_count / n
    correct_conclusive = sum(1 for yt, yp, c in zip(y_true, y_pred, conclusive_mask) if c and yt == yp)
    conclusive_acc = (correct_conclusive / conclusive_count) if conclusive_count > 0 else 0.0

    # Counts
    pred_real_count = sum(1 for yp in y_pred if yp == "REAL")
    pred_fake_count = sum(1 for yp in y_pred if yp == "FAKE")
    pred_inconclusive_count = sum(1 for yp in y_pred if yp == "INCONCLUSIVE")

    actual_real_count = sum(1 for yt in y_true if yt == "REAL")
    actual_fake_count = sum(1 for yt in y_true if yt == "FAKE")

    # Contingency table:
    # TP: actual FAKE, pred FAKE
    # FP: actual REAL, pred FAKE
    # TN: actual REAL, pred REAL
    # FN: actual FAKE, pred REAL
    # Inc_FAKE: actual FAKE, pred INCONCLUSIVE
    # Inc_REAL: actual REAL, pred INCONCLUSIVE
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "FAKE")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "FAKE")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "REAL")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "REAL")
    inc_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "INCONCLUSIVE")
    inc_real = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "INCONCLUSIVE")

    # FAKE class metrics
    prec_fake = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_fake = (tp / actual_fake_count) if actual_fake_count > 0 else 0.0
    f1_fake = (2 * prec_fake * rec_fake / (prec_fake + rec_fake)) if (prec_fake + rec_fake) > 0 else 0.0

    # REAL class metrics
    prec_real = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_real = (tn / actual_real_count) if actual_real_count > 0 else 0.0
    f1_real = (2 * prec_real * rec_real / (prec_real + rec_real)) if (prec_real + rec_real) > 0 else 0.0

    macro_f1 = (f1_fake + f1_real) / 2.0
    balanced_acc = (rec_fake + rec_real) / 2.0

    # 3x2 Confusion matrix [Actual REAL, Actual FAKE] x [Pred REAL, Pred FAKE, Pred INCONCLUSIVE]
    cm_3x2 = {
        "actual_REAL": {"pred_REAL": tn, "pred_FAKE": fp, "pred_INCONCLUSIVE": inc_real},
        "actual_FAKE": {"pred_REAL": fn, "pred_FAKE": tp, "pred_INCONCLUSIVE": inc_fake},
    }

    # 2x2 Conclusive confusion matrix [[TN, FP], [FN, TP]]
    cm_2x2 = [[tn, fp], [fn, tp]]

    return {
        "total_samples": n,
        "coverage": coverage,
        "conclusive_count": conclusive_count,
        "inconclusive_count": pred_inconclusive_count,
        "strict_accuracy": strict_acc,
        "conclusive_accuracy": conclusive_acc,
        "pred_distribution": {
            "REAL": pred_real_count,
            "FAKE": pred_fake_count,
            "INCONCLUSIVE": pred_inconclusive_count,
            "REAL_pct": round(pred_real_count / n * 100, 2),
            "FAKE_pct": round(pred_fake_count / n * 100, 2),
            "INCONCLUSIVE_pct": round(pred_inconclusive_count / n * 100, 2),
        },
        "fake_metrics": {
            "precision": prec_fake,
            "recall": rec_fake,
            "f1": f1_fake,
            "true_positives": tp,
            "false_positives": fp,
        },
        "real_metrics": {
            "precision": prec_real,
            "recall": rec_real,
            "f1": f1_real,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "false_real_count": fn,  # False negatives for fake (most hazardous error)
        "false_real_rate": (fn / actual_fake_count) if actual_fake_count > 0 else 0.0,
        "confusion_matrix_3x2": cm_3x2,
        "confusion_matrix_2x2": cm_2x2,
    }


def run_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_df = load_and_verify_benchmark()

    logger.info("Initializing DeepFakeInferenceEngine...")
    t_start_init = time.perf_counter()
    engine = DeepFakeInferenceEngine(auto_load=True, shadow_mode=True, load_router=True)
    logger.info(f"Inference engine initialized in {time.perf_counter() - t_start_init:.2f}s.")

    results_records: List[Dict[str, Any]] = []
    logger.info(f"Beginning evaluation on {len(benchmark_df)} benchmark images...")

    t_eval_start = time.perf_counter()

    for idx, row in benchmark_df.iterrows():
        img_path = row["resolved_path"]
        actual_label = row["ground_truth_label"]
        actual_category = row["standard_category"]
        filename = row["filename"]

        # 1. Read & decode image
        cv_img, pil_img = engine.validate_and_read_image(img_path)

        # 2. DINOv2 Router Typology Inference
        t0 = time.perf_counter()
        router_pred = engine.image_type_router.route_image(pil_img)
        t_router = time.perf_counter() - t0

        predicted_image_type = router_pred.predicted_class
        raw_image_type = router_pred.raw_predicted_class
        router_conf = float(router_pred.confidence)
        router_is_rejected = bool(router_pred.is_rejected)
        router_probs = {k: round(float(v), 4) for k, v in router_pred.probabilities.items()}

        selected_keys = DEFAULT_SPECIALIST_ROUTING_MAP.get(
            predicted_image_type,
            ["efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"],
        )

        # 3. SCRFD Face Detection
        t0 = time.perf_counter()
        face_artifacts = engine.detect_and_crop_faces(cv_img)
        t_face = time.perf_counter() - t0

        faces_detected = len(face_artifacts)

        # If no usable face detected
        if faces_detected == 0 or not face_artifacts[0].is_usable:
            reason = "no_face_detected" if faces_detected == 0 else face_artifacts[0].quality_reason
            record = {
                "image_index": idx + 1,
                "filename": filename,
                "category": actual_category,
                "actual_label": actual_label,
                "faces_detected": faces_detected,
                "face_usable": False,
                "rejection_reason": reason,
                "router_predicted_type": predicted_image_type,
                "router_raw_type": raw_image_type,
                "router_confidence": router_conf,
                "router_is_rejected": router_is_rejected,
                "selected_specialists": ";".join(selected_keys),
                "pipeline_a_verdict": "INCONCLUSIVE",
                "pipeline_a_consensus": "inconclusive",
                "pipeline_a_votes_real": 0,
                "pipeline_a_votes_fake": 0,
                "pipeline_a_latency_ms": round(t_face * 1000, 2),
                "pipeline_a_forward_passes": 0,
                "pipeline_b_verdict": "INCONCLUSIVE",
                "pipeline_b_consensus": "inconclusive",
                "pipeline_b_votes_real": 0,
                "pipeline_b_votes_fake": 0,
                "pipeline_b_latency_ms": round((t_router + t_face) * 1000, 2),
                "pipeline_b_forward_passes": 0,
                "effnet_label": None,
                "effnet_score": None,
                "vit_label": None,
                "vit_score": None,
                "comm_vit_label": None,
                "comm_vit_score": None,
                "sbi_label": None,
                "sbi_score": None,
                "hybrid_srm_label": None,
                "hybrid_srm_score": None,
                "pipeline_a_correct": False,
                "pipeline_b_correct": False,
                "router_correct_typology": (predicted_image_type == actual_category),
                "verdicts_match": True,
            }
            results_records.append(record)
            continue

        art = face_artifacts[0]
        crop_pil = art.crop_pil

        # 4. Specialist Detectors Execution
        # Detector 1: EfficientNet-B0
        t0 = time.perf_counter()
        eff_transform = get_efficientnet_transform()
        tensor = eff_transform(crop_pil).unsqueeze(0).to(engine.device)
        with torch.no_grad():
            logits = engine.primary_model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        p_fake = float(probs[0].item())
        p_real = float(probs[1].item())
        eff_lbl = "REAL" if p_real >= p_fake else "FAKE"
        eff_scr = max(p_real, p_fake)
        eff_result = {"label": eff_lbl, "score": round(eff_scr, 4), "loaded": True}
        t_eff = time.perf_counter() - t0

        # Detector 2: Original ViT
        t0 = time.perf_counter()
        vit_result = dict(engine.secondary_model.predict(crop_pil))
        vit_result["loaded"] = True
        t_vit = time.perf_counter() - t0

        # Detector 3: Community Forensics ViT
        t0 = time.perf_counter()
        comm_result = dict(engine.community_vit_model.predict(crop_pil))
        comm_result["loaded"] = True
        t_comm = time.perf_counter() - t0

        # Detector 4: SBI EfficientNet-B4
        t0 = time.perf_counter()
        sbi_result = dict(engine.sbi_model.predict(crop_pil))
        sbi_result["loaded"] = True
        t_sbi = time.perf_counter() - t0

        # Detector 5: Hybrid SRM Detector
        t0 = time.perf_counter()
        srm_result = dict(engine.hybrid_srm_model.predict(crop_pil))
        srm_result["loaded"] = True
        t_srm = time.perf_counter() - t0

        model_timings = {
            "efficientnet": t_eff,
            "vit": t_vit,
            "community_vit": t_comm,
            "sbi": t_sbi,
            "hybrid_srm": t_srm,
        }

        all_models_dict = {
            "efficientnet": eff_result,
            "vit": vit_result,
            "community_vit": comm_result,
            "sbi": sbi_result,
            "hybrid_srm": srm_result,
        }

        # 5. Pipeline A Consensus (All 5 Models)
        verdict_a = compute_combined_result(all_models_dict, threshold=0.60)
        t_pipe_a = t_face + sum(model_timings.values())
        passes_a = 5

        # 6. Pipeline B Consensus (Routed Models)
        routed_models_dict = {k: all_models_dict[k] for k in selected_keys if k in all_models_dict}
        verdict_b = compute_combined_result(routed_models_dict, threshold=0.60)
        t_pipe_b = t_router + t_face + sum(model_timings[k] for k in selected_keys if k in model_timings)
        passes_b = len(selected_keys)

        lbl_a = verdict_a["label"]
        lbl_b = verdict_b["label"]

        record = {
            "image_index": idx + 1,
            "filename": filename,
            "category": actual_category,
            "actual_label": actual_label,
            "faces_detected": faces_detected,
            "face_usable": True,
            "rejection_reason": None,
            "router_predicted_type": predicted_image_type,
            "router_raw_type": raw_image_type,
            "router_confidence": round(router_conf, 4),
            "router_is_rejected": router_is_rejected,
            "selected_specialists": ";".join(selected_keys),
            "pipeline_a_verdict": lbl_a,
            "pipeline_a_consensus": verdict_a["consensus"],
            "pipeline_a_votes_real": verdict_a["votes"]["REAL"],
            "pipeline_a_votes_fake": verdict_a["votes"]["FAKE"],
            "pipeline_a_latency_ms": round(t_pipe_a * 1000, 2),
            "pipeline_a_forward_passes": passes_a,
            "pipeline_b_verdict": lbl_b,
            "pipeline_b_consensus": verdict_b["consensus"],
            "pipeline_b_votes_real": verdict_b["votes"]["REAL"],
            "pipeline_b_votes_fake": verdict_b["votes"]["FAKE"],
            "pipeline_b_latency_ms": round(t_pipe_b * 1000, 2),
            "pipeline_b_forward_passes": passes_b,
            "effnet_label": eff_result["label"],
            "effnet_score": eff_result["score"],
            "vit_label": vit_result["label"],
            "vit_score": vit_result["score"],
            "comm_vit_label": comm_result["label"],
            "comm_vit_score": comm_result["score"],
            "sbi_label": sbi_result["label"],
            "sbi_score": sbi_result["score"],
            "hybrid_srm_label": srm_result["label"],
            "hybrid_srm_score": srm_result["score"],
            "pipeline_a_correct": (lbl_a == actual_label),
            "pipeline_b_correct": (lbl_b == actual_label),
            "router_correct_typology": (predicted_image_type == actual_category),
            "verdicts_match": (lbl_a == lbl_b),
        }
        results_records.append(record)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(benchmark_df):
            elapsed = time.perf_counter() - t_eval_start
            logger.info(f"Progress: [{idx + 1}/{len(benchmark_df)}] evaluated. Elapsed: {elapsed:.1f}s.")

    # Convert results to DataFrame
    eval_df = pd.DataFrame(results_records)

    # Save CSV
    csv_path = OUTPUT_DIR / "routing_comparison.csv"
    eval_df.to_csv(csv_path, index=False)
    logger.info(f"Saved evaluation CSV to {csv_path}")

    # Compute overall metrics
    y_true = eval_df["actual_label"].tolist()
    y_pred_a = eval_df["pipeline_a_verdict"].tolist()
    y_pred_b = eval_df["pipeline_b_verdict"].tolist()

    metrics_a = compute_binary_metrics(y_true, y_pred_a)
    metrics_b = compute_binary_metrics(y_true, y_pred_b)

    # Compute latency and efficiency
    mean_lat_a = float(eval_df["pipeline_a_latency_ms"].mean())
    mean_lat_b = float(eval_df["pipeline_b_latency_ms"].mean())
    total_passes_a = int(eval_df["pipeline_a_forward_passes"].sum())
    total_passes_b = int(eval_df["pipeline_b_forward_passes"].sum())
    mean_passes_a = float(eval_df["pipeline_a_forward_passes"].mean())
    mean_passes_b = float(eval_df["pipeline_b_forward_passes"].mean())

    latency_savings_pct = (1.0 - mean_lat_b / mean_lat_a) * 100.0 if mean_lat_a > 0 else 0.0
    passes_savings_pct = (1.0 - total_passes_b / total_passes_a) * 100.0 if total_passes_a > 0 else 0.0

    efficiency = {
        "pipeline_a": {
            "mean_latency_ms": round(mean_lat_a, 2),
            "total_detector_passes": total_passes_a,
            "mean_passes_per_image": round(mean_passes_a, 2),
        },
        "pipeline_b": {
            "mean_latency_ms": round(mean_lat_b, 2),
            "total_detector_passes": total_passes_b,
            "mean_passes_per_image": round(mean_passes_b, 2),
        },
        "savings": {
            "latency_reduction_pct": round(latency_savings_pct, 2),
            "passes_reduction_pct": round(passes_savings_pct, 2),
        },
    }

    # Per-category breakdown
    categories = ["REAL", "STYLEGAN_GAN", "DIFFUSION", "FACE_SWAP", "INPAINTING"]
    per_cat_metrics = {}

    for cat in categories:
        cat_df = eval_df[eval_df["category"] == cat]
        yt_cat = cat_df["actual_label"].tolist()
        yp_a_cat = cat_df["pipeline_a_verdict"].tolist()
        yp_b_cat = cat_df["pipeline_b_verdict"].tolist()

        m_a = compute_binary_metrics(yt_cat, yp_a_cat)
        m_b = compute_binary_metrics(yt_cat, yp_b_cat)

        router_correct = int(cat_df["router_correct_typology"].sum())
        router_acc = router_correct / len(cat_df) if len(cat_df) > 0 else 0.0

        per_cat_metrics[cat] = {
            "sample_count": len(cat_df),
            "actual_label": cat_df.iloc[0]["actual_label"] if len(cat_df) > 0 else "UNKNOWN",
            "router_accuracy": round(router_acc, 4),
            "router_correct_count": router_correct,
            "pipeline_a": {
                "strict_accuracy": round(m_a["strict_accuracy"], 4),
                "conclusive_accuracy": round(m_a["conclusive_accuracy"], 4),
                "conclusive_coverage": round(m_a["coverage"], 4),
                "predictions": m_a["pred_distribution"],
            },
            "pipeline_b": {
                "strict_accuracy": round(m_b["strict_accuracy"], 4),
                "conclusive_accuracy": round(m_b["conclusive_accuracy"], 4),
                "conclusive_coverage": round(m_b["coverage"], 4),
                "predictions": m_b["pred_distribution"],
            },
            "strict_accuracy_delta": round(m_b["strict_accuracy"] - m_a["strict_accuracy"], 4),
        }

    # Failure Analysis
    # 1. Categories where routing improves / degrades
    improving_cats = [c for c, m in per_cat_metrics.items() if m["strict_accuracy_delta"] > 0]
    degrading_cats = [c for c, m in per_cat_metrics.items() if m["strict_accuracy_delta"] < 0]
    neutral_cats = [c for c, m in per_cat_metrics.items() if m["strict_accuracy_delta"] == 0]

    # 2. Images incorrectly classified by the router
    router_errors = eval_df[~eval_df["router_correct_typology"]].copy()
    router_errors_list = []
    for _, r in router_errors.head(20).iterrows():
        router_errors_list.append({
            "image_index": int(r["image_index"]),
            "filename": r["filename"],
            "actual_category": r["category"],
            "router_predicted": r["router_predicted_type"],
            "router_raw": r["router_raw_type"],
            "confidence": float(r["router_confidence"]),
            "is_rejected": bool(r["router_is_rejected"]),
            "selected_specialists": r["selected_specialists"],
            "verdict_b": r["pipeline_b_verdict"],
        })

    # 3. Cases where selected specialist misses a fake image (False REAL in Pipeline B)
    fake_missed_b = eval_df[(eval_df["actual_label"] == "FAKE") & (eval_df["pipeline_b_verdict"] == "REAL")].copy()
    specialist_misses_list = []
    for _, r in fake_missed_b.head(20).iterrows():
        specialist_misses_list.append({
            "image_index": int(r["image_index"]),
            "filename": r["filename"],
            "category": r["category"],
            "router_predicted": r["router_predicted_type"],
            "selected_specialists": r["selected_specialists"],
            "pipeline_b_consensus": r["pipeline_b_consensus"],
            "pipeline_a_verdict": r["pipeline_a_verdict"],
        })

    # 4. Cases where Pipeline A succeeds but Pipeline B fails
    a_wins = eval_df[(eval_df["pipeline_a_correct"]) & (~eval_df["pipeline_b_correct"])].copy()
    a_wins_list = []
    for _, r in a_wins.head(20).iterrows():
        a_wins_list.append({
            "image_index": int(r["image_index"]),
            "filename": r["filename"],
            "category": r["category"],
            "actual_label": r["actual_label"],
            "router_predicted": r["router_predicted_type"],
            "selected_specialists": r["selected_specialists"],
            "verdict_a": r["pipeline_a_verdict"],
            "verdict_b": r["pipeline_b_verdict"],
        })

    # 5. Cases where Pipeline B succeeds but Pipeline A fails
    b_wins = eval_df[(~eval_df["pipeline_a_correct"]) & (eval_df["pipeline_b_correct"])].copy()
    b_wins_list = []
    for _, r in b_wins.head(20).iterrows():
        b_wins_list.append({
            "image_index": int(r["image_index"]),
            "filename": r["filename"],
            "category": r["category"],
            "actual_label": r["actual_label"],
            "router_predicted": r["router_predicted_type"],
            "selected_specialists": r["selected_specialists"],
            "verdict_a": r["pipeline_a_verdict"],
            "verdict_b": r["pipeline_b_verdict"],
        })

    failures_summary = {
        "improving_categories": improving_cats,
        "degrading_categories": degrading_cats,
        "neutral_categories": neutral_cats,
        "total_router_misclassifications": len(router_errors),
        "router_overall_accuracy": round(float(eval_df["router_correct_typology"].mean()), 4),
        "total_specialist_misses_b": len(fake_missed_b),
        "pipeline_a_succeeds_b_fails_count": len(a_wins),
        "pipeline_b_succeeds_a_fails_count": len(b_wins),
        "router_errors_sample": router_errors_list,
        "specialist_misses_sample": specialist_misses_list,
        "a_wins_sample": a_wins_list,
        "b_wins_sample": b_wins_list,
    }

    # Consolidated JSON bundle
    deliverable_json = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_dataset": {
            "name": "Frozen 250-Image Out-of-Distribution Benchmark",
            "path": str(BENCHMARK_DIR),
            "total_images": len(benchmark_df),
            "classes": categories,
            "images_per_class": 50,
        },
        "pipeline_a": {
            "name": "All-Models Consensus (5 Specialists)",
            "models": ["efficientnet", "vit", "community_vit", "sbi", "hybrid_srm"],
            "metrics": metrics_a,
            "efficiency": efficiency["pipeline_a"],
        },
        "pipeline_b": {
            "name": "DINOv2 Intelligent Routing",
            "backbone": "DINOv2-base (CLS 768-dim)",
            "routing_map": DEFAULT_SPECIALIST_ROUTING_MAP,
            "metrics": metrics_b,
            "efficiency": efficiency["pipeline_b"],
        },
        "efficiency_comparison": efficiency["savings"],
        "per_category_performance": per_cat_metrics,
        "failure_analysis": failures_summary,
    }

    json_path = OUTPUT_DIR / "routing_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(deliverable_json, f, indent=2)
    logger.info(f"Saved evaluation JSON to {json_path}")

    # Generate Plots
    generate_evaluation_plots(eval_df, per_cat_metrics, metrics_a, metrics_b, efficiency)

    # Generate Markdown Report
    generate_markdown_report(deliverable_json)

    logger.info("Evaluation complete! All deliverables generated successfully.")


def generate_evaluation_plots(eval_df, per_cat_metrics, metrics_a, metrics_b, efficiency):
    """Generate high-resolution comparative visualization charts."""
    # 1. Confusion Matrix Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 3x2 matrix for Pipeline A
    cm_a = np.array([
        [metrics_a["confusion_matrix_3x2"]["actual_REAL"]["pred_REAL"],
         metrics_a["confusion_matrix_3x2"]["actual_REAL"]["pred_FAKE"],
         metrics_a["confusion_matrix_3x2"]["actual_REAL"]["pred_INCONCLUSIVE"]],
        [metrics_a["confusion_matrix_3x2"]["actual_FAKE"]["pred_REAL"],
         metrics_a["confusion_matrix_3x2"]["actual_FAKE"]["pred_FAKE"],
         metrics_a["confusion_matrix_3x2"]["actual_FAKE"]["pred_INCONCLUSIVE"]]
    ])

    # 3x2 matrix for Pipeline B
    cm_b = np.array([
        [metrics_b["confusion_matrix_3x2"]["actual_REAL"]["pred_REAL"],
         metrics_b["confusion_matrix_3x2"]["actual_REAL"]["pred_FAKE"],
         metrics_b["confusion_matrix_3x2"]["actual_REAL"]["pred_INCONCLUSIVE"]],
        [metrics_b["confusion_matrix_3x2"]["actual_FAKE"]["pred_REAL"],
         metrics_b["confusion_matrix_3x2"]["actual_FAKE"]["pred_FAKE"],
         metrics_b["confusion_matrix_3x2"]["actual_FAKE"]["pred_INCONCLUSIVE"]]
    ])

    cols = ["Pred REAL", "Pred FAKE", "Pred INCONCLUSIVE"]
    rows = ["Actual REAL\n(50)", "Actual FAKE\n(200)"]

    for ax, cm, title in zip(axes, [cm_a, cm_b], ["Pipeline A (All 5 Specialists)", "Pipeline B (DINOv2 Routed)"]):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=200)
        ax.set_xticks(range(3))
        ax.set_xticklabels(cols, fontsize=10, fontweight="bold")
        ax.set_yticks(range(2))
        ax.set_yticklabels(rows, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        for i in range(2):
            for j in range(3):
                val = cm[i, j]
                pct = val / (50 if i == 0 else 200) * 100
                color = "white" if val > 90 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.suptitle("Confusion Matrix Comparison (with Explicit Inconclusive State)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix_comparison.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

    # 2. Per-Category Accuracy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    categories = ["REAL", "STYLEGAN_GAN", "DIFFUSION", "FACE_SWAP", "INPAINTING"]
    x = np.arange(len(categories))
    width = 0.35

    acc_a = [per_cat_metrics[c]["pipeline_a"]["strict_accuracy"] * 100 for c in categories]
    acc_b = [per_cat_metrics[c]["pipeline_b"]["strict_accuracy"] * 100 for c in categories]

    rects1 = ax.bar(x - width/2, acc_a, width, label="Pipeline A (All Models)", color="#2563eb", alpha=0.9)
    rects2 = ax.bar(x + width/2, acc_b, width, label="Pipeline B (DINOv2 Routed)", color="#10b981", alpha=0.9)

    ax.set_ylabel("Strict Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Manipulation-Category Accuracy: All-Models vs DINOv2 Routing", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="upper left")

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    cat_plot_path = OUTPUT_DIR / "per_category_comparison.png"
    plt.savefig(cat_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-category plot to {cat_plot_path}")

    # 3. Efficiency Comparison Plot (Latency vs Forward Passes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Latency comparison
    latencies = [efficiency["pipeline_a"]["mean_latency_ms"], efficiency["pipeline_b"]["mean_latency_ms"]]
    labels = ["Pipeline A\n(All 5 Specialists)", "Pipeline B\n(DINOv2 Routed)"]
    colors = ["#ef4444", "#3b82f6"]

    bars1 = ax1.bar(labels, latencies, color=colors, width=0.5, alpha=0.85)
    ax1.set_ylabel("Mean Latency per Image (ms)", fontsize=10, fontweight="bold")
    ax1.set_title(f"Inference Latency\n({efficiency['savings']['latency_reduction_pct']:.1f}% Reduction)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(latencies) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f} ms", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Passes comparison
    passes = [efficiency["pipeline_a"]["mean_passes_per_image"], efficiency["pipeline_b"]["mean_passes_per_image"]]
    bars2 = ax2.bar(labels, passes, color=colors, width=0.5, alpha=0.85)
    ax2.set_ylabel("Mean Forward Passes per Image", fontsize=10, fontweight="bold")
    ax2.set_title(f"Detector Forward Passes\n({efficiency['savings']['passes_reduction_pct']:.1f}% Reduction)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 6)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f} passes", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Computational Efficiency Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    eff_plot_path = OUTPUT_DIR / "efficiency_comparison.png"
    plt.savefig(eff_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved efficiency plot to {eff_plot_path}")


def generate_markdown_report(data: Dict[str, Any]):
    """Generate professional, metric-dense markdown comparison report."""
    pa = data["pipeline_a"]
    pb = data["pipeline_b"]
    ma = pa["metrics"]
    mb = pb["metrics"]
    eff = data["efficiency_comparison"]
    cats = data["per_category_performance"]
    fail = data["failure_analysis"]

    report_content = f"""# Comparative Evaluation Report: All-Models vs. DINOv2 Selective Routing

> **Evaluation Objective**: Determine empirically whether intelligent DINOv2 selective routing maintains forensic detection efficacy while reducing computational cost compared to exhaustive all-specialist consensus.  
> **Evaluation Scope**: Identical 250-image frozen out-of-distribution benchmark (`C:\\Users\\abijith\\Downloads\\DeepFake-Forensics-Evaluation\\evaluation_dataset`).  
> **Execution Mode**: Production consensus logic (`compute_combined_result()`, threshold=0.60), zero benchmark modification, zero test-time training or tuning.

---

## 1. Executive Summary & Verdict

| Metric | Pipeline A (All 5 Specialists) | Pipeline B (DINOv2 Selective Routing) | Delta (B - A) | Relative Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Strict Accuracy (All N=250)** | **{ma['strict_accuracy']*100:.2f}%** | **{mb['strict_accuracy']*100:.2f}%** | **{(mb['strict_accuracy']-ma['strict_accuracy'])*100:+.2f}%** | {"Improvement" if mb['strict_accuracy']>ma['strict_accuracy'] else ("Neutral" if mb['strict_accuracy']==ma['strict_accuracy'] else "Degradation")} |
| **Conclusive Accuracy** | {ma['conclusive_accuracy']*100:.2f}% | {mb['conclusive_accuracy']*100:.2f}% | {(mb['conclusive_accuracy']-ma['conclusive_accuracy'])*100:+.2f}% | Evaluated on decisive outputs |
| **Coverage (Decisive Rate)** | {ma['coverage']*100:.2f}% | {mb['coverage']*100:.2f}% | {(mb['coverage']-ma['coverage'])*100:+.2f}% | Abstention rate: {mb['inconclusive_count']/250*100:.1f}% |
| **Fake Precision** | {ma['fake_metrics']['precision']*100:.2f}% | {mb['fake_metrics']['precision']*100:.2f}% | {(mb['fake_metrics']['precision']-ma['fake_metrics']['precision'])*100:+.2f}% | Positive predictive value |
| **Fake Recall (Sensitivity)** | {ma['fake_metrics']['recall']*100:.2f}% | {mb['fake_metrics']['recall']*100:.2f}% | {(mb['fake_metrics']['recall']-ma['fake_metrics']['recall'])*100:+.2f}% | Detected out of 200 fakes |
| **Fake F1-Score** | {ma['fake_metrics']['f1']*100:.2f}% | {mb['fake_metrics']['f1']*100:.2f}% | {(mb['fake_metrics']['f1']-ma['fake_metrics']['f1'])*100:+.2f}% | Harmonic mean on fakes |
| **Real Precision** | {ma['real_metrics']['precision']*100:.2f}% | {mb['real_metrics']['precision']*100:.2f}% | {(mb['real_metrics']['precision']-ma['real_metrics']['precision'])*100:+.2f}% | Specificity accuracy |
| **Real Recall (Specificity)** | {ma['real_metrics']['recall']*100:.2f}% | {mb['real_metrics']['recall']*100:.2f}% | {(mb['real_metrics']['recall']-ma['real_metrics']['recall'])*100:+.2f}% | Detected out of 50 reals |
| **Macro F1-Score** | **{ma['macro_f1']*100:.2f}%** | **{mb['macro_f1']*100:.2f}%** | **{(mb['macro_f1']-ma['macro_f1'])*100:+.2f}%** | Unweighted class mean |
| **Balanced Accuracy** | {ma['balanced_accuracy']*100:.2f}% | {mb['balanced_accuracy']*100:.2f}% | {(mb['balanced_accuracy']-ma['balanced_accuracy'])*100:+.2f}% | Mean of Real & Fake recall |
| **False REAL Predictions** | **{ma['false_real_count']} ({ma['false_real_rate']*100:.1f}%)** | **{mb['false_real_count']} ({mb['false_real_rate']*100:.1f}%)** | **{mb['false_real_count']-ma['false_real_count']:+d}** | Critical security hazard |
| **Mean Latency per Image** | **{pa['efficiency']['mean_latency_ms']:.1f} ms** | **{pb['efficiency']['mean_latency_ms']:.1f} ms** | **{pb['efficiency']['mean_latency_ms']-pa['efficiency']['mean_latency_ms']:+.1f} ms** | **{eff['latency_reduction_pct']:.1f}% Faster** |
| **Forward Passes per Image** | **{pa['efficiency']['mean_passes_per_image']:.2f}** | **{pb['efficiency']['mean_passes_per_image']:.2f}** | **{pb['efficiency']['mean_passes_per_image']-pa['efficiency']['mean_passes_per_image']:+.2f}** | **{eff['passes_reduction_pct']:.1f}% Compute Saved** |

### Bottom-Line Forensic Assessment:
**Is intelligent routing useful?**
- **Computational Efficiency**: DINOv2 selective routing achieves a **{eff['passes_reduction_pct']:.1f}% reduction in detector forward passes** (from {pa['efficiency']['mean_passes_per_image']:.1f} to {pb['efficiency']['mean_passes_per_image']:.2f} per face) and cuts average inference time by **{eff['latency_reduction_pct']:.1f}%** ({pb['efficiency']['mean_latency_ms']:.1f} ms vs {pa['efficiency']['mean_latency_ms']:.1f} ms).
- **Detection Performance**: Strict accuracy changed by **{(mb['strict_accuracy']-ma['strict_accuracy'])*100:+.2f}%** ({mb['strict_accuracy']*100:.2f}% vs {ma['strict_accuracy']*100:.2f}%), while Macro F1 changed by **{(mb['macro_f1']-ma['macro_f1'])*100:+.2f}%**.
- **Security Posture (False Reals)**: Pipeline B produced {mb['false_real_count']} false REAL predictions vs Pipeline A's {ma['false_real_count']} ({mb['false_real_count']-ma['false_real_count']:+d}).

---

## 2. Verdict Distribution & Confusion Matrices

### REAL / FAKE / INCONCLUSIVE Distribution

| Verdict | Pipeline A (All Models) | Pipeline A (%) | Pipeline B (DINOv2 Routed) | Pipeline B (%) |
| :--- | :---: | :---: | :---: | :---: |
| **REAL** | {ma['pred_distribution']['REAL']} | {ma['pred_distribution']['REAL_pct']:.1f}% | {mb['pred_distribution']['REAL']} | {mb['pred_distribution']['REAL_pct']:.1f}% |
| **FAKE** | {ma['pred_distribution']['FAKE']} | {ma['pred_distribution']['FAKE_pct']:.1f}% | {mb['pred_distribution']['FAKE']} | {mb['pred_distribution']['FAKE_pct']:.1f}% |
| **INCONCLUSIVE** | {ma['pred_distribution']['INCONCLUSIVE']} | {ma['pred_distribution']['INCONCLUSIVE_pct']:.1f}% | {mb['pred_distribution']['INCONCLUSIVE']} | {mb['pred_distribution']['INCONCLUSIVE_pct']:.1f}% |
| **Total** | {ma['total_samples']} | 100.0% | {mb['total_samples']} | 100.0% |

### Detailed 3x2 Confusion Matrices (with Inconclusive Abstentions)

#### Pipeline A (All 5 Specialists)
```text
                  Predicted REAL    Predicted FAKE    Predicted INCONCLUSIVE    Total
Actual REAL             {ma['confusion_matrix_3x2']['actual_REAL']['pred_REAL']:<18} {ma['confusion_matrix_3x2']['actual_REAL']['pred_FAKE']:<17} {ma['confusion_matrix_3x2']['actual_REAL']['pred_INCONCLUSIVE']:<25} 50
Actual FAKE             {ma['confusion_matrix_3x2']['actual_FAKE']['pred_REAL']:<18} {ma['confusion_matrix_3x2']['actual_FAKE']['pred_FAKE']:<17} {ma['confusion_matrix_3x2']['actual_FAKE']['pred_INCONCLUSIVE']:<25} 200
```

#### Pipeline B (DINOv2 Selective Routing)
```text
                  Predicted REAL    Predicted FAKE    Predicted INCONCLUSIVE    Total
Actual REAL             {mb['confusion_matrix_3x2']['actual_REAL']['pred_REAL']:<18} {mb['confusion_matrix_3x2']['actual_REAL']['pred_FAKE']:<17} {mb['confusion_matrix_3x2']['actual_REAL']['pred_INCONCLUSIVE']:<25} 50
Actual FAKE             {mb['confusion_matrix_3x2']['actual_FAKE']['pred_REAL']:<18} {mb['confusion_matrix_3x2']['actual_FAKE']['pred_FAKE']:<17} {mb['confusion_matrix_3x2']['actual_FAKE']['pred_INCONCLUSIVE']:<25} 200
```

---

## 3. Per-Manipulation-Category Performance Breakdown

| Category | Actual Label | Samples | Router Typology Acc | Pipeline A Strict Acc | Pipeline B Strict Acc | Accuracy Delta | Pipeline A (R / F / Inc) | Pipeline B (R / F / Inc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for c, d in cats.items():
        pa_p = d["pipeline_a"]["predictions"]
        pb_p = d["pipeline_b"]["predictions"]
        report_content += (
            f"| **{c}** | {d['actual_label']} | {d['sample_count']} | "
            f"{d['router_accuracy']*100:.1f}% ({d['router_correct_count']}/{d['sample_count']}) | "
            f"{d['pipeline_a']['strict_accuracy']*100:.1f}% | "
            f"{d['pipeline_b']['strict_accuracy']*100:.1f}% | "
            f"**{d['strict_accuracy_delta']*100:+.1f}%** | "
            f"{pa_p['REAL']} / {pa_p['FAKE']} / {pa_p['INCONCLUSIVE']} | "
            f"{pb_p['REAL']} / {pb_p['FAKE']} / {pb_p['INCONCLUSIVE']} |\n"
        )

    report_content += f"""
---

## 4. Routing Failure & Divergence Analysis

### Categories Where Routing Changes Detection Efficacy
- **Categories where Routing Improves Efficacy**: {', '.join(fail['improving_categories']) if fail['improving_categories'] else 'None'}
- **Categories where Routing Degrades Efficacy**: {', '.join(fail['degrading_categories']) if fail['degrading_categories'] else 'None'}
- **Neutral Categories**: {', '.join(fail['neutral_categories']) if fail['neutral_categories'] else 'None'}

### Router Typology Classification
- **Router Overall Accuracy on Benchmark**: **{fail['router_overall_accuracy']*100:.2f}%**
- **Total Router Typology Misclassifications**: {fail['total_router_misclassifications']} out of 250 ({fail['total_router_misclassifications']/250*100:.1f}%)

#### Exemplar Router Misclassifications:
| Image Index | Filename | Ground Truth Category | Router Prediction | Confidence | Assigned Specialists | Pipeline B Verdict |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
"""
    for item in fail["router_errors_sample"][:8]:
        report_content += f"| {item['image_index']} | `{item['filename']}` | {item['actual_category']} | **{item['router_predicted']}** | {item['confidence']:.4f} | `{item['selected_specialists']}` | {item['verdict_b']} |\n"

    report_content += f"""
### Specialist Misses (False REALs Under Selective Routing)
Total fake images where the selected specialist models mistakenly voted REAL: **{fail['total_specialist_misses_b']} images**.
When an image is misrouted or when the designated specialist has an out-of-domain false-real blind spot, the absence of the other 4 models leaves no dissenting votes to force an INCONCLUSIVE split.

#### Exemplar Specialist Misses:
| Image Index | Filename | Manipulation Category | Router Typology | Selected Specialists | Pipeline B Verdict | Pipeline A Verdict |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
"""
    for item in fail["specialist_misses_sample"][:8]:
        report_content += f"| {item['image_index']} | `{item['filename']}` | {item['category']} | {item['router_predicted']} | `{item['selected_specialists']}` | **{item['pipeline_b_consensus']}** | {item['pipeline_a_verdict']} |\n"

    report_content += f"""
### Pipeline Divergence Cases
- **Pipeline A Succeeds, Pipeline B Fails**: **{fail['pipeline_a_succeeds_b_fails_count']} cases** (All-models consensus was correct, but selective routing failed)
- **Pipeline B Succeeds, Pipeline A Fails**: **{fail['pipeline_b_succeeds_a_fails_count']} cases** (Selective routing successfully removed distracting noisy models)

---

## 5. Artifacts and Visualization Assets

- **Full Evaluation Metrics (JSON)**: [`routing_comparison.json`](file:///{str(OUTPUT_DIR / 'routing_comparison.json').replace('\\', '/')})
- **Per-Image Predictions Log (CSV)**: [`routing_comparison.csv`](file:///{str(OUTPUT_DIR / 'routing_comparison.csv').replace('\\', '/')})
- **Confusion Matrix Plot**: [`confusion_matrix_comparison.png`](file:///{str(OUTPUT_DIR / 'confusion_matrix_comparison.png').replace('\\', '/')})
- **Per-Category Performance Plot**: [`per_category_comparison.png`](file:///{str(OUTPUT_DIR / 'per_category_comparison.png').replace('\\', '/')})
- **Computational Efficiency Plot**: [`efficiency_comparison.png`](file:///{str(OUTPUT_DIR / 'efficiency_comparison.png').replace('\\', '/')})
"""

    report_path = OUTPUT_DIR / "routing_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Saved evaluation markdown report to {report_path}")


if __name__ == "__main__":
    run_evaluation()
