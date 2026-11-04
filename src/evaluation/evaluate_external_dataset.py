"""Evaluate the production face pipeline on an external ``real/`` / ``fake/`` dataset.

Run: ``python -m src.evaluation.evaluate_external_dataset DATASET_DIR``
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.image_detection.inference import DEFAULT_UNCERTAINTY_THRESHOLD, DeepFakeInferenceEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_dataset(dataset_dir: Path) -> Iterable[Dict[str, str]]:
    """Yield images from ``real/`` and ``fake[/generation-type]/`` folders."""
    for label, root_name in (("REAL", "real"), ("FAKE", "fake")):
        root = dataset_dir / root_name
        if not root.is_dir():
            raise ValueError(f"Dataset must contain a '{root_name}/' folder: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            relative_parent = path.relative_to(root).parent
            category = "real" if label == "REAL" else (relative_parent.parts[0] if relative_parent.parts else "fake")
            yield {"path": str(path), "actual_label": label, "category": category}


def fuse(efficientnet: Dict[str, Any], vit: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Mirror the production policy: no score averaging; both votes must agree."""
    e_score, v_score = efficientnet.get("score"), vit.get("score")
    if e_score is None or v_score is None:
        return {"label": "INCONCLUSIVE", "score": None, "reason": "model_unavailable_or_no_face"}
    score = round(min(float(e_score), float(v_score)), 4)
    if e_score < threshold or v_score < threshold:
        return {"label": "INCONCLUSIVE", "score": score, "reason": "low_confidence"}
    if efficientnet["label"] != vit["label"]:
        return {"label": "INCONCLUSIVE", "score": score, "reason": "model_disagreement"}
    return {"label": efficientnet["label"], "score": score, "reason": None}


def metrics(records: List[Dict[str, Any]], prediction_key: str) -> Dict[str, Any]:
    """Metrics use definitive predictions; coverage reports the excluded cases."""
    definitive = [r for r in records if r[prediction_key] in {"REAL", "FAKE"}]
    tp = sum(r["actual_label"] == "FAKE" and r[prediction_key] == "FAKE" for r in definitive)
    tn = sum(r["actual_label"] == "REAL" and r[prediction_key] == "REAL" for r in definitive)
    fp = sum(r["actual_label"] == "REAL" and r[prediction_key] == "FAKE" for r in definitive)
    fn = sum(r["actual_label"] == "FAKE" and r[prediction_key] == "REAL" for r in definitive)
    total = len(records)
    count = len(definitive)
    return {
        "total_images": total, "definitive_predictions": count,
        "inconclusive_count": total - count,
        "coverage_percent": round(100 * count / total, 2) if total else 0.0,
        "accuracy_percent": round(100 * (tp + tn) / count, 2) if count else 0.0,
        "precision_percent": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
        "recall_percent": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0,
        "f1_percent": round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": {"labels": ["REAL", "FAKE"], "matrix": [[tn, fp], [fn, tp]]},
    }


class ExternalDatasetEvaluator:
    def __init__(self, engine=None, vit_detector=None, threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD):
        self.engine = engine or DeepFakeInferenceEngine(auto_load=True)
        if vit_detector is None:
            from src.image_detection.vit_detector import ViTDetector
            vit_detector = ViTDetector(device=self.engine.device)
        self.vit_detector = vit_detector
        self.threshold = threshold

    def evaluate_image(self, path: Path) -> Dict[str, Any]:
        result = self.engine.predict_image(path, uncertainty_threshold=self.threshold, include_gradcam=False)
        efficientnet = {"label": result.final_prediction.label, "score": result.final_prediction.score}
        usable = [a for a in self.engine.crop_artifacts.values() if a.is_usable]
        if not usable:
            vit = {"label": "INCONCLUSIVE", "score": None}
        else:
            votes = [self.vit_detector.predict(artifact.crop_pil) for artifact in usable]
            labels = {vote["label"] for vote in votes}
            vit = {"label": votes[0]["label"] if len(labels) == 1 else "INCONCLUSIVE",
                   "score": min(vote["score"] for vote in votes)}
        final = fuse(efficientnet, vit, self.threshold)
        return {"efficientnet": efficientnet, "vit": vit, "final": final}

    def run(self, dataset_dir: Path, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for item in iter_dataset(dataset_dir):
            outcome = self.evaluate_image(Path(item["path"]))
            records.append({**item,
                "efficientnet_prediction": outcome["efficientnet"]["label"], "efficientnet_score": outcome["efficientnet"]["score"],
                "vit_prediction": outcome["vit"]["label"], "vit_score": outcome["vit"]["score"],
                "final_prediction": outcome["final"]["label"], "final_score": outcome["final"]["score"], "final_reason": outcome["final"]["reason"]})
        fields = list(records[0]) if records else ["path", "actual_label", "category"]
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
        grouped = defaultdict(list)
        for record in records: grouped[record["category"]].append(record)
        summary = {"dataset": str(dataset_dir), "threshold": self.threshold, "images": len(records),
                   "overall": {"efficientnet": metrics(records, "efficientnet_prediction"), "vit": metrics(records, "vit_prediction"), "fusion": metrics(records, "final_prediction")},
                   "by_fake_category": {name: {"efficientnet": metrics(rows, "efficientnet_prediction"), "vit": metrics(rows, "vit_prediction"), "fusion": metrics(rows, "final_prediction")} for name, rows in grouped.items() if name != "real"}}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_markdown(summary, output_dir / "report.md")
        return summary

    @staticmethod
    def _write_markdown(summary: Dict[str, Any], path: Path) -> None:
        lines = ["# External Dataset Evaluation", "", f"Images: {summary['images']}", "", "## Overall", "", "| Model | Accuracy | Precision | Recall | F1 | Coverage | Inconclusive |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, values in summary["overall"].items():
            lines.append(f"| {name} | {values['accuracy_percent']:.2f}% | {values['precision_percent']:.2f}% | {values['recall_percent']:.2f}% | {values['f1_percent']:.2f}% | {values['coverage_percent']:.2f}% | {values['inconclusive_count']} |")
        for category, model_results in summary["by_fake_category"].items():
            lines.extend(["", f"## Fake category: {category}", "", "| Model | Accuracy | F1 | Coverage | Inconclusive |", "|---|---:|---:|---:|---:|"])
            for name, values in model_results.items(): lines.append(f"| {name} | {values['accuracy_percent']:.2f}% | {values['f1_percent']:.2f}% | {values['coverage_percent']:.2f}% | {values['inconclusive_count']} |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_evaluation"))
    args = parser.parse_args()
    summary = ExternalDatasetEvaluator().run(args.dataset_dir, args.output_dir)
    print(f"Evaluated {summary['images']} image(s). Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
