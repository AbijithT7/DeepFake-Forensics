"""Tests for external-dataset discovery, metrics, and artifact generation."""
from __future__ import annotations

import unittest
from unittest.mock import mock_open, patch
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.evaluate_external_dataset import ExternalDatasetEvaluator, iter_dataset, metrics


class FakeEngine:
    device = "cpu"

    def predict_image(self, path, **_kwargs):
        label = "REAL" if "real" in Path(path).name else "FAKE"
        self.crop_artifacts = {1: SimpleNamespace(is_usable=True, crop_pil=str(path))}
        return SimpleNamespace(final_prediction=SimpleNamespace(label=label, score=.9))


class FakeViT:
    def predict(self, crop):
        label = "REAL" if "real" in Path(crop).name else "FAKE"
        return {"label": label, "score": .8}


class TestExternalEvaluation(unittest.TestCase):
    def test_mock_dataset_writes_all_artifacts_and_categories(self):
        mock_items = [
            {"path": "mock/real/real.jpg", "actual_label": "REAL", "category": "real"},
            {"path": "mock/fake/diffusion/fake.jpg", "actual_label": "FAKE", "category": "diffusion"},
            {"path": "mock/fake/swap/fake2.png", "actual_label": "FAKE", "category": "swap"},
        ]
        with patch("src.evaluation.evaluate_external_dataset.iter_dataset", return_value=mock_items), \
             patch.object(Path, "mkdir"), patch.object(Path, "open", mock_open()), \
             patch.object(Path, "write_text") as write_text:
            evaluator = ExternalDatasetEvaluator(engine=FakeEngine(), vit_detector=FakeViT())
            summary = evaluator.run(Path("mock"), Path("output"))
            self.assertEqual(summary["images"], 3)
            self.assertEqual(set(summary["by_fake_category"]), {"diffusion", "swap"})
            self.assertEqual(summary["overall"]["fusion"]["coverage_percent"], 100.0)
            self.assertGreaterEqual(write_text.call_count, 2)

    def test_metrics_excludes_inconclusive_from_classification_scores(self):
        rows = [{"actual_label": "REAL", "prediction": "REAL"}, {"actual_label": "FAKE", "prediction": "INCONCLUSIVE"}]
        result = metrics(rows, "prediction")
        self.assertEqual(result["coverage_percent"], 50.0)
        self.assertEqual(result["inconclusive_count"], 1)
        self.assertEqual(result["accuracy_percent"], 100.0)

    def test_dataset_requires_real_and_fake_roots(self):
        with patch.object(Path, "is_dir", return_value=False):
            with self.assertRaises(ValueError): list(iter_dataset(Path("mock")))


if __name__ == "__main__":
    unittest.main()
