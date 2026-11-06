"""
API Integration Tests for DeepFake Forensics FastAPI Backend.

Verifies:
1. GET /health endpoint (backend and model loading status).
2. GET /models/info endpoint (metadata and specifications).
3. POST /predict/image with valid real image (REAL).
4. POST /predict/image with valid AI-generated fake image (FAKE).
5. POST /predict/image with no-face image (status INCONCLUSIVE).
6. POST /predict/image with multiple-face image (separate per-face analysis).
7. POST /predict/image with invalid file formats (.pdf).
8. POST /predict/image with corrupted bytes (HTTP 400).
9. POST /predict/image with empty file (HTTP 400).
10. POST /predict/image with custom uncertainty threshold.
11. GET / and /static frontend assets serving.
12. POST /predict/image with Grad-CAM payload generation.
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.api.app import app


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

        cls.real_image_path = PROJECT_ROOT / "tests" / "sample_image_real.jpg"
        cls.fake_image_path = PROJECT_ROOT / "tests" / "sample_image_fake.jpg"
        cls.noface_image_path = PROJECT_ROOT / "tests" / "sample_landscape_noface.jpg"
        cls.multiface_image_path = PROJECT_ROOT / "tests" / "sample_multiple_faces.jpg"

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_health_endpoint(self):
        """Test GET /health returns operational status and model readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["primary_model_loaded"])
        self.assertTrue(data["hybrid_srm_loaded"])
        self.assertTrue(data["router_loaded"])
        self.assertEqual(data["router_mode"], "shadow")
        self.assertEqual(data["primary_model"], "EfficientNet-B0")
        self.assertIn("SCRFD", data["face_detector"])
        self.assertIn("device", data)
        self.assertIn("cuda_available", data)

    def test_02_models_info_endpoint(self):
        """Test GET /models/info returns detailed model specifications."""
        response = self.client.get("/models/info")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["default_model"], "efficientnet")
        self.assertEqual(data["primary_model"]["name"], "EfficientNet-B0")
        self.assertEqual(data["primary_model"]["classes"], ["FAKE", "REAL"])
        self.assertEqual(data["primary_model"]["score_type"], "uncalibrated_model_score")
        self.assertIn("hybrid_srm_model", data)
        self.assertEqual(data["hybrid_srm_model"]["name"], "Hybrid SRM Detector (Phase 2)")
        self.assertEqual(data["hybrid_srm_model"]["status"], "loaded")
        self.assertIn("router", data)
        self.assertEqual(data["router"]["status"], "loaded")
        self.assertEqual(data["router"]["mode"], "shadow")
        self.assertIn("routing_strategy", data["router"])

    def test_03_predict_real_image(self):
        """Test POST /predict/image with a valid real sample."""
        with open(self.real_image_path, "rb") as f:
            response = self.client.post(
                "/predict/image",
                files={"file": ("real_face.jpg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(data["faces_detected"], 1)
        self.assertEqual(data["model_name"], "EfficientNet-B0")

        pred = data["predictions"][0]
        self.assertEqual(pred["face_id"], 1)
        self.assertEqual(pred["label"], "REAL")
        self.assertEqual(pred["score_type"], "uncalibrated_model_score")
        self.assertGreaterEqual(pred["efficientnet"]["score"], 0.90)
        self.assertGreaterEqual(pred["score"], 0.60)
        self.assertIsNotNone(pred["box"])
        self.assertIn("hybrid_srm", pred)
        self.assertEqual(pred["hybrid_srm"]["label"], "REAL")
        self.assertTrue(pred["hybrid_srm"]["loaded"])

    def test_04_predict_fake_image(self):
        """Test POST /predict/image with a valid fake sample."""
        with open(self.fake_image_path, "rb") as f:
            response = self.client.post(
                "/predict/image",
                files={"file": ("fake_face.jpg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "inconclusive")
        self.assertGreaterEqual(data["faces_detected"], 1)
        pred = data["predictions"][0]
        self.assertEqual(pred["label"], "INCONCLUSIVE")
        self.assertEqual(pred["fusion"]["reason"], "model_disagreement")
        self.assertEqual(pred["score_type"], "uncalibrated_model_score")
        self.assertIn("hybrid_srm", pred)
        self.assertEqual(pred["hybrid_srm"]["label"], "FAKE")
        self.assertTrue(pred["hybrid_srm"]["loaded"])

    def test_05_predict_no_face_image(self):
        """Test POST /predict/image on a landscape with no faces returns INCONCLUSIVE."""
        with open(self.noface_image_path, "rb") as f:
            response = self.client.post(
                "/predict/image",
                files={"file": ("landscape.jpg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "inconclusive")
        self.assertEqual(data["faces_detected"], 0)
        self.assertIn("No human faces detected", data["message"])

        pred = data["predictions"][0]
        self.assertEqual(pred["label"], "INCONCLUSIVE")
        self.assertEqual(pred["reason"], "no_face_detected")
        self.assertIsNone(pred["score"])

    def test_06_predict_multiple_faces(self):
        """Test POST /predict/image on multi-face image analyzes each face."""
        with open(self.multiface_image_path, "rb") as f:
            response = self.client.post(
                "/predict/image",
                files={"file": ("two_faces.jpg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["faces_detected"], 2)
        self.assertEqual(len(data["predictions"]), 2)

        p1, p2 = data["predictions"][0], data["predictions"][1]
        self.assertEqual(p1["face_id"], 1)
        self.assertEqual(p2["face_id"], 2)
        self.assertNotEqual(p1["box"], p2["box"])

    def test_07_invalid_file_extension(self):
        """Test POST /predict/image rejects unsupported extensions."""
        response = self.client.post(
            "/predict/image",
            files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.json()["detail"])

    def test_08_corrupt_image_bytes(self):
        """Test POST /predict/image returns HTTP 400 for unreadable image bytes."""
        response = self.client.post(
            "/predict/image",
            files={"file": ("corrupt.jpg", b"This is not a jpeg image content", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid or unsupported image file", response.json()["detail"])

    def test_09_empty_file_upload(self):
        """Test POST /predict/image returns HTTP 400 for empty upload."""
        response = self.client.post(
            "/predict/image",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty (0 bytes)", response.json()["detail"])

    def test_10_custom_uncertainty_threshold(self):
        """Test POST /predict/image respects custom uncertainty_threshold."""
        with open(self.real_image_path, "rb") as f:
            response = self.client.post(
                "/predict/image?uncertainty_threshold=0.99999",
                files={"file": ("real_face.jpg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uncertainty_threshold"], 0.99999)
        pred = data["predictions"][0]
        if pred["score"] < 0.99999:
            self.assertEqual(pred["label"], "INCONCLUSIVE")
            self.assertEqual(pred["reason"], "low_confidence")

    def test_11_invalid_query_parameters(self):
        """Test invalid query parameters produce 422 Unprocessable Entity."""
        with open(self.real_image_path, "rb") as f:
            # Threshold out of range (< 0.50)
            res = self.client.post(
                "/predict/image?uncertainty_threshold=0.2",
                files={"file": ("face.jpg", f, "image/jpeg")},
            )
            self.assertEqual(res.status_code, 422)

    def test_12_frontend_static_serving(self):
        """Test that the web frontend UI and static assets are served properly."""
        resp_root = self.client.get("/")
        self.assertEqual(resp_root.status_code, 200)
        self.assertIn("DeepFake Forensics", resp_root.text)
        self.assertIn("styles.css", resp_root.text)
        self.assertIn("app.js", resp_root.text)

        resp_css = self.client.get("/static/styles.css")
        self.assertEqual(resp_css.status_code, 200)
        self.assertIn("outcome-badge", resp_css.text)

        resp_js = self.client.get("/static/app.js")
        self.assertEqual(resp_js.status_code, 200)
        self.assertIn("API_BASE_URL", resp_js.text)

    def test_13_gradcam_payload_generation(self):
        """Test that POST /predict/image generates Grad-CAM spatial attribution artifacts."""
        with open(self.real_image_path, "rb") as f:
            response = self.client.post(
                "/predict/image?include_gradcam=true",
                files={"file": ("real_face.jpg", f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["faces_detected"], 1)

        pred = data["predictions"][0]
        self.assertTrue(pred["face_crop_base64"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(pred["gradcam_heatmap_base64"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(pred["gradcam_overlay_base64"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
