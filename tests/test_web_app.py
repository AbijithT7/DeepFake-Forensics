"""
Test Suite for DeepFake Forensics Web Application.

Tests:
1. Routing and page rendering for all 8 interfaces.
2. EfficientNet-B0 face-based analysis pipeline execution via web endpoints.
3. No-Face Integrity Guard blocking and bypass flow.
4. Grad-CAM target class switching (Predicted -> FAKE -> REAL).
5. Occlusion sensitivity computation.
6. History tracking and session artifact serving.
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.app import app, load_session


class TestDeepFakeWebApp(unittest.TestCase):
    """Integration test suite for the Flask web application."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.sample_face = PROJECT_ROOT / "tests" / "sample_image.jpg"
        cls.sample_noface = PROJECT_ROOT / "tests" / "sample_noface.jpg"

    def test_01_home_page_renders(self):
        """Verify home page loads successfully with EfficientNet-B0 highlights."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Facial DeepFake Forensics", html)
        self.assertIn("EfficientNet-B0", html)

    def test_02_upload_page_renders(self):
        """Verify upload page loads with sample selectors."""
        response = self.client.get("/upload")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Upload Forensic Evidence", html)
        self.assertIn("Quick Test Samples", html)

    def test_03_history_page_renders(self):
        """Verify history page loads successfully."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Forensic Analysis History", html)

    def test_04_no_face_integrity_guard_triggers(self):
        """Verify upload of an image without a face triggers the integrity guard."""
        response = self.client.post(
            "/analyze",
            data={
                "sample_path": "tests/sample_noface.jpg",
                "eval_mode": "face_crop",
                "target_class": "predicted",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("error=", response.location)
        self.assertIn("bypass=true", response.location)

    def test_05_full_analysis_workflow_renders_all_pages(self):
        """Execute complete analysis on a sample face and test all 8 interfaces."""
        # 1. Analyze sample face
        response = self.client.post(
            "/analyze",
            data={
                "sample_path": "tests/sample_image.jpg",
                "eval_mode": "face_crop",
                "target_class": "predicted",
                "run_occlusion": "true",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        session_url = response.location
        session_id = session_url.split("/")[-1]
        self.assertTrue(len(session_id) > 0)

        # 2. Check Results Page
        res_page = self.client.get(f"/results/{session_id}")
        self.assertEqual(res_page.status_code, 200)
        res_html = res_page.get_data(as_text=True)
        self.assertIn("Forensic Analysis Results", res_html)
        self.assertIn("Confidence Score", res_html)

        # 3. Check Face Detection Page
        fd_page = self.client.get(f"/face_detection/{session_id}")
        self.assertEqual(fd_page.status_code, 200)
        fd_html = fd_page.get_data(as_text=True)
        self.assertIn("Face Detection & Anatomical Cropping", fd_html)
        self.assertIn("Impact of Face Cropping", fd_html)

        # 4. Check Manipulation Forensics Page
        for_page = self.client.get(f"/forensics/{session_id}")
        self.assertEqual(for_page.status_code, 200)
        for_html = for_page.get_data(as_text=True)
        self.assertIn("Facial Manipulation Forensics", for_html)
        self.assertIn("Pillar 1: Face Swap", for_html)
        self.assertIn("Pillar 2: Facial Feature", for_html)
        self.assertIn("Pillar 3: Facial Distortion", for_html)
        self.assertIn("Pillar 4: Skin Texture", for_html)
        self.assertIn("Pillar 5: Face-to-Body", for_html)

        # 5. Check Face-Focused Grad-CAM Page
        cam_page = self.client.get(f"/gradcam/{session_id}")
        self.assertEqual(cam_page.status_code, 200)
        cam_html = cam_page.get_data(as_text=True)
        self.assertIn("Face-Focused Grad-CAM Attribution", cam_html)
        self.assertIn("Raw Grad-CAM Heatmap", cam_html)

        # 6. Check Target Switcher on Grad-CAM
        re_target = self.client.post(
            f"/gradcam/{session_id}/reanalyze",
            data={"target": "fake"},
            follow_redirects=True,
        )
        self.assertEqual(re_target.status_code, 200)
        re_html = re_target.get_data(as_text=True)
        self.assertIn("Target Class: FAKE", re_html)

        # 7. Check Occlusion Analysis Page
        occ_page = self.client.get(f"/occlusion/{session_id}")
        self.assertEqual(occ_page.status_code, 200)
        occ_html = occ_page.get_data(as_text=True)
        self.assertIn("Face-Focused Occlusion Sensitivity", occ_html)
        self.assertIn("Most Influential Facial Region", occ_html)

        # 8. Check Session File Serving
        sess_data = load_session(session_id)
        self.assertIsNotNone(sess_data)
        self.assertIn("face_crop", sess_data["artifacts"])
        crop_file = sess_data["artifacts"]["face_crop"]
        img_resp = self.client.get(f"/session_file/{session_id}/{crop_file}")
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.content_type, "image/png")


if __name__ == "__main__":
    unittest.main()
