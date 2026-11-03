"""
Automated Test Suite for Face-Based Detection & Cropping.

Verifies:
1. Face detection accuracy on single-face portraits.
2. Face detection on multi-face images.
3. Accurate rejection of non-face images (geometric patterns & full scenes).
4. Face cropping bounds, margin padding, and aspect ratio integrity.
5. Model input transforms: 128x128 for Custom CNN, 224x224 for EfficientNet-B0.
6. End-to-end model inference on face crops.
7. Pipeline integrity guard: safe halt when no face is detected.
8. Multi-face pipeline processing and artifact creation.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.face_detector import FaceDetector
from src.explainability.pipeline import ExplainabilityPipeline
from src.image_detection.predict import (
    CLASS_NAMES as CNN_CLASSES,
    get_transform as get_cnn_transform,
    load_model as load_cnn_model,
)
from src.image_detection.predict_efficientnet import (
    CLASS_NAMES as EFF_CLASSES,
    get_eval_transform as get_eff_transform,
    load_efficientnet_model,
)


def test_face_detection_single_face():
    """Verify face detector locates single faces in known test images."""
    detector = FaceDetector()
    img_path = PROJECT_ROOT / "tests" / "sample_image.jpg"
    img = Image.open(img_path)
    boxes = detector.detect_faces(img)

    assert len(boxes) == 1, f"Expected 1 face in sample_image.jpg, got {len(boxes)}"
    x, y, w, h = boxes[0]
    assert w > 100 and h > 100, f"Unreasonably small face box: {boxes[0]}"


def test_face_detection_multiple_faces():
    """Verify face detector detects all faces in multi-face scenes."""
    detector = FaceDetector()
    img_path = PROJECT_ROOT / "tests" / "sample_multiface.jpg"
    img = Image.open(img_path)
    boxes = detector.detect_faces(img)

    assert len(boxes) >= 2, f"Expected at least 2 faces in sample_multiface.jpg, got {len(boxes)}"


def test_face_detection_no_face_rejection():
    """Verify face detector returns empty list when no face is present."""
    detector = FaceDetector()

    # Abstract texture
    nf_path = PROJECT_ROOT / "tests" / "sample_noface.jpg"
    boxes_nf = detector.detect_faces(Image.open(nf_path))
    assert len(boxes_nf) == 0, f"Expected 0 faces in sample_noface.jpg, got {len(boxes_nf)}"

    # Full landscape/building scene
    scene_path = PROJECT_ROOT / "tests" / "sample_scene.jpg"
    boxes_scene = detector.detect_faces(Image.open(scene_path))
    assert len(boxes_scene) == 0, f"Expected 0 faces in sample_scene.jpg, got {len(boxes_scene)}"


def test_face_cropping_and_dimensions():
    """Verify face cropping properly crops, adds margin, and stays within image boundaries."""
    detector = FaceDetector()
    img = Image.new("RGB", (500, 500), color=(200, 200, 200))
    box = (100, 100, 150, 150)

    cropped, actual_b = detector.crop_face(img, box, margin=0.10)
    assert isinstance(cropped, Image.Image)
    assert cropped.size[0] > 0 and cropped.size[1] > 0
    # Coordinates must be within original bounds
    x, y, w, h = actual_b
    assert x >= 0 and y >= 0
    assert x + w <= 500 and y + h <= 500


def test_face_crop_preprocessing_and_model_inference():
    """Verify face crop is correctly transformed and evaluated by both models."""
    detector = FaceDetector()
    device = torch.device("cpu")

    cnn_model, _ = load_cnn_model(device=device)
    eff_model, _ = load_efficientnet_model(device=device)

    img = Image.open(PROJECT_ROOT / "tests" / "sample_image.jpg")
    boxes = detector.detect_faces(img)
    face_crop, _ = detector.crop_face(img, boxes[0])

    # CNN transform: 128x128
    t_cnn = get_cnn_transform()(face_crop).unsqueeze(0).to(device)
    assert t_cnn.shape == (1, 3, 128, 128), f"Expected shape (1, 3, 128, 128), got {t_cnn.shape}"

    # EffNet transform: 224x224
    t_eff = get_eff_transform()(face_crop).unsqueeze(0).to(device)
    assert t_eff.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {t_eff.shape}"

    # Forward passes
    with torch.no_grad():
        out_cnn = cnn_model(t_cnn)
        probs_cnn = torch.softmax(out_cnn, dim=1)[0]
        assert probs_cnn.shape == (2,)
        assert 0.0 <= probs_cnn[0].item() <= 1.0

        out_eff = eff_model(t_eff)
        probs_eff = torch.softmax(out_eff, dim=1)[0]
        assert probs_eff.shape == (2,)
        assert 0.0 <= probs_eff[0].item() <= 1.0


def test_pipeline_no_face_integrity_guard():
    """Verify pipeline halts cleanly without error when no face is found."""
    pipeline = ExplainabilityPipeline(device="cpu")
    res = pipeline.run_image("tests/sample_noface.jpg", full_image=False)

    assert res["status"] == "NO_FACE_DETECTED"
    assert res["faces_detected"] == 0
    assert res["cnn_result"] is None
    assert res["effnet_result"] is None


def test_pipeline_multi_face_analysis():
    """Verify pipeline analyzes multiple faces separately and creates sub-face artifacts."""
    pipeline = ExplainabilityPipeline(device="cpu")
    out_dir = PROJECT_ROOT / "outputs" / "gradcam" / "test_multiface_suite"
    res = pipeline.run_image("tests/sample_multiface.jpg", output_subdir=out_dir)

    assert res["status"] == "SUCCESS"
    assert res["faces_detected"] >= 2
    assert len(res["faces"]) >= 2

    # Check that face subdirectories were created
    assert (out_dir / "face_0" / "face_crop.png").is_file()
    assert (out_dir / "face_1" / "face_crop.png").is_file()
    assert (out_dir / "face_side_by_side.png").is_file()


if __name__ == "__main__":
    print("Running face detection and cropping test suite...")
    test_face_detection_single_face()
    test_face_detection_multiple_faces()
    test_face_detection_no_face_rejection()
    test_face_cropping_and_dimensions()
    test_face_crop_preprocessing_and_model_inference()
    test_pipeline_no_face_integrity_guard()
    test_pipeline_multi_face_analysis()
    print("\nALL FACE DETECTION TESTS PASSED SUCCESSFULLY!")
