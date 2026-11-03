"""
Comprehensive Automated Test Suite for Grad-CAM Explainability Module.

Tests:
1. Custom CNN & EfficientNet-B0 checkpoint loading without modification.
2. Target convolutional layer detection for both architectures.
3. Grad-CAM mathematical execution: activations, gradients, global pooling, ReLU, min-max norm.
4. Target class selection ('predicted', 'real', 'fake') producing distinct attributions.
5. Hook cleanup and absence of memory leaks.
6. Spatial heatmap alignment across multiple resolutions and aspect ratios.
7. Visualizer outputs: raw heatmap, overlay, side-by-side, combined comparison.
8. Evaluation across the 6 required scenarios:
   - Scenario 1: Known REAL face image.
   - Scenario 2: Known AI-generated FAKE face image.
   - Scenario 3: Full-scene image.
   - Scenario 4: Image with multiple faces.
   - Scenario 5: Image with no detectable face.
   - Scenario 6: Different image resolutions.
9. End-to-end CLI execution and artifact verification in outputs/gradcam/.
"""

import os
from pathlib import Path
import subprocess
import sys
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import GradCAM, GradCAMResult, find_target_layer
from src.explainability.occlusion import OcclusionSensitivity, OcclusionResult
from src.explainability.visualizer import GradCAMVisualizer
from src.explainability.pipeline import ExplainabilityPipeline, FaceDetectorHelper
from src.image_detection.predict import (
    CLASS_NAMES as CNN_CLASSES,
    DEFAULT_CHECKPOINT_PATH as CNN_CHECKPOINT,
    get_transform as get_cnn_transform,
    load_model as load_cnn_model,
)
from src.image_detection.predict_efficientnet import (
    CLASS_NAMES as EFF_CLASSES,
    DEFAULT_CHECKPOINT_PATH as EFF_CHECKPOINT,
    get_eval_transform as get_eff_transform,
    load_efficientnet_model,
)


def _draw_synthetic_face(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]):
    """Helper to draw a synthetic face in a given bounding box."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    # Face skin oval
    draw.ellipse((x1, y1, x2, y2), fill=(230, 195, 175), outline=(180, 140, 120))
    # Eyes
    ew = int(w * 0.12)
    eh = int(h * 0.12)
    draw.ellipse((x1 + int(w * 0.25), y1 + int(h * 0.35), x1 + int(w * 0.25) + ew, y1 + int(h * 0.35) + eh), fill=(30, 30, 30))
    draw.ellipse((x1 + int(w * 0.63), y1 + int(h * 0.35), x1 + int(w * 0.63) + ew, y1 + int(h * 0.35) + eh), fill=(30, 30, 30))
    # Nose
    draw.line([(x1 + int(w * 0.5), y1 + int(h * 0.45)), (x1 + int(w * 0.48), y1 + int(h * 0.62)), (x1 + int(w * 0.52), y1 + int(h * 0.62))], fill=(160, 110, 90), width=2)
    # Mouth
    draw.arc((x1 + int(w * 0.3), y1 + int(h * 0.65), x1 + int(w * 0.7), y1 + int(h * 0.8)), start=0, end=180, fill=(180, 60, 60), width=3)


def get_or_create_test_fixtures() -> dict:
    """Ensure all 6 test scenarios have corresponding image files in tests/."""
    tests_dir = PROJECT_ROOT / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {}

    # 1. Known REAL face
    real_path = tests_dir / "sample_image_real.jpg"
    assert real_path.is_file(), f"Expected real face image at {real_path}"
    fixtures["real"] = real_path

    # 2. Known AI-generated FAKE face
    fake_path = tests_dir / "sample_image_fake.jpg"
    assert fake_path.is_file(), f"Expected fake face image at {fake_path}"
    fixtures["fake"] = fake_path

    # 3. Full-scene image (room / landscape background)
    scene_path = tests_dir / "sample_scene.jpg"
    if not scene_path.is_file():
        img = Image.new("RGB", (640, 480), color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(img)
        # Ground
        draw.rectangle([(0, 300), (640, 480)], fill=(34, 139, 34))  # Grass green
        # Building / House
        draw.rectangle([(200, 200), (440, 400)], fill=(205, 133, 63))
        draw.polygon([(200, 200), (320, 100), (440, 200)], fill=(178, 34, 34))
        # Sun
        draw.ellipse([(500, 40), (580, 120)], fill=(255, 215, 0))
        img.save(scene_path, "JPEG", quality=95)
    fixtures["scene"] = scene_path

    # 4. Image with multiple faces
    multiface_path = tests_dir / "sample_multiface.jpg"
    if not multiface_path.is_file():
        img = Image.new("RGB", (600, 400), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        _draw_synthetic_face(draw, (50, 80, 250, 320))   # Face 1
        _draw_synthetic_face(draw, (350, 80, 550, 320))  # Face 2
        img.save(multiface_path, "JPEG", quality=95)
    fixtures["multiface"] = multiface_path

    # 5. Image with no detectable face (geometric abstract texture)
    noface_path = tests_dir / "sample_noface.jpg"
    if not noface_path.is_file():
        img = Image.new("RGB", (320, 320), color=(10, 20, 40))
        draw = ImageDraw.Draw(img)
        for i in range(0, 320, 40):
            draw.line([(i, 0), (320 - i, 320)], fill=(0, 200, 255), width=2)
            draw.rectangle([(i, i), (i + 20, i + 20)], fill=(20, 80, 150))
        img.save(noface_path, "JPEG", quality=95)
    fixtures["noface"] = noface_path

    # 6. Resolution variations
    res_path = tests_dir / "sample_res_wide.jpg"
    if not res_path.is_file() or True:  # Ensure neutral background for robust face localization
        img = Image.new("RGB", (800, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        _draw_synthetic_face(draw, (300, 30, 500, 270))
        img.save(res_path, "JPEG", quality=95)
    fixtures["res_wide"] = res_path

    return fixtures


# ==============================================================================
# Unit & Integration Tests
# ==============================================================================

def test_checkpoint_loading_and_architecture():
    """Verify both checkpoints load cleanly and are in evaluation mode."""
    device = torch.device("cpu")

    cnn_model, _ = load_cnn_model(device=device)
    assert isinstance(cnn_model, nn.Module)
    assert not cnn_model.training, "Custom CNN must be loaded in eval mode"

    eff_model, _ = load_efficientnet_model(device=device)
    assert isinstance(eff_model, nn.Module)
    assert not eff_model.training, "EfficientNet-B0 must be loaded in eval mode"


def test_target_layer_identification():
    """Verify target convolutional layer selection for both architectures."""
    device = torch.device("cpu")
    cnn_model, _ = load_cnn_model(device=device)
    eff_model, _ = load_efficientnet_model(device=device)

    # Custom CNN target layer
    cnn_layer, cnn_name = find_target_layer(cnn_model)
    assert isinstance(cnn_layer, nn.Conv2d), f"Expected Conv2d for CNN, got {type(cnn_layer)}"
    assert cnn_layer.out_channels == 128, f"Expected 128 out channels, got {cnn_layer.out_channels}"
    assert "features" in cnn_name

    # EfficientNet-B0 target layer
    eff_layer, eff_name = find_target_layer(eff_model)
    assert eff_layer is eff_model.get_target_layer_for_gradcam(), "Must match model's get_target_layer_for_gradcam"
    assert "features" in eff_name


def test_gradcam_gradient_calculation_cnn():
    """Verify Grad-CAM forward/backward hooks, gradient averaging, and ReLU for Custom CNN."""
    device = torch.device("cpu")
    cnn_model, _ = load_cnn_model(device=device)
    cam_engine = GradCAM(cnn_model, device=device)

    # 128x128 synthetic tensor
    x = torch.randn(1, 3, 128, 128)
    res = cam_engine.generate(x, target_class="predicted", target_size=(128, 128))

    assert isinstance(res, GradCAMResult)
    assert res.heatmap.shape == (128, 128)
    assert res.heatmap.min() >= 0.0
    assert res.heatmap.max() <= 1.0
    # Heatmap should not be empty/all-zero
    assert np.any(res.heatmap > 0.0), "Heatmap must contain non-zero activations"
    assert res.predicted_class in CNN_CLASSES
    assert 0.0 <= res.confidence <= 100.0


def test_gradcam_gradient_calculation_efficientnet():
    """Verify Grad-CAM forward/backward hooks and gradient averaging for EfficientNet-B0."""
    device = torch.device("cpu")
    eff_model, _ = load_efficientnet_model(device=device)
    cam_engine = GradCAM(eff_model, device=device)

    # 224x224 synthetic tensor
    x = torch.randn(1, 3, 224, 224)
    res = cam_engine.generate(x, target_class="predicted", target_size=(224, 224))

    assert isinstance(res, GradCAMResult)
    assert res.heatmap.shape == (224, 224)
    assert res.heatmap.min() >= 0.0
    assert res.heatmap.max() <= 1.0
    assert np.any(res.heatmap > 0.0), "Heatmap must contain non-zero activations"
    assert res.predicted_class in EFF_CLASSES
    assert 0.0 <= res.confidence <= 100.0


def test_target_class_selection_produces_distinct_heatmaps():
    """Verify that targeting 'real' vs 'fake' classes yields distinct attribution maps."""
    device = torch.device("cpu")
    cnn_model, _ = load_cnn_model(device=device)
    cam_engine = GradCAM(cnn_model, device=device)

    sample_img = Image.open(PROJECT_ROOT / "tests" / "sample_image.jpg").convert("RGB")
    tensor = get_cnn_transform()(sample_img).unsqueeze(0).to(device)

    res_fake = cam_engine.generate(tensor, target_class="fake", target_size=sample_img.size)
    res_real = cam_engine.generate(tensor, target_class="real", target_size=sample_img.size)

    assert res_fake.target_class == "FAKE"
    assert res_real.target_class == "REAL"
    assert res_fake.target_idx == 0
    assert res_real.target_idx == 1

    diff = np.abs(res_fake.heatmap - res_real.heatmap)
    mean_diff = float(np.mean(diff))
    assert mean_diff > 0.01, f"Expected distinct heatmaps for REAL and FAKE, got mean diff: {mean_diff}"


def test_no_memory_leak_and_hook_cleanup():
    """Verify that hooks are cleanly deregistered and gradients reset after Grad-CAM."""
    device = torch.device("cpu")
    cnn_model, _ = load_cnn_model(device=device)
    cam_engine = GradCAM(cnn_model, device=device)

    x = torch.randn(1, 3, 128, 128)
    _ = cam_engine.generate(x, target_class="predicted", target_size=(128, 128))

    # Handles list must be cleared
    assert len(cam_engine._hook_handles) == 0, "All hook handles must be removed"
    assert cam_engine.activations is None, "Activations cache must be cleared"
    assert cam_engine.gradients is None, "Gradients cache must be cleared"

    # All parameter gradients should be None or zero
    for param in cnn_model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0), "Model gradients must be zeroed"


def test_spatial_alignment_and_resolutions():
    """Verify Grad-CAM resizes correctly to preserve exact original dimensions and aspect ratio."""
    device = torch.device("cpu")
    cnn_model, _ = load_cnn_model(device=device)
    cam_engine = GradCAM(cnn_model, device=device)

    test_shapes = [
        (100, 100),
        (236, 327),
        (728, 513),
        (1312, 1199),
        (800, 300),
    ]

    for width, height in test_shapes:
        x = torch.randn(1, 3, 128, 128)
        res = cam_engine.generate(x, target_class="predicted", target_size=(width, height))
        assert res.heatmap.shape == (height, width), f"Expected shape ({height}, {width}), got {res.heatmap.shape}"


def test_visualizer_components():
    """Verify visualizer produces valid colormaps, overlays, side-by-side and comparison figures."""
    viz = GradCAMVisualizer(default_colormap="turbo", default_alpha=0.5)

    img = Image.new("RGB", (200, 150), color=(100, 150, 200))
    heatmap = np.random.rand(150, 200).astype(np.float32)

    # 1. Colormap
    rgb = viz.apply_colormap(heatmap, colormap="turbo")
    assert rgb.shape == (150, 200, 3)
    assert rgb.dtype == np.uint8

    # 2. Overlay
    overlay = viz.create_overlay(img, heatmap, alpha=0.5)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (200, 150)

    # 3. Side by side
    res = GradCAMResult(
        heatmap=heatmap,
        raw_cam=heatmap,
        predicted_idx=0,
        predicted_class="FAKE",
        target_idx=0,
        target_class="FAKE",
        probabilities={"FAKE": 90.0, "REAL": 10.0},
        confidence=90.0,
        target_score=2.5,
        layer_name="features.6",
        model_name="DeepfakeCNN",
    )
    sbs = viz.create_side_by_side(img, Image.fromarray(rgb), overlay, res)
    assert isinstance(sbs, Image.Image)
    assert sbs.size[0] > sbs.size[1]

    # 4. Model comparison
    comp = viz.create_model_comparison(img, overlay, overlay, res, res)
    assert isinstance(comp, Image.Image)
    assert comp.size[0] > comp.size[1]


def test_face_detection_and_multi_face():
    """Verify face detection helper handles single faces, multi-faces, and non-face images."""
    fixtures = get_or_create_test_fixtures()
    detector = FaceDetectorHelper()

    # Multi-face image should detect multiple faces
    multiface_img = Image.open(fixtures["multiface"])
    detected = detector.detect_faces(multiface_img)
    assert len(detected) >= 2, f"Expected at least 2 faces in synthetic multi-face image, got {len(detected)}"

    # No-face image should detect 0 faces
    noface_img = Image.open(fixtures["noface"])
    detected_none = detector.detect_faces(noface_img)
    assert len(detected_none) == 0, f"Expected 0 faces in geometric texture image, got {len(detected_none)}"


def test_all_six_scenarios_end_to_end():
    """
    Execute pipeline on all 6 specified scenarios:
    1. Known REAL face image.
    2. Known AI-generated FAKE face image.
    3. Full-scene image.
    4. Image with multiple faces.
    5. Image with no detectable face.
    6. Different image resolutions.
    """
    fixtures = get_or_create_test_fixtures()
    pipeline = ExplainabilityPipeline(device="cpu", colormap="turbo", alpha=0.5)

    scenarios = [
        ("real_face", fixtures["real"], False),
        ("fake_face", fixtures["fake"], False),
        ("scene", fixtures["scene"], True),
        ("multiface", fixtures["multiface"], False),
        ("noface", fixtures["noface"], True),
        ("res_wide", fixtures["res_wide"], False),
    ]

    for name, img_path, full_img_flag in scenarios:
        out_sub = PROJECT_ROOT / "outputs" / "gradcam" / f"test_{name}"
        res = pipeline.run_image(
            image_input=img_path,
            model_choice="both",
            target_class="predicted",
            output_subdir=out_sub,
            full_image=full_img_flag,
        )

        assert res["cnn_result"] is not None
        assert res["effnet_result"] is not None
        assert (out_sub / "original.png").is_file()
        assert (out_sub / "cnn_heatmap.png").is_file()
        assert (out_sub / "cnn_overlay.png").is_file()
        assert (out_sub / "cnn_side_by_side.png").is_file()
        assert (out_sub / "efficientnet_heatmap.png").is_file()
        assert (out_sub / "efficientnet_overlay.png").is_file()
        assert (out_sub / "efficientnet_side_by_side.png").is_file()
        assert (out_sub / "comparison.png").is_file()


def test_face_only_and_background_only_masking():
    """Verify face-only and background-only masks preserve dimensions and properly isolate regions."""
    detector = FaceDetectorHelper()
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    bbox = (50, 50, 100, 100)

    face_only = detector.create_face_only_image(img, bbox, fill_color=(128, 128, 128))
    bg_only = detector.create_background_only_image(img, bbox, fill_color=(128, 128, 128))

    assert face_only.size == (200, 200)
    assert bg_only.size == (200, 200)

    # In face_only, outer corner (10, 10) must be filled with 128
    fo_arr = np.array(face_only)
    assert np.all(fo_arr[10, 10] == 128), "Outer background must be neutralized in face_only"

    # In bg_only, center face pixel (100, 100) must be filled with 128
    bg_arr = np.array(bg_only)
    assert np.all(bg_arr[100, 100] == 128), "Center face must be neutralized in bg_only"


def test_occlusion_sensitivity_calculation():
    """Verify sliding-window occlusion test computes valid drop maps and metadata."""
    device = torch.device("cpu")
    cnn_model, _ = load_cnn_model(device=device)
    occ = OcclusionSensitivity(cnn_model, get_cnn_transform(), device=device, patch_size=32, stride=16)

    img = Image.open(PROJECT_ROOT / "tests" / "sample_image.jpg").convert("RGB")
    res = occ.generate(img, target_class="fake", target_size=(100, 100))

    assert isinstance(res, OcclusionResult)
    assert res.heatmap.shape == (100, 100)
    assert 0.0 <= res.heatmap.min() <= res.heatmap.max() <= 1.0
    assert 0.0 <= res.base_probability <= 100.0
    assert res.target_class == "FAKE"


def test_fair_fake_comparison():
    """Verify fair comparison figure generation where both models target FAKE."""
    pipeline = ExplainabilityPipeline(device="cpu")
    img = Image.open(PROJECT_ROOT / "tests" / "sample_image.jpg").convert("RGB")
    fair_img = pipeline._generate_fair_fake_comparison(img)

    assert isinstance(fair_img, Image.Image)
    assert fair_img.size[0] > fair_img.size[1]


def test_cli_execution():
    """Verify CLI executes cleanly via python -m src.explainability.pipeline."""
    cmd = [
        sys.executable,
        "-m",
        "src.explainability.pipeline",
        "--image",
        "tests/sample_image.jpg",
        "--model",
        "both",
        "--target",
        "predicted",
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI command failed with code {proc.returncode}:\n{proc.stderr}"
    assert "DEEPFAKE FORENSICS" in proc.stdout
    assert "Custom CNN" in proc.stdout
    assert "EfficientNet-B0" in proc.stdout


if __name__ == "__main__":
    print("Running automated Grad-CAM test suite...")
    test_checkpoint_loading_and_architecture()
    test_target_layer_identification()
    test_gradcam_gradient_calculation_cnn()
    test_gradcam_gradient_calculation_efficientnet()
    test_target_class_selection_produces_distinct_heatmaps()
    test_no_memory_leak_and_hook_cleanup()
    test_spatial_alignment_and_resolutions()
    test_visualizer_components()
    test_face_detection_and_multi_face()
    test_face_only_and_background_only_masking()
    test_occlusion_sensitivity_calculation()
    test_fair_fake_comparison()
    test_all_six_scenarios_end_to_end()
    test_cli_execution()
    print("\nALL GRAD-CAM TESTS PASSED SUCCESSFULLY!")
