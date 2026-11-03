"""
Test script for DeepFake CNN Model & Prediction Pipeline.

Verifies:
1. Model architecture creation
2. Checkpoint weight loading from models/deepfake_cnn.pth
3. Automatic CUDA / CPU device selection
4. Inference pipeline and output formatting (class + confidence)
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw
import torch

from src.image_detection.model import DeepfakeCNN
from src.image_detection.predict import (
    CLASS_NAMES,
    get_device,
    load_model,
    predict_image,
)


def create_sample_image(output_path: Path) -> Path:
    """
    Creates a sample 128x128 RGB test image if none exists.
    Draws a simple synthetic face pattern.
    """
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (128, 128), color=(240, 220, 200))
    draw = ImageDraw.Draw(img)

    # Simple facial oval and features for testing
    draw.ellipse((24, 20, 104, 110), fill=(225, 195, 175), outline=(180, 140, 120))
    # Eyes
    draw.ellipse((42, 50, 52, 60), fill=(40, 40, 40))
    draw.ellipse((76, 50, 86, 60), fill=(40, 40, 40))
    # Nose
    draw.line([(64, 60), (62, 75), (66, 75)], fill=(150, 100, 80), width=2)
    # Mouth
    draw.arc((50, 80, 78, 95), start=0, end=180, fill=(180, 60, 60), width=2)

    img.save(output_path, "JPEG", quality=95)
    print(f"[TEST] Created sample test image at: {output_path}")
    return output_path


def test_model_architecture():
    """Verify DeepfakeCNN can be instantiated and produces expected output shapes."""
    print("\n--- Test 1: Model Architecture ---")
    model = DeepfakeCNN(num_classes=2)
    sample_input = torch.randn(1, 3, 128, 128)
    output = model(sample_input)
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    print(f"[PASS] DeepfakeCNN forward pass output shape: {output.shape}")


def test_checkpoint_loading():
    """Verify weights load correctly from models/deepfake_cnn.pth."""
    print("\n--- Test 2: Checkpoint Loading ---")
    device = get_device()
    model, used_device = load_model(device=device)
    assert model is not None
    assert isinstance(model, DeepfakeCNN)
    print(f"[PASS] Model successfully loaded on {used_device}")
    return model, used_device


def test_prediction_pipeline(model=None, device=None):
    """Verify end-to-end prediction pipeline on a sample image."""
    print("\n--- Test 3: Prediction Pipeline on Sample Image ---")
    sample_img_path = PROJECT_ROOT / "tests" / "sample_image.jpg"
    create_sample_image(sample_img_path)

    result = predict_image(
        image_path=sample_img_path,
        model=model,
        device=device,
    )

    print(f" Image:       {result['image_path']}")
    print(f" Prediction:  {result['class_name']}")
    print(f" Confidence:  {result['confidence']:.2f}%")
    print(f" Probabilities: {result['probabilities']}")

    assert result["class_name"] in CLASS_NAMES, f"Unexpected class: {result['class_name']}"
    assert 0.0 <= result["confidence"] <= 100.0, f"Invalid confidence: {result['confidence']}"
    assert result["predicted_label"] in (0, 1), f"Invalid label index: {result['predicted_label']}"

    # Probabilities should sum to approximately 100%
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 100.0) < 0.1, f"Probabilities do not sum to 100%: {prob_sum}"
    print("[PASS] Prediction returned valid class, confidence, and probability distribution.")


def run_all_tests():
    print("=" * 60)
    print("DeepFake-Forensics: Running Model & Prediction Tests")
    print("=" * 60)

    test_model_architecture()
    model, device = test_checkpoint_loading()
    test_prediction_pipeline(model=model, device=device)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
