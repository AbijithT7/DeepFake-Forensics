"""
EfficientNet-B0 DeepFake Prediction Module.

Performs standalone inference on an input image using the trained EfficientNet-B0 model:
- Preprocessing: Resize to 224x224, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Device: Automatically uses NVIDIA CUDA if available, otherwise CPU
- Output: Predicted class ('REAL' or 'FAKE') and confidence percentage
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detection.efficientnet_model import (
    CLASS_NAMES,
    DeepfakeEfficientNetB0,
    get_eval_transform,
)

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using device: CUDA ({device_name})")
        return torch.device("cuda")
    print("[INFO] Using device: CPU")
    return torch.device("cpu")


def load_efficientnet_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[DeepfakeEfficientNetB0, torch.device]:
    """
    Load the trained EfficientNet-B0 checkpoint for inference.
    """
    if device is None:
        device = get_device()

    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"EfficientNet checkpoint not found at: {path}\n"
            "Please ensure training has completed and 'models/efficientnet_b0_best.pth' exists."
        )

    print(f"[INFO] Loading EfficientNet-B0 checkpoint from: {path}")
    model = DeepfakeEfficientNetB0(num_classes=len(CLASS_NAMES), pretrained=False)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("[INFO] EfficientNet-B0 loaded and ready for evaluation.")
    return model, device


def predict_efficientnet_image(
    image_path: Union[str, Path],
    model: Optional[DeepfakeEfficientNetB0] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[str, float, dict]]:
    """
    Predict whether an input image is REAL or FAKE using EfficientNet-B0.

    Args:
        image_path: Path to the image file.
        model: Optional pre-loaded DeepfakeEfficientNetB0 model.
        checkpoint_path: Path to checkpoint file.
        device: Execution device.

    Returns:
        Dict containing class_name, confidence, probabilities, predicted_label, image_path.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    if model is None:
        model, device = load_efficientnet_model(checkpoint_path=checkpoint_path, device=device)
    elif device is None:
        device = next(model.parameters()).device

    transform = get_eval_transform()
    with Image.open(img_path) as img:
        rgb_img = img.convert("RGB")
        input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100.0

    prob_dict = {
        name: probabilities[i].item() * 100.0
        for i, name in enumerate(CLASS_NAMES)
    }

    return {
        "model": "EfficientNet-B0",
        "class_name": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "predicted_label": predicted_idx,
        "image_path": str(img_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict REAL or FAKE for an image using EfficientNet-B0."
    )
    parser.add_argument("image_path", nargs="?", default=None, help="Path to input image.")
    parser.add_argument("--image", "-i", dest="image_flag", default=None, help="Path to input image.")
    parser.add_argument(
        "--model",
        "-m",
        dest="checkpoint_path",
        default=None,
        help="Path to model checkpoint (defaults to models/efficientnet_b0_best.pth).",
    )
    args = parser.parse_args()

    target_image = args.image_flag or args.image_path
    if not target_image:
        parser.print_help()
        print("\nError: Please provide an image path to predict.")
        sys.exit(1)

    result = predict_efficientnet_image(
        image_path=target_image,
        checkpoint_path=args.checkpoint_path,
    )

    print("\n" + "=" * 50)
    print(f" Model:      {result['model']}")
    print(f" Image:      {result['image_path']}")
    print(f" Prediction: {result['class_name']}")
    print(f" Confidence: {result['confidence']:.2f}%")
    print("-" * 50)
    print(" Class Probabilities:")
    for cls_name, prob in result["probabilities"].items():
        print(f"  - {cls_name:<6}: {prob:6.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
