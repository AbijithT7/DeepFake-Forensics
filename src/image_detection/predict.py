"""
DeepFake Prediction Module.

Loads the trained DeepfakeCNN model checkpoint and performs inference on an input image:
- Preprocessing: Resize to 128x128, RGB, ToTensor, Normalize(mean=0.5, std=0.5)
- Device: Automatically uses NVIDIA CUDA if available, otherwise CPU
- Output: Predicted class ('REAL' or 'FAKE') and confidence percentage
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to sys.path to support running directly or as a module
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from src.image_detection.model import DeepfakeCNN
except ImportError:
    try:
        from model import DeepfakeCNN
    except ImportError:
        from .model import DeepfakeCNN


# Class labels matching torchvision ImageFolder alphabetical sorting:
# Index 0: 'Fake' -> FAKE
# Index 1: 'Real' -> REAL
CLASS_NAMES = ["FAKE", "REAL"]

# Default checkpoint path relative to project root
DEFAULT_CHECKPOINT_PATH = Path("models/deepfake_cnn.pth")


def get_device() -> torch.device:
    """
    Detect and return the execution device.
    Automatically uses NVIDIA CUDA if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using device: CUDA ({device_name})")
        return device
    else:
        print("[INFO] Using device: CPU")
        return torch.device("cpu")


def get_transform() -> transforms.Compose:
    """
    Return the exact preprocessing pipeline from training:
    1. Resize to 128x128
    2. Convert to PyTorch Tensor
    3. Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def find_checkpoint(checkpoint_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Locate the model checkpoint file.
    Checks the provided path, then default paths.
    """
    candidates = []
    if checkpoint_path is not None:
        candidates.append(Path(checkpoint_path))

    # Standard default paths
    candidates.extend([
        DEFAULT_CHECKPOINT_PATH,
        Path(__file__).resolve().parent.parent.parent / "models" / "deepfake_cnn.pth",
        Path("models/deepfake_cnn.pth.zip"),
        Path(__file__).resolve().parent.parent.parent / "models" / "deepfake_cnn.pth.zip",
    ])

    for path in candidates:
        if path.is_file():
            return path

    searched = "\n - " + "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Checkpoint file not found. Searched in:{searched}\n"
        "Please ensure 'deepfake_cnn.pth' exists in the 'models/' directory."
    )


def load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Tuple[DeepfakeCNN, torch.device]:
    """
    Load the DeepfakeCNN model with weights from the checkpoint.
    """
    if device is None:
        device = get_device()

    resolved_path = find_checkpoint(checkpoint_path)
    print(f"[INFO] Loading checkpoint from: {resolved_path}")

    model = DeepfakeCNN(num_classes=len(CLASS_NAMES))

    # Load weights
    try:
        # PyTorch 2.6+ defaults to weights_only=True
        state_dict = torch.load(resolved_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(resolved_path, map_location=device)

    # In case checkpoint is wrapped under a key like 'state_dict' or 'model'
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("[INFO] Model loaded and ready for evaluation.")
    return model, device


def predict_image(
    image_path: Union[str, Path],
    model: Optional[DeepfakeCNN] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[str, float, dict]]:
    """
    Predict whether an input image is REAL or FAKE.

    Args:
        image_path: Path to the image file.
        model: Optional pre-loaded DeepfakeCNN model.
        checkpoint_path: Path to checkpoint (used if model is not provided).
        device: Device to run inference on.

    Returns:
        Dict containing:
            - 'class_name': 'REAL' or 'FAKE'
            - 'confidence': Confidence percentage (0-100)
            - 'probabilities': Dict of class names to probabilities
            - 'predicted_label': 0 (FAKE) or 1 (REAL)
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    if model is None:
        model, device = load_model(checkpoint_path=checkpoint_path, device=device)
    elif device is None:
        device = next(model.parameters()).device

    # Load image and ensure RGB mode
    with Image.open(img_path) as img:
        rgb_image = img.convert("RGB")
        transform = get_transform()
        input_tensor = transform(rgb_image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100.0

    prob_dict = {
        name: probabilities[i].item() * 100.0
        for i, name in enumerate(CLASS_NAMES)
    }

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "predicted_label": predicted_idx,
        "image_path": str(img_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict REAL or FAKE for an image using DeepfakeCNN."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--image",
        "-i",
        dest="image_flag",
        default=None,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--model",
        "-m",
        dest="checkpoint_path",
        default=None,
        help="Path to the model checkpoint (defaults to models/deepfake_cnn.pth)."
    )
    args = parser.parse_args()

    target_image = args.image_flag or args.image_path
    if not target_image:
        parser.print_help()
        print("\nError: Please provide an image path to predict.")
        sys.exit(1)

    result = predict_image(
        image_path=target_image,
        checkpoint_path=args.checkpoint_path
    )

    print("\n" + "=" * 45)
    print(f" Image:      {result['image_path']}")
    print(f" Prediction: {result['class_name']}")
    print(f" Confidence: {result['confidence']:.2f}%")
    print("-" * 45)
    print(" Class Probabilities:")
    for cls_name, prob in result["probabilities"].items():
        print(f"  - {cls_name:<6}: {prob:6.2f}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
