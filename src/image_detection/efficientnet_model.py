"""
EfficientNet-B0 Model Architecture for DeepFake Detection.

Transfer learning model using torchvision's EfficientNet-B0 backbone:
- Pretrained on ImageNet-1K
- Global Average Pooling (AdaptiveAvgPool2d) for spatial translation invariance
- Custom binary classification head with Dropout
- Standard ImageNet preprocessing and training augmentations
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ["FAKE", "REAL"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)


def get_train_transform() -> transforms.Compose:
    """
    Return training transforms with domain-appropriate data augmentations:
    - Resize to 224x224
    - Random horizontal flip (p=0.5)
    - Random rotation (+/- 10 degrees)
    - Color jitter (brightness, contrast, saturation)
    - Tensor conversion & ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Return deterministic evaluation (validation/test/inference) transforms:
    - Resize to 224x224
    - Tensor conversion & ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DeepfakeEfficientNetB0(nn.Module):
    """
    DeepFake Detector based on pretrained EfficientNet-B0.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_p: float = 0.3):
        super(DeepfakeEfficientNetB0, self).__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # In EfficientNet-B0, backbone.classifier is:
        # Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.backbone(x)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """
        Return the terminal convolutional stage for Grad-CAM visualization.
        In EfficientNet-B0, features[8] is the last 1280-channel Conv2dNormActivation block.
        """
        return self.backbone.features[8]
