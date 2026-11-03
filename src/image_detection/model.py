"""
DeepFake CNN Model Architecture.

Exact neural network architecture recreated from the Kaggle training notebook:
- 3 Convolutional blocks with ReLU and MaxPool2d (input: 3x128x128 -> output: 128x16x16)
- Classifier with Flatten, Linear(32768, 128), ReLU, Dropout(0.5), Linear(128, 2)
"""

import torch
import torch.nn as nn


class DeepfakeCNN(nn.Module):
    """
    Convolutional Neural Network for DeepFake detection.
    
    Recreated precisely from the Kaggle notebook checkpoint.
    """

    def __init__(self, num_classes: int = 2):
        super(DeepfakeCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
