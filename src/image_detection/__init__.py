"""
Image detection package for DeepFake-Forensics.
Unified on EfficientNet-B0 and SCRFD.
"""

from src.image_detection.efficientnet_model import DeepfakeEfficientNetB0
from src.image_detection.face_detector import SCRFDFaceDetector
from src.image_detection.inference import DeepFakeInferenceEngine

__all__ = [
    "DeepfakeEfficientNetB0",
    "SCRFDFaceDetector",
    "DeepFakeInferenceEngine",
]
