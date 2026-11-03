"""Image Authenticity Detection Module.

This package contains components for image-level deepfake and manipulation detection,
including face extraction, CNN backbones, feature extraction, and classification.
"""

from .model import DeepfakeCNN

__all__ = ["DeepfakeCNN", "predict_image", "load_model", "get_transform", "get_device"]


def __getattr__(name):
    if name in ("predict_image", "load_model", "get_transform", "get_device"):
        from . import predict
        return getattr(predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


