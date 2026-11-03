"""Explainability & Interpretability Module (XAI).

Provides explainable AI tools for generating attribution maps,
Grad-CAM heatmaps, occlusion sensitivity maps, and forensic visual evidence.
"""

from typing import TYPE_CHECKING
from src.explainability.gradcam import GradCAM, GradCAMResult, find_target_layer
from src.explainability.occlusion import OcclusionSensitivity, OcclusionResult
from src.explainability.visualizer import GradCAMVisualizer

if TYPE_CHECKING:
    from src.explainability.pipeline import ExplainabilityPipeline

__all__ = [
    "GradCAM",
    "GradCAMResult",
    "find_target_layer",
    "OcclusionSensitivity",
    "OcclusionResult",
    "GradCAMVisualizer",
    "ExplainabilityPipeline",
]


def __getattr__(name: str):
    if name == "ExplainabilityPipeline":
        from src.explainability.pipeline import ExplainabilityPipeline
        return ExplainabilityPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
