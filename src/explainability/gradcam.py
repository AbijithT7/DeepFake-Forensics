"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module for EfficientNet-B0.

Calculates the gradient of the target class score with respect to feature activation
maps of the terminal convolutional layer (features[8]), producing high-fidelity
spatial attribution heatmaps for model explainability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["FAKE", "REAL"]


@dataclass
class GradCAMResult:
    """Structured container for Grad-CAM execution outputs."""

    heatmap: np.ndarray  # 2D float32 array in [0.0, 1.0], shape (H, W)
    raw_cam: np.ndarray  # 2D float32 array at conv feature map resolution
    predicted_idx: int
    predicted_class: str
    target_idx: int
    target_class: str
    probabilities: Dict[str, float]  # Percentages (0 - 100)
    confidence: float  # Top predicted confidence percentage
    target_score: float  # Unnormalized logit score of the target class
    layer_name: str
    model_name: str


def find_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> Tuple[nn.Module, str]:
    """
    Identify and return the optimal terminal convolutional layer for Grad-CAM.

    Priority order:
    1. Explicit layer_name string if supplied.
    2. Model's built-in get_target_layer_for_gradcam() if defined (e.g. DeepfakeEfficientNetB0 -> features[8]).
    3. Last nn.Conv2d module in the network hierarchy.
    """
    if layer_name:
        try:
            target = model.get_submodule(layer_name)
            return target, layer_name
        except AttributeError:
            parts = layer_name.split(".")
            curr = model
            for part in parts:
                if part.isdigit():
                    curr = curr[int(part)]
                else:
                    curr = getattr(curr, part)
            return curr, layer_name

    # Check for custom method on DeepfakeEfficientNetB0
    if hasattr(model, "get_target_layer_for_gradcam") and callable(model.get_target_layer_for_gradcam):
        target = model.get_target_layer_for_gradcam()
        for name, mod in model.named_modules():
            if mod is target:
                return target, name
        return target, "backbone.features[8]"

    # Generic search: find the last Conv2d layer
    last_conv_name = None
    last_conv_mod = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_conv_name = name
            last_conv_mod = mod

    if last_conv_mod is not None:
        return last_conv_mod, last_conv_name

    raise ValueError(
        f"Could not automatically detect a convolutional layer in {model.__class__.__name__}. "
        "Please provide an explicit target_layer."
    )


class GradCAM:
    """
    Grad-CAM engine for EfficientNet-B0 deepfake explainability.

    Features:
    - Robust forward & tensor backward hooks with automatic cleanup to prevent memory leaks.
    - Unnormalized logit gradients (recommended by Selvaraju et al.) to prevent softmax saturation.
    - Global average pooling of gradients over spatial dimensions.
    - Positive rectification (ReLU) to focus solely on evidence supporting the target class.
    - Min-max normalization with epsilon division guard.
    - Bilinear resizing to original face crop dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Grad-CAM engine.

        Args:
            model: PyTorch model in evaluation mode.
            target_layer: Specific layer module or module name. If None, auto-detected.
            device: Execution device. If None, inferred from model parameters.
        """
        self.model = model
        self.model.eval()

        if isinstance(target_layer, (str, type(None))):
            self.target_layer, self.layer_name = find_target_layer(model, layer_name=target_layer)
        else:
            self.target_layer = target_layer
            self.layer_name = self.target_layer.__class__.__name__
            for name, mod in model.named_modules():
                if mod is target_layer:
                    self.layer_name = name
                    break

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        self._clear_hooks()

        def forward_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            self.activations = out.detach().clone()

            def tensor_backward_hook(grad: torch.Tensor):
                self.gradients = grad.detach().clone()

            if out.requires_grad:
                out.register_hook(tensor_backward_hook)

        handle = self.target_layer.register_forward_hook(forward_hook)
        self._hook_handles.append(handle)

    def _clear_hooks(self) -> None:
        """Remove all registered hook handles and clear cached tensors."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self.activations = None
        self.gradients = None

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._clear_hooks()

    @staticmethod
    def resolve_target_class(
        target_class: Optional[Union[str, int]],
        predicted_idx: int,
    ) -> Tuple[int, str]:
        """
        Resolve user target class input to a valid class index and label.
        """
        if target_class is None or target_class == "predicted":
            return predicted_idx, CLASS_NAMES[predicted_idx]

        if isinstance(target_class, int):
            if target_class in (0, 1):
                return target_class, CLASS_NAMES[target_class]
            raise ValueError(f"Invalid target class index: {target_class}. Must be 0 or 1.")

        cleaned = str(target_class).strip().lower()
        if cleaned in ("0", "fake"):
            return 0, "FAKE"
        elif cleaned in ("1", "real"):
            return 1, "REAL"
        elif cleaned in ("predicted", "pred", "auto"):
            return predicted_idx, CLASS_NAMES[predicted_idx]
        else:
            raise ValueError(
                f"Unrecognized target class: '{target_class}'. "
                "Accepted values: 'predicted', 'fake', 'real', 0, 1."
            )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[str, int]] = "predicted",
        target_size: Optional[Tuple[int, int]] = None,  # (width, height)
    ) -> GradCAMResult:
        """
        Generate a Grad-CAM attribution heatmap for the specified input tensor.

        Args:
            input_tensor: 4D PyTorch tensor of shape (1, C, H, W).
            target_class: Target class to explain ('predicted', 'real', 'fake', 0, 1).
            target_size: Optional (width, height) of original face crop for bilinear interpolation.

        Returns:
            GradCAMResult containing normalized heatmap, probabilities, and metadata.
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"input_tensor must have shape (1, C, H, W), got {input_tensor.shape}")

        input_tensor = input_tensor.to(self.device)
        self.model.eval()

        self._register_hooks()
        try:
            with torch.enable_grad():
                input_tensor.requires_grad_(True)
                logits = self.model(input_tensor)

                probabilities_tensor = F.softmax(logits, dim=1)[0]
                probabilities = {
                    name: probabilities_tensor[i].item() * 100.0
                    for i, name in enumerate(CLASS_NAMES)
                }

                predicted_idx = torch.argmax(probabilities_tensor).item()
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence = probabilities_tensor[predicted_idx].item() * 100.0

                target_idx, target_class_name = self.resolve_target_class(target_class, predicted_idx)
                target_score = logits[0, target_idx]

                self.model.zero_grad()
                if input_tensor.grad is not None:
                    input_tensor.grad.zero_()

                target_score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    f"Grad-CAM failed to capture activations or gradients on layer '{self.layer_name}'. "
                    "Ensure the layer is reached during forward/backward execution."
                )

            # Global average pooling over spatial dimensions:
            # Gradients shape: (1, C, H_feat, W_feat) -> weights: (1, C, 1, 1)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

            # Weighted combination of activation channels
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

            # Positive rectification (ReLU) - focus on features positively contributing to target class
            cam = F.relu(cam)

            raw_cam_np = cam.squeeze().cpu().numpy().astype(np.float32)

            # Min-Max normalization
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            else:
                cam = torch.zeros_like(cam)

            # Bilinear interpolation to target face crop dimensions (W, H)
            if target_size is not None:
                width, height = target_size
                cam = F.interpolate(
                    cam,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

            heatmap_np = cam.squeeze().cpu().numpy().astype(np.float32)
            heatmap_np = np.clip(heatmap_np, 0.0, 1.0)

            model_name = getattr(self.model, "__class__", type(self.model)).__name__

            return GradCAMResult(
                heatmap=heatmap_np,
                raw_cam=raw_cam_np,
                predicted_idx=predicted_idx,
                predicted_class=predicted_class,
                target_idx=target_idx,
                target_class=target_class_name,
                probabilities=probabilities,
                confidence=confidence,
                target_score=float(target_score.item()),
                layer_name=self.layer_name,
                model_name=model_name,
            )

        finally:
            self._clear_hooks()
            self.model.zero_grad()
