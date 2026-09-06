"""Frozen production augmentations shared by all YOLO OS-SSL arms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class GeometryParams:
    top: int
    left: int
    height: int
    width: int
    flip: bool


class ProductionPairTransform:
    """Apply one shared geometry and independent frozen non-geometric views.

    Inputs may be ``[C,H,W]`` tensors in ``[0,1]``/``[0,255]`` or Pillow RGB
    images.  The output is always three-channel float32 in ``[0,1]``.
    """

    image_size = 256
    scale = (0.6, 1.0)
    ratio = (0.75, 4.0 / 3.0)
    hflip_probability = 0.5
    color_probability = 0.8
    color_magnitude = 0.2
    blur_probability = 0.5
    blur_sigma = (0.1, 2.0)

    def __init__(self, *, image_size: int = 256) -> None:
        if image_size != self.image_size:
            raise ValueError("YOLO-OS-SSL production transform requires image_size=256.")

    @staticmethod
    def _rand(generator: torch.Generator | None) -> float:
        return float(torch.rand((), generator=generator).item())

    @staticmethod
    def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
        return low + (high - low) * ProductionPairTransform._rand(generator)

    @staticmethod
    def _as_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            value = image.detach().clone()
        else:
            # PairManifestDataset always supplies RGB Pillow images.  Using
            # Pillow's bytes directly keeps NumPy out of the runtime contract.
            try:
                width, height = image.size
                raw = image.tobytes()
                if hasattr(torch, "frombuffer"):
                    value = torch.frombuffer(memoryview(raw), dtype=torch.uint8).clone()
                else:  # PyTorch 1.8 in the frozen MMDetection2/R50 runtime
                    value = torch.ByteTensor(torch.ByteStorage.from_buffer(raw)).clone()
                value = value.reshape(height, width, 3).permute(2, 0, 1)
            except (AttributeError, RuntimeError, ValueError) as exc:
                raise ValueError("Non-tensor input must be a Pillow RGB-compatible image.") from exc
        if value.ndim != 3:
            raise ValueError(f"Expected image shape [C,H,W] or [H,W,C], got {tuple(value.shape)}.")
        if value.shape[0] not in {1, 3} and value.shape[-1] in {1, 3}:
            value = value.permute(2, 0, 1)
        if value.shape[0] == 1:
            value = value.repeat(3, 1, 1)
        if value.shape[0] != 3:
            raise ValueError(f"YOLO-OS-SSL requires one or three input channels, got {value.shape[0]}.")
        value = value.to(dtype=torch.float32)
        if value.numel() and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
            value = value / 255.0
        return value.clamp_(0.0, 1.0)

    def sample_geometry(self, image: torch.Tensor, *, generator: torch.Generator | None = None) -> GeometryParams:
        """Sample the one random-resized-crop and flip applied to both branches."""

        _, height, width = image.shape
        area = height * width
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = area * self._uniform(self.scale[0], self.scale[1], generator)
            aspect = math.exp(self._uniform(log_ratio[0], log_ratio[1], generator))
            crop_width = int(round(math.sqrt(target_area * aspect)))
            crop_height = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = int(torch.randint(0, height - crop_height + 1, (), generator=generator).item())
                left = int(torch.randint(0, width - crop_width + 1, (), generator=generator).item())
                return GeometryParams(top, left, crop_height, crop_width, self._rand(generator) < self.hflip_probability)

        # Exact torchvision RandomResizedCrop-style central fallback.
        input_ratio = width / height
        if input_ratio < self.ratio[0]:
            crop_width = width
            crop_height = int(round(crop_width / self.ratio[0]))
        elif input_ratio > self.ratio[1]:
            crop_height = height
            crop_width = int(round(crop_height * self.ratio[1]))
        else:
            crop_width, crop_height = width, height
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        return GeometryParams(top, left, crop_height, crop_width, self._rand(generator) < self.hflip_probability)

    def apply_geometry(self, image: Any, params: GeometryParams) -> torch.Tensor:
        value = self._as_tensor(image)
        cropped = value[:, params.top : params.top + params.height, params.left : params.left + params.width]
        if cropped.shape[-2:] != (params.height, params.width):
            raise ValueError("Geometry parameters crop outside image bounds.")
        resized = functional.interpolate(
            cropped.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        return torch.flip(resized, dims=(2,)) if params.flip else resized

    @staticmethod
    def _gaussian_kernel(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel[:, None] * kernel[None, :]

    def apply_non_geometric(self, image: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Apply branch-local brightness/contrast and Gaussian blur."""

        value = image
        if self._rand(generator) < self.color_probability:
            brightness = self._uniform(1.0 - self.color_magnitude, 1.0 + self.color_magnitude, generator)
            contrast = self._uniform(1.0 - self.color_magnitude, 1.0 + self.color_magnitude, generator)
            value = value * brightness
            mean = value.mean(dim=(-2, -1), keepdim=True)
            value = (value - mean) * contrast + mean
            value = value.clamp(0.0, 1.0)
        if self._rand(generator) < self.blur_probability:
            sigma = self._uniform(self.blur_sigma[0], self.blur_sigma[1], generator)
            kernel = self._gaussian_kernel(sigma, device=value.device, dtype=value.dtype)
            radius = kernel.shape[0] // 2
            weight = kernel.expand(value.shape[0], 1, -1, -1)
            value = functional.conv2d(value.unsqueeze(0), weight, padding=radius, groups=value.shape[0]).squeeze(0)
        return value.clamp(0.0, 1.0)

    def __call__(
        self, first: Any, second: Any, *, generator: torch.Generator | None = None, return_geometry: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, GeometryParams]:
        first_tensor = self._as_tensor(first)
        second_tensor = self._as_tensor(second)
        if first_tensor.shape[-2:] != second_tensor.shape[-2:]:
            raise ValueError("Paired branches must have identical spatial dimensions for shared geometry.")
        geometry = self.sample_geometry(first_tensor, generator=generator)
        first_view = self.apply_non_geometric(self.apply_geometry(first_tensor, geometry), generator=generator)
        second_view = self.apply_non_geometric(self.apply_geometry(second_tensor, geometry), generator=generator)
        if return_geometry:
            return first_view, second_view, geometry
        return first_view, second_view
