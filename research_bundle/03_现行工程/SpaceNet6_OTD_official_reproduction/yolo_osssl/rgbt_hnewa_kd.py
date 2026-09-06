"""CPU-testable Hnewa-CMKD-MSE-inspired YOLO11 feature distillation.

The frozen port uses a six-channel weak+strong fusion teacher, reads the three
native post-neck tensors exposed by Detect, and applies the literal equal-level
raw feature MSE with ``alpha=0.5``.  This module deliberately does not own a
trainer or detector loss, and its runtime distiller installs no hooks, so no
teacher-side module can enter the exported student graph.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import nn


HNEWA_YOLO11_TAP_LAYERS = (16, 19, 22)
HNEWA_FEATURE_MSE_ALPHA = 0.5
YOLO11_FIRST_CONV_WEIGHT_KEY = "model.0.conv.weight"


class HnewaCMKDError(RuntimeError):
    """Raised when the frozen Hnewa-inspired feature contract drifts."""


def six_channel_fusion(weak_images: torch.Tensor, strong_images: torch.Tensor) -> torch.Tensor:
    """Concatenate aligned three-channel modalities as the fusion-teacher input."""

    if weak_images.ndim != 4 or strong_images.ndim != 4:
        raise HnewaCMKDError("weak and strong images must be [B,3,H,W]")
    if weak_images.shape != strong_images.shape or weak_images.shape[1] != 3:
        raise HnewaCMKDError("weak and strong images must have matching [B,3,H,W] shapes")
    if weak_images.device != strong_images.device or weak_images.dtype != strong_images.dtype:
        raise HnewaCMKDError("weak and strong images must share device and dtype")
    return torch.cat((weak_images, strong_images), dim=1)


def _yolo_layer_stack(model: nn.Module) -> nn.ModuleList | nn.Sequential:
    """Resolve either an Ultralytics DetectionModel or its YOLO wrapper."""

    candidates = (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None))
    for candidate in candidates:
        if isinstance(candidate, (nn.ModuleList, nn.Sequential)):
            return candidate
    raise HnewaCMKDError("model does not expose the Ultralytics YOLO layer stack")


class YOLO11PyramidTap:
    """Layer-identity oracle for CPU tests, not the runtime KD interface."""

    def __init__(self, model: nn.Module) -> None:
        layers = _yolo_layer_stack(model)
        if len(layers) <= HNEWA_YOLO11_TAP_LAYERS[-1]:
            raise HnewaCMKDError("YOLO layer stack does not contain layers 16/19/22")
        self._features: dict[int, torch.Tensor] = {}
        self._handles = [
            layers[index].register_forward_hook(self._make_hook(index))
            for index in HNEWA_YOLO11_TAP_LAYERS
        ]

    def _make_hook(self, index: int):
        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise HnewaCMKDError(f"YOLO layer {index} did not return one tensor")
            self._features[index] = output

        return capture

    def pop(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(self._features) != set(HNEWA_YOLO11_TAP_LAYERS):
            raise HnewaCMKDError("one complete YOLO forward must populate layers 16/19/22")
        features = tuple(self._features[index] for index in HNEWA_YOLO11_TAP_LAYERS)
        self._features.clear()
        return features  # type: ignore[return-value]

    def clear(self) -> None:
        self._features.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._features.clear()


@dataclass(frozen=True)
class HnewaFeatureMSEResult:
    """Literal per-level, equal-level-mean, and alpha-weighted losses."""

    per_level: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    raw_mean: torch.Tensor
    weighted: torch.Tensor


def _raw_prediction(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (tuple, list)):
        mappings = [item for item in value if isinstance(item, Mapping)]
        if len(mappings) == 1:
            return mappings[0]
    raise HnewaCMKDError("YOLO forward did not expose one raw prediction mapping")


def yolo11_prediction_features(
    model: nn.Module,
    prediction: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read native Detect inputs after asserting its frozen source layers."""

    layers = _yolo_layer_stack(model)
    detect_sources = getattr(layers[-1], "f", None)
    normalized_sources = list(detect_sources) if isinstance(detect_sources, (tuple, list)) else detect_sources
    if normalized_sources != list(HNEWA_YOLO11_TAP_LAYERS):
        raise HnewaCMKDError("YOLO Detect.f must be [16,19,22]")
    features = _raw_prediction(prediction).get("feats")
    if not isinstance(features, (tuple, list)) or len(features) != 3:
        raise HnewaCMKDError("YOLO raw prediction must expose three Detect feature maps")
    if not all(isinstance(feature, torch.Tensor) and feature.ndim == 4 for feature in features):
        raise HnewaCMKDError("YOLO Detect features must be [B,C,H,W] tensors")
    return features[0], features[1], features[2]


def hnewa_feature_mse(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
) -> HnewaFeatureMSEResult:
    """Compute ``0.5 * mean(MSE(P3), MSE(P4), MSE(P5))`` exactly."""

    if len(student_features) != 3 or len(teacher_features) != 3:
        raise HnewaCMKDError("Hnewa-inspired YOLO11 KD requires exactly P3/P4/P5")
    losses: list[torch.Tensor] = []
    for student, teacher in zip(student_features, teacher_features):
        if student.ndim != 4 or student.shape != teacher.shape:
            raise HnewaCMKDError("student and teacher P3/P4/P5 shapes must match exactly")
        losses.append(functional.mse_loss(student, teacher.detach()))
    per_level = (losses[0], losses[1], losses[2])
    raw_mean = torch.stack(per_level).mean()
    return HnewaFeatureMSEResult(per_level, raw_mean, HNEWA_FEATURE_MSE_ALPHA * raw_mean)


def freeze_hnewa_teacher(teacher: nn.Module) -> None:
    """Give the distiller exclusive eval/no-gradient ownership of its teacher."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def combine_hnewa_total_loss(
    native_batch_loss: torch.Tensor,
    feature_mse: HnewaFeatureMSEResult,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Match Ultralytics' ``loss * B`` scale for the batch-mean feature MSE."""

    if batch_size <= 0:
        raise HnewaCMKDError("batch_size must be positive")
    return native_batch_loss + float(batch_size) * feature_mse.weighted


def transplant_yolo11_fusion_teacher_weights(
    fusion_teacher: nn.Module,
    three_channel_source: nn.Module | Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Load every compatible YOLO11 tensor while explicitly preserving the 6ch stem."""

    source_state = (
        three_channel_source.state_dict()
        if isinstance(three_channel_source, nn.Module)
        else three_channel_source
    )
    teacher_state = fusion_teacher.state_dict()
    if YOLO11_FIRST_CONV_WEIGHT_KEY not in source_state or YOLO11_FIRST_CONV_WEIGHT_KEY not in teacher_state:
        raise HnewaCMKDError("YOLO11 first convolution state key is missing")
    source_stem = source_state[YOLO11_FIRST_CONV_WEIGHT_KEY]
    teacher_stem = teacher_state[YOLO11_FIRST_CONV_WEIGHT_KEY]
    if source_stem.ndim != 4 or teacher_stem.ndim != 4 or source_stem.shape[1] != 3 or teacher_stem.shape[1] != 6:
        raise HnewaCMKDError("fusion transplant requires a 3ch source and default 6ch teacher stem")

    default_stem = teacher_stem.detach().clone()
    merged: dict[str, torch.Tensor] = {}
    loaded: list[str] = []
    for key, target in teacher_state.items():
        source = source_state.get(key)
        if key != YOLO11_FIRST_CONV_WEIGHT_KEY and isinstance(source, torch.Tensor) and source.shape == target.shape:
            merged[key] = source.detach().clone()
            loaded.append(key)
        else:
            merged[key] = target.detach().clone()
    fusion_teacher.load_state_dict(merged, strict=True)
    if not torch.equal(fusion_teacher.state_dict()[YOLO11_FIRST_CONV_WEIGHT_KEY], default_stem):
        raise HnewaCMKDError("fusion teacher first convolution changed during weight transplant")
    return tuple(loaded)


def build_six_channel_yolo11_fusion_teacher(
    three_channel_student: nn.Module,
    *,
    detection_model_factory: Any | None = None,
) -> nn.Module:
    """Construct a default 6ch YOLO11 and transplant only non-stem compatible weights."""

    config = getattr(three_channel_student, "yaml", None)
    nc = getattr(three_channel_student, "nc", None)
    if not isinstance(nc, int) and isinstance(config, dict):
        nc = config.get("nc")
    if not isinstance(config, dict) or not isinstance(nc, int):
        raise HnewaCMKDError("student must expose Ultralytics yaml and integer nc")
    if detection_model_factory is None:
        from .yolo import require_locked_ultralytics

        _, detection_model_factory, _ = require_locked_ultralytics()
    fusion_teacher = detection_model_factory(copy.deepcopy(config), ch=6, nc=nc, verbose=False)
    transplant_yolo11_fusion_teacher_weights(fusion_teacher, three_channel_student)
    return fusion_teacher


class HnewaFeatureDistiller:
    """Plain-object raw-feature owner; it is intentionally not an ``nn.Module``.

    The caller supplies the ordinary student prediction.  ``loss`` reads its
    native ``prediction['feats']``, performs exactly one frozen teacher forward,
    and reads the corresponding teacher raw features.  No persistent forward
    hook is installed, so EMA, deepcopy, profiling, and export remain native.
    """

    def __init__(self, student: nn.Module, fusion_teacher: nn.Module) -> None:
        self.student = student
        self.fusion_teacher = fusion_teacher
        freeze_hnewa_teacher(self.fusion_teacher)

    def loss(self, student_prediction: Any, fusion_images: torch.Tensor) -> HnewaFeatureMSEResult:
        if fusion_images.ndim != 4 or fusion_images.shape[1] != 6:
            raise HnewaCMKDError("fusion teacher input must be [B,6,H,W]")
        if any(parameter.requires_grad for parameter in self.fusion_teacher.parameters()):
            raise HnewaCMKDError("fusion teacher parameters must remain frozen")
        student_features = yolo11_prediction_features(self.student, student_prediction)
        self.fusion_teacher.eval()
        with torch.no_grad():
            teacher_prediction = self.fusion_teacher(fusion_images)
        teacher_features = tuple(
            feature.detach()
            for feature in yolo11_prediction_features(self.fusion_teacher, teacher_prediction)
        )
        return hnewa_feature_mse(student_features, teacher_features)

    def close(self) -> None:
        """Symmetric runtime lifecycle hook; there are no persistent handles to remove."""

    def __enter__(self) -> "HnewaFeatureDistiller":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
