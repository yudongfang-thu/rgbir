"""Ultralytics-bound YOLO11n template and backbone safety checks."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


REQUIRED_ULTRALYTICS_VERSION = "8.4.115"
BACKBONE_KEY_RE = re.compile(r"^model\.(?:[0-9]|10)\.")
BACKBONE_MODULE_COUNT = 11
FULL_YOLO11N_MODULE_COUNT = 24


class YOLOContractError(RuntimeError):
    """Raised when a YOLO runtime, artifact, or module graph drifts from lock."""


def require_locked_ultralytics(*, allow_api_drift: bool = False):
    """Import Ultralytics only for an actual YOLO run and enforce production lock."""

    try:
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
    except ImportError as exc:  # pragma: no cover - depends on production runtime
        raise YOLOContractError(
            "YOLO-OS-SSL production requires ultralytics==8.4.115; import failed."
        ) from exc
    version = str(getattr(ultralytics, "__version__", "unknown"))
    if version != REQUIRED_ULTRALYTICS_VERSION and not allow_api_drift:
        raise YOLOContractError(
            f"Ultralytics production lock is {REQUIRED_ULTRALYTICS_VERSION}, found {version}. "
            "A non-production runtime may only be used with explicit API-drift acknowledgement."
        )
    return YOLO, DetectionModel, version


def verify_yolo11n_weights(path: str | Path) -> str:
    weights_path = Path(path)
    if not weights_path.is_file():
        raise YOLOContractError(f"YOLO11n weights are not a file: {weights_path}")
    return str(weights_path.resolve())


def backbone_keys(state: Mapping[str, torch.Tensor]) -> set[str]:
    return {key for key in state if BACKBONE_KEY_RE.match(key)}


def assert_yolo_module_graph(model: nn.Module) -> None:
    """Dynamically validate the concrete module graph, including local API drift."""

    modules = getattr(model, "model", None)
    if not isinstance(modules, (nn.ModuleList, nn.Sequential)):
        raise YOLOContractError("DetectionModel has no ModuleList/Sequential .model.")
    if len(modules) != FULL_YOLO11N_MODULE_COUNT:
        raise YOLOContractError(
            f"YOLO11n module graph expected {FULL_YOLO11N_MODULE_COUNT} modules, got {len(modules)}."
        )
    if modules[10].__class__.__name__ != "C2PSA":
        raise YOLOContractError(
            f"YOLO11n backbone module 10 must be C2PSA, got {modules[10].__class__.__name__}."
        )
    invalid_sources = [index for index in range(BACKBONE_MODULE_COUNT) if getattr(modules[index], "f", -1) != -1]
    if invalid_sources:
        raise YOLOContractError(
            f"YOLO11n backbone needs sequential modules 0..10; non-sequential f values at {invalid_sources}."
        )
    keys = backbone_keys(model.state_dict())
    if not keys:
        raise YOLOContractError("YOLO11n template exposes no full-key backbone state.")
    expected_prefixes = {f"model.{index}." for index in range(BACKBONE_MODULE_COUNT)}
    seen_prefixes = {f"model.{key.split('.')[1]}." for key in keys}
    if seen_prefixes != expected_prefixes:
        raise YOLOContractError(
            f"YOLO11n backbone key prefixes drifted: expected {sorted(expected_prefixes)}, got {sorted(seen_prefixes)}."
        )


def build_yolo11n_template(
    weights: str | Path,
    *,
    nc: int,
    allow_api_drift: bool = False,
) -> tuple[nn.Module, str, str]:
    """Instantiate one complete detector template while retaining locked backbone bytes.

    Detector heads whose shape changes with ``nc`` retain the new template's
    initialized values.  Every compatible tensor is cloned from the locked
    yolo11n checkpoint and the fully assembled template is then loaded with
    ``strict=True``.
    """

    if nc <= 0:
        raise YOLOContractError(f"nc must be positive, got {nc}.")
    weight_source = verify_yolo11n_weights(weights)
    YOLO, DetectionModel, version = require_locked_ultralytics(allow_api_drift=allow_api_drift)
    loaded = YOLO(str(weights)).model
    loaded_yaml = copy.deepcopy(getattr(loaded, "yaml", None))
    if not isinstance(loaded_yaml, dict):
        raise YOLOContractError("Locked YOLO checkpoint model has no YAML dictionary.")
    loaded_yaml["nc"] = nc
    try:
        template = DetectionModel(loaded_yaml, ch=3, nc=nc, verbose=False)
    except TypeError:  # supported only for explicitly acknowledged local API drift
        if not allow_api_drift:
            raise
        template = DetectionModel(loaded_yaml, ch=3, nc=nc)
    source_state = loaded.state_dict()
    template_state = template.state_dict()
    merged_state = {
        key: (source_state[key].detach().clone() if key in source_state and source_state[key].shape == value.shape else value.detach().clone())
        for key, value in template_state.items()
    }
    template.load_state_dict(merged_state, strict=True)
    assert_yolo_module_graph(template)
    return template, weight_source, version


class YOLO11nBackbone(nn.Module):
    """Sequential modules 0..10, keeping native YOLO full state-key names."""

    def __init__(self, template: nn.Module) -> None:
        super().__init__()
        assert_yolo_module_graph(template)
        modules = getattr(template, "model")
        self.model = nn.ModuleList(copy.deepcopy(list(modules[:BACKBONE_MODULE_COUNT])))
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self(torch.zeros(1, 3, 256, 256))
        self.train(was_training)
        if output.shape != (1, 256, 8, 8):
            raise YOLOContractError(
                "YOLO11n backbone output for a 256x256 input must be [1,256,8,8], "
                f"got {tuple(output.shape)}."
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for module in self.model:
            value = module(value)
        if value.ndim != 4 or value.shape[1] != 256:
            raise YOLOContractError(
                f"YOLO11n backbone output must be [B,256,H,W], got {tuple(value.shape)}."
            )
        return value


def assert_full_backbone_state(template: nn.Module, online_backbone_state: Mapping[str, torch.Tensor]) -> set[str]:
    """Require checkpoint keys to exactly match the template's 0..10 key set."""

    assert_yolo_module_graph(template)
    expected = backbone_keys(template.state_dict())
    actual = set(online_backbone_state)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise YOLOContractError(
            "online_backbone_state key set differs from complete template backbone: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return expected


def save_template_state(path: str | Path, template: nn.Module) -> None:
    """Persist one full detector template state for all dataset arms and seeds."""

    assert_yolo_module_graph(template)
    state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    torch.save({"template_state": state}, path)


def load_template_state(path: str | Path, template: nn.Module) -> None:
    """Restore a full template exactly; partial template loading is forbidden."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - legacy torch only
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("template_state"), dict):
        raise YOLOContractError("Template state file lacks a template_state mapping.")
    template.load_state_dict(payload["template_state"], strict=True)
    assert_yolo_module_graph(template)
