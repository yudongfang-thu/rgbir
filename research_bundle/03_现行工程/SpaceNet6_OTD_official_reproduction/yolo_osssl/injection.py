"""Strict, full-template-only YOLO backbone injection."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .yolo import YOLOContractError, assert_full_backbone_state, backbone_keys


class InjectionError(YOLOContractError):
    """Raised before an unsafe or partial backbone injection can occur."""


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key].cpu(), right[key].cpu()) for key in left)


def strict_inject_online_backbone(
    template: nn.Module, online_backbone_state: Mapping[str, torch.Tensor]
) -> nn.Module:
    """Clone full template state, replace only 0..10, and strictly reload it.

    No Ultralytics load helper or partial Torch load path is used here.
    """

    try:
        expected = assert_full_backbone_state(template, online_backbone_state)
    except YOLOContractError as exc:
        raise InjectionError(str(exc)) from exc
    template_state = template.state_dict()
    non_backbone_before = {key: value for key, value in template_state.items() if key not in expected}
    non_backbone_snapshot = clone_state(non_backbone_before)
    replacement: dict[str, torch.Tensor] = {}
    for key in expected:
        source = online_backbone_state[key]
        destination = template_state[key]
        if not isinstance(source, torch.Tensor):
            raise InjectionError(f"online_backbone_state[{key!r}] is not a Tensor.")
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise InjectionError(
                f"Backbone tensor mismatch for {key}: checkpoint {tuple(source.shape)}/{source.dtype}, "
                f"template {tuple(destination.shape)}/{destination.dtype}."
            )
        replacement[key] = source.detach().to(device=destination.device).clone()
    complete_state = {key: value.detach().clone() for key, value in template_state.items()}
    complete_state.update(replacement)
    template.load_state_dict(complete_state, strict=True)
    injected_state = template.state_dict()
    for key in expected:
        expected_tensor = online_backbone_state[key].detach().to(device=injected_state[key].device)
        if not torch.equal(injected_state[key], expected_tensor):
            raise InjectionError(f"Backbone tensor verification failed after strict injection: {key}")
    non_backbone_after = {key: value for key, value in injected_state.items() if key not in expected}
    if not states_equal(non_backbone_after, non_backbone_snapshot):
        raise InjectionError("Non-backbone template state changed during backbone injection.")
    return template


def extract_online_backbone_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Read the mandatory full-key final/recovery checkpoint field."""

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - legacy torch only
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("online_backbone_state"), dict):
        raise InjectionError("SSL checkpoint lacks dict-valued online_backbone_state.")
    state = payload["online_backbone_state"]
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise InjectionError("online_backbone_state must map string keys to Tensors.")
    return dict(state)
