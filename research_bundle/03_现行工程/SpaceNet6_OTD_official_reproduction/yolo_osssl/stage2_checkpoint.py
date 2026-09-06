"""Small model-state helpers for the Stage-II QAT lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


class Stage2CheckpointError(RuntimeError):
    pass


COMMON_STATE_FORMAT = "stage2-common-qat-v1"
FINAL_STATE_FORMAT = "stage2-final-qat-v1"
DEFAULT_PRECISION = {"weight_bitwidth": 8, "activation_bitwidth": 8}


def checkpoint_precision(payload: Mapping[str, Any]) -> dict[str, int]:
    """Return stored precision, treating legacy Stage-II states as W8A8."""

    value = payload.get("precision", DEFAULT_PRECISION)
    if not isinstance(value, Mapping):
        raise Stage2CheckpointError("invalid Stage-II precision metadata")
    return {
        "weight_bitwidth": int(value["weight_bitwidth"]),
        "activation_bitwidth": int(value["activation_bitwidth"]),
    }


def capture_quantizer_state(quant_sim: Any) -> Mapping[str, Any]:
    state_dict = getattr(quant_sim, "quantizer_state_dict", None)
    return state_dict() if callable(state_dict) else {}


def restore_quantizer_state(quant_sim: Any, state: Mapping[str, Any]) -> None:
    model = getattr(quant_sim, "model", None)
    if model is not None and state:
        model.load_state_dict(state, strict=False)


def capture_common_state(
    *,
    model: torch.nn.Module,
    quant_sim: Any,
    common_epochs: int = 30,
    precision: Mapping[str, int] = DEFAULT_PRECISION,
) -> dict[str, Any]:
    """Store the shared post-E30 QAT model/encoding state for every arm."""

    if int(common_epochs) != 30:
        raise Stage2CheckpointError("Stage-II common QAT phase is fixed at 30 epochs")
    return {
        "format": COMMON_STATE_FORMAT,
        "common_epochs": int(common_epochs),
        "precision": checkpoint_precision({"precision": precision}),
        "model": model.state_dict(),
        "quantizer_state": dict(capture_quantizer_state(quant_sim)),
    }


def save_common_state(path: str | Path, state: Mapping[str, Any]) -> None:
    """Persist the simple shared state after E30 recalibration and freezing."""

    target = Path(path)
    if state.get("format") != COMMON_STATE_FORMAT or int(state.get("common_epochs", -1)) != 30:
        raise Stage2CheckpointError("refusing to save a non-E30 common state")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), target)


def load_common_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    quant_sim: Any,
    precision: Mapping[str, int] = DEFAULT_PRECISION,
) -> dict[str, Any]:
    """Restore only the common model and frozen AIMET encoding state."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != COMMON_STATE_FORMAT
        or int(payload.get("common_epochs", -1)) != 30
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("quantizer_state"), Mapping)
    ):
        raise Stage2CheckpointError("invalid Stage-II E30 common state")
    if checkpoint_precision(payload) != checkpoint_precision({"precision": precision}):
        raise Stage2CheckpointError("Stage-II common precision does not match the requested QuantSim")
    model.load_state_dict(payload["model"], strict=True)
    restore_quantizer_state(quant_sim, payload["quantizer_state"])
    return payload


def capture_final_state(
    *,
    model: torch.nn.Module,
    quant_sim: Any,
    arm: str,
    tau_det: float,
    mask_density: Mapping[str, Any],
    canary_steps: int | None = None,
    precision: Mapping[str, int] = DEFAULT_PRECISION,
) -> dict[str, Any]:
    """Store the final fake-Q model used by evaluation and export."""

    return {
        "format": FINAL_STATE_FORMAT,
        "method_id": "DCODFL-QAT-v1",
        "arm": str(arm),
        "precision": checkpoint_precision({"precision": precision}),
        "model": model.state_dict(),
        "quantizer_state": dict(capture_quantizer_state(quant_sim)),
        "tau_det": float(tau_det),
        "mask_density": dict(mask_density),
        "canary_steps": int(canary_steps) if canary_steps is not None else None,
    }


def save_final_state(path: str | Path, state: Mapping[str, Any]) -> None:
    target = Path(path)
    if state.get("format") != FINAL_STATE_FORMAT:
        raise Stage2CheckpointError("invalid Stage-II final state")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), target)
