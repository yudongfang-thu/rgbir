"""Fit-only utilities for pre-distillation signed-transfer diagnosis.

The production probe compares two one-step branches from the same detector and
optimizer state.  It restores parameters, buffers, optimizer state, RNG state,
and train/eval mode after every comparison, so the probe cannot advance the
formal detector trajectory.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import yaml


class TransferDiagnosticError(RuntimeError):
    pass


@dataclass
class RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


@dataclass
class BranchState:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    rng: RNGState
    training: bool


@dataclass(frozen=True)
class OneStepUtility:
    native_probe_loss: float
    kd_probe_loss: float

    @property
    def utility(self) -> float:
        return self.native_probe_loss - self.kd_probe_loss


@dataclass(frozen=True)
class RandomDirectionResult:
    probe_loss: float
    direction_norm: float
    random_norm: float


def write_fit_only_dataset_yaml(source: str, output: str, *, train: str | None = None) -> None:
    """Write an Ultralytics dataset view whose every split resolves to fit.

    ``train`` may point to a frozen image-list subset.  This is used by
    property diagnostics so Ultralytics never scans labels outside the
    authorized property-fit role while constructing its dataset/cache.
    """

    with open(source, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "train" not in payload:
        raise TransferDiagnosticError("dataset YAML does not define a train split")
    if train is not None:
        payload["train"] = train
    payload["val"] = payload["train"]
    payload["test"] = payload["train"]
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def target_compatibility_metrics(
    pairs: Iterable[tuple[int, torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Aggregate fixed GT reliability and student-target soft-label compatibility."""

    rows = list(pairs)
    if not rows:
        raise TransferDiagnosticError("target compatibility requires at least one object")
    target_true: list[float] = []
    target_brier: list[float] = []
    target_margin: list[float] = []
    target_foreground: list[float] = []
    true_gap: list[float] = []
    rank_agreement: list[float] = []
    for class_id, student_logits, target_logits in rows:
        student = student_logits.detach().float().reshape(-1)
        target = target_logits.detach().float().reshape(-1)
        if student.shape != target.shape or target.numel() < 2 or not 0 <= int(class_id) < target.numel():
            raise TransferDiagnosticError("target compatibility received malformed logits")
        probability = target.sigmoid()
        truth = torch.zeros_like(probability)
        truth[int(class_id)] = 1.0
        target_true.append(float(-torch.nn.functional.logsigmoid(target[int(class_id)]).cpu()))
        target_brier.append(float((probability - truth).square().mean().cpu()))
        non_target = [index for index in range(target.numel()) if index != int(class_id)]
        target_margin.append(float((target[int(class_id)] - target[non_target].max()).cpu()))
        target_foreground.append(float((1.0 - (1.0 - probability).prod()).cpu()))
        true_gap.append(float((student[int(class_id)] - target[int(class_id)]).abs().cpu()))
        agreements: list[float] = []
        for left in range(len(non_target)):
            for right in range(left + 1, len(non_target)):
                target_sign = torch.sign(target[non_target[left]] - target[non_target[right]])
                student_sign = torch.sign(student[non_target[left]] - student[non_target[right]])
                agreements.append(float(target_sign == student_sign))
        if agreements:
            rank_agreement.append(float(np.mean(agreements)))
    output = {
        "objects": float(len(rows)),
        "teacher_true_class_nll": float(np.mean(target_true)),
        "teacher_brier": float(np.mean(target_brier)),
        "teacher_true_vs_rest_margin": float(np.mean(target_margin)),
        "teacher_foreground_probability": float(np.mean(target_foreground)),
        "student_target_true_class_gap": float(np.mean(true_gap)),
    }
    output["non_target_pairwise_rank_agreement"] = (
        float(np.mean(rank_agreement)) if rank_agreement else float("nan")
    )
    return output


def capture_rng_state() -> RNGState:
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def capture_branch_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> BranchState:
    return BranchState(
        model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        rng=capture_rng_state(),
        training=bool(model.training),
    )


def restore_branch_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: BranchState) -> None:
    model.load_state_dict(state.model, strict=True)
    optimizer.load_state_dict(state.optimizer)
    restore_rng_state(state.rng)
    model.train(state.training)


def _finite_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 0 or not bool(torch.isfinite(value).detach().cpu()):
        raise TransferDiagnosticError(f"{name} must be one finite scalar")
    return value


def _one_branch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update_loss: Callable[[], torch.Tensor],
    probe_loss: Callable[[], torch.Tensor],
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _finite_scalar(update_loss(), "update loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        measured = _finite_scalar(probe_loss(), "probe loss")
    return float(measured.detach().cpu())


def one_step_update_utility(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    native_update_loss: Callable[[], torch.Tensor],
    kd_update_loss: Callable[[], torch.Tensor],
    native_probe_loss: Callable[[], torch.Tensor],
    base_state: BranchState | None = None,
) -> OneStepUtility:
    """Compare native and native+KD one-step branches without persistent mutation."""

    base = capture_branch_state(model, optimizer) if base_state is None else base_state
    try:
        restore_branch_state(model, optimizer, base)
        native_value = _one_branch(model, optimizer, native_update_loss, native_probe_loss)
        restore_branch_state(model, optimizer, base)
        kd_value = _one_branch(model, optimizer, kd_update_loss, native_probe_loss)
    finally:
        restore_branch_state(model, optimizer, base)
    return OneStepUtility(native_probe_loss=native_value, kd_probe_loss=kd_value)


def random_direction_probe_loss(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    native_update_loss: Callable[[], torch.Tensor],
    direction_loss: Callable[[], torch.Tensor],
    native_probe_loss: Callable[[], torch.Tensor],
    seed: int,
    base_state: BranchState,
) -> RandomDirectionResult:
    """Take native plus an equal-norm random gradient step, then restore state."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        restore_branch_state(model, optimizer, base_state)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _finite_scalar(native_update_loss(), "native update loss").backward()
        direction = torch.autograd.grad(
            _finite_scalar(direction_loss(), "random-control direction loss"),
            parameters,
            allow_unused=True,
        )
        direction_sq = sum(
            (gradient.detach().double().square().sum() for gradient in direction if gradient is not None),
            torch.zeros((), dtype=torch.float64, device=parameters[0].device),
        )
        direction_norm = direction_sq.sqrt()
        if float(direction_norm.detach().cpu()) == 0.0:
            raise TransferDiagnosticError("random-control direction has zero norm")
        random_rows = [None if gradient is None else torch.randn_like(gradient) for gradient in direction]
        random_sq = sum(
            (row.detach().double().square().sum() for row in random_rows if row is not None),
            torch.zeros((), dtype=torch.float64, device=parameters[0].device),
        )
        scale = direction_norm / random_sq.sqrt()
        for parameter, row in zip(parameters, random_rows):
            if row is None:
                continue
            addition = row * scale.to(dtype=row.dtype)
            parameter.grad = addition if parameter.grad is None else parameter.grad + addition
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            measured = _finite_scalar(native_probe_loss(), "random-control probe loss")
        return RandomDirectionResult(
            probe_loss=float(measured.detach().cpu()),
            direction_norm=float(direction_norm.detach().cpu()),
            random_norm=float((random_sq.sqrt() * scale).detach().cpu()),
        )
    finally:
        restore_branch_state(model, optimizer, base_state)


def _parameter_partition(name: str, *, backbone_modules: int | None, head_module: int | None) -> str:
    parts = name.split(".")
    module_index = None
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        module_index = int(parts[1])
    elif len(parts) >= 3 and parts[0] == "model" and parts[1] == "model" and parts[2].isdigit():
        module_index = int(parts[2])
    if module_index is None:
        return "other"
    if head_module is not None and module_index == head_module:
        return "head"
    if backbone_modules is not None and module_index < backbone_modules:
        return "backbone"
    return "neck"


def gradient_compatibility(
    native_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    backbone_modules: int | None = None,
    head_module: int | None = None,
) -> dict[str, dict[str, float]]:
    """Return native-vs-KD gradient cosine and norm ratio by detector partition."""

    selected = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if not selected:
        raise TransferDiagnosticError("gradient compatibility received no trainable parameters")
    parameters = [parameter for _, parameter in selected]
    native_gradients = torch.autograd.grad(native_loss, parameters, retain_graph=True, allow_unused=True)
    kd_gradients = torch.autograd.grad(kd_loss, parameters, retain_graph=True, allow_unused=True)
    accumulators: dict[str, dict[str, torch.Tensor]] = {}
    for (name, _), native_gradient, kd_gradient in zip(selected, native_gradients, kd_gradients):
        if native_gradient is None or kd_gradient is None:
            continue
        partition = _parameter_partition(name, backbone_modules=backbone_modules, head_module=head_module)
        for key in (partition, "all"):
            row = accumulators.setdefault(
                key,
                {
                    "dot": native_gradient.new_zeros((), dtype=torch.float64),
                    "native_sq": native_gradient.new_zeros((), dtype=torch.float64),
                    "kd_sq": native_gradient.new_zeros((), dtype=torch.float64),
                },
            )
            native64 = native_gradient.detach().double()
            kd64 = kd_gradient.detach().double()
            row["dot"] += (native64 * kd64).sum()
            row["native_sq"] += native64.square().sum()
            row["kd_sq"] += kd64.square().sum()
    if "all" not in accumulators:
        raise TransferDiagnosticError("native and KD losses share no trainable parameters")
    output: dict[str, dict[str, float]] = {}
    tiny = torch.finfo(torch.float64).tiny
    for key, row in accumulators.items():
        native_norm = row["native_sq"].sqrt()
        kd_norm = row["kd_sq"].sqrt()
        cosine = row["dot"] / (native_norm * kd_norm).clamp_min(tiny)
        output[key] = {
            "cosine": float(cosine.cpu()),
            "native_norm": float(native_norm.cpu()),
            "kd_norm": float(kd_norm.cpu()),
            "norm_ratio": float((kd_norm / native_norm.clamp_min(tiny)).cpu()),
            "conflict": float(cosine < 0),
        }
    return output


def balanced_group_folds(group_sizes: Mapping[str, int], folds: int) -> dict[str, int]:
    """Assign whole groups to deterministic size-balanced folds."""

    if folds < 2 or len(group_sizes) < folds or any(int(value) <= 0 for value in group_sizes.values()):
        raise TransferDiagnosticError("group folds require positive sizes and at least one group per fold")
    loads = [0] * folds
    assignment: dict[str, int] = {}
    for group, size in sorted(group_sizes.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        fold = min(range(folds), key=lambda index: (loads[index], index))
        assignment[str(group)] = fold
        loads[fold] += int(size)
    return assignment


def bootstrap_median_interval(
    values_by_group: Mapping[str, Iterable[float]],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Group bootstrap the median of all object-balanced batch-pair utilities."""

    groups = sorted(values_by_group)
    arrays = {group: np.asarray(list(values_by_group[group]), dtype=np.float64) for group in groups}
    if not groups or any(array.size == 0 or not np.isfinite(array).all() for array in arrays.values()):
        raise TransferDiagnosticError("bootstrap groups must contain finite observations")
    observed = float(np.median(np.concatenate([arrays[group] for group in groups])))
    generator = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = generator.choice(groups, size=len(groups), replace=True)
        replicates[index] = np.median(np.concatenate([arrays[str(group)] for group in sampled]))
    return {
        "median": observed,
        "ci95_low": float(np.quantile(replicates, 0.025)),
        "ci95_high": float(np.quantile(replicates, 0.975)),
        "groups": float(len(groups)),
        "observations": float(sum(array.size for array in arrays.values())),
    }
