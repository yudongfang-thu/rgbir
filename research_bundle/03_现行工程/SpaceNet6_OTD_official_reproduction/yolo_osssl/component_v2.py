"""Frozen-target component KD used by the OGSOD--SiXiang M1 panel.

The module deliberately keeps the M1 targets narrow.  In particular,
prototype arms consume a precomputed numerical bank and never receive an EO
image or instantiate a teacher model during detector training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from .b0_distillation import frozen_teacher_prediction, native_student_loss
from .component_kd import ObjectMetadata, _scores, object_metadata


M1_ARMS = (
    "A0_native",
    "A1_paired_optical_anchor",
    "A2_paired_optical_object_mean",
    "A3_same_object_sar_mean",
    "A4_optical_class_prototype",
    "A5_sar_class_prototype",
    "A6_optical_semantic_null",
    "A7_wrong_class_optical_prototype",
)


@dataclass(frozen=True)
class M1ArmSpec:
    component: str
    target_source: str
    requires_eo: bool = False
    requires_teacher: bool = False
    requires_prototype: bool = False
    wrong_class: bool = False


ARM_SPECS = {
    "A0_native": M1ArmSpec("native", "native"),
    "A1_paired_optical_anchor": M1ArmSpec("paired_anchor", "optical", True, True),
    "A2_paired_optical_object_mean": M1ArmSpec("paired_object_mean", "optical", True, True),
    "A3_same_object_sar_mean": M1ArmSpec("paired_object_mean", "sar", False, True),
    "A4_optical_class_prototype": M1ArmSpec("class_prototype", "optical", False, False, True),
    "A5_sar_class_prototype": M1ArmSpec("class_prototype", "sar", False, False, True),
    "A6_optical_semantic_null": M1ArmSpec("semantic_identity_null", "optical", False, False, True),
    "A7_wrong_class_optical_prototype": M1ArmSpec("class_prototype", "optical", False, False, True, True),
}


class ComponentV2Error(RuntimeError):
    pass


def arm_spec(arm: str) -> M1ArmSpec:
    try:
        return ARM_SPECS[arm]
    except KeyError as exc:
        raise ComponentV2Error(f"unknown M1 arm: {arm}") from exc


def requires_paired_images(arm: str) -> bool:
    return arm_spec(arm).requires_eo


def _bernoulli_kl(student: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    student_logits = student.float() / float(temperature)
    target_probability = torch.sigmoid(target.detach().float() / float(temperature))
    return target_probability * (
        target_probability.clamp_min(torch.finfo(torch.float32).tiny).log() - functional.logsigmoid(student_logits)
    ) + (1.0 - target_probability) * (
        (1.0 - target_probability).clamp_min(torch.finfo(torch.float32).tiny).log()
        - functional.logsigmoid(-student_logits)
    )


def object_mean_targets(
    scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
) -> dict[tuple[int, int], torch.Tensor]:
    """One class-logit vector per GT object, averaged over its assigned anchors."""

    return {
        (item.image_index, item.gt_index): scores[item.image_index, anchors[(item.image_index, item.gt_index)]].mean(
            0, keepdim=True
        )
        for item in metadata
    }


def semantic_identity_null(target: torch.Tensor, class_id: int) -> torch.Tensor:
    """Cyclically permute non-target logits while keeping every scalar invariant."""

    if target.ndim != 2 or target.shape[0] != 1 or not 0 <= int(class_id) < target.shape[1]:
        raise ComponentV2Error("semantic null expects one valid [1,nc] object target")
    non_target = [index for index in range(target.shape[1]) if index != int(class_id)]
    if len(non_target) < 2:
        raise ComponentV2Error("semantic identity null requires at least three classes")
    output = target.clone()
    output[:, non_target] = target[:, non_target[1:] + non_target[:1]]
    return output


def object_balanced_target_kd(
    student_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    targets: Mapping[tuple[int, int], torch.Tensor],
    *,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, int]:
    """Bernoulli KL, average objects first and anchors only within an object."""

    losses: list[torch.Tensor] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        target = targets.get(key)
        if target is None:
            continue
        student = student_scores[item.image_index, anchors[key]]
        losses.append(_bernoulli_kl(student, target, temperature).mean())
    if not losses:
        return student_scores.float().sum() * 0.0, 0
    return float(temperature) ** 2 * torch.stack(losses).mean() * student_scores.shape[0], len(losses)


def object_balanced_paired_anchor_kd(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, int]:
    losses = [
        _bernoulli_kl(
            student_scores[item.image_index, anchors[(item.image_index, item.gt_index)]],
            teacher_scores[item.image_index, anchors[(item.image_index, item.gt_index)]],
            temperature,
        ).mean()
        for item in metadata
    ]
    if not losses:
        return student_scores.float().sum() * 0.0, 0
    return float(temperature) ** 2 * torch.stack(losses).mean() * student_scores.shape[0], len(losses)


def _bucket(bank: Mapping[str, Any], key: tuple[int, ...], group_id: str, *, exact: bool) -> tuple[torch.Tensor, int]:
    global_key = "exact" if exact else "class"
    group_key = "exact_by_group" if exact else "class_by_group"
    total = bank[global_key].get(key)
    own = bank[group_key].get(str(group_id), {}).get(key)
    if total is None:
        return torch.empty(0), 0
    own_sum = torch.zeros_like(total["sum"]) if own is None else own["sum"]
    own_count = 0 if own is None else int(own["count"])
    return total["sum"] - own_sum, int(total["count"]) - own_count


def leave_group_out_prototype(
    bank: Mapping[str, Any],
    *,
    group_id: str,
    class_id: int,
    level: int,
    area_bin: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, str | None]:
    """Return exact `(class, FPN, area)` LOO, then class-only LOO fallback."""

    exact_sum, exact_count = _bucket(bank, (int(class_id), int(level), int(area_bin)), group_id, exact=True)
    if exact_count > 0:
        return (exact_sum / exact_count).to(device=device).unsqueeze(0), "exact"
    class_sum, class_count = _bucket(bank, (int(class_id),), group_id, exact=False)
    if class_count > 0:
        return (class_sum / class_count).to(device=device).unsqueeze(0), "class_only"
    return None, None


def prototype_targets(
    bank: Mapping[str, Any],
    metadata: Sequence[ObjectMetadata],
    *,
    device: torch.device,
    wrong_class: bool = False,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[str, int]]:
    nc = int(bank["nc"])
    targets: dict[tuple[int, int], torch.Tensor] = {}
    coverage = {"exact": 0, "class_only": 0, "missing": 0}
    for item in metadata:
        query_class = (item.class_id + 1) % nc if wrong_class else item.class_id
        target, source = leave_group_out_prototype(
            bank,
            group_id=item.group_id,
            class_id=query_class,
            level=item.level,
            area_bin=item.area_bin,
            device=device,
        )
        if target is None:
            coverage["missing"] += 1
            continue
        targets[(item.image_index, item.gt_index)] = target
        coverage[str(source)] += 1
    return targets, coverage


def load_prototype_bank(path: Path, *, expected_source: str) -> Mapping[str, Any]:
    bank = torch.load(path, map_location="cpu")
    if not isinstance(bank, Mapping) or bank.get("schema") != "component_v2_prototype_bank":
        raise ComponentV2Error("prototype bank schema is invalid")
    if bank.get("source") != expected_source:
        raise ComponentV2Error(f"prototype bank source is {bank.get('source')}, expected {expected_source}")
    return bank


@dataclass(frozen=True)
class ComponentV2Result:
    native_loss: torch.Tensor
    kd_loss: torch.Tensor
    total_loss: torch.Tensor
    detached_loss: Any
    objects: int
    used: int
    coverage: Mapping[str, int]
    target_pairs: tuple[tuple[int, torch.Tensor, torch.Tensor], ...] = ()


def component_v2_criterion(
    criterion: Any,
    prediction: Any,
    batch: Mapping[str, Any],
    *,
    arm: str,
    teacher: torch.nn.Module | None,
    prototype_bank: Mapping[str, Any] | None,
    kd_lambda: float,
    area_quartiles: Sequence[float],
    teacher_counter: list[int] | None = None,
) -> ComponentV2Result:
    spec = arm_spec(arm)
    native = native_student_loss(criterion, prediction, batch)
    zero = native.native_loss * 0.0
    if spec.component == "native" or not bool(native.assignment.fg_mask.any()):
        return ComponentV2Result(native.native_loss, zero, native.native_loss, native.detached_loss, 0, 0, {})
    metadata, anchors = object_metadata(native.assignment, native.parsed_prediction, batch, area_quartiles=area_quartiles)
    student_scores = _scores(native.parsed_prediction)
    targets: dict[tuple[int, int], torch.Tensor]
    coverage: dict[str, int]
    if spec.requires_teacher:
        if teacher is None:
            raise ComponentV2Error(f"{arm} requires its frozen {spec.target_source} teacher")
        images = batch.get("eo_img") if spec.requires_eo else batch.get("img")
        if not isinstance(images, torch.Tensor):
            raise ComponentV2Error(f"{arm} requires EO input")
        teacher_prediction = frozen_teacher_prediction(teacher, images)
        if teacher_counter is not None:
            teacher_counter[0] += 1
        teacher_scores = _scores(teacher_prediction)
        if spec.component == "paired_anchor":
            kd_loss, used = object_balanced_paired_anchor_kd(student_scores, teacher_scores, metadata, anchors)
            coverage = {"paired_anchor": used}
        else:
            targets = object_mean_targets(teacher_scores, metadata, anchors)
            if spec.component == "semantic_identity_null":
                targets = {
                    (item.image_index, item.gt_index): semantic_identity_null(
                        targets[(item.image_index, item.gt_index)], item.class_id
                    )
                    for item in metadata
                }
            kd_loss, used = object_balanced_target_kd(student_scores, metadata, anchors, targets)
            coverage = {"object_mean": used}
    else:
        if prototype_bank is None:
            raise ComponentV2Error(f"{arm} requires a precomputed prototype bank")
        targets, coverage = prototype_targets(
            prototype_bank,
            metadata,
            device=student_scores.device,
            wrong_class=spec.wrong_class,
        )
        if spec.component == "semantic_identity_null":
            targets = {
                (item.image_index, item.gt_index): semantic_identity_null(
                    targets[(item.image_index, item.gt_index)], item.class_id
                )
                for item in metadata
            }
        if coverage["missing"]:
            raise ComponentV2Error(f"{arm} prototype coverage is incomplete: {coverage['missing']} missing")
        kd_loss, used = object_balanced_target_kd(student_scores, metadata, anchors, targets)
    target_pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        student_object = student_scores[item.image_index, anchors[key]].mean(0)
        if spec.component == "paired_anchor":
            target_object = teacher_scores[item.image_index, anchors[key]].mean(0)
        else:
            target = targets.get(key)
            if target is None:
                continue
            target_object = target.reshape(-1, target.shape[-1]).mean(0)
        target_pairs.append((item.class_id, student_object.detach(), target_object.detach()))
    total = native.native_loss + float(kd_lambda) * kd_loss
    if not torch.isfinite(total):
        raise ComponentV2Error(f"{arm} produced a nonfinite loss")
    return ComponentV2Result(
        native.native_loss, kd_loss, total, native.detached_loss, len(metadata), used, coverage, tuple(target_pairs)
    )


class ComponentV2Loss:
    def __init__(
        self,
        native: Any,
        *,
        arm: str,
        teacher: torch.nn.Module | None,
        prototype_bank: Mapping[str, Any] | None,
        kd_lambda: float,
        area_quartiles: Sequence[float],
        fixed_roster: Path | None,
    ) -> None:
        self.native = native
        self.arm = arm
        self.teacher = teacher
        self.prototype_bank = prototype_bank
        self.kd_lambda = float(kd_lambda)
        self.area_quartiles = tuple(float(value) for value in area_quartiles)
        self.fixed_roster = None if fixed_roster is None else str(fixed_roster)
        self.teacher_counter = [0]
        self.stats_rows: list[dict[str, Any]] = []

    def __call__(self, prediction: Any, batch: Mapping[str, Any]):
        if not torch.is_grad_enabled():
            return self.native(prediction, batch)
        result = component_v2_criterion(
            self.native,
            prediction,
            batch,
            arm=self.arm,
            teacher=self.teacher,
            prototype_bank=self.prototype_bank,
            kd_lambda=self.kd_lambda,
            area_quartiles=self.area_quartiles,
            teacher_counter=self.teacher_counter,
        )
        self.stats_rows.append(
            {
                "objects": result.objects,
                "used": result.used,
                "coverage": dict(result.coverage),
                "finite": bool(torch.isfinite(result.total_loss).detach().cpu()),
            }
        )
        detached = dict(result.detached_loss)
        detached["component_v2_kd"] = result.kd_loss.detach()
        return result.total_loss, detached
