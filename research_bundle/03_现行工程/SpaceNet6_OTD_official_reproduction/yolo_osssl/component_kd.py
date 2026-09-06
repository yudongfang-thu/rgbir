"""Object-balanced classification and DFL witnesses for causal cross-modal KD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from .b0_distillation import (
    B0Assignment,
    criterion_reg_max,
    dfl_logits,
    frozen_teacher_prediction,
    native_student_loss,
)


COMPONENT_KD_ARMS = (
    "C0_native",
    "C1_cls_paired",
    "C2_cls_same_class",
    "C2W_cls_wrong_class",
    "C3_cls_sar",
    "C4_dfl_paired",
    "C5_dfl_shuffled",
    "D1_confidence_paired",
    "D2_contrast_paired",
    "D3_contrast_same_class",
    "D4_contrast_wrong_class",
    "R0_joint_uniform",
    "R1_component_router",
    "R2_component_random",
    "R3_quality_router",
    "R4_router_shuffled",
    "R5_router_same_modal",
)
DECOMPOSITION_ARMS = {
    "D1_confidence_paired",
    "D2_contrast_paired",
    "D3_contrast_same_class",
    "D4_contrast_wrong_class",
}
ROUTER_ARMS = {
    "R0_joint_uniform",
    "R1_component_router",
    "R2_component_random",
    "R3_quality_router",
    "R4_router_shuffled",
    "R5_router_same_modal",
}
REFERENCE_ARMS = ROUTER_ARMS - {"R0_joint_uniform"}
OPTICAL_ARMS = {
    "C1_cls_paired",
    "C2_cls_same_class",
    "C2W_cls_wrong_class",
    "C4_dfl_paired",
    "C5_dfl_shuffled",
    *DECOMPOSITION_ARMS,
    *ROUTER_ARMS,
}
SAR_ARMS = {"C3_cls_sar", "R5_router_same_modal"}
PAIRED_DATA_ARMS = OPTICAL_ARMS


class ComponentKDError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObjectMetadata:
    image_index: int
    gt_index: int
    class_id: int
    level: int
    area_bin: int
    group_id: str


@dataclass(frozen=True)
class ComponentKDResult:
    native_loss: torch.Tensor
    kd_loss: torch.Tensor
    total_loss: torch.Tensor
    detached_loss: Any
    objects: int
    shuffled_coverage: float
    class_kd_loss: torch.Tensor | None = None
    dfl_kd_loss: torch.Tensor | None = None
    route_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    route_breakdown: Mapping[str, int] | None = None


def _scores(prediction: Mapping[str, Any]) -> torch.Tensor:
    value = prediction.get("scores")
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ComponentKDError("raw class logits must be [B,nc,A]")
    return value.permute(0, 2, 1).contiguous()


def _level_ranges(prediction: Mapping[str, Any]) -> list[tuple[int, int]]:
    features = prediction.get("feats")
    if not isinstance(features, (list, tuple)) or len(features) != 3:
        raise ComponentKDError("raw prediction must expose P3/P4/P5 features")
    ranges: list[tuple[int, int]] = []
    start = 0
    for feature in features:
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise ComponentKDError("feature level must be [B,C,H,W]")
        stop = start + int(feature.shape[-2] * feature.shape[-1])
        ranges.append((start, stop))
        start = stop
    return ranges


def _padded_gt_rows(batch: Mapping[str, Any], batch_size: int) -> dict[tuple[int, int], tuple[int, float]]:
    batch_indices = batch["batch_idx"].reshape(-1).long()
    classes = batch["cls"].reshape(-1).long()
    boxes = batch["bboxes"].reshape(-1, 4)
    output: dict[tuple[int, int], tuple[int, float]] = {}
    for image_index in range(batch_size):
        rows = torch.nonzero(batch_indices == image_index, as_tuple=False).flatten()
        for gt_index, row in enumerate(rows.tolist()):
            output[(image_index, gt_index)] = (
                int(classes[row]),
                float((boxes[row, 2] * boxes[row, 3]).detach().cpu()),
            )
    return output


def _group_ids(batch: Mapping[str, Any], batch_size: int) -> list[str]:
    pair_info = batch.get("pair_info")
    if pair_info is None:
        return [f"image_{index}" for index in range(batch_size)]
    if not isinstance(pair_info, Sequence) or len(pair_info) != batch_size:
        raise ComponentKDError("pair_info must contain one record per image")
    return [str(value.get("group_id", value.get("sar_source", f"image_{index}"))) for index, value in enumerate(pair_info)]


def object_metadata(
    assignment: B0Assignment,
    prediction: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    area_quartiles: Sequence[float],
) -> tuple[list[ObjectMetadata], dict[tuple[int, int], torch.Tensor]]:
    """Recover one metadata row and foreground-anchor vector per assigned GT."""

    if len(area_quartiles) != 3:
        raise ComponentKDError("area quartiles must contain three values")
    batch_size = int(assignment.fg_mask.shape[0])
    gt_rows = _padded_gt_rows(batch, batch_size)
    group_ids = _group_ids(batch, batch_size)
    levels = _level_ranges(prediction)
    anchors: dict[tuple[int, int], torch.Tensor] = {}
    metadata: list[ObjectMetadata] = []
    for image_index in range(batch_size):
        gt_values = assignment.target_gt_idx[image_index][assignment.fg_mask[image_index]].unique(sorted=True)
        for gt_tensor in gt_values:
            gt_index = int(gt_tensor)
            key = (image_index, gt_index)
            selected = torch.nonzero(
                assignment.fg_mask[image_index] & (assignment.target_gt_idx[image_index] == gt_index),
                as_tuple=False,
            ).flatten()
            if selected.numel() == 0 or key not in gt_rows:
                continue
            level_counts = [int(((selected >= start) & (selected < stop)).sum()) for start, stop in levels]
            level = int(np.argmax(level_counts)) + 3
            class_id, area = gt_rows[key]
            area_bin = int(np.digitize(area, np.asarray(area_quartiles, dtype=np.float64)))
            anchors[key] = selected
            metadata.append(ObjectMetadata(image_index, gt_index, class_id, level, area_bin, group_ids[image_index]))
    return metadata, anchors


def matched_object_donors(
    metadata: Sequence[ObjectMetadata],
    *,
    class_relation: str = "same",
) -> dict[tuple[int, int], tuple[int, int]]:
    """Cross-group deterministic donors matched by FPN level and area bin."""

    if class_relation not in {"same", "different"}:
        raise ComponentKDError("class_relation must be same or different")
    ordered = sorted(metadata, key=lambda item: (item.group_id, item.image_index, item.gt_index))
    donors: dict[tuple[int, int], tuple[int, int]] = {}
    for position, source in enumerate(ordered):
        for step in range(1, len(ordered) + 1):
            donor = ordered[(position + step) % len(ordered)]
            class_matches = donor.class_id == source.class_id
            if (
                donor.group_id != source.group_id
                and donor.level == source.level
                and donor.area_bin == source.area_bin
                and class_matches == (class_relation == "same")
            ):
                donors[(source.image_index, source.gt_index)] = (donor.image_index, donor.gt_index)
                break
    return donors


def _bernoulli_kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    student_logits = student.float() / float(temperature)
    teacher_probability = torch.sigmoid(teacher.detach().float() / float(temperature))
    student_log_p = functional.logsigmoid(student_logits)
    student_log_not_p = functional.logsigmoid(-student_logits)
    teacher_log_p = torch.log(teacher_probability.clamp_min(torch.finfo(torch.float32).tiny))
    teacher_log_not_p = torch.log((1.0 - teacher_probability).clamp_min(torch.finfo(torch.float32).tiny))
    return teacher_probability * (teacher_log_p - student_log_p) + (1.0 - teacher_probability) * (
        teacher_log_not_p - student_log_not_p
    )


def score_common(logits: torch.Tensor) -> torch.Tensor:
    """Return the exact common-mode scalar in a class-logit decomposition."""

    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ComponentKDError("score decomposition requires at least two classes")
    return logits.mean(dim=-1, keepdim=True)


def score_contrast(logits: torch.Tensor) -> torch.Tensor:
    """Remove the common foreground-confidence shift from class logits."""

    return logits - score_common(logits)


def _categorical_kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log = functional.log_softmax(student.float() / float(temperature), dim=-1)
    teacher_probability = functional.softmax(teacher.detach().float() / float(temperature), dim=-1)
    return (
        teacher_probability
        * (teacher_probability.clamp_min(torch.finfo(torch.float32).tiny).log() - student_log)
    ).sum(-1)


def _object_teacher_target(
    teacher_scores: torch.Tensor,
    item: ObjectMetadata,
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    donor_by_object: Mapping[tuple[int, int], tuple[int, int]] | None,
    teacher_target_by_object: Mapping[tuple[int, int], torch.Tensor] | None,
) -> torch.Tensor | None:
    key = (item.image_index, item.gt_index)
    if teacher_target_by_object is not None:
        return teacher_target_by_object.get(key)
    if donor_by_object is None:
        return teacher_scores[item.image_index, anchors[key]]
    donor_key = donor_by_object.get(key)
    if donor_key is None:
        return None
    return teacher_scores[donor_key[0], anchors[donor_key]].mean(0, keepdim=True)


def object_balanced_confidence_kd(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    donor_by_object: Mapping[tuple[int, int], tuple[int, int]] | None = None,
    teacher_target_by_object: Mapping[tuple[int, int], torch.Tensor] | None = None,
    object_mask: Mapping[tuple[int, int], bool] | None = None,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        if object_mask is not None and not object_mask.get(key, False):
            continue
        teacher_target = _object_teacher_target(
            teacher_scores,
            item,
            anchors,
            donor_by_object=donor_by_object,
            teacher_target_by_object=teacher_target_by_object,
        )
        if teacher_target is None:
            continue
        student = score_common(student_scores[item.image_index, anchors[key]])
        teacher = score_common(teacher_target)
        losses.append(_bernoulli_kl(student, teacher, temperature).mean())
    if not losses:
        return student_scores.float().sum() * 0.0, 0
    return float(temperature) ** 2 * torch.stack(losses).mean() * student_scores.shape[0], len(losses)


def object_balanced_contrast_kd(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    donor_by_object: Mapping[tuple[int, int], tuple[int, int]] | None = None,
    teacher_target_by_object: Mapping[tuple[int, int], torch.Tensor] | None = None,
    object_mask: Mapping[tuple[int, int], bool] | None = None,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        if object_mask is not None and not object_mask.get(key, False):
            continue
        teacher_target = _object_teacher_target(
            teacher_scores,
            item,
            anchors,
            donor_by_object=donor_by_object,
            teacher_target_by_object=teacher_target_by_object,
        )
        if teacher_target is None:
            continue
        student = score_contrast(student_scores[item.image_index, anchors[key]])
        teacher = score_contrast(teacher_target)
        losses.append(_categorical_kl(student, teacher, temperature).mean())
    if not losses:
        return student_scores.float().sum() * 0.0, 0
    return float(temperature) ** 2 * torch.stack(losses).mean() * student_scores.shape[0], len(losses)


def object_balanced_classification_kd(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    donor_by_object: Mapping[tuple[int, int], tuple[int, int]] | None = None,
    teacher_target_by_object: Mapping[tuple[int, int], torch.Tensor] | None = None,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        donor_key = donor_by_object.get(key) if donor_by_object is not None else key
        source_anchors = anchors[key]
        if teacher_target_by_object is not None:
            teacher_target = teacher_target_by_object.get(key)
            if teacher_target is None:
                continue
        elif donor_key is None:
            continue
        elif donor_by_object is not None:
            donor_anchors = anchors[donor_key]
            teacher_target = teacher_scores[donor_key[0], donor_anchors].mean(0, keepdim=True)
        else:
            teacher_target = teacher_scores[item.image_index, source_anchors]
        loss = _bernoulli_kl(
            student_scores[item.image_index, source_anchors], teacher_target, temperature
        ).mean()
        losses.append(loss)
    if not losses:
        return student_scores.float().sum() * 0.0, 0
    return float(temperature) ** 2 * torch.stack(losses).mean() * student_scores.shape[0], len(losses)


def object_balanced_dfl_kd(
    student_dfl: torch.Tensor,
    teacher_dfl: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    donor_by_object: Mapping[tuple[int, int], tuple[int, int]] | None = None,
    teacher_target_by_object: Mapping[tuple[int, int], torch.Tensor] | None = None,
    object_mask: Mapping[tuple[int, int], bool] | None = None,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        if object_mask is not None and not object_mask.get(key, False):
            continue
        donor_key = donor_by_object.get(key) if donor_by_object is not None else key
        source_anchors = anchors[key]
        if teacher_target_by_object is not None:
            teacher_logits = teacher_target_by_object.get(key)
            if teacher_logits is None:
                continue
        elif donor_key is None:
            continue
        elif donor_by_object is None:
            teacher_logits = teacher_dfl[item.image_index, source_anchors]
        else:
            donor_anchors = anchors[donor_key]
            teacher_logits = teacher_dfl[donor_key[0], donor_anchors].mean(0, keepdim=True)
        teacher_probability = functional.softmax(teacher_logits.detach().float(), dim=-1)
        student_log_probability = functional.log_softmax(student_dfl[item.image_index, source_anchors].float(), dim=-1)
        if teacher_probability.shape[0] == 1:
            teacher_probability = teacher_probability.expand_as(student_log_probability)
        per_value = teacher_probability * (
            torch.log(teacher_probability.clamp_min(torch.finfo(torch.float32).tiny)) - student_log_probability
        )
        losses.append(per_value.sum(-1).mean())
    if not losses:
        return student_dfl.float().sum() * 0.0, 0
    return torch.stack(losses).mean() * student_dfl.shape[0], len(losses)


DonorBank = MutableMapping[tuple[int, int, int], dict[str, torch.Tensor]]


def matched_targets_with_bank(
    teacher_values: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    bank: DonorBank,
    *,
    class_relation: str = "same",
) -> dict[tuple[int, int], torch.Tensor]:
    """Resolve strict cross-group donors, then retain object means for later batches."""

    donors = matched_object_donors(metadata, class_relation=class_relation)
    targets: dict[tuple[int, int], torch.Tensor] = {}
    for item in metadata:
        source_key = (item.image_index, item.gt_index)
        donor_key = donors.get(source_key)
        if donor_key is not None:
            targets[source_key] = teacher_values[donor_key[0], anchors[donor_key]].mean(0, keepdim=True)
            continue
        candidate_strata = [
            stratum
            for stratum in bank
            if stratum[1:] == (item.level, item.area_bin)
            and (stratum[0] == item.class_id) == (class_relation == "same")
        ]
        for stratum in sorted(candidate_strata):
            for donor_group, donor_target in sorted(bank[stratum].items()):
                if donor_group != item.group_id:
                    targets[source_key] = donor_target
                    break
            if source_key in targets:
                break
    for item in metadata:
        key = (item.image_index, item.gt_index)
        stratum = (item.class_id, item.level, item.area_bin)
        bank.setdefault(stratum, {})[item.group_id] = teacher_values[item.image_index, anchors[key]].mean(
            0, keepdim=True
        ).detach()
    return targets


def object_true_class_nll(
    scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    target_by_object: Mapping[tuple[int, int], torch.Tensor] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """Categorical true-class NLL for each assigned object."""

    output: dict[tuple[int, int], torch.Tensor] = {}
    for item in metadata:
        key = (item.image_index, item.gt_index)
        values = target_by_object.get(key) if target_by_object is not None else scores[item.image_index, anchors[key]]
        if values is None:
            continue
        output[key] = -functional.log_softmax(score_contrast(values).float(), dim=-1)[:, item.class_id].mean()
    return output


def object_true_class_confidence(
    scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    target_by_object: Mapping[tuple[int, int], torch.Tensor] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    output: dict[tuple[int, int], torch.Tensor] = {}
    for item in metadata:
        key = (item.image_index, item.gt_index)
        values = target_by_object.get(key) if target_by_object is not None else scores[item.image_index, anchors[key]]
        if values is None:
            continue
        output[key] = functional.softmax(score_contrast(values).float(), dim=-1)[:, item.class_id].mean()
    return output


def object_dfl_gt_nll(
    criterion: Any,
    prediction: Mapping[str, Any],
    assignment: B0Assignment,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
) -> dict[tuple[int, int], torch.Tensor]:
    """Native two-bin DFL cross-entropy averaged over anchors for each GT object."""

    from ultralytics.utils.tal import bbox2dist

    dfl_loss = getattr(getattr(criterion, "bbox_loss", None), "dfl_loss", None)
    if dfl_loss is None:
        raise ComponentKDError("native criterion does not expose DFL loss")
    reg_max = criterion_reg_max(criterion)
    target_grid = assignment.target_bboxes.float() / assignment.stride_tensor
    target_ltrb = bbox2dist(assignment.anchor_points, target_grid, reg_max - 1)
    logits = dfl_logits(prediction, reg_max=reg_max)
    output: dict[tuple[int, int], torch.Tensor] = {}
    for item in metadata:
        key = (item.image_index, item.gt_index)
        selected = anchors[key]
        loss = dfl_loss(
            logits[item.image_index, selected].reshape(-1, reg_max).float(),
            target_ltrb[item.image_index, selected].reshape(-1, 4).float(),
        )
        output[key] = loss.float().mean()
    return output


def reference_component_masks(
    metadata: Sequence[ObjectMetadata],
    reference_class_nll: Mapping[tuple[int, int], torch.Tensor],
    teacher_class_nll: Mapping[tuple[int, int], torch.Tensor],
    reference_dfl_nll: Mapping[tuple[int, int], torch.Tensor],
    teacher_dfl_nll: Mapping[tuple[int, int], torch.Tensor],
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], bool]]:
    class_mask: dict[tuple[int, int], bool] = {}
    dfl_mask: dict[tuple[int, int], bool] = {}
    for item in metadata:
        key = (item.image_index, item.gt_index)
        class_mask[key] = bool(
            key in reference_class_nll
            and key in teacher_class_nll
            and teacher_class_nll[key].detach() < reference_class_nll[key].detach()
        )
        dfl_mask[key] = bool(
            key in reference_dfl_nll
            and key in teacher_dfl_nll
            and teacher_dfl_nll[key].detach() < reference_dfl_nll[key].detach()
        )
    return class_mask, dfl_mask


def route_state_counts(
    metadata: Sequence[ObjectMetadata],
    class_mask: Mapping[tuple[int, int], bool],
    dfl_mask: Mapping[tuple[int, int], bool],
) -> tuple[int, int, int, int]:
    counts = [0, 0, 0, 0]
    for item in metadata:
        key = (item.image_index, item.gt_index)
        has_class = bool(class_mask.get(key, False))
        has_dfl = bool(dfl_mask.get(key, False))
        index = 2 if has_class and has_dfl else 0 if has_class else 1 if has_dfl else 3
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def route_state_breakdown(
    metadata: Sequence[ObjectMetadata],
    class_mask: Mapping[tuple[int, int], bool],
    dfl_mask: Mapping[tuple[int, int], bool],
) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in metadata:
        key = (item.image_index, item.gt_index)
        has_class = bool(class_mask.get(key, False))
        has_dfl = bool(dfl_mask.get(key, False))
        state = "both" if has_class and has_dfl else "class_only" if has_class else "dfl_only" if has_dfl else "none"
        for category in (f"class_{item.class_id}", f"P{item.level}", f"area_{item.area_bin}"):
            output[f"{state}/{category}"] = output.get(f"{state}/{category}", 0) + 1
    return output


def random_matched_object_routes(
    metadata: Sequence[ObjectMetadata],
    class_mask: Mapping[tuple[int, int], bool],
    dfl_mask: Mapping[tuple[int, int], bool],
    *,
    generator: torch.Generator,
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], bool]]:
    """Permute four-state route labels per image, preserving exact overlap."""

    output_class: dict[tuple[int, int], bool] = {}
    output_dfl: dict[tuple[int, int], bool] = {}
    image_indices = sorted({item.image_index for item in metadata})
    for image_index in image_indices:
        items = [item for item in metadata if item.image_index == image_index]
        states = [
            (
                bool(class_mask.get((item.image_index, item.gt_index), False)),
                bool(dfl_mask.get((item.image_index, item.gt_index), False)),
            )
            for item in items
        ]
        order = torch.randperm(len(states), generator=generator).tolist()
        for item, state_index in zip(items, order):
            key = (item.image_index, item.gt_index)
            output_class[key], output_dfl[key] = states[state_index]
    return output_class, output_dfl


def quality_matched_object_routes(
    metadata: Sequence[ObjectMetadata],
    class_mask: Mapping[tuple[int, int], bool],
    dfl_mask: Mapping[tuple[int, int], bool],
    class_quality: Mapping[tuple[int, int], torch.Tensor],
    dfl_quality: Mapping[tuple[int, int], torch.Tensor],
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], bool]]:
    """Match per-image component doses using teacher-only class and DFL quality."""

    output_class = {(item.image_index, item.gt_index): False for item in metadata}
    output_dfl = {(item.image_index, item.gt_index): False for item in metadata}
    for image_index in sorted({item.image_index for item in metadata}):
        keys = [(item.image_index, item.gt_index) for item in metadata if item.image_index == image_index]
        n_class = sum(bool(class_mask.get(key, False)) for key in keys)
        n_dfl = sum(bool(dfl_mask.get(key, False)) for key in keys)
        class_candidates = [key for key in keys if key in class_quality]
        dfl_candidates = [key for key in keys if key in dfl_quality]
        class_order = sorted(class_candidates, key=lambda key: (-float(class_quality[key].detach()), key))
        dfl_order = sorted(dfl_candidates, key=lambda key: (float(dfl_quality[key].detach()), key))
        for key in class_order[:n_class]:
            output_class[key] = True
        for key in dfl_order[:n_dfl]:
            output_dfl[key] = True
    return output_class, output_dfl


def _class_targets_for_mode(
    teacher_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    bank: DonorBank,
    *,
    mode: str,
) -> dict[tuple[int, int], torch.Tensor] | None:
    if mode == "paired":
        return None
    if mode not in {"same_class", "wrong_class"}:
        raise ComponentKDError("class target mode must be paired, same_class, or wrong_class")
    return matched_targets_with_bank(
        teacher_scores,
        metadata,
        anchors,
        bank,
        class_relation="same" if mode == "same_class" else "different",
    )


def _router_criterion(
    criterion: Any,
    native: Any,
    batch: Mapping[str, Any],
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    arm: str,
    optical_teacher: torch.nn.Module,
    sar_teacher: torch.nn.Module | None,
    reference_teacher: torch.nn.Module | None,
    lambda_cls: float,
    lambda_dfl: float,
    class_target_mode: str,
    class_donor_bank: DonorBank,
    dfl_donor_bank: DonorBank,
    random_generator: torch.Generator,
) -> ComponentKDResult:
    optical_prediction = frozen_teacher_prediction(optical_teacher, batch["eo_img"])
    optical_scores = _scores(optical_prediction)
    reg_max = criterion_reg_max(criterion)
    optical_dfl = dfl_logits(optical_prediction, reg_max=reg_max)
    base_class_targets = _class_targets_for_mode(
        optical_scores,
        metadata,
        anchors,
        class_donor_bank,
        mode=class_target_mode,
    )
    if arm == "R0_joint_uniform":
        class_mask = {(item.image_index, item.gt_index): True for item in metadata}
        dfl_mask = dict(class_mask)
    else:
        if reference_teacher is None:
            raise ComponentKDError(f"{arm} requires a frozen SAR reference")
        reference_prediction = frozen_teacher_prediction(reference_teacher, batch["img"])
        reference_class = object_true_class_nll(_scores(reference_prediction), metadata, anchors)
        teacher_class = object_true_class_nll(
            optical_scores,
            metadata,
            anchors,
            target_by_object=base_class_targets,
        )
        reference_dfl = object_dfl_gt_nll(criterion, reference_prediction, native.assignment, metadata, anchors)
        teacher_dfl = object_dfl_gt_nll(criterion, optical_prediction, native.assignment, metadata, anchors)
        class_mask, dfl_mask = reference_component_masks(
            metadata,
            reference_class,
            teacher_class,
            reference_dfl,
            teacher_dfl,
        )
        if arm == "R2_component_random":
            class_mask, dfl_mask = random_matched_object_routes(
                metadata,
                class_mask,
                dfl_mask,
                generator=random_generator,
            )
        elif arm == "R3_quality_router":
            class_quality = object_true_class_confidence(
                optical_scores,
                metadata,
                anchors,
                target_by_object=base_class_targets,
            )
            class_mask, dfl_mask = quality_matched_object_routes(
                metadata,
                class_mask,
                dfl_mask,
                class_quality,
                teacher_dfl,
            )
    target_scores = optical_scores
    target_dfl = optical_dfl
    class_targets = base_class_targets
    dfl_targets = None
    if arm == "R4_router_shuffled":
        null_mode = "same_class" if class_target_mode == "paired" else "wrong_class"
        class_targets = _class_targets_for_mode(
            optical_scores,
            metadata,
            anchors,
            class_donor_bank,
            mode=null_mode,
        )
        dfl_targets = matched_targets_with_bank(
            optical_dfl,
            metadata,
            anchors,
            dfl_donor_bank,
            class_relation="same",
        )
    elif arm == "R5_router_same_modal":
        if sar_teacher is None:
            raise ComponentKDError("R5_router_same_modal requires a frozen SAR teacher")
        sar_prediction = frozen_teacher_prediction(sar_teacher, batch["img"])
        target_scores = _scores(sar_prediction)
        target_dfl = dfl_logits(sar_prediction, reg_max=reg_max)
        class_targets = None
    class_loss, class_used = object_balanced_contrast_kd(
        _scores(native.parsed_prediction),
        target_scores,
        metadata,
        anchors,
        teacher_target_by_object=class_targets,
        object_mask=class_mask,
    )
    dfl_loss, dfl_used = object_balanced_dfl_kd(
        dfl_logits(native.parsed_prediction, reg_max=reg_max),
        target_dfl,
        metadata,
        anchors,
        teacher_target_by_object=dfl_targets,
        object_mask=dfl_mask,
    )
    total = native.native_loss + float(lambda_cls) * class_loss + float(lambda_dfl) * dfl_loss
    coverage = float(min(class_used, dfl_used) / len(metadata)) if metadata else 0.0
    return ComponentKDResult(
        native.native_loss,
        class_loss + dfl_loss,
        total,
        native.detached_loss,
        len(metadata),
        coverage,
        class_loss,
        dfl_loss,
        route_state_counts(metadata, class_mask, dfl_mask),
        route_state_breakdown(metadata, class_mask, dfl_mask),
    )


def component_kd_criterion(
    criterion: Any,
    prediction: Any,
    batch: Mapping[str, Any],
    *,
    arm: str,
    optical_teacher: torch.nn.Module | None,
    sar_teacher: torch.nn.Module | None,
    reference_teacher: torch.nn.Module | None,
    lambda_cls: float,
    lambda_dfl: float,
    area_quartiles: Sequence[float],
    class_target_mode: str = "paired",
    class_donor_bank: DonorBank | None = None,
    dfl_donor_bank: DonorBank | None = None,
    random_generator: torch.Generator | None = None,
) -> ComponentKDResult:
    if arm not in COMPONENT_KD_ARMS:
        raise ComponentKDError(f"unknown component KD arm {arm}")
    native = native_student_loss(criterion, prediction, batch)
    if arm == "C0_native" or not bool(native.assignment.fg_mask.any()):
        zero = native.native_loss * 0.0
        return ComponentKDResult(native.native_loss, zero, native.native_loss, native.detached_loss, 0, 0.0)
    metadata, anchors = object_metadata(native.assignment, native.parsed_prediction, batch, area_quartiles=area_quartiles)
    class_donor_bank = class_donor_bank if class_donor_bank is not None else {}
    dfl_donor_bank = dfl_donor_bank if dfl_donor_bank is not None else {}
    random_generator = random_generator if random_generator is not None else torch.Generator().manual_seed(42)
    if arm in ROUTER_ARMS:
        if optical_teacher is None or "eo_img" not in batch:
            raise ComponentKDError(f"{arm} requires paired EO input and an optical teacher")
        return _router_criterion(
            criterion,
            native,
            batch,
            metadata,
            anchors,
            arm=arm,
            optical_teacher=optical_teacher,
            sar_teacher=sar_teacher,
            reference_teacher=reference_teacher,
            lambda_cls=lambda_cls,
            lambda_dfl=lambda_dfl,
            class_target_mode=class_target_mode,
            class_donor_bank=class_donor_bank,
            dfl_donor_bank=dfl_donor_bank,
            random_generator=random_generator,
        )
    if arm in OPTICAL_ARMS:
        if optical_teacher is None or "eo_img" not in batch:
            raise ComponentKDError(f"{arm} requires paired EO input and an optical teacher")
        teacher_prediction = frozen_teacher_prediction(optical_teacher, batch["eo_img"])
    elif arm in SAR_ARMS:
        if sar_teacher is None:
            raise ComponentKDError(f"{arm} requires the frozen SAR teacher")
        teacher_prediction = frozen_teacher_prediction(sar_teacher, batch["img"])
    else:
        raise ComponentKDError(f"teacher routing is undefined for {arm}")
    shuffled = arm in {
        "C2_cls_same_class",
        "C2W_cls_wrong_class",
        "C5_dfl_shuffled",
        "D3_contrast_same_class",
        "D4_contrast_wrong_class",
    }
    class_relation = "different" if arm in {"C2W_cls_wrong_class", "D4_contrast_wrong_class"} else "same"
    donor_bank = dfl_donor_bank if arm == "C5_dfl_shuffled" else class_donor_bank
    donors = (
        matched_object_donors(metadata, class_relation=class_relation)
        if shuffled and donor_bank is None
        else None
    )
    class_arms = {"C1_cls_paired", "C2_cls_same_class", "C2W_cls_wrong_class", "C3_cls_sar"}
    if arm in class_arms | DECOMPOSITION_ARMS:
        teacher_scores = _scores(teacher_prediction)
        donor_targets = (
            matched_targets_with_bank(
                teacher_scores,
                metadata,
                anchors,
                donor_bank,
                class_relation=class_relation,
            )
            if shuffled and donor_bank is not None
            else None
        )
        student_scores = _scores(native.parsed_prediction)
        if arm == "D1_confidence_paired":
            kd_loss, used = object_balanced_confidence_kd(
                student_scores,
                teacher_scores,
                metadata,
                anchors,
            )
        elif arm in DECOMPOSITION_ARMS:
            kd_loss, used = object_balanced_contrast_kd(
                student_scores,
                teacher_scores,
                metadata,
                anchors,
                donor_by_object=donors,
                teacher_target_by_object=donor_targets,
            )
        else:
            kd_loss, used = object_balanced_classification_kd(
                student_scores,
                teacher_scores,
                metadata,
                anchors,
                donor_by_object=donors,
                teacher_target_by_object=donor_targets,
            )
        coefficient = float(lambda_cls)
        class_loss = kd_loss
        dfl_loss = None
    else:
        reg_max = criterion_reg_max(criterion)
        teacher_dfl = dfl_logits(teacher_prediction, reg_max=reg_max)
        donor_targets = (
            matched_targets_with_bank(
                teacher_dfl,
                metadata,
                anchors,
                donor_bank,
                class_relation=class_relation,
            )
            if shuffled and donor_bank is not None
            else None
        )
        kd_loss, used = object_balanced_dfl_kd(
            dfl_logits(native.parsed_prediction, reg_max=reg_max),
            teacher_dfl,
            metadata,
            anchors,
            donor_by_object=donors,
            teacher_target_by_object=donor_targets,
        )
        coefficient = float(lambda_dfl)
        class_loss = None
        dfl_loss = kd_loss
    coverage = float(used / len(metadata)) if metadata else 0.0
    return ComponentKDResult(
        native.native_loss,
        kd_loss,
        native.native_loss + coefficient * kd_loss,
        native.detached_loss,
        len(metadata),
        coverage,
        class_loss,
        dfl_loss,
    )


class ComponentKDLoss:
    def __init__(
        self,
        native: Any,
        *,
        arm: str,
        optical_teacher: torch.nn.Module | None,
        sar_teacher: torch.nn.Module | None,
        reference_teacher: torch.nn.Module | None,
        lambda_cls: float,
        lambda_dfl: float,
        area_quartiles: Sequence[float],
        class_target_mode: str = "paired",
        random_seed: int = 42,
    ) -> None:
        self.native = native
        self.arm = arm
        self.optical_teacher = optical_teacher
        self.sar_teacher = sar_teacher
        self.reference_teacher = reference_teacher
        self.lambda_cls = float(lambda_cls)
        self.lambda_dfl = float(lambda_dfl)
        self.area_quartiles = tuple(float(value) for value in area_quartiles)
        self.class_target_mode = class_target_mode
        self.class_donor_bank: DonorBank = {}
        self.dfl_donor_bank: DonorBank = {}
        self.random_generator = torch.Generator().manual_seed(int(random_seed))
        self.stats_rows: list[dict[str, Any]] = []

    def __call__(self, prediction: Any, batch: Mapping[str, Any]):
        # Ultralytics invokes the training criterion during metric-only
        # validation, where paired EO is intentionally absent.  KD is a
        # training objective; validation always reports the native loss.
        if not torch.is_grad_enabled():
            return self.native(prediction, batch)
        result = component_kd_criterion(
            self.native,
            prediction,
            batch,
            arm=self.arm,
            optical_teacher=self.optical_teacher,
            sar_teacher=self.sar_teacher,
            reference_teacher=self.reference_teacher,
            lambda_cls=self.lambda_cls,
            lambda_dfl=self.lambda_dfl,
            area_quartiles=self.area_quartiles,
            class_target_mode=self.class_target_mode,
            class_donor_bank=self.class_donor_bank,
            dfl_donor_bank=self.dfl_donor_bank,
            random_generator=self.random_generator,
        )
        class_only, dfl_only, both, none = result.route_counts
        self.stats_rows.append(
            {
                "objects": float(result.objects),
                "coverage": result.shuffled_coverage,
                "class_only": float(class_only),
                "dfl_only": float(dfl_only),
                "both": float(both),
                "none": float(none),
                "route_breakdown": dict(result.route_breakdown or {}),
            }
        )
        detached = dict(result.detached_loss)
        detached["component_kd"] = result.kd_loss.detach()
        if result.class_kd_loss is not None:
            detached["class_component_kd"] = result.class_kd_loss.detach()
        if result.dfl_kd_loss is not None:
            detached["dfl_component_kd"] = result.dfl_kd_loss.detach()
        return result.total_loss, detached
