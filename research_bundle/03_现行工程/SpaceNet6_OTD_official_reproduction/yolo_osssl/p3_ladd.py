"""P3-only learnability-aware decomposition used by the FP method zoo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as functional

from .b0_distillation import B0Assignment


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class P3LADDModules(nn.Module):
    """Training-only P3 decomposition, reachability, and reconstruction blocks."""

    def __init__(self, channels: int, nc: int = 1) -> None:
        super().__init__()
        self.teacher_pre = ConvNormAct(channels, channels)
        self.teacher_common = nn.Conv2d(channels, channels, 1)
        self.teacher_private = nn.Sequential(ConvNormAct(channels, channels), nn.Conv2d(channels, channels, 1))
        self.teacher_reconstruct = nn.Sequential(
            ConvNormAct(2 * channels, channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.teacher_task = nn.Conv2d(channels, nc, 1)
        self.student_common = nn.Conv2d(channels, channels, 1)
        self.student_private = nn.Conv2d(channels, channels, 1)
        self.student_reconstruct = nn.Sequential(
            ConvNormAct(2 * channels, channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.reach_probe = nn.Conv2d(channels, channels, 1)
        self.stage = "a"

    def _teacher_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.teacher_pre,
            self.teacher_common,
            self.teacher_private,
            self.teacher_reconstruct,
            self.teacher_task,
            self.reach_probe,
        )

    def _student_modules(self) -> tuple[nn.Module, ...]:
        return self.student_common, self.student_private, self.student_reconstruct

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.stage == "b":
            for module in self._teacher_modules():
                module.eval()
        return self

    def teacher_split(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.teacher_pre(feature)
        common = self.teacher_common(hidden)
        private = self.teacher_private(hidden)
        reconstruction = self.teacher_reconstruct(torch.cat((common, private), dim=1))
        task_logits = self.teacher_task(common)
        return common, private, reconstruction, task_logits

    def student_split(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        common = self.student_common(feature)
        private = self.student_private(feature)
        reconstruction = self.student_reconstruct(torch.cat((common, private), dim=1))
        return common, private, reconstruction

    def configure_stage(self, stage: str) -> None:
        if stage not in {"a", "b"}:
            raise ValueError(f"unknown LADD stage {stage!r}")
        self.stage = stage
        trainable = self._teacher_modules() if stage == "a" else self._student_modules()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in trainable:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.train(self.training)


@dataclass(frozen=True)
class LADDResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _p3_maps(assignment: B0Assignment, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    extent = height * width
    foreground = assignment.fg_mask[:, :extent].reshape(assignment.fg_mask.shape[0], 1, height, width)
    scores = assignment.target_scores[:, :extent].sum(-1).reshape(assignment.fg_mask.shape[0], 1, height, width)
    return foreground, scores


def _weighted_feature_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    foreground: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    weights = foreground.to(student.dtype) * scores.to(student.dtype)
    per_token = (student.float() - teacher.detach().float()).pow(2).mean(1, keepdim=True)
    return (per_token * weights.float()).sum() / weights.float().sum().clamp_min(1e-6)


def p3_ladd_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    assignment: B0Assignment,
    modules: P3LADDModules,
    *,
    stage: str,
    delta: float = 0.2,
    negative_cap: float = 2.0,
) -> LADDResult:
    """Compute the fixed P3 LADD Stage-A or Stage-B objective."""

    if stage not in {"a", "b"}:
        raise ValueError(f"unknown LADD stage {stage!r}")
    if student_feature.shape != teacher_feature.shape or student_feature.ndim != 4:
        raise ValueError("student and optical P3 features must have matching NCHW shapes")
    foreground, target_scores = _p3_maps(assignment, student_feature.shape[-2], student_feature.shape[-1])
    z_t, u_t, teacher_reconstruction, task_logits = modules.teacher_split(teacher_feature.detach())
    z_s, _, student_reconstruction = modules.student_split(student_feature)
    q_s = modules.reach_probe(student_feature.detach() if stage == "b" else student_feature)
    if stage == "b":
        q_s = q_s.detach()

    teacher_rec = functional.l1_loss(teacher_reconstruction.float(), teacher_feature.detach().float())
    student_rec = functional.l1_loss(student_reconstruction.float(), student_feature.float())
    q_cmp = functional.normalize(q_s.float(), dim=1)
    z_cmp = functional.normalize(z_t.float(), dim=1)
    u_cmp = functional.normalize(u_t.float(), dim=1)
    distance_positive = (q_cmp - z_cmp).pow(2).sum(1, keepdim=True)
    distance_negative = (q_cmp - u_cmp).pow(2).sum(1, keepdim=True).clamp(max=float(negative_cap))
    reach_match = distance_positive.mean()
    reach_rank = functional.softplus(float(delta) + distance_positive - distance_negative).mean()
    task = functional.binary_cross_entropy_with_logits(task_logits.float(), target_scores.float(), reduction="sum")
    task = task / target_scores.float().sum().clamp_min(1.0)
    kd = _weighted_feature_mse(z_s, z_t, foreground, target_scores)

    core = 0.1 * teacher_rec + reach_match + reach_rank + task
    total = core if stage == "a" else kd + 0.1 * student_rec
    return LADDResult(
        total=total,
        components={
            "ladd_teacher_rec": teacher_rec,
            "ladd_reach_match": reach_match,
            "ladd_reach_rank": reach_rank,
            "ladd_task": task,
            "ladd_kd": kd,
            "ladd_student_rec": student_rec,
        },
    )


class P3RawFeatureAdapter(nn.Module):
    """Single 1x1 student adapter for the LADD raw-feature null control."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.projection(feature)
