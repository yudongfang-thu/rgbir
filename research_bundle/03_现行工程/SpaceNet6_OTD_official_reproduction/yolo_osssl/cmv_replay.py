"""Frozen CMV selections for the diagnostic 20-step replay.

The paired CMV branch defines one reference trajectory.  Target controls use
its exact selected anchors.  Selector controls use the same SAR-safe candidate
support and match its selected-object dose independently in every image.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fm_resource_distillation import (
    CMVState,
    matched_object_score_mask,
    matched_random_object_mask,
    mixed_teacher_target,
)


REPLAY_VARIANTS = (
    "paired_cmv",
    "shuffled_veto",
    "random_veto",
    "arbitration",
    "optical_target",
    "sar_discrepancy",
)


@dataclass(frozen=True)
class FrozenCMVReplay:
    """Detached masks, targets and weights shared by one replay micro-batch."""

    support: torch.Tensor
    selected: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    weights: torch.Tensor
    batch_indices: torch.Tensor
    target_indices: torch.Tensor


def object_dose_by_image(
    selected: torch.Tensor,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> dict[int, int]:
    """Count selected targets per image, not selected anchors."""

    dose: dict[int, int] = {}
    if not selected.numel():
        return dose
    pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
    objects, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    for image in torch.unique(batch_indices, sorted=True).tolist():
        object_ids = (objects[:, 0] == int(image)).nonzero(as_tuple=False).flatten()
        dose[int(image)] = sum(bool(selected[inverse == object_id].any()) for object_id in object_ids)
    return dose


def freeze_cmv_replay(
    paired: CMVState,
    shuffled: CMVState,
    student_logits: torch.Tensor,
    weights: torch.Tensor,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    generator: torch.Generator,
) -> FrozenCMVReplay:
    """Freeze the reference masks and route all replay attribution variants.

    ``paired_cmv``, ``arbitration`` and ``optical_target`` share the exact
    paired CMV mask.  The shuffled, random and SAR-discrepancy selectors are
    restricted to the same SAR-safe support and match the paired object dose
    per image.  Thus a selector comparison changes *which* safe objects are
    retained without changing how many are retained.
    """

    reference = paired.gate.selected.detach().clone()
    support = paired.gate.safe.detach().clone()

    # An unreliable optical teacher does not veto in CMV, so rank it ahead of
    # reliability-conditioned agreement.  Reliable teachers are ordered by
    # correction agreement, making the shuffled route the matched-dose form of
    # the same veto rule.
    shuffled_score = torch.where(
        shuffled.gate.optical_reliable,
        shuffled.gate.agreement.float(),
        shuffled.gate.agreement.new_full(shuffled.gate.agreement.shape, 2.0),
    )
    shuffled_selected = matched_object_score_mask(
        shuffled_score,
        batch_indices,
        target_indices,
        reference,
        candidate=support,
    )
    random_selected = matched_random_object_mask(
        support,
        reference,
        batch_indices,
        target_indices,
        generator=generator,
    )
    sar_discrepancy = (
        paired.sar_target.float()
        * (
            paired.sar_target.float().clamp_min(torch.finfo(torch.float32).tiny).log()
            - torch.log_softmax(student_logits.detach().float(), dim=-1)
        )
    ).sum(-1).mean(-1)
    discrepancy_selected = matched_object_score_mask(
        sar_discrepancy,
        batch_indices,
        target_indices,
        reference,
        candidate=support,
    )

    selected = {
        "paired_cmv": reference,
        "shuffled_veto": shuffled_selected.detach(),
        "random_veto": random_selected.detach(),
        "arbitration": reference.clone(),
        "optical_target": reference.clone(),
        "sar_discrepancy": discrepancy_selected.detach(),
    }
    targets = {
        "paired_cmv": paired.sar_target.detach(),
        "shuffled_veto": paired.sar_target.detach(),
        "random_veto": paired.sar_target.detach(),
        "arbitration": mixed_teacher_target(
            paired.sar_target,
            paired.optical_target,
            paired.gate.optical_reliable,
        ),
        "optical_target": paired.optical_target.detach(),
        "sar_discrepancy": paired.sar_target.detach(),
    }
    reference_dose = object_dose_by_image(reference, batch_indices, target_indices)
    for name in ("shuffled_veto", "random_veto", "sar_discrepancy"):
        if object_dose_by_image(selected[name], batch_indices, target_indices) != reference_dose:
            raise RuntimeError(f"{name} did not preserve the paired per-image object dose")
        if bool((selected[name] & ~support).any()):
            raise RuntimeError(f"{name} selected an object outside the frozen SAR-safe support")

    return FrozenCMVReplay(
        support=support,
        selected=selected,
        targets=targets,
        weights=weights.detach().clone(),
        batch_indices=batch_indices.detach().clone(),
        target_indices=target_indices.detach().clone(),
    )
