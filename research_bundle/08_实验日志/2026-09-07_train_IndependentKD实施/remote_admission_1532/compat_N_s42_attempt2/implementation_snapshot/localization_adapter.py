"""Standalone L1/L_GT integration; no classification KD is evaluated here."""
from __future__ import annotations

from pathlib import Path
import sys
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from geometry_support import GeometryContract, build_paired_geometry_mask, SUPPORT_VERSION
from localization_loss import (LocalizationConfig, build_localization_selection, dfl_view,
                               gt_dfl_distribution, localization_kd, _check_aligned)


class LocalizationAdapter:
    """Loss is scalar, unweighted and not multiplied by actual batch size.

    Geometry is never inferred from tensors. An absent/unverified contract
    raises for formal callers; diagnostics can explicitly request false masks.
    """
    def __init__(self, contract=None, config=None, require_verified=True):
        self.config = config if isinstance(config, LocalizationConfig) else LocalizationConfig(**(config or {}))
        self.contract = GeometryContract.load(contract) if isinstance(contract, (str, Path)) else contract
        self.contract = self.contract if self.contract is not None else GeometryContract.unverified()
        self.require_verified = bool(require_verified)
        if self.require_verified and (not self.contract.verified or not self.contract.entries):
            raise ValueError("L1/L_GT require nonempty independently accepted exact-image geometry")

    def select(self, teacher, reference, batch, strides=(8, 16, 32), return_records=False):
        mask, geometry = build_paired_geometry_mask(reference, batch, strides, self.config, self.contract)
        selection = build_localization_selection(teacher, reference, batch, strides=strides,
            config=self.config, mode="teacher", geometry_eligible=mask,
            geometry_verified=self.contract.verified, return_records=return_records)
        selection.stats["geometry_support"] = geometry
        selection.stats["geometry_support_version"] = SUPPORT_VERSION
        return selection

    def compute(self, student_raw, teacher_raw, reference_raw, batch,
                strides=(8, 16, 32), task="L1", return_records=False):
        if task not in ("L1", "L_GT"):
            raise ValueError("LocalizationAdapter only accepts standalone L1 or L_GT")
        _check_aligned(student_raw, reference_raw, self.config, strides)
        selection = self.select(teacher_raw, reference_raw, batch, strides, return_records)
        indices = selection.selected_indices
        if len(indices):
            bi, ai = selection.batch_indices[indices], selection.anchor_indices[indices]
            student = dfl_view(student_raw["boxes"])[bi, ai]
            if task == "L_GT":
                target = gt_dfl_distribution(selection.rgb_distances[indices], student.shape[-1],
                    self.config.temperature, self.config.support_epsilon)
            else:
                target = F.softmax(dfl_view(teacher_raw["boxes"])[bi, ai].detach() / self.config.temperature, -1)
            loss = localization_kd(student, target, selection.stats["normalizer"], self.config.temperature)
        else:
            loss = student_raw["boxes"].float().sum() * 0.0
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite standalone localization loss")
        stats = dict(selection.stats, task=task, classification_kd_evaluated=False,
                     target_kind="teacher_tempered_dfl" if task == "L1" else "same_mask_tempered_gt_dfl",
                     loss_unweighted=float(loss.detach()))
        return loss, stats

    __call__ = compute
