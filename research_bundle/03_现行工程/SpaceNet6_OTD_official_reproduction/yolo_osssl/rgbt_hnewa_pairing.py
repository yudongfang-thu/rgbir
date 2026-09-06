"""Shared-geometry RGB-T loading for the frozen Hnewa-inspired baseline."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch.utils.data import Dataset

from yolo_osssl.paired_detection import PairedDetectionDataset, transform_paired_record


def _rgbt_pair_info(pair_info: Mapping[str, Any]) -> dict[str, Any]:
    renamed = {
        "weak_source": str(pair_info["sar_source"]),
        "strong_source": str(pair_info["eo_source"]),
    }
    if "shuffled_eo_source" in pair_info:
        renamed["shuffled_strong_source"] = str(pair_info["shuffled_eo_source"])
    if "group_id" in pair_info:
        renamed["group_id"] = str(pair_info["group_id"])
    return renamed


def _named_rgbt_row(
    weak: Mapping[str, Any],
    strong_img: torch.Tensor,
    pair_info: Mapping[str, Any],
    shuffled_strong_img: torch.Tensor | None = None,
) -> dict[str, Any]:
    row = dict(weak)
    weak_img = row.get("img")
    if not isinstance(weak_img, torch.Tensor):
        raise TypeError("transformed weak record must contain tensor img")
    row["weak_img"] = weak_img
    row["strong_img"] = strong_img
    if shuffled_strong_img is not None:
        row["shuffled_strong_img"] = shuffled_strong_img
    row["pair_info"] = _rgbt_pair_info(pair_info)
    return row


def transform_rgbt_record(
    weak_record: Mapping[str, Any],
    strong_record: Mapping[str, Any],
    transform: Callable[[dict[str, Any]], Mapping[str, Any]],
    shuffled_strong_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one replayed transform to a weak/strong pair and optional donor.

    The returned ``weak_img`` is the exact native ``img`` tensor.  A shuffled
    control is a separate ``shuffled_strong_img`` tensor, so callers can replace
    only the strong half of ``cat([weak_img, strong_img], dim=1)``.
    """

    transformed = transform_paired_record(
        weak_record,
        strong_record,
        transform,
        donor_record=shuffled_strong_record,
    )
    if shuffled_strong_record is None:
        weak, strong_img, pair_info = transformed
        return _named_rgbt_row(weak, strong_img, pair_info)
    weak, strong_img, shuffled_strong_img, pair_info = transformed
    return _named_rgbt_row(weak, strong_img, pair_info, shuffled_strong_img)


class RGBTSharedGeometryDataset(Dataset[dict[str, Any]]):
    """Modality-neutral adapter over the existing Ultralytics paired loader.

    ``img`` remains the native Ultralytics weak-modality input.  ``weak_img`` is
    an alias for it; ``strong_img`` and optional ``shuffled_strong_img`` carry
    the synchronized teacher views.  Six-channel fusion is intentionally left
    to the training-time teacher interface.
    """

    def __init__(
        self,
        base: Any,
        strong_by_weak: Mapping[str, str],
        shuffled_strong_by_weak: Mapping[str, str] | None = None,
        group_by_weak: Mapping[str, str] | None = None,
    ) -> None:
        self.paired = PairedDetectionDataset(
            base,
            eo_by_sar=strong_by_weak,
            shuffled_eo_by_sar=shuffled_strong_by_weak,
            group_by_sar=group_by_weak,
        )

    def __len__(self) -> int:
        return len(self.paired)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.paired, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.paired[index])
        strong_img = row.pop("eo_img")
        shuffled_strong_img = row.pop("shuffled_eo_img", None)
        return _named_rgbt_row(row, strong_img, row.pop("pair_info"), shuffled_strong_img)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [dict(row) for row in batch]
        strong_images = [row.pop("strong_img") for row in rows]
        pair_info = [row.pop("pair_info") for row in rows]
        has_shuffled = ["shuffled_strong_img" in row for row in rows]
        if any(has_shuffled) and not all(has_shuffled):
            raise ValueError("a batch cannot mix paired and shuffled RGB-T rows")
        shuffled_images = [row.pop("shuffled_strong_img") for row in rows] if all(has_shuffled) else None
        for row in rows:
            row.pop("weak_img")

        output = self.paired.base.collate_fn(rows)
        output["weak_img"] = output["img"]
        output["strong_img"] = torch.stack(strong_images)
        output["pair_info"] = pair_info
        if shuffled_images is not None:
            output["shuffled_strong_img"] = torch.stack(shuffled_images)
        return output
