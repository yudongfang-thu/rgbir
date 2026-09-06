"""Small pure helpers shared by the fixed-GT-anchor roster scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


class FixedRosterError(RuntimeError):
    pass


def roster_key(image_id: str, gt_index: int) -> tuple[str, int]:
    return str(Path(image_id).resolve()), int(gt_index)


def index_roster(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = roster_key(str(row["image_id"]), int(row["gt_index"]))
        if key in indexed:
            raise FixedRosterError(f"duplicate fixed-roster object {key}")
        anchors = row.get("anchor_indices")
        if not isinstance(anchors, list) or not anchors:
            raise FixedRosterError(f"fixed-roster object {key} has no anchors")
        indexed[key] = row
    return indexed
