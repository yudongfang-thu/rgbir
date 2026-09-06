"""GT-free ECPT pseudo-label construction primitives."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from yolo_osssl.error_complement import area_bin, box_iou


def prediction_signature(row: dict, nc: int) -> np.ndarray:
    """Prediction-only class-count and class-area signature."""

    count = np.zeros(nc, dtype=np.float64)
    area = np.zeros(nc, dtype=np.float64)
    image_area = max(int(row["width"]) * int(row["height"]), 1)
    for item in row["eo_predictions"]:
        class_id = int(item["class_id"])
        if 0 <= class_id < nc:
            x1, y1, x2, y2 = item["box"]
            count[class_id] += 1.0
            area[class_id] += max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
    return np.concatenate((np.log1p(count), np.log1p(area)))


def build_prediction_only_donors(rows: Sequence[dict], nc: int, group_mode: str = "any") -> dict[str, str]:
    """Choose deterministic non-self donors without labels or D0 donor maps."""

    if group_mode not in {"any", "same_group", "cross_group"}:
        raise ValueError(f"unsupported donor group mode: {group_mode}")

    signatures = {str(row["pair_id"]): prediction_signature(row, nc) for row in rows}
    by_id = {str(row["pair_id"]): row for row in rows}
    result = {}
    for pair_id in sorted(by_id):
        target = by_id[pair_id]
        eligible = [
            donor_id for donor_id, donor in by_id.items()
            if donor_id != pair_id
            and (
                group_mode == "any"
                or (group_mode == "same_group" and str(donor["group_id"]) == str(target["group_id"]))
                or (group_mode == "cross_group" and str(donor["group_id"]) != str(target["group_id"]))
            )
        ]
        if not eligible:
            raise RuntimeError(f"no eligible wrong donor for {pair_id}")
        result[pair_id] = min(
            eligible,
            key=lambda donor_id: (
                float(np.square(signatures[pair_id] - signatures[donor_id]).sum()), donor_id
            ),
        )
    return result


def residual(source: Sequence[dict], base: Sequence[dict], min_confidence: float) -> list[dict]:
    """Keep source boxes not covered by a same-class base box."""

    return [
        item for item in source
        if float(item["confidence"]) >= min_confidence
        and not any(
            int(item["class_id"]) == int(other["class_id"])
            and box_iou(item["box"], other["box"]) >= 0.5
            for other in base
        )
    ]


def scale_predictions(predictions: Sequence[dict], source: dict, target: dict) -> list[dict]:
    x_scale = int(target["width"]) / int(source["width"])
    y_scale = int(target["height"]) / int(source["height"])
    return [
        {
            **item,
            "box": [
                float(item["box"][0]) * x_scale,
                float(item["box"][1]) * y_scale,
                float(item["box"][2]) * x_scale,
                float(item["box"][3]) * y_scale,
            ],
        }
        for item in predictions
    ]


def matched_p5_boxes(
    target: dict,
    donor: dict,
    thresholds: dict,
    area_edges: Sequence[float],
) -> dict[str, list[dict]]:
    """Return common base plus equal-dose SAR/exact/wrong additions."""

    sar90 = [
        item for item in target["sar_predictions"]
        if float(item["confidence"]) >= float(thresholds["sar90"]["threshold"])
    ]
    pools = {
        "sar": residual(target["sar_predictions"], sar90, float(thresholds["sar80"]["threshold"])),
        "exact": residual(target["eo_predictions"], sar90, float(thresholds["eo90"]["threshold"])),
        "wrong": residual(
            scale_predictions(donor["eo_predictions"], donor, target),
            sar90,
            float(thresholds["eo90"]["threshold"]),
        ),
    }
    grouped: dict[str, dict[tuple[int, int], list[dict]]] = {
        relation: defaultdict(list) for relation in pools
    }
    for relation, items in pools.items():
        for item in items:
            x1, y1, x2, y2 = item["box"]
            key = (
                int(item["class_id"]),
                area_bin(max(0.0, x2 - x1) * max(0.0, y2 - y1), area_edges),
            )
            grouped[relation][key].append(item)
    additions = {relation: [] for relation in pools}
    for key in sorted(set(grouped["sar"]) | set(grouped["exact"]) | set(grouped["wrong"])):
        count = min(len(grouped[relation].get(key, ())) for relation in pools)
        for relation in pools:
            additions[relation].extend(
                sorted(
                    grouped[relation].get(key, ()),
                    key=lambda item: (-float(item["confidence"]), tuple(item["box"])),
                )[:count]
            )
    return {relation: [*sar90, *additions[relation]] for relation in additions}


def yolo_lines(boxes: Sequence[dict], width: int, height: int) -> list[str]:
    lines = []
    for item in boxes:
        x1, y1, x2, y2 = (float(value) for value in item["box"])
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        if x2 <= x1 or y2 <= y1:
            continue
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
        if all(math.isfinite(value) for value in values):
            lines.append(f"{int(item['class_id'])} " + " ".join(f"{value:.8f}" for value in values))
    return lines
