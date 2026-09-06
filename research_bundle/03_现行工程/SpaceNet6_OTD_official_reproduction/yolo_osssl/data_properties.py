"""Deterministic, checkpoint-free measurements for paired optical/SAR data."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


DATASETS = {"ogsod", "spacenet6"}
PROPERTY_ROLES = {"property_fit", "property_confirm"}
DECISIONS = {
    "ALLOW_EXACT_LOCAL_SPEC",
    "ALLOW_LOCAL_SET_SPEC",
    "ALLOW_OBJECT_CONTEXT_RESIDUAL_SPEC",
    "DEFER_CONTEXT_ONLY",
    "KILL_NO_SHARED_PROPERTY",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def acquisition_from_pair_id(pair_id: str) -> str:
    normalized = pair_id.replace("{modality}", "SAR-Intensity")
    return normalized.split("_tile_", 1)[0]


def parse_yolo_boxes(path: Path, width: int, height: int) -> tuple[list[dict], int]:
    boxes: list[dict] = []
    invalid = 0
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if len(fields) < 5:
            invalid += 1
            continue
        class_id = int(float(fields[0]))
        cx, cy, bw, bh = (float(value) for value in fields[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            invalid += 1
            continue
        boxes.append(
            {
                "index": index,
                "class_id": class_id,
                "box": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "area": (x2 - x1) * (y2 - y1),
            }
        )
    return boxes, invalid


def box_union_fraction(boxes: Sequence[dict], width: int, height: int) -> float:
    if not boxes:
        return 0.0
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in item["box"])
        mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1
    return float(mask.mean())


def nearest_neighbor_distances(boxes: Sequence[dict]) -> list[float]:
    if len(boxes) < 2:
        return [float("nan")] * len(boxes)
    centers = np.asarray(
        [[(b["box"][0] + b["box"][2]) / 2.0, (b["box"][1] + b["box"][3]) / 2.0] for b in boxes],
        dtype=np.float64,
    )
    distances = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1).tolist()


def image_stratum(boxes: Sequence[dict], image_area: float) -> tuple[str, int, int]:
    classes = sorted({int(item["class_id"]) for item in boxes})
    presence = ",".join(map(str, classes)) if classes else "none"
    count = len(boxes)
    count_bin = 0 if count <= 1 else 1 if count <= 3 else 2 if count <= 7 else 3
    median_area = float(np.median([item["area"] for item in boxes])) / max(image_area, 1.0) if boxes else 0.0
    area_bin = 0 if median_area < 0.001 else 1 if median_area < 0.005 else 2 if median_area < 0.02 else 3
    return presence, count_bin, area_bin


def stratified_exact_split(rows: Sequence[dict], target_fit: int, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < target_fit < len(rows):
        raise ValueError("target_fit must be between zero and the row count")
    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        strata[tuple(row["property_stratum"])].append(dict(row))
    fit: list[dict] = []
    confirm: list[dict] = []
    remainders: list[tuple[float, tuple, dict]] = []
    ratio = target_fit / len(rows)
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda item: str(item["pair_id"]))
        order = rng.permutation(len(group))
        shuffled = [group[int(index)] for index in order]
        raw = ratio * len(group)
        take = int(math.floor(raw))
        fit.extend(shuffled[:take])
        confirm.extend(shuffled[take:])
        for item in shuffled[take:]:
            remainders.append((raw - take, key, item))
    need = target_fit - len(fit)
    if need:
        ranked = sorted(remainders, key=lambda item: (-item[0], item[1], str(item[2]["pair_id"])))
        promote_ids = {id(item[2]) for item in ranked[:need]}
        promoted = [item for item in confirm if id(item) in promote_ids]
        confirm = [item for item in confirm if id(item) not in promote_ids]
        fit.extend(promoted)
    return sorted(fit, key=lambda r: str(r["pair_id"])), sorted(confirm, key=lambda r: str(r["pair_id"]))


def acquisition_split(rows: Sequence[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(dict(row))
    rng = np.random.default_rng(seed)
    tie = {group: float(rng.random()) for group in groups}
    target = fraction * len(rows)
    selected: set[str] = set()
    current = 0
    for group in sorted(groups, key=lambda key: (-len(groups[key]), tie[key], key)):
        size = len(groups[group])
        if current < target and abs((current + size) - target) <= abs(current - target) or not selected:
            selected.add(group)
            current += size
    fit = [dict(row) for row in rows if str(row["group_id"]) in selected]
    confirm = [dict(row) for row in rows if str(row["group_id"]) not in selected]
    return sorted(fit, key=lambda r: str(r["pair_id"])), sorted(confirm, key=lambda r: str(r["pair_id"]))


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_gray(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def quantize16(image: np.ndarray) -> np.ndarray:
    data = image.astype(np.float32)
    low, high = np.percentile(data, [1.0, 99.0])
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip(((data - low) / (high - low) * 15.999), 0, 15).astype(np.uint8)


def normalized_mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        raise ValueError("NMI inputs must have the same non-empty shape")
    a, b = quantize16(left).ravel(), quantize16(right).ravel()
    joint = np.bincount(a * 16 + b, minlength=256).reshape(16, 16).astype(np.float64)
    joint /= max(joint.sum(), 1.0)
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    denom = pa[:, None] * pb[None, :]
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))
    ha = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    return 0.0 if ha + hb <= 1e-12 else float(2.0 * mi / (ha + hb))


def sobel_layout(image: np.ndarray, size: int = 16) -> np.ndarray:
    data = image.astype(np.float32)
    gx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3)
    return resize_gray(cv2.magnitude(gx, gy), size).reshape(-1)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64).reshape(-1)
    b = right.astype(np.float64).reshape(-1)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-12 else float(np.dot(a, b) / denom)


def descriptor_scores(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "nmi": normalized_mutual_information(left, right),
        "sobel": cosine_similarity(sobel_layout(left), sobel_layout(right)),
    }


def overlapping_shift(left: np.ndarray, right: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape:
        raise ValueError("shift inputs must share a shape")
    height, width = left.shape
    lx1, lx2 = max(0, -dx), min(width, width - dx)
    ly1, ly2 = max(0, -dy), min(height, height - dy)
    if lx2 <= lx1 or ly2 <= ly1:
        raise ValueError("shift has no overlap")
    return left[ly1:ly2, lx1:lx2], right[ly1 + dy : ly2 + dy, lx1 + dx : lx2 + dx]


def local_best(left: np.ndarray, right: np.ndarray, offsets: Sequence[int], descriptor: str) -> dict[str, float | int | bool]:
    values = sorted({0, *[int(v) for magnitude in offsets for v in (-magnitude, magnitude)]})
    candidates = [(0, 0), *[(value, 0) for value in values if value], *[(0, value) for value in values if value]]
    best_score = -float("inf")
    best_dx = best_dy = 0
    for dx, dy in candidates:
        a, b = overlapping_shift(left, right, dx, dy)
        score = normalized_mutual_information(a, b) if descriptor == "nmi" else cosine_similarity(sobel_layout(a), sobel_layout(b))
        if score > best_score:
            best_score, best_dx, best_dy = score, dx, dy
    boundary = max(abs(best_dx), abs(best_dy)) == max(abs(v) for v in values)
    return {"score": float(best_score), "dx": best_dx, "dy": best_dy, "boundary": boundary}


def local_best_box(
    sar_crop: np.ndarray,
    optical_image: np.ndarray,
    box: Sequence[float],
    offsets: Sequence[int],
    descriptor: str,
) -> dict[str, float | int | bool]:
    values = sorted({0, *[int(v) for magnitude in offsets for v in (-magnitude, magnitude)]})
    candidates = [(0, 0), *[(value, 0) for value in values if value], *[(0, value) for value in values if value]]
    best_score = -float("inf")
    best_dx = best_dy = 0
    curve: list[dict[str, float | int]] = []
    for dx, dy in candidates:
        try:
            target = crop_box(optical_image, box, size=sar_crop.shape[0], dx=dx, dy=dy)
        except ValueError:
            continue
        score = normalized_mutual_information(sar_crop, target) if descriptor == "nmi" else cosine_similarity(sobel_layout(sar_crop), sobel_layout(target))
        curve.append({"dx": dx, "dy": dy, "score": float(score)})
        if score > best_score:
            best_score, best_dx, best_dy = score, dx, dy
    boundary = max(abs(best_dx), abs(best_dy)) == max(offsets)
    return {"score": float(best_score), "dx": best_dx, "dy": best_dy, "boundary": boundary, "curve": curve}


def integer_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    ix2, iy2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError("empty box")
    return ix1, iy1, ix2, iy2


def crop_box(image: np.ndarray, box: Sequence[float], size: int = 32, dx: int = 0, dy: int = 0) -> np.ndarray:
    height, width = image.shape
    moved = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
    x1, y1, x2, y2 = integer_box(moved, width, height)
    return resize_gray(image[y1:y2, x1:x2], size)


def boxes_intersect(left: Sequence[float], right: Sequence[float]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def nearest_context_box(
    target: Sequence[float], boxes: Sequence[Sequence[float]], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = target
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for radius in (1.0, 1.5, 2.0, 3.0):
        for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ncx, ncy = cx + ux * radius * bw, cy + uy * radius * bh
            candidate = (ncx - bw / 2.0, ncy - bh / 2.0, ncx + bw / 2.0, ncy + bh / 2.0)
            if candidate[0] < 0 or candidate[1] < 0 or candidate[2] > width or candidate[3] > height:
                continue
            if any(boxes_intersect(candidate, other) for other in boxes):
                continue
            candidates.append((math.hypot(ncx - cx, ncy - cy), candidate))
        if candidates:
            break
    return min(candidates, key=lambda item: (item[0], item[1]))[1] if candidates else None


def foreground_features(image: np.ndarray, box: Sequence[float], image_area: float) -> np.ndarray:
    crop = crop_box(image, box, size=32)
    intensity = resize_gray(crop, 8).reshape(-1) / 255.0
    sobel = sobel_layout(crop, size=8)
    norm = float(np.linalg.norm(sobel))
    if norm > 0:
        sobel = sobel / norm
    area = ((box[2] - box[0]) * (box[3] - box[1])) / max(image_area, 1.0)
    return np.concatenate([intensity, sobel, np.asarray([area], dtype=np.float32)]).astype(np.float32)


def select_hard_donors(rows: Sequence[dict], dataset: str) -> dict[str, dict[str, str | None]]:
    if dataset not in DATASETS:
        raise ValueError(dataset)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        candidates = [item for item in rows if item["pair_id"] != row["pair_id"]]
        if not candidates:
            raise ValueError("at least two pairs are required")

        def distance(item: dict) -> tuple:
            class_penalty = 0 if item.get("class_presence") == row.get("class_presence") else 1
            count_delta = abs(int(item["object_count"]) - int(row["object_count"]))
            area_delta = abs(float(item["median_area_fraction"]) - float(row["median_area_fraction"]))
            coverage_delta = abs(float(item["foreground_fraction"]) - float(row["foreground_fraction"]))
            return class_penalty if dataset == "ogsod" else 0, count_delta, area_delta, coverage_delta, str(item["pair_id"])

        hard = min(candidates, key=distance)
        mapping = {"hard_wrong": str(hard["pair_id"])}
        if dataset == "spacenet6":
            same = [item for item in candidates if item["group_id"] == row["group_id"]]
            cross = [item for item in candidates if item["group_id"] != row["group_id"]]
            mapping["same_acquisition_wrong"] = str(min(same, key=distance)["pair_id"]) if same else None
            mapping["cross_acquisition_wrong"] = str(min(cross, key=distance)["pair_id"]) if cross else None
            if mapping["cross_acquisition_wrong"] is None:
                raise ValueError("SpaceNet6 requires at least two acquisitions")
            mapping["hard_wrong"] = mapping["cross_acquisition_wrong"]
        result[str(row["pair_id"])] = mapping
    return result


def grouped_bootstrap(values: dict[str, list[float]], draws: int, seed: int) -> dict[str, float]:
    clean = {key: [float(v) for v in vals if np.isfinite(v)] for key, vals in values.items()}
    clean = {key: vals for key, vals in clean.items() if vals}
    if not clean:
        return {"effect": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "groups": 0}
    keys = sorted(clean)
    unit_means = np.asarray([np.mean(clean[key]) for key in keys], dtype=np.float64)
    observed = float(unit_means.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    chunk = max(1, min(256, int(2_000_000 / len(keys))))
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        samples[start:stop] = unit_means[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"effect": observed, "ci95_low": float(low), "ci95_high": float(high), "groups": len(keys)}


def robust_standardized_effect(values: Sequence[float]) -> float:
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(data):
        return -float("inf")
    center = float(np.median(data))
    scale = float(np.median(np.abs(data - center))) * 1.4826
    return center / max(scale, 1e-6)
