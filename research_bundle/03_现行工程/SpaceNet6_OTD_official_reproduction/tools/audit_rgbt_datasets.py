#!/usr/bin/env python3
"""Audit LLVIP and DroneVehicle before RGB-T distillation experiments.

The audit is deliberately read-only with respect to the datasets. It checks
pairing, annotation coverage, split identity, geometry, and a deterministic
sample of cross-modal image statistics. It does not create training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DRONE_CLASS_ALIASES = {
    "car": "car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
    "feright car": "freight car",
    "feright_car": "freight car",
    "feright": "freight car",
    "freight car": "freight car",
    "freight_car": "freight car",
}
DRONE_CLASSES = {"car", "truck", "bus", "van", "freight car"}


def image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def xml_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == ".xml"
    }


def deterministic_subset(keys: Iterable[str], limit: int) -> list[str]:
    ordered = sorted(keys)
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=limit, dtype=int)
    return [ordered[index] for index in indices]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p90": float(np.percentile(finite, 90)),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float32).ravel()
    right_flat = right.astype(np.float32).ravel()
    left_flat -= left_flat.mean()
    right_flat -= right_flat.mean()
    denom = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    return float(np.dot(left_flat, right_flat) / denom) if denom else float("nan")


def normalized_mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    joint, _, _ = np.histogram2d(
        left.ravel(), right.ravel(), bins=32, range=((0, 256), (0, 256))
    )
    joint /= joint.sum()
    left_prob = joint.sum(axis=1)
    right_prob = joint.sum(axis=0)
    nz = joint > 0
    independent = left_prob[:, None] * right_prob[None, :]
    mutual_information = float(np.sum(joint[nz] * np.log(joint[nz] / independent[nz])))
    left_entropy = float(-np.sum(left_prob[left_prob > 0] * np.log(left_prob[left_prob > 0])))
    right_entropy = float(-np.sum(right_prob[right_prob > 0] * np.log(right_prob[right_prob > 0])))
    denom = math.sqrt(left_entropy * right_entropy)
    return mutual_information / denom if denom else float("nan")


def edge_magnitude(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def border_stats(image: np.ndarray, width: int = 100) -> dict[str, float] | None:
    height, image_width = image.shape
    if height <= 2 * width or image_width <= 2 * width:
        return None
    mask = np.ones_like(image, dtype=bool)
    mask[width:-width, width:-width] = False
    border = image[mask].astype(np.float32)
    return {
        "mean": float(border.mean()),
        "std": float(border.std()),
        "near_endpoint_fraction": float(np.mean((border <= 5) | (border >= 250))),
    }


def crop_image(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    return image[border:-border, border:-border]


def pair_image_metrics(
    left_files: dict[str, Path],
    right_files: dict[str, Path],
    sample_size: int,
    crop_border: int = 0,
) -> dict:
    paired = sorted(set(left_files) & set(right_files))
    selected = deterministic_subset(paired, sample_size)
    raw_correlations: list[float] = []
    edge_correlations: list[float] = []
    mutual_information: list[float] = []
    shifts: list[float] = []
    phase_responses: list[float] = []
    left_border_endpoint: list[float] = []
    right_border_endpoint: list[float] = []
    shapes: Counter[str] = Counter()
    unreadable: list[str] = []

    for stem in selected:
        left = cv2.imread(str(left_files[stem]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_files[stem]), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            unreadable.append(stem)
            continue
        shapes[f"{left.shape[1]}x{left.shape[0]}|{right.shape[1]}x{right.shape[0]}"] += 1
        left_analysis = crop_image(left, crop_border)
        right_analysis = crop_image(right, crop_border)
        left_small = cv2.resize(left_analysis, (256, 256), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_analysis, (256, 256), interpolation=cv2.INTER_AREA)
        left_edge = edge_magnitude(left_small)
        right_edge = edge_magnitude(right_small)
        raw_correlations.append(pearson(left_small, right_small))
        edge_correlations.append(pearson(left_edge, right_edge))
        mutual_information.append(normalized_mutual_information(left_small, right_small))
        shift, response = cv2.phaseCorrelate(left_edge, right_edge)
        shifts.append(float(math.hypot(shift[0], shift[1])))
        phase_responses.append(float(response))
        left_border = border_stats(left)
        right_border = border_stats(right)
        if left_border is not None:
            left_border_endpoint.append(left_border["near_endpoint_fraction"])
        if right_border is not None:
            right_border_endpoint.append(right_border["near_endpoint_fraction"])

    return {
        "paired_population": len(paired),
        "sample_requested": sample_size,
        "sample_readable": len(selected) - len(unreadable),
        "analysis_crop_border_px": crop_border,
        "unreadable_stems": unreadable,
        "shape_pairs": dict(shapes),
        "raw_intensity_correlation": summarize(raw_correlations),
        "edge_magnitude_correlation": summarize(edge_correlations),
        "normalized_mutual_information": summarize(mutual_information),
        "edge_phase_shift_at_256px": summarize(shifts),
        "edge_phase_response": summarize(phase_responses),
        "left_100px_border_endpoint_fraction": summarize(left_border_endpoint),
        "right_100px_border_endpoint_fraction": summarize(right_border_endpoint),
        "interpretation_note": (
            "Phase-correlation displacement is a cross-modal registration proxy, not "
            "ground-truth alignment. Low response makes the displacement unreliable."
        ),
    }


def adjacent_frame_metrics(
    files: dict[str, Path], sample_size: int, crop_border: int = 0
) -> dict:
    numeric = sorted((int(stem), stem) for stem in files if stem.isdigit())
    candidates = [
        (left_stem, right_stem)
        for (left_id, left_stem), (right_id, right_stem) in zip(numeric, numeric[1:])
        if right_id == left_id + 1
    ]
    selected_ids = deterministic_subset(
        [f"{left}|{right}" for left, right in candidates], sample_size
    )
    correlations: list[float] = []
    unreadable = 0
    for pair_id in selected_ids:
        left_stem, right_stem = pair_id.split("|", maxsplit=1)
        left = cv2.imread(str(files[left_stem]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(files[right_stem]), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            unreadable += 1
            continue
        left_small = cv2.resize(
            crop_image(left, crop_border), (64, 64), interpolation=cv2.INTER_AREA
        )
        right_small = cv2.resize(
            crop_image(right, crop_border), (64, 64), interpolation=cv2.INTER_AREA
        )
        correlations.append(pearson(left_small, right_small))
    return {
        "consecutive_numeric_pairs": len(candidates),
        "sample_readable": len(selected_ids) - unreadable,
        "analysis_crop_border_px": crop_border,
        "thumbnail_correlation": summarize(correlations),
        "fraction_above_0_95": (
            float(np.mean(np.asarray(correlations) > 0.95)) if correlations else None
        ),
        "interpretation_note": (
            "High adjacent-frame correlation indicates sequence dependence; it is not "
            "a duplicate-image decision rule."
        ),
    }


def sampled_mean_intensity(files: dict[str, Path], stems: list[str], limit: int = 32) -> dict:
    means = []
    for stem in deterministic_subset(stems, limit):
        image = cv2.imread(str(files[stem]), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            means.append(float(image.mean()))
    return summarize(means)


def parse_annotation(path: Path, polygon: bool) -> dict:
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width"))) if size is not None else None
    height = int(float(size.findtext("height"))) if size is not None else None
    objects = []
    for node in root.findall("object"):
        raw_class = (node.findtext("name") or "").strip()
        class_name = DRONE_CLASS_ALIASES.get(raw_class, raw_class) if polygon else raw_class
        if polygon:
            region = node.find("polygon")
            points = []
            if region is not None:
                geometry_type = "polygon"
                for index in range(1, 5):
                    x = float(region.findtext(f"x{index}"))
                    y = float(region.findtext(f"y{index}"))
                    points.append((x, y))
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                box = [min(xs), min(ys), max(xs), max(ys)]
            elif node.find("bndbox") is not None:
                geometry_type = "bndbox"
                region = node.find("bndbox")
                box = [
                    float(region.findtext("xmin")),
                    float(region.findtext("ymin")),
                    float(region.findtext("xmax")),
                    float(region.findtext("ymax")),
                ]
                points = [
                    (box[0], box[1]),
                    (box[2], box[1]),
                    (box[2], box[3]),
                    (box[0], box[3]),
                ]
            elif node.find("point") is not None:
                geometry_type = "point"
                region = node.find("point")
                points = [(float(region.findtext("x")), float(region.findtext("y")))]
                box = None
            else:
                geometry_type = "missing"
                box = None
        else:
            geometry_type = "bndbox"
            region = node.find("bndbox")
            points = []
            box = None
            if region is not None:
                box = [
                    float(region.findtext("xmin")),
                    float(region.findtext("ymin")),
                    float(region.findtext("xmax")),
                    float(region.findtext("ymax")),
                ]
        objects.append(
            {
                "class": class_name,
                "raw_class": raw_class,
                "geometry_type": geometry_type,
                "box": box,
                "points": points,
            }
        )
    return {
        "width": width,
        "height": height,
        "folder": (root.findtext("folder") or "").strip(),
        "filename": (root.findtext("filename") or "").strip(),
        "objects": objects,
    }


def annotation_summary(files: dict[str, Path], polygon: bool) -> tuple[dict, dict[str, dict]]:
    raw_classes: Counter[str] = Counter()
    canonical_classes: Counter[str] = Counter()
    geometry_types: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    source_groups: Counter[str] = Counter()
    empty = 0
    invalid_geometry = 0
    out_of_bounds = 0
    inside_crop = 0
    crosses_crop = 0
    outside_crop = 0
    official_eval_included = 0
    official_eval_skipped: Counter[str] = Counter()
    parse_errors: list[str] = []
    parsed: dict[str, dict] = {}

    for stem, path in files.items():
        try:
            annotation = parse_annotation(path, polygon=polygon)
        except (ET.ParseError, TypeError, ValueError):
            parse_errors.append(stem)
            continue
        parsed[stem] = annotation
        width = annotation["width"]
        height = annotation["height"]
        dimensions[f"{width}x{height}"] += 1
        source_groups[annotation["folder"] or "<empty>"] += 1
        if not annotation["objects"]:
            empty += 1
        for obj in annotation["objects"]:
            raw_classes[obj["raw_class"]] += 1
            canonical_classes[obj["class"]] += 1
            geometry_types[obj["geometry_type"]] += 1
            if polygon:
                if obj["raw_class"] == "*":
                    official_eval_skipped["name_star"] += 1
                elif obj["geometry_type"] in {"polygon", "bndbox"}:
                    official_eval_included += 1
                else:
                    official_eval_skipped[obj["geometry_type"]] += 1
            box = obj["box"]
            if box is None:
                continue
            if box[0] >= box[2] or box[1] >= box[3]:
                invalid_geometry += 1
                continue
            if width is not None and height is not None:
                if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                    out_of_bounds += 1
            if polygon:
                crop = (100.0, 100.0, 740.0, 612.0)
                if box[2] <= crop[0] or box[3] <= crop[1] or box[0] >= crop[2] or box[1] >= crop[3]:
                    outside_crop += 1
                elif box[0] < crop[0] or box[1] < crop[1] or box[2] > crop[2] or box[3] > crop[3]:
                    crosses_crop += 1
                else:
                    inside_crop += 1

    return (
        {
            "xml_count": len(files),
            "parsed_count": len(parsed),
            "parse_error_stems": parse_errors,
            "empty_annotation_count": empty,
            "raw_class_instances": dict(raw_classes),
            "canonical_class_instances": dict(canonical_classes),
            "unknown_class_instances": {
                name: count
                for name, count in canonical_classes.items()
                if polygon and name not in DRONE_CLASSES
            },
            "geometry_types": dict(geometry_types),
            "declared_dimensions": dict(dimensions),
            "source_groups": dict(source_groups),
            "invalid_geometry_count": invalid_geometry,
            "out_of_bounds_count": out_of_bounds,
            "crop_100px_box_relation": {
                "inside": inside_crop,
                "crosses_boundary": crosses_crop,
                "outside": outside_crop,
            },
            "official_xml_eval_policy": {
                "included_polygon_or_bndbox": official_eval_included,
                "skipped": dict(official_eval_skipped),
            },
        },
        parsed,
    )


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def paired_annotation_agreement(left: dict[str, dict], right: dict[str, dict]) -> dict:
    paired = sorted(set(left) & set(right))
    same_class_multiset = 0
    matched_ious: list[float] = []
    images_with_all_matches_above_05 = 0
    for stem in paired:
        left_objects = left[stem]["objects"]
        right_objects = right[stem]["objects"]
        if Counter(obj["class"] for obj in left_objects) == Counter(
            obj["class"] for obj in right_objects
        ):
            same_class_multiset += 1

        image_ious = []
        for class_name in sorted(
            set(obj["class"] for obj in left_objects)
            | set(obj["class"] for obj in right_objects)
        ):
            left_boxes = [
                obj["box"]
                for obj in left_objects
                if obj["class"] == class_name and obj["box"] is not None
            ]
            right_boxes = [
                obj["box"]
                for obj in right_objects
                if obj["class"] == class_name and obj["box"] is not None
            ]
            candidates = sorted(
                (
                    box_iou(left_box, right_box),
                    left_index,
                    right_index,
                )
                for left_index, left_box in enumerate(left_boxes)
                for right_index, right_box in enumerate(right_boxes)
            )
            used_left: set[int] = set()
            used_right: set[int] = set()
            for iou, left_index, right_index in reversed(candidates):
                if left_index in used_left or right_index in used_right:
                    continue
                used_left.add(left_index)
                used_right.add(right_index)
                image_ious.append(iou)
        matched_ious.extend(image_ious)
        if image_ious and min(image_ious) >= 0.5:
            images_with_all_matches_above_05 += 1

    return {
        "paired_annotation_files": len(paired),
        "same_class_multiset_count": same_class_multiset,
        "same_class_multiset_fraction": same_class_multiset / len(paired) if paired else None,
        "greedy_same_class_hbb_iou": summarize(matched_ious),
        "images_with_all_greedy_matches_iou_ge_0_5": images_with_all_matches_above_05,
        "interpretation_note": (
            "IoU uses axis-aligned envelopes of the source polygons and only describes "
            "cross-modal annotation agreement; it is not an OBB evaluation metric."
        ),
    }


def set_audit(left: set[str], right: set[str]) -> dict:
    return {
        "left_count": len(left),
        "right_count": len(right),
        "paired_count": len(left & right),
        "left_only_count": len(left - right),
        "right_only_count": len(right - left),
        "left_only": sorted(left - right),
        "right_only": sorted(right - left),
    }


def overlap_audit(left: set[str], right: set[str], example_limit: int = 30) -> dict:
    overlap = sorted(left & right)
    return {"count": len(overlap), "examples": overlap[:example_limit]}


def source_key_map(parsed: dict[str, dict]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for stem, annotation in parsed.items():
        key = f"{annotation['folder']}|{annotation['filename']}"
        mapping[key].append(stem)
    return mapping


def cross_split_source_similarity(
    left_map: dict[str, list[str]],
    right_map: dict[str, list[str]],
    left_files: dict[str, Path],
    right_files: dict[str, Path],
    sample_size: int,
) -> dict:
    overlap = set(left_map) & set(right_map)
    selected = deterministic_subset(overlap, sample_size)
    best_correlations: list[float] = []
    for key in selected:
        candidates = []
        for left_stem in left_map[key][:4]:
            left_path = left_files.get(left_stem)
            if left_path is None:
                continue
            left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            if left is None:
                continue
            left_small = cv2.resize(
                crop_image(left, 100), (64, 64), interpolation=cv2.INTER_AREA
            )
            for right_stem in right_map[key][:4]:
                right_path = right_files.get(right_stem)
                if right_path is None:
                    continue
                right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
                if right is None:
                    continue
                right_small = cv2.resize(
                    crop_image(right, 100), (64, 64), interpolation=cv2.INTER_AREA
                )
                candidates.append(pearson(left_small, right_small))
        if candidates:
            best_correlations.append(max(candidates))
    array = np.asarray(best_correlations)
    return {
        "overlapping_source_key_count": len(overlap),
        "sample_readable": len(best_correlations),
        "best_thumbnail_correlation": summarize(best_correlations),
        "fraction_above_0_95": float(np.mean(array > 0.95)) if len(array) else None,
        "fraction_above_0_99": float(np.mean(array > 0.99)) if len(array) else None,
        "interpretation_note": (
            "This checks whether repeated XML source keys also correspond to very similar "
            "same-modality images; it does not declare duplicate identity."
        ),
    }


def audit_llvip(root: Path, sample_size: int) -> dict:
    split_results = {}
    all_image_stems: set[str] = set()
    annotations = xml_files(root / "Annotations")
    annotation_stats, parsed_annotations = annotation_summary(annotations, polygon=False)
    prefix_sets = {}
    train_prefix_stats = {}

    for split in ("train", "test"):
        visible = image_files(root / "visible" / split)
        infrared = image_files(root / "infrared" / split)
        all_image_stems |= set(visible) | set(infrared)
        prefix_sets[split] = {stem[:2] for stem in visible if len(stem) >= 2}
        split_annotation_stems = set(annotations) & (set(visible) | set(infrared))
        split_results[split] = {
            "visible_vs_infrared": set_audit(set(visible), set(infrared)),
            "annotation_coverage": set_audit(
                set(visible) & set(infrared), split_annotation_stems
            ),
            "sequence_prefix_counts": dict(Counter(stem[:2] for stem in visible)),
            "pair_image_metrics": pair_image_metrics(visible, infrared, sample_size),
            "visible_adjacent_frames": adjacent_frame_metrics(visible, sample_size),
            "infrared_adjacent_frames": adjacent_frame_metrics(infrared, sample_size),
        }
        if split == "train":
            for prefix in sorted(prefix_sets[split]):
                stems = sorted(stem for stem in visible if stem.startswith(prefix))
                train_prefix_stats[prefix] = {
                    "pair_count": len(stems),
                    "person_instances": sum(
                        len(parsed_annotations[stem]["objects"])
                        for stem in stems
                        if stem in parsed_annotations
                    ),
                    "visible_mean_intensity": sampled_mean_intensity(visible, stems),
                    "infrared_mean_intensity": sampled_mean_intensity(infrared, stems),
                }

    return {
        "root": str(root),
        "splits": split_results,
        "train_test_image_stem_overlap": sorted(
            set(image_files(root / "visible" / "train"))
            & set(image_files(root / "visible" / "test"))
        ),
        "train_test_sequence_prefix_overlap": sorted(prefix_sets["train"] & prefix_sets["test"]),
        "train_sequence_prefix_stats": train_prefix_stats,
        "annotation_vs_all_images": set_audit(all_image_stems, set(annotations)),
        "annotations": annotation_stats,
    }


def audit_drone(root: Path, sample_size: int) -> dict:
    split_results = {}
    split_image_stems = {}
    split_source_groups = {}
    split_source_keys = {"visible": {}, "infrared": {}}
    split_source_maps = {"visible": {}, "infrared": {}}
    split_images = {"visible": {}, "infrared": {}}

    for split in ("train", "val", "test"):
        visible = image_files(root / split / f"{split}img")
        infrared = image_files(root / split / f"{split}imgr")
        visible_labels = xml_files(root / split / f"{split}label")
        infrared_labels = xml_files(root / split / f"{split}labelr")
        infrared_stats, infrared_parsed = annotation_summary(infrared_labels, polygon=True)
        visible_stats, visible_parsed = annotation_summary(visible_labels, polygon=True)
        split_image_stems[split] = set(infrared) | set(visible)
        split_images["visible"][split] = visible
        split_images["infrared"][split] = infrared
        split_source_groups[split] = set(infrared_stats["source_groups"]) | set(
            visible_stats["source_groups"]
        )
        split_source_maps["visible"][split] = source_key_map(visible_parsed)
        split_source_maps["infrared"][split] = source_key_map(infrared_parsed)
        split_source_keys["visible"][split] = set(split_source_maps["visible"][split])
        split_source_keys["infrared"][split] = set(split_source_maps["infrared"][split])
        split_results[split] = {
            "visible_vs_infrared": set_audit(set(visible), set(infrared)),
            "infrared_image_vs_label": set_audit(set(infrared), set(infrared_labels)),
            "visible_image_vs_label": set_audit(set(visible), set(visible_labels)),
            "infrared_annotations": infrared_stats,
            "visible_annotations": visible_stats,
            "cross_modal_annotation_agreement": paired_annotation_agreement(
                visible_parsed, infrared_parsed
            ),
            "pair_image_metrics": pair_image_metrics(
                visible, infrared, sample_size, crop_border=100
            ),
            "visible_adjacent_frames": adjacent_frame_metrics(
                visible, sample_size, crop_border=100
            ),
            "infrared_adjacent_frames": adjacent_frame_metrics(
                infrared, sample_size, crop_border=100
            ),
        }

    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_vs_{right}"] = {
            "numeric_stems": overlap_audit(split_image_stems[left], split_image_stems[right]),
            "source_groups": overlap_audit(
                split_source_groups[left], split_source_groups[right]
            ),
            "visible_source_keys": overlap_audit(
                split_source_keys["visible"][left], split_source_keys["visible"][right]
            ),
            "infrared_source_keys": overlap_audit(
                split_source_keys["infrared"][left], split_source_keys["infrared"][right]
            ),
            "visible_source_key_image_similarity": cross_split_source_similarity(
                split_source_maps["visible"][left],
                split_source_maps["visible"][right],
                split_images["visible"][left],
                split_images["visible"][right],
                sample_size,
            ),
            "infrared_source_key_image_similarity": cross_split_source_similarity(
                split_source_maps["infrared"][left],
                split_source_maps["infrared"][right],
                split_images["infrared"][left],
                split_images["infrared"][right],
                sample_size,
            ),
        }
    return {
        "root": str(root),
        "directory_semantics": {
            "visible_images": "{split}img",
            "infrared_images": "{split}imgr",
            "visible_annotations": "{split}label",
            "infrared_annotations": "{split}labelr",
            "evidence": (
                "The *imgr images have identical color channels, class totals match the "
                "official infrared counts, and the official UA-CMDet config maps its "
                "infrared branch to *imgr."
            ),
        },
        "splits": split_results,
        "split_overlap": overlap,
    }


def metric(summary: dict, key: str) -> str:
    value = summary.get(key)
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(report: dict) -> str:
    lines = [
        "# RGB-T dataset audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This is a pre-training integrity and descriptive audit. Image correlations and",
        "phase shifts are proxies; they do not establish teacher quality or method efficacy.",
        "",
        "## LLVIP",
        "",
        "| split | visible | infrared | paired | missing pairs | edge corr median | phase response median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, data in report["llvip"]["splits"].items():
        pair = data["visible_vs_infrared"]
        images = data["pair_image_metrics"]
        lines.append(
            f"| {split} | {pair['left_count']} | {pair['right_count']} | "
            f"{pair['paired_count']} | {pair['left_only_count'] + pair['right_only_count']} | "
            f"{metric(images['edge_magnitude_correlation'], 'median')} | "
            f"{metric(images['edge_phase_response'], 'median')} |"
        )
    lines += [
        "",
        f"- Annotation files: {report['llvip']['annotations']['xml_count']}",
        "- Class instances: "
        f"{report['llvip']['annotations']['canonical_class_instances']}",
        "- Train/test sequence-prefix overlap: "
        f"{report['llvip']['train_test_sequence_prefix_overlap']}",
        "",
        "## DroneVehicle",
        "",
        "| split | visible | infrared | paired | missing pairs | visible labels | infrared labels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, data in report["dronevehicle"]["splits"].items():
        pair = data["visible_vs_infrared"]
        lines.append(
            f"| {split} | {pair['left_count']} | {pair['right_count']} | "
            f"{pair['paired_count']} | {pair['left_only_count'] + pair['right_only_count']} | "
            f"{data['visible_annotations']['xml_count']} | "
            f"{data['infrared_annotations']['xml_count']} |"
        )
    lines += [
        "",
        "## Protocol consequences",
        "",
        "- LLVIP method development needs a sequence-grouped dev split carved only from",
        "  the official training sequences; keep the official test sequences sealed.",
        "- DroneVehicle source polygons and modality-specific annotations must be audited",
        "  before freezing any HBB conversion or shared-label policy.",
        "- Teacher strength is an empirical baseline result, not a dataset assumption.",
        "- Do not start distillation until archive validation, exact pairing, geometry,",
        "  and split-group checks all have an explicit pass or documented exclusion.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvip-root", type=Path, required=True)
    parser.add_argument("--drone-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sample-pairs", type=int, default=256)
    args = parser.parse_args()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_pairs_per_split": args.sample_pairs,
        "llvip": audit_llvip(args.llvip_root, args.sample_pairs),
        "dronevehicle": audit_drone(args.drone_root, args.sample_pairs),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
