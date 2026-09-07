"""Lease-bound T/R diagnosis on a frozen train/val roster; never reads test.

D1 uses class-agnostic one-to-one spatial assignment of pre-NMS predictions
with confidence >= .05, IoU >= .1. D2 uses the implemented frozen-reference
anchor selector. Neither is a detection benchmark nor evidence of KD gains.
No registration contract means explicitly UNVERIFIED D2 opportunities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from localization_loss import (LocalizationConfig, build_localization_selection,
                               _decode_boxes, _iou, _labels, _layout, _match_objects)
from geometry_contract import GeometryContract

PROTOCOL = "rgbir_d1_pre_nms_class_agnostic_d2_frozen_anchor_v1"
PRIMARY_LUMINANCE_THRESHOLD = 78.283
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_rows(path, rows):
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def reject_test_path(path):
    """Check lexical and actual targets before opening data, including symlinks."""
    p = Path(path)
    for candidate in (p, p.resolve()):
        if any(part.lower() in ("test", "testing", "test2017", "test-dev") for part in candidate.parts):
            raise ValueError("Sealed test path is outside this diagnostic: " + str(p))


def dataset_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "test" in value:
        raise ValueError("A train/val-only dataset YAML is required")
    root = Path(value.get("path", Path(path).parent))
    if not root.is_absolute():
        root = Path(path).parent / root
    value["root"] = root
    return value


def split_images(data, split):
    if split not in ("train", "val"):
        raise ValueError("Only train and val may be diagnosed")
    sources = data[split] if isinstance(data[split], list) else [data[split]]
    images = []
    for source in sources:
        path = Path(source)
        if not path.is_absolute():
            path = data["root"] / path
        reject_test_path(path)
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == ".txt":
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip())
                if not item.is_absolute():
                    item = path.parent / item
                images.append(item)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError("Unresolved diagnostic split: " + str(path))
    result = sorted(set(images), key=str)
    for item in result:
        reject_test_path(item)
    if not result:
        raise ValueError("Selected split is empty")
    return result


def label_path(image):
    parts = list(Path(image).parts)
    if "images" not in parts:
        raise ValueError("Processed YOLO image path must contain images: " + str(image))
    pos = len(parts) - 1 - parts[::-1].index("images")
    parts[pos] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(path):
    reject_test_path(path)
    rows = [list(map(float, line.split())) for line in Path(path).read_text().splitlines() if line.strip()]
    array = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite YOLO labels: " + str(path))
    if len(array) and (not np.equal(array[:, 0], np.floor(array[:, 0])).all() or
                       not ((array[:, 1:] >= 0) & (array[:, 1:] <= 1)).all() or
                       not (array[:, 3:] > 0).all()):
        raise ValueError("Invalid normalized XYWH labels: " + str(path))
    return array


def source_groups(data, images, dataset, split):
    root = data["root"]
    candidates = [root.parent / ("rgb_" + split + "_source_groups.tsv"),
                  root / ("rgb_" + split + "_source_groups.tsv")]
    mapping = {}
    for path in candidates:
        if path.is_file():
            for line in path.read_text().splitlines():
                stem, group = line.split("\t")[:2]
                mapping[stem] = group
            break
    if dataset.lower() == "dronevehicle" and split == "train" and not mapping:
        raise ValueError("Drone train source-group governance TSV is required")
    if dataset.lower() == "llvip":
        roster_name = "fit.tsv" if split == "train" else "dev.tsv"
        for ancestor in [root] + list(root.parents)[:5]:
            path = ancestor / "splits/grouped_v1" / roster_name
            if path.is_file():
                with path.open() as handle:
                    mapping.update({row["stem"]: row["sequence_prefix"] for row in csv.DictReader(handle, delimiter="\t")})
                break
    result = {}
    for image in images:
        if image.stem in mapping:
            result[str(image)] = mapping[image.stem]
        elif dataset.lower() == "llvip":
            result[str(image)] = image.stem[:2]
        else:
            result[str(image)] = "unavailable:" + image.parent.name
    return result


def stratified_rows(rows, limit):
    """Proportional source-group quotas, then deterministic uniform frame ranks."""
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("max-images must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    names = sorted(groups)
    counts = {name: 1 if limit >= len(names) else 0 for name in names}
    remaining = limit - sum(counts.values())
    capacity = {name: len(groups[name]) - counts[name] for name in names}
    total_capacity = sum(capacity.values())
    ideal = {name: remaining * capacity[name] / total_capacity for name in names}
    for name in names:
        counts[name] += int(ideal[name])
    extra = limit - sum(counts.values())
    ranked = sorted(names, key=lambda name: (-(ideal[name] - int(ideal[name])), name))
    for name in ranked[:extra]:
        counts[name] += 1
    selected = []
    for name in names:
        items = sorted(groups[name], key=lambda row: row["rgb_path"])
        count = counts[name]
        if count:
            positions = np.linspace(0, len(items)-1, count).astype(int)
            selected.extend(items[int(pos)] for pos in positions)
    return selected


def frozen_roster(cfg, split, source_roster=None, max_images=None):
    rgb_data = dataset_config(cfg["paths"]["student_data_yaml"])
    ir_data = dataset_config(cfg["paths"]["privileged_data_yaml"])
    rgb_images, ir_images = split_images(rgb_data, split), split_images(ir_data, split)
    ir_by_stem = {}
    for image in ir_images:
        if image.stem in ir_by_stem:
            raise ValueError("Ambiguous IR stem; explicit paired mapping is needed")
        ir_by_stem[image.stem] = str(image)
    mapping_path = Path(cfg["paths"]["paired_train_mapping"])
    if split == "val":
        mapping_path = mapping_path.with_name(mapping_path.name.replace("_train", "_val"))
    mapping = json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}
    allowed_ir = {str(p) for p in ir_images}
    groups = source_groups(rgb_data, rgb_images, cfg["dataset"], split)
    rows = []
    for image in rgb_images:
        target = mapping.get(str(image), ir_by_stem.get(image.stem))
        if target is None or target not in allowed_ir:
            raise ValueError("Pair is absent from declared IR split: " + str(image))
        rows.append({"rgb_path": str(image), "ir_path": str(target), "stem": image.stem,
                     "source_group": groups[str(image)], "split": split,
                     "rgb_label": str(label_path(image)), "ir_label": str(label_path(target))})
    if source_roster is not None:
        source_roster = Path(source_roster)
        if source_roster.suffix.lower() == ".json":
            source = json.loads(source_roster.read_text())
            if isinstance(source, dict):
                source = source.get("roster", source.get("rows", source.get("images")))
        else:
            source = [line.strip() for line in source_roster.read_text().splitlines() if line.strip()]
        if not isinstance(source, list):
            raise ValueError("Unsupported frozen roster; expected list/roster/rows/images")
        by_path = {row["rgb_path"]: row for row in rows}
        by_actual = {str(Path(row["rgb_path"]).resolve()): row for row in rows}
        selected, seen = [], set()
        for item in source:
            path = item if isinstance(item, str) else item.get("rgb_path", item.get("im_file", item.get("image")))
            if not path:
                raise ValueError("Roster row has no RGB image path")
            row = by_path.get(str(path), by_actual.get(str(Path(path).resolve())))
            if row is None:
                raise ValueError("Roster image is not in declared " + split + ": " + str(path))
            if row["rgb_path"] in seen:
                raise ValueError("Duplicate image in frozen roster")
            seen.add(row["rgb_path"])
            selected.append(row)
        if max_images is not None and len(selected) > max_images:
            raise ValueError("Supplied frozen roster exceeds max-images; do not silently resample it")
        rows = selected
    else:
        rows = stratified_rows(rows, max_images if max_images is not None else (2048 if split == "train" else 200))
    return {"protocol": PROTOCOL, "dataset": cfg["dataset"], "split": split,
            "population_images": len(rgb_images), "sampling": "supplied_frozen_roster" if source_roster else "proportional_source_groups_uniform_frame_ranks",
            "source_roster": str(source_roster) if source_roster else None,
            "count": len(rows), "roster": rows, "official_test_accessed": False}


def letterbox_image_and_labels(image, labels, size=640):
    """Native-style centered letterbox, scaleup=True, no random augmentation."""
    import cv2
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = (int(round(width * ratio)), int(round(height * ratio)))
    left = int(round((size - resized[0]) / 2 - .1))
    top = int(round((size - resized[1]) / 2 - .1))
    right, bottom = size - resized[0] - left, size - resized[1] - top
    if resized != (width, height):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    matrix = np.array([[ratio, 0, left], [0, ratio, top], [0, 0, 1]], dtype=np.float64)
    transformed = labels.copy()
    if len(transformed):
        transformed[:, 1] = (labels[:, 1] * width * ratio + left) / size
        transformed[:, 2] = (labels[:, 2] * height * ratio + top) / size
        transformed[:, 3] = labels[:, 3] * width * ratio / size
        transformed[:, 4] = labels[:, 4] * height * ratio / size
    return image, transformed, matrix


def load_batch(rows, size):
    import cv2
    from PIL import Image
    rgb_images, ir_images, rgb_rows, ir_rows, metadata, brightness = [], [], [], [], [], []
    for bi, row in enumerate(rows):
        rgb, ir = cv2.imread(row["rgb_path"]), cv2.imread(row["ir_path"])
        if rgb is None or ir is None:
            raise ValueError("Image decode failed: " + row["rgb_path"])
        rgb_shape, ir_shape = rgb.shape[:2], ir.shape[:2]
        # Preserve the original frozen-threshold measurement exactly; OpenCV
        # grayscale rounding/decoding is not silently substituted for PIL L.
        with Image.open(row["rgb_path"]) as source_image:
            bright = float(np.asarray(source_image.convert("L")).mean())
        brightness.append({"mean_rgb_luma": bright,
                           "luminance_proxy": "low" if bright < PRIMARY_LUMINANCE_THRESHOLD else "high",
                           "brightness_proxy_bin": min(int(bright // 64), 3)})
        rgb, rgb_gt, rgb_matrix = letterbox_image_and_labels(rgb, read_yolo_labels(row["rgb_label"]), size)
        ir, ir_gt, ir_matrix = letterbox_image_and_labels(ir, read_yolo_labels(row["ir_label"]), size)
        rgb_images.append(np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1)))
        ir_images.append(np.ascontiguousarray(ir[:, :, ::-1].transpose(2, 0, 1)))
        rgb_rows.extend([[bi] + values for values in rgb_gt.tolist()])
        ir_rows.extend([[bi] + values for values in ir_gt.tolist()])
        metadata.append({"original_shape": list(rgb_shape), "ir_original_shape": list(ir_shape),
                         "output_shape": [size, size], "rgb_matrix": rgb_matrix.tolist(),
                         "ir_matrix": ir_matrix.tolist(), "augmentation": "centered_letterbox_only",
                         "weak_source": str(Path(row["rgb_path"]).resolve()),
                         "strong_source": str(Path(row["ir_path"]).resolve())})
    def pack(values):
        tensor = torch.tensor(values, dtype=torch.float32).reshape(-1, 6)
        return {"batch_idx": tensor[:, 0].long(), "cls": tensor[:, 1:2].long(), "bboxes": tensor[:, 2:]}
    batch = pack(rgb_rows)
    batch.update(img=torch.from_numpy(np.stack(rgb_images)).float() / 255,
                 strong_img=torch.from_numpy(np.stack(ir_images)).float() / 255,
                 teacher_batch=pack(ir_rows), im_file=[row["rgb_path"] for row in rows],
                 pair_info=metadata)
    return batch, brightness


@torch.no_grad()
def spatial_assign(gt_boxes, prediction_boxes, confidence, classes, minimum_conf=.05, minimum_iou=.1):
    eligible = torch.nonzero(confidence >= minimum_conf, as_tuple=False).flatten()
    candidate_boxes = prediction_boxes[eligible]
    zeros_gt = torch.zeros(len(gt_boxes), dtype=torch.long, device=gt_boxes.device)
    zeros_candidate = torch.zeros(len(eligible), dtype=torch.long, device=gt_boxes.device)
    gi, pi = _match_objects(gt_boxes, zeros_gt, candidate_boxes, zeros_candidate, minimum_iou)
    result = {}
    for g, p in zip(gi.tolist(), pi.tolist()):
        anchor = int(eligible[p])
        result[g] = {"anchor_index": anchor, "class": int(classes[anchor]),
                     "confidence": float(confidence[anchor]), "iou": float(_iou(gt_boxes[g:g+1], prediction_boxes[anchor:anchor+1])[0, 0]),
                     "box": prediction_boxes[anchor].tolist()}
    return result, len(eligible)


def reference_state(prediction, gt_class):
    if prediction is None:
        return "no_coarse_candidate"
    if prediction["class"] != gt_class:
        return "class_error"
    if prediction["confidence"] < .25:
        return "low_confidence"
    return "localization_gap" if prediction["iou"] < .70 else "well_localized"


@torch.no_grad()
def d1_records(teacher, reference, batch, rows, brightness, cfg, strides):
    centers, sv, _, size = _layout(reference, cfg, strides)
    batch_size, nc, _ = reference["scores"].shape
    ri, rc, rb = _labels(batch, batch_size, nc, centers.device, size)
    ti, tc, tb = _labels(batch["teacher_batch"], batch_size, nc, centers.device, size)
    rbox, tbox = _decode_boxes(reference, centers, sv), _decode_boxes(teacher, centers, sv)
    rp, rclass = reference["scores"].float().sigmoid().max(1)
    tp, tclass = teacher["scores"].float().sigmoid().max(1)
    records, totals = [], Counter()
    for bi, row in enumerate(rows):
        rglobal = torch.nonzero(ri == bi, as_tuple=False).flatten()
        tglobal = torch.nonzero(ti == bi, as_tuple=False).flatten()
        rgb, ir = rb[rglobal], tb[tglobal]
        rassign, rcount = spatial_assign(rgb, rbox[bi], rp[bi], rclass[bi])
        tassign, tcount = spatial_assign(ir, tbox[bi], tp[bi], tclass[bi])
        pair_r, pair_t = _match_objects(rgb, rc[rglobal], ir, tc[tglobal], .5)
        pairs = dict(zip(pair_r.tolist(), pair_t.tolist()))
        totals.update(images=1, rgb_gt_count=len(rgb), ir_gt_count=len(ir), paired_gt_count=len(pairs),
                      reference_confidence_filtered_predictions=rcount, teacher_confidence_filtered_predictions=tcount)
        for local, gi in enumerate(rglobal.tolist()):
            cls = int(rc[gi])
            ref = rassign.get(local)
            ir_local = pairs.get(local)
            teacher_prediction = tassign.get(ir_local) if ir_local is not None else None
            state = reference_state(ref, cls)
            pair_iou = float(_iou(rgb[local:local+1], ir[ir_local:ir_local+1])[0, 0]) if ir_local is not None else None
            target_iou = float(_iou(rgb[local:local+1], rgb.new_tensor([teacher_prediction["box"]]))[0, 0]) if teacher_prediction is not None else None
            opportunity = (state == "localization_gap" and teacher_prediction is not None and
                           teacher_prediction["class"] == cls and teacher_prediction["confidence"] >= .25 and
                           teacher_prediction["iou"] >= .50 and target_iou >= .60 and target_iou - ref["iou"] > .05)
            area = float((rgb[local, 2:] - rgb[local, :2]).prod())
            scale_bin = "small" if area < 32**2 else ("medium" if area < 96**2 else "large")
            record = {"image": row["rgb_path"], "ir_image": row["ir_path"], "stem": row["stem"],
                      "source_group": row["source_group"], "rgb_gt_local_index": local,
                      "class": cls, "gt_box_input": rgb[local].tolist(), "area_input_px2": area,
                      "scale_bin": scale_bin, "reference_state": state, "reference": ref,
                      "ir_gt_local_index": ir_local, "paired_gt_iou": pair_iou,
                      "correspondence": "same_class_gt_iou_ge_0.5" if ir_local is not None else "unreliable_or_unpaired",
                      "teacher": teacher_prediction, "teacher_to_rgb_iou": target_iou,
                      "object_level_localization_opportunity": bool(opportunity), **brightness[bi]}
            records.append(record)
            totals[state] += 1
            totals["unreliable_or_unpaired"] += int(ir_local is None)
            totals["object_level_localization_opportunity"] += int(opportunity)
    return records, totals


def geometry_mask(contract, batch, prediction, strides):
    if contract is None:
        return None
    boxes = batch["bboxes"]
    height, width = batch["img"].shape[-2:]
    scale = boxes.new_tensor([width, height])
    xyxy = torch.cat(((boxes[:, :2]-boxes[:, 2:]/2)*scale, (boxes[:, :2]+boxes[:, 2:]/2)*scale), -1)
    chunks = []
    for feature, stride in zip(prediction["feats"], strides):
        mask = contract.object_mask(batch["im_file"], xyxy, batch["batch_idx"], batch["pair_info"], stride)
        chunks.append(mask[:, None].expand(-1, feature.shape[-2]*feature.shape[-1]))
    return torch.cat(chunks, 1)


def run(args):
    from legacy_bridge import legacy
    cfg = yaml.safe_load(args.config.read_text())
    if str(torch.__version__) != cfg["torch_version"] or str(legacy.ultralytics.__version__) != cfg["ultralytics_version"]:
        raise RuntimeError("Pinned framework environment changed")
    lease = legacy.require_bound_lease_from_environment()
    if len(lease["gpus"]) != 1:
        raise ValueError("Diagnostic requires exactly one bound GPU")
    roster = frozen_roster(cfg, args.split, args.roster, args.max_images)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "frozen_roster.json", roster)
    shutil.copy2(args.config, args.output / "protocol_config.yaml")
    contract = GeometryContract.load(args.geometry_contract) if args.geometry_contract else None
    verified = bool(contract is not None and contract.verified)
    local_cfg = LocalizationConfig(**cfg.get("localization", {"input_size": cfg["imgsz"]}))
    rgb_data = dataset_config(cfg["paths"]["student_data_yaml"])
    names = rgb_data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    reference = legacy.load_frozen(cfg["reference"], cfg["paths"]["student_data_yaml"], names).cuda()
    teacher = legacy.load_frozen(cfg["teacher"], cfg["paths"]["privileged_data_yaml"], names).cuda()
    strides = tuple(int(v) for v in reference.stride)
    if tuple(int(v) for v in teacher.stride) != strides:
        raise ValueError("Teacher/reference strides differ")
    started = time.time()
    d1_totals, d2_totals, by_group, d1_groups = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    diagnostic_rows = roster["roster"]
    (args.output / "d1_objects.jsonl").touch()
    (args.output / "d2_anchors.jsonl").touch()
    (args.output / "images.jsonl").touch()
    with torch.inference_mode():
        for start in range(0, len(diagnostic_rows), args.batch):
            rows = diagnostic_rows[start:start + args.batch]
            batch, brightness = load_batch(rows, cfg["imgsz"])
            batch["img"], batch["strong_img"] = batch["img"].cuda(), batch["strong_img"].cuda()
            reference_raw = legacy.raw_prediction(reference(batch["img"]))
            teacher_raw = legacy.raw_prediction(teacher(batch["strong_img"]))
            objects, counts = d1_records(teacher_raw, reference_raw, batch, rows, brightness, local_cfg, strides)
            d1_totals.update(counts)
            for obj in objects:
                for key in ("class", "source_group", "scale_bin", "luminance_proxy", "brightness_proxy_bin"):
                    group = str(key) + ":" + str(obj[key])
                    d1_groups[group].update(rgb_gt_count=1)
                    d1_groups[group][obj["reference_state"]] += 1
                    d1_groups[group]["object_level_localization_opportunity"] += int(obj["object_level_localization_opportunity"])
            append_rows(args.output / "d1_objects.jsonl", objects)
            selection = build_localization_selection(teacher_raw, reference_raw, batch,
                strides=strides, config=local_cfg, geometry_eligible=geometry_mask(contract, batch, reference_raw, strides),
                geometry_verified=verified, return_records=True)
            stats = selection.stats
            d2_totals.update({key: value for key, value in stats.items() if key.endswith("_count") and isinstance(value, int)})
            base_records = stats["base_records"]
            for record in base_records:
                bi = record["batch_index"]
                record.update(image=rows[bi]["rgb_path"], ir_image=rows[bi]["ir_path"],
                              source_group=rows[bi]["source_group"], geometry_verified=verified, **brightness[bi])
                dists = record["rgb_distances"]
                area = (dists[0]+dists[2])*(dists[1]+dists[3])*record["stride"]**2
                record.update(area_input_px2=area, scale_bin="small" if area < 32**2 else ("medium" if area < 96**2 else "large"))
                # Batch indices/global GT rows are augmented-batch coordinates;
                # image + local GT gives stable identity across batch sizes.
                record["object_id"] = record["image"] + ":" + str(record["rgb_gt_local_index"])
                for key in ("class", "source_group", "scale_bin", "luminance_proxy", "brightness_proxy_bin"):
                    group = str(key) + ":" + str(record[key])
                    by_group[group].update(base_count=1, selected_count=int(record["selected"]))
            append_rows(args.output / "d2_anchors.jsonl", base_records)
            append_rows(args.output / "images.jsonl", [dict(row, **bright, pair_info=meta) for row, bright, meta in zip(rows, brightness, batch["pair_info"])])
            write_json(args.output / "progress.json", {"images_completed": start + len(rows), "images_total": len(diagnostic_rows),
                "seconds": time.time() - started, "d1_totals": dict(d1_totals), "totals": dict(d2_totals),
                "geometry_verified": verified, "official_test_accessed": False})
            print(json.dumps({"images": start + len(rows), "total": len(diagnostic_rows),
                              "d2_base": d2_totals["base_count"], "d2_selected": d2_totals["selected_count"]}), flush=True)
    summary = {"protocol": PROTOCOL, "status": "completed", "dataset": cfg["dataset"], "split": args.split,
        "geometry_verified": verified, "geometry_status": "empirical_contract_mask" if verified else "UNVERIFIED_DIAGNOSTIC",
        "geometry_contract": str(args.geometry_contract) if args.geometry_contract else None,
        "formal_localization_ready": bool(verified and d2_totals["selected_count"] > 0),
        "formal_readiness_scope": "D2 opportunity only; calibration/canary/receipt still required",
        "images": len(diagnostic_rows), "d1_totals": dict(d1_totals), "totals": dict(d2_totals),
        "d1_groups": {key: dict(value) for key, value in d1_groups.items()},
        "d2_groups": {key: dict(value) for key, value in by_group.items()},
        "teacher": cfg["teacher"], "reference": cfg["reference"], "config": str(args.config),
        "roster": str(args.output / "frozen_roster.json"), "batch_size": args.batch,
        "augmentation": "centered_letterbox_scaleup_true_only", "amp": False,
        "primary_luminance_threshold": PRIMARY_LUMINANCE_THRESHOLD,
        "primary_luminance_rule": "low < 78.283; high >= 78.283; proxy only",
        "luminance_measurement": "mean of original PIL Image.convert('L') pixels",
        "secondary_brightness_edges": [0, 64, 128, 192, 256], "brightness_is_day_night_label": False,
        "seconds": time.time()-started, "gpu_allocated_peak_mib": torch.cuda.max_memory_allocated()/2**20,
        "gpu_reserved_peak_mib": torch.cuda.max_memory_reserved()/2**20,
        "resources": legacy.bound_lease_resource_record_from_environment(), "official_test_accessed": False,
        "d1_prediction_policy": {"pre_nms": True, "confidence_min": .05, "spatial_iou_min": .1,
                                 "assignment": "class_agnostic_max_cardinality_then_total_iou", "prediction_truncation": False}}
    write_json(args.output / "summary.json", summary)
    legacy.emit_bound_run_receipt(run_dir=args.output / "run_evidence", method_identity=cfg["method_identity"],
        dataset=cfg["dataset"], data_role="development_" + args.split, seed=cfg["seed"], run_kind="feature",
        trainers=[Path(__file__)], losses=[HERE / "localization_loss.py", HERE / "legacy_oev1/object_evidence_loss.py"],
        configs=[args.config, Path(cfg["paths"]["student_data_yaml"]), Path(cfg["paths"]["privileged_data_yaml"])],
        split_rosters=[args.output / "frozen_roster.json"], metric_files=[args.output / "summary.json"],
        environment={"torch": str(torch.__version__), "ultralytics": legacy.ultralytics.__version__},
        inputs={"protocol": PROTOCOL, "teacher_weights": cfg["teacher"], "reference_weights": cfg["reference"],
                "geometry_verified": verified, "teacher_labels_used_by_kd": True})


def self_test():
    """Essential CPU IO/selection fixtures. Does not import Ultralytics."""
    import tempfile
    checks = []
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    labels = np.array([[0, .5, .5, .5, .5]], dtype=np.float32)
    out, transformed, matrix = letterbox_image_and_labels(image, labels, 64)
    assert out.shape == (64, 64, 3)
    assert np.allclose(matrix, [[1, 0, 0], [0, 1, 16], [0, 0, 1]])
    assert np.allclose(transformed[0], [0, .5, .5, .5, .25])
    checks.append("letterbox_image_labels_and_actual_transform")
    rows = [{"rgb_path": str(i), "source_group": "a" if i < 20 else "b"} for i in range(100)]
    selected = stratified_rows(rows, 10)
    assert len(selected) == 10 and len({r["rgb_path"] for r in selected}) == 10
    assert {r["source_group"] for r in selected} == {"a", "b"}
    assert selected == stratified_rows(rows, 10)
    checks.append("deterministic_proportional_source_sampling")
    gt = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
    predictions = gt.clone()
    assigned, count = spatial_assign(gt, predictions, torch.tensor([.8, .1]), torch.tensor([1, 0]))
    assert count == 2 and len(assigned) == 2
    assert reference_state(assigned[0], 0) == "class_error"
    assert reference_state(assigned[1], 0) == "low_confidence"
    checks.append("class_agnostic_assignment_retains_class_and_confidence_errors")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for modality in ("rgb", "ir"):
            for role in ("train", "val"):
                (root/modality/"images"/role).mkdir(parents=True)
                (root/modality/"labels"/role).mkdir(parents=True)
                (root/modality/"images"/role/"00001.jpg").touch()
                (root/modality/"labels"/role/"00001.txt").write_text("0 .5 .5 .25 .25\n")
            (root/(modality+".yaml")).write_text(yaml.safe_dump({"path": str(root/modality), "train": "images/train", "val": "images/val", "names": {0: "object"}}))
        cfg = {"dataset": "fixture", "paths": {"student_data_yaml": str(root/"rgb.yaml"),
               "privileged_data_yaml": str(root/"ir.yaml"), "paired_train_mapping": str(root/"rgb_to_ir_train.json")}}
        roster = frozen_roster(cfg, "val", max_images=200)
        assert roster["count"] == 1 and Path(roster["roster"][0]["ir_path"]).parts[-4:] == ("ir", "images", "val", "00001.jpg")
        assert read_yolo_labels(roster["roster"][0]["rgb_label"]).shape == (1, 5)
        source = root/"source.json"
        write_json(source, roster)
        repeated = frozen_roster(cfg, "val", source, 200)
        assert repeated["roster"] == roster["roster"]
        source.write_text(json.dumps([str(root/"rgb/images/train/00001.jpg")]))
        try:
            frozen_roster(cfg, "val", source, 200)
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-split roster accepted")
        checks.append("paired_split_roster_yolo_labels_and_cross_split_rejection")
    try:
        reject_test_path("/dataset/images/test/00001.jpg")
    except ValueError:
        checks.append("test_path_rejected")
    else:
        raise AssertionError("Test path accepted")
    print(json.dumps({"status": "passed", "checks": checks, "gpu_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--geometry-contract", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.config is None or args.output is None or args.batch < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error("--config and --output are required; batch/max-images must be positive")
    try:
        run(args)
    except Exception as error:
        if args.output and args.output.is_dir() and not (args.output / "summary.json").exists():
            write_json(args.output / "failure.json", {"error": repr(error), "status": "failed", "official_test_accessed": False})
        raise


if __name__ == "__main__":
    main()
