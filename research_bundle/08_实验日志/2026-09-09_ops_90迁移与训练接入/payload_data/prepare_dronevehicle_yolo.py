#!/usr/bin/env python3
"""Build the frozen DroneVehicle cropped YOLO-HBB v1 views.

The conversion keeps RGB and IR annotations separate.  It mirrors the
official XML evaluation semantics for classes and geometry, then changes only
the representation: the common 100 px border is cropped and polygon/bndbox
annotations become cropped HBB labels.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2


VERSION = "dronevehicle-yolo-hbb-v1"
SOURCE_SIZE = (840, 712)
CROP_RECT = (100.0, 100.0, 740.0, 612.0)
CROP_SIZE = (640, 512)
CLASS_NAMES = ("car", "freight car", "truck", "bus", "van")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_ALIASES = {
    "car": "car",
    "freight car": "freight car",
    "feright car": "freight car",
    "feright": "freight car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

SPLIT_SPECS = {
    "train": {
        "rgb": ("trainimg", "trainlabel"),
        "infrared": ("trainimgr", "trainlabelr"),
    },
    "val": {
        "rgb": ("valimg", "vallabel"),
        "infrared": ("valimgr", "vallabelr"),
    },
    "test": {
        "rgb": ("testimg", "testlabel"),
        "infrared": ("testimgr", "testlabelr"),
    },
}


def canonical_class(raw_name: str) -> str | None:
    """Return the official canonical spelling; ``None`` is the official '*'."""
    normalized = " ".join(
        raw_name.strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if normalized == "*":
        return None
    try:
        return CLASS_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"unknown DroneVehicle class {raw_name!r}") from error


def _inside(point: tuple[float, float], edge: str, value: float) -> bool:
    x, y = point
    return {
        "left": x >= value,
        "right": x <= value,
        "top": y >= value,
        "bottom": y <= value,
    }[edge]


def _intersection(
    start: tuple[float, float], end: tuple[float, float], edge: str, value: float
) -> tuple[float, float]:
    x0, y0 = start
    x1, y1 = end
    if edge in {"left", "right"}:
        if x1 == x0:
            return value, y0
        ratio = (value - x0) / (x1 - x0)
        return value, y0 + ratio * (y1 - y0)
    if y1 == y0:
        return x0, value
    ratio = (value - y0) / (y1 - y0)
    return x0 + ratio * (x1 - x0), value


def clip_polygon_to_rect(
    points: Iterable[tuple[float, float]], rect: tuple[float, float, float, float] = CROP_RECT
) -> list[tuple[float, float]]:
    """Clip a polygon to ``rect`` with Sutherland-Hodgman clipping."""
    clipped = list(points)
    for edge, value in (
        ("left", rect[0]),
        ("right", rect[2]),
        ("top", rect[1]),
        ("bottom", rect[3]),
    ):
        if not clipped:
            break
        output: list[tuple[float, float]] = []
        previous = clipped[-1]
        previous_inside = _inside(previous, edge, value)
        for current in clipped:
            current_inside = _inside(current, edge, value)
            if current_inside != previous_inside:
                output.append(_intersection(previous, current, edge, value))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        clipped = output
    return clipped


def _finite_points(points: Iterable[tuple[float, float]]) -> bool:
    return all(math.isfinite(x) and math.isfinite(y) for x, y in points)


def _parse_polygon(node: ET.Element) -> list[tuple[float, float]] | None:
    polygon = node.find("polygon")
    if polygon is None:
        return None
    points = []
    for index in range(1, 5):
        x_text, y_text = polygon.findtext(f"x{index}"), polygon.findtext(f"y{index}")
        if x_text is None or y_text is None:
            return []
        points.append((float(x_text), float(y_text)))
    return points


def _parse_bndbox(node: ET.Element) -> tuple[float, float, float, float] | None:
    box = node.find("bndbox")
    if box is None:
        return None
    values = tuple(float(box.findtext(key) or "nan") for key in ("xmin", "ymin", "xmax", "ymax"))
    return values  # type: ignore[return-value]


def _envelope(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _clip_bndbox(
    box: tuple[float, float, float, float], rect: tuple[float, float, float, float] = CROP_RECT
) -> tuple[float, float, float, float]:
    return max(box[0], rect[0]), max(box[1], rect[1]), min(box[2], rect[2]), min(box[3], rect[3])


def _to_yolo(box: tuple[float, float, float, float], class_name: str) -> str:
    x0, y0, x1, y1 = box
    crop_x0, crop_y0, _, _ = CROP_RECT
    width, height = CROP_SIZE
    center_x = ((x0 + x1) * 0.5 - crop_x0) / width
    center_y = ((y0 + y1) * 0.5 - crop_y0) / height
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height
    return f"{CLASS_TO_ID[class_name]} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"


def labels_from_xml(xml_path: Path) -> tuple[list[str], dict[str, object]]:
    """Convert one XML independently and return labels plus transparent counters."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    source_width = int(float(size.findtext("width") or "0")) if size is not None else 0
    source_height = int(float(size.findtext("height") or "0")) if size is not None else 0
    if (source_width, source_height) != SOURCE_SIZE:
        raise ValueError(f"unexpected source size {(source_width, source_height)} in {xml_path}")

    labels: list[str] = []
    source_geometry: Counter[str] = Counter()
    source_classes: Counter[str] = Counter()
    written_classes: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    drops: Counter[str] = Counter()
    aliases: Counter[str] = Counter()
    official_eval_candidates = 0
    for node in root.findall("object"):
        raw_class = (node.findtext("name") or "").strip()
        class_name = canonical_class(raw_class)
        if class_name is None:
            source_geometry["star"] += 1
            ignored["class_star"] += 1
            continue
        source_classes[class_name] += 1
        if raw_class.strip().lower().replace("_", " ") != class_name:
            aliases[raw_class] += 1
        polygon = _parse_polygon(node)
        bndbox = _parse_bndbox(node) if polygon is None else None
        if polygon is not None:
            source_geometry["polygon"] += 1
            official_eval_candidates += 1
            if not _finite_points(polygon):
                drops["nonfinite_polygon"] += 1
                continue
            clipped_polygon = clip_polygon_to_rect(polygon)
            if len(clipped_polygon) < 3:
                drops["zero_or_invalid_post_clip_polygon"] += 1
                continue
            box = _envelope(clipped_polygon)
        elif bndbox is not None:
            source_geometry["bndbox"] += 1
            official_eval_candidates += 1
            if not all(math.isfinite(value) for value in bndbox):
                drops["nonfinite_bndbox"] += 1
                continue
            box = _clip_bndbox(bndbox)
        elif node.find("point") is not None:
            source_geometry["point"] += 1
            ignored["point"] += 1
            continue
        else:
            source_geometry["missing"] += 1
            drops["missing_geometry"] += 1
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            drops["zero_or_invalid_post_clip_geometry"] += 1
            continue
        label = _to_yolo(box, class_name)
        values = [float(value) for value in label.split()[1:]]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"invalid normalized label from {xml_path}: {label}")
        labels.append(label)
        written_classes[class_name] += 1
    return labels, {
        "source_objects": len(root.findall("object")),
        "official_eval_candidates": official_eval_candidates,
        "source_geometry": dict(source_geometry),
        "source_class_instances": dict(source_classes),
        "written_class_instances": dict(written_classes),
        "ignored": dict(ignored),
        "dropped": dict(drops),
        "alias_usage": dict(aliases),
    }


def _image_map(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _xml_map(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(directory.glob("*.xml"))}


def _merge_counts(target: Counter[str], values: dict[str, object]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def _process_image(
    image_path: Path, xml_path: Path, image_destination: Path, label_destination: Path
) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or (image.shape[1], image.shape[0]) != SOURCE_SIZE:
        raise ValueError(f"unexpected or unreadable source image {image_path}")
    labels, counts = labels_from_xml(xml_path)
    cropped = image[int(CROP_RECT[1]) : int(CROP_RECT[3]), int(CROP_RECT[0]) : int(CROP_RECT[2])]
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_destination), cropped):
        raise RuntimeError(f"failed to write {image_destination}")
    label_destination.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return {"labels_written": len(labels), **counts}


def _summarize_modality(
    source_root: Path, output_root: Path, split: str, modality: str, workers: int
) -> dict[str, object]:
    image_dir_name, label_dir_name = SPLIT_SPECS[split][modality]
    source_images = _image_map(source_root / split / image_dir_name)
    source_xml = _xml_map(source_root / split / label_dir_name)
    if set(source_images) != set(source_xml):
        raise ValueError(f"image/XML stem mismatch for {split}/{modality}")

    output_images = output_root / modality / "images" / split
    output_labels = output_root / modality / "labels" / split
    aggregate = {
        "source_image_files": len(source_images),
        "source_xml_files": len(source_xml),
        "source_pair_stems": len(source_images),
        "output_image_files": 0,
        "output_label_files": 0,
        "labels_written": 0,
        "source_objects": 0,
        "official_eval_candidates": 0,
        "source_geometry": Counter(),
        "source_class_instances": Counter(),
        "written_class_instances": Counter(),
        "ignored": Counter(),
        "dropped": Counter(),
        "alias_usage": Counter(),
    }

    def work(stem: str) -> dict[str, object]:
        image_path = source_images[stem]
        return _process_image(
            image_path,
            source_xml[stem],
            output_images / f"{stem}{image_path.suffix.lower()}",
            output_labels / f"{stem}.txt",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(work, sorted(source_images)):
            aggregate["labels_written"] += int(result["labels_written"])
            aggregate["source_objects"] += int(result["source_objects"])
            aggregate["official_eval_candidates"] += int(result["official_eval_candidates"])
            for key in (
                "source_geometry",
                "source_class_instances",
                "written_class_instances",
                "ignored",
                "dropped",
                "alias_usage",
            ):
                _merge_counts(aggregate[key], result[key])  # type: ignore[arg-type]
    aggregate["output_image_files"] = len(_image_map(output_images))
    aggregate["output_label_files"] = len(list(output_labels.glob("*.txt")))
    for key in (
        "source_geometry",
        "source_class_instances",
        "written_class_instances",
        "ignored",
        "dropped",
        "alias_usage",
    ):
        aggregate[key] = dict(aggregate[key])  # type: ignore[arg-type]
    return aggregate


def _write_yaml(path: Path, modality_root: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    path.write_text(
        "\n".join(
            (
                f"path: {modality_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names,
                "",
            )
        ),
        encoding="utf-8",
    )


def write_train_source_groups(source_root: Path, output_root: Path) -> dict[str, object]:
    """Write the train-only XML folder identities without inferring missing ones."""
    summaries: dict[str, object] = {}
    for modality in ("rgb", "infrared"):
        _, label_dir_name = SPLIT_SPECS["train"][modality]
        xml_files = _xml_map(source_root / "train" / label_dir_name)
        destination = output_root / f"{modality}_train_source_groups.tsv"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing sidecar: {destination}")
        rows = []
        missing_folder_count = 0
        for stem, xml_path in sorted(xml_files.items()):
            folder = (ET.parse(xml_path).getroot().findtext("folder") or "").strip()
            if not folder:
                missing_folder_count += 1
                continue
            rows.append(f"{stem}\t{folder}")
        destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        summaries[modality] = {
            "path": str(destination),
            "source_train_xml_files": len(xml_files),
            "rows_written": len(rows),
            "missing_folder_count": missing_folder_count,
        }
    return summaries


def validate_output(output_root: Path) -> dict[str, object]:
    """Decode every converted image and validate every written YOLO field."""
    result: dict[str, object] = {"by_split_modality": {}}
    for split in SPLIT_SPECS:
        for modality in ("rgb", "infrared"):
            images = _image_map(output_root / modality / "images" / split)
            labels = sorted((output_root / modality / "labels" / split).glob("*.txt"))
            bad_images: list[str] = []
            bad_labels: list[str] = []
            label_rows = 0
            for stem, image_path in images.items():
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None or (image.shape[1], image.shape[0]) != CROP_SIZE:
                    bad_images.append(stem)
            for label_path in labels:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        bad_labels.append(label_path.stem)
                        continue
                    try:
                        class_id = int(parts[0])
                        values = [float(value) for value in parts[1:]]
                    except ValueError:
                        bad_labels.append(label_path.stem)
                        continue
                    if class_id not in range(len(CLASS_NAMES)) or not all(
                        math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
                    ):
                        bad_labels.append(label_path.stem)
                    label_rows += 1
            key = f"{split}/{modality}"
            result["by_split_modality"][key] = {
                "image_files": len(images),
                "label_files": len(labels),
                "decoded_640x512": len(images) - len(bad_images),
                "bad_image_stems": bad_images,
                "label_rows": label_rows,
                "bad_label_stems": sorted(set(bad_labels)),
            }
    return result


def _deterministic_stems(stems: list[str], count: int) -> list[str]:
    if len(stems) <= count:
        return stems
    return [stems[round(index * (len(stems) - 1) / (count - 1))] for index in range(count)]


def write_visuals(output_root: Path, visual_root: Path, per_split: int) -> dict[str, object]:
    visual_root.mkdir(parents=True, exist_ok=True)
    records = []
    for split in SPLIT_SPECS:
        for modality in ("rgb", "infrared"):
            image_dir = output_root / modality / "images" / split
            label_dir = output_root / modality / "labels" / split
            images = _image_map(image_dir)
            for stem in _deterministic_stems(sorted(images), per_split):
                image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"unreadable converted image {images[stem]}")
                for line in (label_dir / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
                    class_id_text, cx_text, cy_text, width_text, height_text = line.split()
                    class_id = int(class_id_text)
                    cx, cy, width, height = map(float, (cx_text, cy_text, width_text, height_text))
                    x0 = round((cx - width / 2) * CROP_SIZE[0])
                    y0 = round((cy - height / 2) * CROP_SIZE[1])
                    x1 = round((cx + width / 2) * CROP_SIZE[0])
                    y1 = round((cy + height / 2) * CROP_SIZE[1])
                    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    cv2.putText(
                        image,
                        CLASS_NAMES[class_id],
                        (x0, max(y0 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                output_path = visual_root / f"{split}_{modality}_{stem}.jpg"
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(f"failed to write visual {output_path}")
                records.append(
                    {
                        "split": split,
                        "modality": modality,
                        "stem": stem,
                        "image": str(output_path),
                        "label": str(label_dir / f"{stem}.txt"),
                    }
                )
    visual_receipt = {
        "version": f"{VERSION}_visuals",
        "selection": "sorted stems sampled at evenly spaced deterministic indices",
        "per_split_modality": per_split,
        "records": records,
    }
    (visual_root / "visual_receipt.json").write_text(
        json.dumps(visual_receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return visual_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--visuals-per-split", type=int, default=3)
    parser.add_argument(
        "--write-train-source-groups-only",
        action="store_true",
        help="Add train-only XML folder sidecars and receipt metadata without conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    receipt_path = args.receipt.resolve()
    visual_root = args.visual_root.resolve() if args.visual_root is not None else None
    if args.workers < 1 or args.visuals_per_split < 1:
        raise ValueError("workers and visuals-per-split must be positive")
    if args.write_train_source_groups_only:
        if not output_root.is_dir() or not receipt_path.is_file():
            raise FileNotFoundError("sidecar mode requires an existing output root and receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("version") != VERSION:
            raise ValueError(f"unexpected receipt version in {receipt_path}")
        receipt["train_source_groups"] = write_train_source_groups(source_root, output_root)
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"version": VERSION, "receipt": str(receipt_path)}, ensure_ascii=False))
        return
    if visual_root is None:
        raise ValueError("--visual-root is required for full conversion")
    for path in (output_root, receipt_path, visual_root):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    output_root.mkdir(parents=True)
    receipt = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_input_read_only": True,
        "output_root": str(output_root),
        "receipt_path": str(receipt_path),
        "visual_root": str(visual_root),
        "source_image_size": {"width": SOURCE_SIZE[0], "height": SOURCE_SIZE[1]},
        "crop": {"x": [100, 740], "y": [100, 612], "output_size": list(CROP_SIZE)},
        "modalities": {
            split: {
                modality: {"image_directory": image_dir, "label_directory": label_dir}
                for modality, (image_dir, label_dir) in modalities.items()
            }
            for split, modalities in SPLIT_SPECS.items()
        },
        "class_order": list(CLASS_NAMES),
        "class_order_primary_code": "LADD_public/tools/prepare_dronevehicle_cclkd_hbb.py:CLASS_NAMES",
        "class_aliases": CLASS_ALIASES,
        "official_evaluation_semantics": {
            "included_geometry": ["polygon", "bndbox"],
            "ignored_geometry": ["point"],
            "ignored_class": "*",
            "polygon_conversion": "clip polygon to crop rectangle, then take HBB envelope",
            "bndbox_conversion": "intersect bndbox with crop rectangle",
        },
        "by_split_modality": {},
    }
    for split in SPLIT_SPECS:
        for modality in ("rgb", "infrared"):
            receipt["by_split_modality"][f"{split}/{modality}"] = _summarize_modality(
                source_root, output_root, split, modality, args.workers
            )
    _write_yaml(output_root / "rgb.yaml", output_root / "rgb")
    _write_yaml(output_root / "infrared.yaml", output_root / "infrared")
    receipt["train_source_groups"] = write_train_source_groups(source_root, output_root)
    receipt["output_validation"] = validate_output(output_root)
    receipt["visuals"] = write_visuals(output_root, visual_root, args.visuals_per_split)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"version": VERSION, "receipt": str(receipt_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
