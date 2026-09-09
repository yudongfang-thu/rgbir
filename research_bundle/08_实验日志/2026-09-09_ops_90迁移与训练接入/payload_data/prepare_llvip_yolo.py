#!/usr/bin/env python3
"""Prepare grouped LLVIP YOLO datasets for visible and infrared anchors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def yolo_labels(annotation_path: Path) -> tuple[list[str], int, Counter[str]]:
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    labels = []
    excluded = 0
    classes: Counter[str] = Counter()
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        classes[class_name] += 1
        if class_name != "person":
            raise ValueError(f"unexpected LLVIP class {class_name!r} in {annotation_path}")
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        if xmin >= xmax or ymin >= ymax:
            excluded += 1
            continue
        center_x = (xmin + xmax) / (2 * width)
        center_y = (ymin + ymax) / (2 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(
            f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return labels, excluded, classes


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def write_yaml(path: Path, dataset_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/fit",
                "val: images/dev",
                "test: images/test",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"output already exists: {args.output_root}")

    receipt = {
        "version": "llvip_yolo_grouped_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_dir": str(args.manifest_dir),
        "output_root": str(args.output_root),
        "splits": {},
        "class_instances_before_exclusion": Counter(),
        "excluded_degenerate_boxes": 0,
        "excluded_by_split": {},
    }

    for split in ("fit", "dev", "test"):
        records = load_manifest(args.manifest_dir / f"{split}.tsv")
        excluded_split = 0
        output_objects = 0
        for record in records:
            stem = record["stem"]
            labels, excluded, classes = yolo_labels(Path(record["annotation"]))
            receipt["class_instances_before_exclusion"].update(classes)
            receipt["excluded_degenerate_boxes"] += excluded
            excluded_split += excluded
            output_objects += len(labels)
            label_text = "\n".join(labels) + ("\n" if labels else "")
            for modality in ("visible", "infrared"):
                image_destination = (
                    args.output_root / modality / "images" / split / f"{stem}.jpg"
                )
                label_destination = (
                    args.output_root / modality / "labels" / split / f"{stem}.txt"
                )
                link(Path(record[modality]), image_destination)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.write_text(label_text, encoding="utf-8")
        receipt["splits"][split] = {
            "image_pairs": len(records),
            "output_objects_per_modality": output_objects,
        }
        receipt["excluded_by_split"][split] = excluded_split

    for modality in ("visible", "infrared"):
        write_yaml(
            args.output_root / f"{modality}.yaml",
            args.output_root / modality,
        )

    receipt["class_instances_before_exclusion"] = dict(
        receipt["class_instances_before_exclusion"]
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
