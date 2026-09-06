#!/usr/bin/env python3
"""Materialize train/dev RGB-T mappings without exposing a test split to training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
DATASET_SPECS = {
    "llvip": {
        "modalities": {"visible": ("visible",), "infrared": ("infrared", "ir")},
        "weak_modality": "visible",
        "source_splits": {"train": "fit", "val": "dev"},
        "names": {0: "person"},
    },
    "dronevehicle": {
        "modalities": {"rgb": ("rgb", "visible"), "infrared": ("infrared", "ir")},
        "weak_modality": "rgb",
        "source_splits": {"train": "train", "val": "val"},
        "names": {0: "car", 1: "freight car", 2: "truck", 3: "bus", 4: "van"},
    },
}


def _image_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory is missing: {directory}")
    paths = sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    by_stem = {path.stem: path for path in paths}
    if len(by_stem) != len(paths):
        raise RuntimeError(f"duplicate image stems in {directory}")
    if not by_stem:
        raise RuntimeError(f"image directory is empty: {directory}")
    return by_stem


def _modality_root(data_root: Path, dataset: str, canonical: str) -> Path:
    for candidate in DATASET_SPECS[dataset]["modalities"][canonical]:
        path = data_root / candidate
        if path.is_dir():
            return path.resolve()
    aliases = ", ".join(DATASET_SPECS[dataset]["modalities"][canonical])
    raise FileNotFoundError(f"{dataset} {canonical} root is missing under {data_root}; tried {aliases}")


def _read_groups(path: Path | None) -> tuple[dict[str, str], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(f"group metadata is missing: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("JSON group metadata must be a stem-to-group mapping")
        return {Path(str(stem)).stem: str(group) for stem, group in payload.items()}, str(path.resolve())
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if rows and len(rows[0]) == 2 and rows[0][0].strip().lower() == "stem" and rows[0][1].strip().lower() in {"folder", "group", "source_group"}:
        rows = rows[1:]
    if not rows or any(len(row) != 2 or not row[0].strip() or not row[1].strip() for row in rows):
        raise RuntimeError("TSV group metadata must be non-empty stem<TAB>folder rows")
    metadata = {Path(row[0].strip()).stem: row[1].strip() for row in rows}
    if len(metadata) != len(rows):
        raise RuntimeError("TSV group metadata has duplicate stems")
    return metadata, str(path.resolve())


def _groups_for(dataset: str, stems: list[str], metadata: dict[str, str]) -> dict[str, str | None]:
    if dataset == "llvip":
        invalid = [stem for stem in stems if len(stem) < 2 or not stem[:2].isdigit()]
        if invalid:
            raise RuntimeError(f"LLVIP stem has no two-digit sequence prefix: {invalid[0]}")
        return {stem: stem[:2] for stem in stems}
    return {stem: metadata.get(stem) for stem in stems}


def _direction_groups(
    dataset: str, data_root: Path, student_modality: str, stems: list[str]
) -> tuple[dict[str, str | None], str | None]:
    if dataset == "llvip":
        return _groups_for(dataset, stems, {}), None
    sidecar = data_root / f"{student_modality}_train_source_groups.tsv"
    metadata, source = _read_groups(sidecar if sidecar.is_file() else None)
    return _groups_for(dataset, stems, metadata), source


def _shuffled_mapping(
    source: dict[str, Path],
    target: dict[str, Path],
    groups: dict[str, str | None],
) -> tuple[dict[str, str], int, int]:
    stems = sorted(source)
    if len(stems) < 2:
        raise RuntimeError("a shuffled donor mapping needs at least two paired images")
    group_buckets: dict[str, list[str]] = {}
    missing_groups: list[str] = []
    for stem in stems:
        group = groups[stem]
        if group is None:
            missing_groups.append(stem)
        else:
            group_buckets.setdefault(group, []).append(stem)
    ordered_buckets = sorted(group_buckets.values(), key=lambda bucket: (-len(bucket), bucket[0]))
    ordered_stems = [stem for bucket in ordered_buckets for stem in bucket] + missing_groups
    largest_bucket = max((len(bucket) for bucket in ordered_buckets), default=1)
    donors = ordered_stems[largest_bucket:] + ordered_stems[:largest_bucket]
    mapping: dict[str, str] = {}
    group_constrained = 0
    fallback = 0
    for stem, donor in zip(ordered_stems, donors, strict=True):
        if groups[stem] is not None and groups[donor] is not None and groups[stem] != groups[donor]:
            group_constrained += 1
        else:
            fallback += 1
        if stem == donor:
            raise RuntimeError("shuffled donor permutation produced a same-stem donor")
        mapping[str(source[stem])] = str(target[donor])
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("shuffled donor mapping must be a one-to-one permutation")
    return mapping, group_constrained, fallback


def _write_data_yaml(path: Path, modality_root: Path, source_splits: dict[str, str], names: dict[int, str]) -> None:
    payload = {
        "path": str(modality_root),
        "train": f"images/{source_splits['train']}",
        "val": f"images/{source_splits['val']}",
        "names": names,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _evidence(path: Path) -> dict[str, int | str]:
    state = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(state.st_size), "mtime_ns": int(state.st_mtime_ns)}


def _validate_labels(image_by_stem: dict[str, Path], label_dir: Path) -> None:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label directory is missing: {label_dir}")
    missing = [stem for stem in image_by_stem if not (label_dir / f"{stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"label missing for {missing[0]} in {label_dir}")


def prepare(
    dataset: str,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    spec = DATASET_SPECS[dataset]
    root = data_root.resolve()
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "mappings").mkdir(exist_ok=True)
    modalities = tuple(spec["modalities"])
    modality_roots = {modality: _modality_root(root, dataset, modality) for modality in modalities}
    source_splits = spec["source_splits"]
    inventories: dict[str, dict[str, dict[str, Path]]] = {}
    for modality, modality_root in modality_roots.items():
        inventories[modality] = {}
        for role, source_split in source_splits.items():
            images = _image_paths(modality_root / "images" / source_split)
            _validate_labels(images, modality_root / "labels" / source_split)
            inventories[modality][role] = images
        _write_data_yaml(output / f"{modality}.data.yaml", modality_root, source_splits, spec["names"])

    train_counts = {modality: len(inventories[modality]["train"]) for modality in modalities}
    val_counts = {modality: len(inventories[modality]["val"]) for modality in modalities}
    if len(set(train_counts.values())) != 1 or len(set(val_counts.values())) != 1:
        raise RuntimeError(f"paired modality counts disagree: train={train_counts}, val={val_counts}")

    directions: dict[str, Any] = {}
    evaluation_role = "dev" if dataset == "llvip" else "val"
    for student in modalities:
        privileged = next(modality for modality in modalities if modality != student)
        source = inventories[student]["train"]
        target = inventories[privileged]["train"]
        if set(source) != set(target):
            missing = sorted(set(source).symmetric_difference(target))
            raise RuntimeError(f"paired train stems disagree for {student}->{privileged}: {missing[0]}")
        groups, metadata_source = _direction_groups(dataset, root, student, sorted(source))
        paired = {str(source[stem]): str(target[stem]) for stem in sorted(source)}
        shuffled, constrained, fallback = _shuffled_mapping(source, target, groups)
        paired_path = output / "mappings" / f"{student}_to_{privileged}_train.json"
        shuffled_path = output / "mappings" / f"{student}_to_{privileged}_shuffled_train.json"
        paired_path.write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shuffled_path.write_text(json.dumps(shuffled, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        direction = {
            "student": student,
            "privileged": privileged,
            "paired_train_pairs": len(paired),
            "shuffled_train_pairs": len(shuffled),
            "shuffled_group_constrained_pairs": constrained,
            "shuffled_fallback_pairs": fallback,
            "group_metadata_source": metadata_source,
            "group_metadata_missing_stems": sum(group is None for group in groups.values()),
            "student_native_gt_only": True,
            "teacher_labels_used_by_kd": False,
            "paired_mapping": _evidence(paired_path),
            "shuffled_mapping": _evidence(shuffled_path),
        }
        if student == spec["weak_modality"]:
            selection_source = inventories[student]["val"]
            selection_target = inventories[privileged]["val"]
            if set(selection_source) != set(selection_target):
                missing = sorted(set(selection_source).symmetric_difference(selection_target))
                raise RuntimeError(
                    f"paired {evaluation_role} stems disagree for {student}->{privileged}: {missing[0]}"
                )
            selection_paired = {
                str(selection_source[stem]): str(selection_target[stem])
                for stem in sorted(selection_source)
            }
            selection_path = output / "mappings" / f"{student}_to_{privileged}_{evaluation_role}.json"
            selection_path.write_text(
                json.dumps(selection_paired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            direction.update(
                {
                    "paired_selection_pairs": len(selection_paired),
                    "paired_selection_mapping": _evidence(selection_path),
                }
            )
        directions[f"{student}_to_{privileged}"] = direction

    receipt: dict[str, Any] = {
        "schema": "rgbt-experiment-prepare-v1",
        "receipt_id": f"rgbt-prepare-{dataset}-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "data_root": str(root),
        "output_root": str(output),
        "source_splits": source_splits,
        "evaluation_role": evaluation_role,
        "test_accessed": False,
        "expected_nc": len(spec["names"]),
        "modalities": {},
        "directions": directions,
    }
    for modality, modality_root in modality_roots.items():
        receipt["modalities"][modality] = {
            "root": _evidence(modality_root),
            "train_images": len(inventories[modality]["train"]),
            "val_images": len(inventories[modality]["val"]),
            "data_yaml": _evidence(output / f"{modality}.data.yaml"),
        }
    (output / "prepare_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = prepare(args.dataset, args.data_root, args.output_root)
    receipt_path = args.receipt or args.output_root / "prepare_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "receipt": str(receipt_path.resolve()), "directions": len(receipt["directions"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
