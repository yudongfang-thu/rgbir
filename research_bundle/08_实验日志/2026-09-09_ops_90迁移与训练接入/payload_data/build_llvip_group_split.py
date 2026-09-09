#!/usr/bin/env python3
"""Create a sequence-grouped LLVIP fit/dev manifest without touching images."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


DEV_PREFIXES = {"01", "04", "07", "12", "25"}


def stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.jpg")}


def object_count(annotation: Path) -> int:
    return len(ET.parse(annotation).getroot().findall("object"))


def rows(root: Path, split: str, selected: list[str]) -> list[dict[str, str]]:
    output = []
    for stem in selected:
        output.append(
            {
                "stem": stem,
                "sequence_prefix": stem[:2],
                "visible": str(root / "visible" / split / f"{stem}.jpg"),
                "infrared": str(root / "infrared" / split / f"{stem}.jpg"),
                "annotation": str(root / "Annotations" / f"{stem}.xml"),
            }
        )
    return output


def write_tsv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    train_visible = stems(args.root / "visible" / "train")
    train_infrared = stems(args.root / "infrared" / "train")
    test_visible = stems(args.root / "visible" / "test")
    test_infrared = stems(args.root / "infrared" / "test")
    annotations = {path.stem for path in (args.root / "Annotations").glob("*.xml")}

    assert train_visible == train_infrared
    assert test_visible == test_infrared
    assert train_visible.isdisjoint(test_visible)
    assert train_visible | test_visible == annotations

    dev_stems = sorted(stem for stem in train_visible if stem[:2] in DEV_PREFIXES)
    fit_stems = sorted(train_visible - set(dev_stems))
    test_stems = sorted(test_visible)

    split_rows = {
        "fit": rows(args.root, "train", fit_stems),
        "dev": rows(args.root, "train", dev_stems),
        "test": rows(args.root, "test", test_stems),
    }
    for name, records in split_rows.items():
        write_tsv(args.output_dir / f"{name}.tsv", records)

    receipt = {
        "version": "llvip_grouped_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.root),
        "group_key": "first two filename digits",
        "dev_prefixes": sorted(DEV_PREFIXES),
        "fit_prefixes": sorted({stem[:2] for stem in fit_stems}),
        "official_test_prefixes": sorted({stem[:2] for stem in test_stems}),
        "counts": {name: len(records) for name, records in split_rows.items()},
        "person_instances": {
            name: sum(object_count(Path(record["annotation"])) for record in records)
            for name, records in split_rows.items()
        },
        "prefix_pair_counts": {
            name: dict(Counter(record["sequence_prefix"] for record in records))
            for name, records in split_rows.items()
        },
        "selection_rationale": (
            "Five complete training prefixes give 2,406 dev pairs (20.0% of the "
            "official training pairs) while spanning dark, mid, and bright visible "
            "intensity regimes observed in the deterministic CPU audit."
        ),
        "protocol": (
            "Use fit for training and dev for method selection. The official test list "
            "is recorded for final frozen evaluation only."
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
