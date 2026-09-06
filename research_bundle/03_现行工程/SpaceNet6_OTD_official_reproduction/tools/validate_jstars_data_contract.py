#!/usr/bin/env python3
"""Validate paired-data manifests and the frozen JSTARS test-exposure ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PAIR_REPORT_SCHEMA = "jstars-paired-data-contract-validation-v1"
EXPOSURE_LEDGER_SCHEMA = "jstars-dataset-exposure-ledger-v1"
EXPOSURE_REPORT_SCHEMA = "jstars-dataset-exposure-validation-v1"
MAX_ISSUE_SAMPLES = 20

FROZEN_EXPOSURE_POLICY = {
    "stf": {
        "role": "REPRO_ONLY",
        "test_status": "REPRODUCTION_TEST",
        "confirmatory": False,
    },
    "llvip": {
        "role": "SEALED_CANDIDATE",
        "test_status": "UNVERIFIED_SEALED",
        "confirmatory": False,
    },
    "dronevehicle": {
        "role": "SEALED_CANDIDATE",
        "test_status": "UNVERIFIED_SEALED",
        "confirmatory": False,
    },
    "vedai": {
        "role": "DEVELOPMENT_ONLY",
        "test_status": "NO_INDEPENDENT_CONFIRMATORY_TEST",
        "confirmatory": False,
    },
    "spacenet6": {
        "role": "NOT_CONFIRMATORY",
        "test_status": "HISTORICALLY_EXPOSED_TEST",
        "confirmatory": False,
    },
}


class ContractError(ValueError):
    """Raised when a contract input cannot be parsed."""


class _Issues:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.samples: list[dict[str, Any]] = []

    def add(self, code: str, *, amount: int = 1, **details: Any) -> None:
        self.counts[code] += amount
        if len(self.samples) < MAX_ISSUE_SAMPLES:
            self.samples.append({"code": code, **details})


def _resolved(raw: str, base: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _normalise_stem(raw: str, dataset: str) -> str:
    stem = Path(raw).stem
    if dataset.lower() == "vedai":
        if stem.endswith(("_co", "_ir")):
            stem = stem[:-3]
        if stem.isdigit():
            return stem.zfill(8)
    return stem


def _read_mapping(path: Path) -> list[tuple[Path, Path]]:
    if not path.is_file():
        raise ContractError(f"paired manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ContractError(f"paired manifest must be a non-empty JSON object: {path}")
    if any(not isinstance(source, str) or not isinstance(target, str) for source, target in payload.items()):
        raise ContractError(f"paired manifest paths must be strings: {path}")
    return [(_resolved(source, path.parent), _resolved(target, path.parent)) for source, target in payload.items()]


def _read_roster(path: Path, dataset: str) -> list[str]:
    if not path.is_file():
        raise ContractError(f"split roster is missing: {path}")
    values = [
        _normalise_stem(line.split()[0], dataset)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ContractError(f"split roster is empty: {path}")
    return values


def build_pair_manifest(
    *,
    dataset: str,
    roster: Path,
    source_dir: Path,
    target_dir: Path,
    source_suffix: str,
    target_suffix: str,
) -> dict[str, str]:
    """Build a complete, ordered source-to-target manifest from one frozen roster."""

    stems = _read_roster(roster, dataset)
    if _duplicate_count(stems):
        raise ContractError(f"split roster contains duplicate stems: {roster}")
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for stem in stems:
        source = (source_dir / f"{stem}{source_suffix}").resolve()
        target = (target_dir / f"{stem}{target_suffix}").resolve()
        if not source.is_file() or not target.is_file():
            missing.append(stem)
            continue
        mapping[str(source)] = str(target)
    if missing:
        examples = ", ".join(missing[:5])
        raise ContractError(
            f"{len(missing)} roster pairs are missing source or target files; examples: {examples}"
        )
    return mapping


def _duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _known_nested_target_issue(
    target: Path,
    *,
    source_modality: str,
    target_modality: str,
) -> dict[str, Any] | None:
    parts = target.parts
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if not image_indices:
        return None
    index = image_indices[-1]
    tail = parts[index + 1 :]
    if len(tail) < 4 or tail[0] != source_modality or tail[1] != target_modality:
        return None
    corrected = Path(*parts[: index + 1], target_modality, *tail[2:])
    return {
        "bad_fragment": f"images/{source_modality}/{target_modality}/{tail[2]}",
        "expected_fragment": f"images/{target_modality}/{tail[2]}",
        "suggested_target": str(corrected),
        "suggested_target_exists": corrected.is_file(),
    }


def _validate_manifest(
    path: Path,
    *,
    split: str,
    dataset: str,
    source_modality: str,
    target_modality: str,
    issues: _Issues,
) -> tuple[dict[str, Any], set[str]]:
    pairs = _read_mapping(path)
    source_paths = [str(source) for source, _ in pairs]
    target_paths = [str(target) for _, target in pairs]
    source_stems = [_normalise_stem(source.name, dataset) for source, _ in pairs]
    target_stems = [_normalise_stem(target.name, dataset) for _, target in pairs]
    source_existing = 0
    target_existing = 0
    stem_matches = 0

    for source, target in pairs:
        if source.is_file():
            source_existing += 1
        else:
            issues.add("MISSING_SOURCE_FILE", split=split, path=str(source))
        if target.is_file():
            target_existing += 1
        else:
            issues.add("MISSING_TARGET_FILE", split=split, path=str(target))
        if _normalise_stem(source.name, dataset) == _normalise_stem(target.name, dataset):
            stem_matches += 1
        else:
            issues.add(
                "PAIR_STEM_MISMATCH",
                split=split,
                source=str(source),
                target=str(target),
            )
        known_issue = _known_nested_target_issue(
            target,
            source_modality=source_modality,
            target_modality=target_modality,
        )
        if known_issue is not None:
            issues.add(
                "TARGET_MODALITY_NESTED_UNDER_SOURCE",
                split=split,
                source=str(source),
                target=str(target),
                **known_issue,
            )

    duplicates = {
        "DUPLICATE_SOURCE_PATH": _duplicate_count(source_paths),
        "DUPLICATE_TARGET_PATH": _duplicate_count(target_paths),
        "DUPLICATE_SOURCE_STEM": _duplicate_count(source_stems),
        "DUPLICATE_TARGET_STEM": _duplicate_count(target_stems),
    }
    for code, count in duplicates.items():
        if count:
            issues.add(code, amount=count, split=split, duplicates=count)

    summary = {
        "path": str(path.resolve()),
        "pairs": len(pairs),
        "source_files_existing": source_existing,
        "target_files_existing": target_existing,
        "stem_matches": stem_matches,
        "unique_source_stems": len(set(source_stems)),
        "unique_target_stems": len(set(target_stems)),
    }
    return summary, set(source_stems)


def validate_pair_contract(
    *,
    dataset: str,
    train_manifest: Path,
    source_modality: str,
    target_modality: str,
    val_manifest: Path | None = None,
    train_roster: Path | None = None,
    val_roster: Path | None = None,
    split_identity: str | None = None,
) -> dict[str, Any]:
    """Validate pair existence, bijection, stem identity, and split separation."""

    issues = _Issues()
    manifest_summaries: dict[str, Any] = {}
    manifest_stems: dict[str, set[str]] = {}
    manifest_summaries["train"], manifest_stems["train"] = _validate_manifest(
        train_manifest,
        split="train",
        dataset=dataset,
        source_modality=source_modality,
        target_modality=target_modality,
        issues=issues,
    )
    if val_manifest is not None:
        manifest_summaries["val"], manifest_stems["val"] = _validate_manifest(
            val_manifest,
            split="val",
            dataset=dataset,
            source_modality=source_modality,
            target_modality=target_modality,
            issues=issues,
        )

    train_val_overlap = sorted(manifest_stems["train"] & manifest_stems.get("val", set()))
    if train_val_overlap:
        issues.add(
            "TRAIN_VAL_STEM_OVERLAP",
            amount=len(train_val_overlap),
            overlap_count=len(train_val_overlap),
            examples=train_val_overlap[:5],
        )

    roster_paths = {"train": train_roster, "val": val_roster}
    roster_summaries: dict[str, Any] = {}
    roster_stems: dict[str, set[str]] = {}
    for split, roster_path in roster_paths.items():
        if roster_path is None:
            continue
        values = _read_roster(roster_path, dataset)
        duplicates = _duplicate_count(values)
        if duplicates:
            issues.add("DUPLICATE_ROSTER_STEM", amount=duplicates, split=split, duplicates=duplicates)
        roster_stems[split] = set(values)
        roster_summaries[split] = {
            "path": str(roster_path.resolve()),
            "entries": len(values),
            "unique_stems": len(roster_stems[split]),
        }
        if split in manifest_stems:
            missing = sorted(roster_stems[split] - manifest_stems[split])
            extra = sorted(manifest_stems[split] - roster_stems[split])
            if missing:
                issues.add(
                    "MANIFEST_MISSING_ROSTER_STEM",
                    amount=len(missing),
                    split=split,
                    count=len(missing),
                    examples=missing[:5],
                )
            if extra:
                issues.add(
                    "MANIFEST_EXTRA_ROSTER_STEM",
                    amount=len(extra),
                    split=split,
                    count=len(extra),
                    examples=extra[:5],
                )

    roster_overlap = sorted(roster_stems.get("train", set()) & roster_stems.get("val", set()))
    if roster_overlap:
        issues.add(
            "TRAIN_VAL_ROSTER_OVERLAP",
            amount=len(roster_overlap),
            overlap_count=len(roster_overlap),
            examples=roster_overlap[:5],
        )
    if split_identity == "official_fold01" and (train_roster is None or val_roster is None):
        issues.add("OFFICIAL_FOLD01_ROSTER_INCOMPLETE")

    return {
        "schema": PAIR_REPORT_SCHEMA,
        "status": "PASS" if not issues.counts else "FAIL",
        "dataset": dataset.lower(),
        "source_modality": source_modality,
        "target_modality": target_modality,
        "split_identity": split_identity,
        "manifests": manifest_summaries,
        "rosters": roster_summaries,
        "train_val_overlap_count": len(train_val_overlap),
        "roster_overlap_count": len(roster_overlap),
        "issue_counts": dict(sorted(issues.counts.items())),
        "issues": issues.samples,
    }


def validate_exposure_ledger(
    ledger: dict[str, Any], *, confirmatory_dataset: str | None = None
) -> dict[str, Any]:
    """Validate the frozen dataset-use identities and an optional confirmation request."""

    issues = _Issues()
    if ledger.get("schema") != EXPOSURE_LEDGER_SCHEMA:
        issues.add("LEDGER_SCHEMA_MISMATCH", actual=ledger.get("schema"))
    rows = ledger.get("datasets")
    if not isinstance(rows, list):
        raise ContractError("exposure ledger datasets must be a list")
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("dataset", "")).strip():
            raise ContractError("every exposure ledger row needs a dataset")
        dataset = str(row["dataset"]).lower()
        if dataset in by_dataset:
            issues.add("DUPLICATE_DATASET", dataset=dataset)
        by_dataset[dataset] = row
        if row.get("test_status") == "HISTORICALLY_EXPOSED_TEST" and row.get("confirmatory") is not False:
            issues.add("HISTORICALLY_EXPOSED_CONFIRMATORY", dataset=dataset)

    for dataset, expected in FROZEN_EXPOSURE_POLICY.items():
        row = by_dataset.get(dataset)
        if row is None:
            issues.add("MISSING_FROZEN_DATASET", dataset=dataset)
            continue
        drift = {
            field: {"expected": value, "actual": row.get(field)}
            for field, value in expected.items()
            if row.get(field) != value
        }
        if drift:
            issues.add("FROZEN_POLICY_DRIFT", dataset=dataset, fields=drift)

    if confirmatory_dataset is not None:
        requested = confirmatory_dataset.lower()
        row = by_dataset.get(requested)
        if row is None or row.get("confirmatory") is not True:
            issues.add("CONFIRMATORY_USE_FORBIDDEN", dataset=requested)

    return {
        "schema": EXPOSURE_REPORT_SCHEMA,
        "status": "PASS" if not issues.counts else "FAIL",
        "datasets_validated": len(by_dataset),
        "confirmatory_dataset_requested": confirmatory_dataset.lower() if confirmatory_dataset else None,
        "issue_counts": dict(sorted(issues.counts.items())),
        "issues": issues.samples,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pairs = subparsers.add_parser("pairs", help="validate paired train/val manifests")
    pairs.add_argument("--dataset", required=True)
    pairs.add_argument("--train-manifest", required=True, type=Path)
    pairs.add_argument("--val-manifest", type=Path)
    pairs.add_argument("--source-modality", required=True)
    pairs.add_argument("--target-modality", required=True)
    pairs.add_argument("--train-roster", type=Path)
    pairs.add_argument("--val-roster", type=Path)
    pairs.add_argument(
        "--vedai-official-fold-dir",
        type=Path,
        help="Use fold01.txt and fold01test.txt from this VEDAI Annotations directory.",
    )
    pairs.add_argument("--split-identity")
    pairs.add_argument("--output", type=Path)

    exposure = subparsers.add_parser("exposure", help="validate the dataset exposure ledger")
    exposure.add_argument("--ledger", required=True, type=Path)
    exposure.add_argument("--confirmatory-dataset")
    exposure.add_argument("--output", type=Path)

    build = subparsers.add_parser(
        "build-pairs", help="build a versioned paired manifest from a frozen roster"
    )
    build.add_argument("--dataset", required=True)
    build.add_argument("--roster", required=True, type=Path)
    build.add_argument("--source-dir", required=True, type=Path)
    build.add_argument("--target-dir", required=True, type=Path)
    build.add_argument("--source-suffix", default=".png")
    build.add_argument("--target-suffix", default=".png")
    build.add_argument("--output", required=True, type=Path)
    return parser


def _write_and_print(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "build-pairs":
            if args.output.exists():
                raise ContractError(f"refusing to overwrite paired manifest: {args.output}")
            mapping = build_pair_manifest(
                dataset=args.dataset,
                roster=args.roster,
                source_dir=args.source_dir,
                target_dir=args.target_dir,
                source_suffix=args.source_suffix,
                target_suffix=args.target_suffix,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "schema": PAIR_REPORT_SCHEMA,
                        "status": "PASS",
                        "dataset": args.dataset.lower(),
                        "pairs": len(mapping),
                        "output": str(args.output.resolve()),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "pairs":
            train_roster = args.train_roster
            val_roster = args.val_roster
            split_identity = args.split_identity
            if args.vedai_official_fold_dir is not None:
                if args.dataset.lower() != "vedai":
                    raise ContractError("--vedai-official-fold-dir is only valid for VEDAI")
                train_roster = args.vedai_official_fold_dir / "fold01.txt"
                val_roster = args.vedai_official_fold_dir / "fold01test.txt"
                split_identity = "official_fold01"
            report = validate_pair_contract(
                dataset=args.dataset,
                train_manifest=args.train_manifest,
                val_manifest=args.val_manifest,
                source_modality=args.source_modality,
                target_modality=args.target_modality,
                train_roster=train_roster,
                val_roster=val_roster,
                split_identity=split_identity,
            )
        else:
            ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
            if not isinstance(ledger, dict):
                raise ContractError("exposure ledger root must be an object")
            report = validate_exposure_ledger(
                ledger, confirmatory_dataset=args.confirmatory_dataset
            )
    except (ContractError, OSError, json.JSONDecodeError) as error:
        report = {
            "schema": PAIR_REPORT_SCHEMA if args.command in {"pairs", "build-pairs"} else EXPOSURE_REPORT_SCHEMA,
            "status": "FAIL",
            "issue_counts": {"INPUT_ERROR": 1},
            "issues": [{"code": "INPUT_ERROR", "message": str(error)}],
        }
    _write_and_print(report, args.output if args.command != "build-pairs" else None)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
