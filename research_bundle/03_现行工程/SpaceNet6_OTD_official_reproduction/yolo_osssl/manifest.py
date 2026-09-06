"""Simple paired-image manifest handling for OS-SSL experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


Arm = Literal["paired", "sar_only", "shuffled"]
SHUFFLE_SEED = 42


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    sar_path: str
    rgb_path: str
    group_id: str
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, source: str = "record") -> "PairRecord":
        required = ("pair_id", "sar_path", "rgb_path", "group_id")
        missing = [name for name in required if not str(value.get(name, "")).strip()]
        if missing:
            raise ManifestError(f"{source} lacks required fields: {', '.join(missing)}")
        return cls(
            pair_id=str(value["pair_id"]),
            sar_path=str(value["sar_path"]),
            rgb_path=str(value["rgb_path"]),
            group_id=str(value["group_id"]),
            extras={key: item for key, item in value.items() if key not in required},
        )

    def to_mapping(self) -> dict[str, Any]:
        output = {
            "pair_id": self.pair_id,
            "sar_path": self.sar_path,
            "rgb_path": self.rgb_path,
            "group_id": self.group_id,
        }
        output.update(self.extras)
        return output


@dataclass(frozen=True)
class ArmPair:
    source: PairRecord
    second: PairRecord
    arm: Arm

    @property
    def pair_id(self) -> str:
        return self.source.pair_id


def _resolve(raw_path: str, base: Path) -> str:
    path = Path(raw_path)
    return str((base / path).resolve()) if not path.is_absolute() else str(path)


def read_pair_manifest(path: str | Path, *, verify_files: bool = False) -> list[PairRecord]:
    manifest = Path(path)
    if not manifest.is_file():
        raise ManifestError(f"Manifest is not a file: {manifest}")
    records: list[PairRecord] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ManifestError(f"Manifest line {line_number} is not an object")
        record = PairRecord.from_mapping(value, source=f"line {line_number}")
        extras = dict(record.extras)
        for name in ("donor_rgb_path", "source_rgb_path"):
            if name in extras:
                extras[name] = _resolve(str(extras[name]), manifest.parent)
        records.append(
            PairRecord(
                record.pair_id,
                _resolve(record.sar_path, manifest.parent),
                _resolve(record.rgb_path, manifest.parent),
                record.group_id,
                extras,
            )
        )
    validate_pair_records(records, verify_files=verify_files)
    return records


def validate_pair_records(records: Sequence[PairRecord], *, verify_files: bool = False) -> None:
    if not records:
        raise ManifestError("Pair manifest is empty")
    if len({record.pair_id for record in records}) != len(records):
        raise ManifestError("Pair manifest contains duplicate pair_id values")
    if verify_files:
        missing = [path for record in records for path in (record.sar_path, record.rgb_path) if not Path(path).is_file()]
        if missing:
            raise ManifestError(f"Manifest references missing image: {missing[0]}")


def build_shuffled_donor_map(records: Sequence[PairRecord], *, seed: int = SHUFFLE_SEED) -> dict[str, PairRecord]:
    """Create a deterministic donor permutation with different pair and collect IDs."""

    validate_pair_records(records)
    donors = list(records)
    rng = random.Random(seed)
    for _ in range(1000):
        rng.shuffle(donors)
        if all(a.pair_id != b.pair_id and a.group_id != b.group_id for a, b in zip(records, donors)):
            return {source.pair_id: donor for source, donor in zip(records, donors)}
    raise ManifestError("Could not construct a cross-collect shuffled donor map")


def select_arm_pairs(records: Sequence[PairRecord], arm: Arm, *, seed: int = SHUFFLE_SEED) -> list[ArmPair]:
    validate_pair_records(records)
    if arm in {"paired", "sar_only"}:
        return [ArmPair(record, record, arm) for record in records]
    if arm != "shuffled":
        raise ManifestError(f"Unsupported SSL arm: {arm}")
    if all("donor_pair_id" in record.extras for record in records):
        return [ArmPair(record, record, arm) for record in records]
    donor_map = build_shuffled_donor_map(records, seed=seed)
    return [ArmPair(record, donor_map[record.pair_id], arm) for record in records]


def frozen_shuffled_records(records: Sequence[PairRecord], *, seed: int = SHUFFLE_SEED) -> list[PairRecord]:
    donor_map = build_shuffled_donor_map(records, seed=seed)
    output: list[PairRecord] = []
    for source in records:
        donor = donor_map[source.pair_id]
        extras = dict(source.extras)
        extras.update({"arm": "shuffled", "donor_pair_id": donor.pair_id, "donor_group_id": donor.group_id})
        output.append(PairRecord(source.pair_id, source.sar_path, donor.rgb_path, source.group_id, extras))
    return output


def write_pair_manifest(records: Iterable[PairRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record.to_mapping(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
