#!/usr/bin/env python3
"""Validate a canonical YOLO OS-SSL JSONL manifest and freeze shuffled donors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.manifest import (
    SHUFFLE_SEED,
    ManifestError,
    frozen_shuffled_records,
    read_pair_manifest,
    select_arm_pairs,
    write_pair_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path, help="Canonical SAR/RGB pair manifest.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Validated ordinary JSONL output.")
    parser.add_argument("--arm", choices=("paired", "sar_only", "shuffled"), required=True)
    parser.add_argument("--verify-files", action="store_true", help="Rehash every listed SAR and RGB image before writing.")
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED, help="Frozen shuffled donor seed (must be 42).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.seed != SHUFFLE_SEED:
            raise ManifestError(f"The frozen donor seed is {SHUFFLE_SEED}; received {args.seed}.")
        records = read_pair_manifest(args.input_jsonl, verify_files=args.verify_files)
        if args.arm == "shuffled":
            output_records = frozen_shuffled_records(records, seed=args.seed)
        else:
            # Calls the identity constructor even though canonical records are retained.
            select_arm_pairs(records, args.arm, seed=args.seed)
            output_records = records
        write_pair_manifest(output_records, args.output_jsonl)
        print(
            json.dumps(
                {
                    "engineering_status": "ok",
                    "arm": args.arm,
                    "records": len(output_records),
                    "output_jsonl": str(args.output_jsonl),
                    "hashes_verified": bool(args.verify_files),
                    "shuffled_seed": SHUFFLE_SEED if args.arm == "shuffled" else None,
                },
                sort_keys=True,
            )
        )
        return 0
    except (ManifestError, OSError) as exc:
        print(json.dumps({"engineering_status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
