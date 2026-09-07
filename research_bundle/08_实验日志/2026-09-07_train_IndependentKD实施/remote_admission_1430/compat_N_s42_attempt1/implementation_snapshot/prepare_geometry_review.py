"""Freeze at most 24 train-only visual-review candidates; never certify geometry.

Priority uses only existing static selected D2 and frozen natural-stream
membership, then balances dataset/source/scale. Model AP and candidate-quality
scores are not read or ranked. Calibration sampling itself is never changed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
import posixpath
from pathlib import Path

MAX_PAIRS = 24


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _key(path):
    if os.name == "nt" and str(path).startswith("/"):
        # A downloaded server manifest keeps its actual POSIX identity. Do not
        # invent E:\\mnt... by resolving Linux paths on this Windows host.
        # This is lexical only; real server aliases must be explicit in input.
        return posixpath.normpath(str(path))
    return str(Path(str(path)).absolute().resolve())


def _lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def natural_membership(spec):
    path = spec.get("natural_batches")
    if path is None:
        return {}, False
    receipt = _load(spec["coverage_receipt"])
    if receipt.get("status") != "COMPLETED" or receipt.get("actual_batches") != 64 or receipt.get("generator_seed") != 20260907:
        raise ValueError("Candidate priority requires a completed natural 64-batch seed20260907 receipt")
    if str(receipt.get("dataset", "")).lower() != str(spec["dataset"]).lower():
        raise ValueError("Natural-stream dataset differs from candidate dataset")
    mapping, batches = defaultdict(list), []
    for batch in _lines(path):
        index = batch["batch"]
        batches.append(index)
        for image in batch["images"]:
            for value in (image["image"], image["rgb_source"]):
                mapping[_key(value)].append(index)
    if batches != list(range(64)):
        raise ValueError("Natural stream must contain exactly batches0..63, without replacement")
    return {key: sorted(set(value)) for key, value in mapping.items()}, True


def dataset_candidates(spec):
    roster = _load(spec["train_roster"])
    if str(roster.get("dataset", "")).lower() != str(spec["dataset"]).lower() or roster.get("split") not in ("train", "fit"):
        raise ValueError("Explicit matching train/fit roster is required")
    if roster.get("official_test_accessed") is True:
        raise ValueError("Test-exposed roster is not admissible")
    membership, has_natural = natural_membership(spec)
    rows, path_lookup = {}, {}
    for index, row in enumerate(roster["roster"]):
        if row.get("split", roster["split"]) not in ("train", "fit"):
            raise ValueError("Non-training image in roster")
        key = _key(row["rgb_path"])
        if key in rows:
            raise ValueError("Duplicate physical train source in candidate roster")
        rows[key] = row
        for name in ("rgb_path", "rgb_source"):
            if row.get(name):
                path_lookup[_key(row[name])] = key
    selected = defaultdict(list)
    ignored = 0
    for record in _lines(spec["d2_jsonl"]):
        if record.get("selected") is not True:
            continue
        if record.get("quality_gate") is not True:
            raise ValueError("Static selected record contradicts frozen quality gate")
        image = record.get("image_id", record.get("image"))
        key = path_lookup.get(_key(image))
        if key is None:
            # Fail rather than letting a dev D2 file silently contribute frames.
            raise ValueError("Selected D2 image is absent from declared train roster: " + str(image))
        selected[key].append(record)
    result = []
    for key, objects in selected.items():
        row = rows[key]
        counts = Counter(str(obj.get("scale_bin", "unknown")) for obj in objects)
        scale = min(counts, key=lambda name: (-counts[name], name))
        source = str(row.get("source_group", row.get("group", "")))
        if not source or source.startswith("unavailable:"):
            raise ValueError("Real source metadata is required; folder name is not a fallback")
        hits = sorted(set(membership.get(key, []) + membership.get(_key(row.get("rgb_source", row["rgb_path"])), [])))
        result.append({"dataset": spec["dataset"], "split": roster["split"],
            "stem": row["stem"], "rgb_path": row["rgb_path"], "ir_path": row["ir_path"],
            "rgb_source": key, "ir_source": _key(row.get("ir_source", row["ir_path"])),
            "source_group": source, "representative_scale": scale, "static_scale_counts": dict(counts),
            "static_selected_object_count": len(objects), "natural_hit_batches": hits,
            "natural_hit": bool(hits), "static_object_ids": [obj.get("object_id") for obj in objects],
            "geometry_verified": False})
    return result, {"dataset": spec["dataset"], "train_roster": str(spec["train_roster"]),
                    "static_d2": str(spec["d2_jsonl"]), "unique_train_candidates": len(result),
                    "natural_receipt_available": has_natural,
                    "candidates_in_natural_stream": sum(row["natural_hit"] for row in result),
                    "natural_batches": spec.get("natural_batches")}


def generate_candidates(input_specs, limit=MAX_PAIRS):
    if type(limit) is not int or not 1 <= limit <= MAX_PAIRS:
        raise ValueError("This bounded audit allows between1 and24 pairs")
    pool, input_receipts = [], []
    for spec in input_specs:
        rows, receipt = dataset_candidates(spec)
        pool.extend(rows)
        input_receipts.append(receipt)
    identities = {(row["dataset"], row["rgb_source"]) for row in pool}
    if len(identities) != len(pool):
        raise ValueError("Duplicate dataset/source in input specifications")
    selected, dataset_counts, source_counts, scale_counts = [], Counter(), Counter(), Counter()
    while pool and len(selected) < limit:
        # Strict natural-hit priority; within that partition, greedily spread
        # dataset/source/scale. Stable path tie-break, no data- or model-RNG use.
        def priority(row):
            d, g, s = row["dataset"], row["source_group"], row["representative_scale"]
            return (not row["natural_hit"], dataset_counts[d], source_counts[(d, g)],
                    scale_counts[(d, s)], str(d), str(g), str(s), row["rgb_source"])
        row = min(pool, key=priority)
        pool.remove(row)
        selected.append(row)
        dataset_counts[row["dataset"]] += 1
        source_counts[(row["dataset"], row["source_group"])] += 1
        scale_counts[(row["dataset"], row["representative_scale"])] += 1
    review_n = math.ceil(len(selected) * 0.2)
    independent = {i * len(selected) // review_n for i in range(review_n)} if review_n else set()
    for i, row in enumerate(selected):
        row.update(audit_id="independent_v2_" + row["dataset"] + "_" + row["stem"],
                   review_order=i, independent_review_required=i in independent,
                   annotation_status="PENDING_VISUAL_INSPECTION", review_status="PENDING")
    complete_inputs = all(row["natural_receipt_available"] for row in input_receipts)
    receipt = {"protocol": "independent_geometry_24_train_static_d2_natural_priority_v1",
        "status": "FROZEN_CANDIDATE_ROSTER" if complete_inputs else "DRAFT_AWAIT_NATURAL_STREAM",
        "requested_limit": limit, "selected_pairs": len(selected), "independent_reviews_required": review_n,
        "inputs": input_receipts, "selection_rule": "natural-hit first; balance dataset/source/scale; stable source-path tie-break",
        "uses_model_AP": False, "uses_quality_value_ranking": False, "modifies_calibration_sampling": False,
        "geometry_verified": False, "annotation_points_created": 0,
        "candidate_count_does_not_prove_16_signal_batches": True, "roster": selected}
    return receipt


def write_package(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "candidate_selection_receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep model/object/scale information out of the visual point reviewer feed.
    visual_keys = ("audit_id", "dataset", "split", "stem", "rgb_path", "ir_path", "rgb_source", "ir_source", "source_group", "independent_review_required")
    visual = {"protocol": result["protocol"], "status": result["status"], "geometry_verified": False,
              "roster": [{key: row[key] for key in visual_keys} for row in result["roster"]]}
    (output / "visual_review_roster.json").write_text(json.dumps(visual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    template = {"protocol": result["protocol"], "annotator": None, "coordinate_basis": "original images only; no GT/prediction boxes",
        "formal_geometry_verified": False, "observations": [{"audit_id": row["audit_id"],
             "status": "PENDING_VISUAL_INSPECTION", "original_shape": None, "uncertainty_raw_px": None,
             "points": [], "rejection_or_scope_notes": None} for row in result["roster"]]}
    (output / "primary_observations_template.json").write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON list or {datasets:[...]} of frozen input paths")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = _load(args.inputs)
    specs = inputs["datasets"] if isinstance(inputs, dict) else inputs
    result = generate_candidates(specs)
    write_package(result, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "roster"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
