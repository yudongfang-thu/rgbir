#!/usr/bin/env python3
"""Rebuild the frozen 94 train/dev views in a new 90 attempt directory.

No test image/XML is read, extracted, converted, or mapped. ZIP central-directory
metadata is inspected to locate the selected official-training members. Historical
receipts are compared as stored; no new digest is computed. Failed output remains
in place and cannot be resumed/overwritten by this entry point.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import csv
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path("/mnt/dataX/ydf/projects/RGBT_campaign_90/data_attempt1")
DRONE_SOURCE = Path("/mnt/dataset/DroneVehicle")
LLVIP_ZIP = Path("/mnt/dataY/ydf/dataset/LLVIP.zip")
PAYLOAD = Path(__file__).resolve().parent / "payload_data"
DEV_PREFIXES = {"01", "04", "07", "12", "25"}
FIT_PREFIXES = {"02", "03", "05", "06", "08", "09", "10", "11",
                "13", "14", "15", "16", "17", "18"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def event(stage, **values):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    with (ROOT / "stages.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PAYLOAD / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_llvip_lists():
    with (PAYLOAD / "llvip_fit_original.tsv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    fit = [record["stem"] for record in records]
    dev_paths = (PAYLOAD / "llvip_dev_original.txt").read_text(encoding="utf-8-sig").splitlines()
    dev = [PurePosixPath(value.strip()).stem for value in dev_paths if value.strip()]
    for role, items, prefixes, count in (("fit", fit, FIT_PREFIXES, 9619),
                                          ("dev", dev, DEV_PREFIXES, 2406)):
        require(len(items) == len(set(items)) == count, f"bad frozen {role} count or duplicates")
        require(all(len(stem) == 6 and stem.isdigit() for stem in items), f"bad {role} stems")
        require({stem[:2] for stem in items} == prefixes, f"bad frozen {role} prefix set")
    require(set(fit).isdisjoint(dev), "fit/dev overlap")
    for record in records:
        stem = record["stem"]
        require(record["sequence_prefix"] == stem[:2], "fit sequence mismatch")
        for modality in ("visible", "infrared"):
            require(PurePosixPath(record[modality]).parts[-3:] == (modality, "train", stem + ".jpg"),
                    "fit source is not the saved official train member")
        require(PurePosixPath(record["annotation"]).parts[-2:] == ("Annotations", stem + ".xml"),
                "fit annotation source mismatch")
    return {"fit": sorted(fit), "dev": sorted(dev)}


def safe_member_parts(name):
    # All central-directory names are checked, but nonselected members are never read.
    require("\\" not in name and "\x00" not in name, f"unsafe ZIP name: {name!r}")
    parts = PurePosixPath(name).parts
    require(bool(parts) and not name.startswith("/") and ".." not in parts,
            f"unsafe ZIP path: {name!r}")
    require(not any(":" in part for part in parts), f"unsafe ZIP drive/stream: {name!r}")
    return parts


def selected_zip_members(archive, lists):
    expected_stems = set(lists["fit"]) | set(lists["dev"])
    selected = {}
    official_images = {"visible": set(), "infrared": set()}
    for info in archive.infolist():
        parts = safe_member_parts(info.filename)
        if info.is_dir():
            continue
        rel = None
        if len(parts) >= 3 and parts[-3] in official_images and parts[-2] == "train":
            require(parts[-1].lower().endswith(".jpg"), f"unexpected official train file {info.filename}")
            stem = PurePosixPath(parts[-1]).stem
            modality = parts[-3]
            official_images[modality].add(stem)
            if stem in expected_stems:
                rel = PurePosixPath(modality, "train", stem + ".jpg")
        elif len(parts) >= 2 and parts[-2] == "Annotations" and parts[-1].endswith(".xml"):
            stem = PurePosixPath(parts[-1]).stem
            if stem in expected_stems:
                rel = PurePosixPath("Annotations", stem + ".xml")
        if rel is None:
            continue
        mode = info.external_attr >> 16
        require(not stat.S_ISLNK(mode), f"refusing ZIP symlink {info.filename}")
        require(not (info.flag_bits & 1), f"encrypted selected ZIP entry {info.filename}")
        require(str(rel) not in selected, f"duplicate selected ZIP destination {rel}")
        selected[str(rel)] = info
    for modality, stems in official_images.items():
        require(stems == expected_stems, f"ZIP {modality}/train differs from full frozen fit+dev roster")
    expected = {f"{m}/train/{s}.jpg" for m in official_images for s in expected_stems}
    expected |= {f"Annotations/{s}.xml" for s in expected_stems}
    require(set(selected) == expected, "selected ZIP image/XML inventory mismatch")
    return selected


def extract_llvip(lists):
    output = ROOT / "raw" / "LLVIP_train_only"
    output.mkdir(parents=True, exist_ok=False)
    total_bytes = 0
    with zipfile.ZipFile(LLVIP_ZIP) as archive:
        selected = selected_zip_members(archive, lists)
        event("llvip_zip_selected", selected_members=len(selected), official_train_pairs=12025)
        for index, (relative, info) in enumerate(sorted(selected.items()), 1):
            target = output / relative
            require(target.resolve().is_relative_to(output.resolve()), "ZIP target escaped extraction root")
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            # ZipExtFile verifies CRC when consumed to EOF; no testzip() or extractall().
            with archive.open(info, "r") as source, target.open("xb") as destination:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    written += len(block)
            require(written == info.file_size, f"ZIP size mismatch {relative}")
            total_bytes += written
            if index % 500 == 0 or index == len(selected):
                event("llvip_extract_progress", done=index, total=len(selected), bytes_written=total_bytes)
    return output, {"selected_members": len(selected), "bytes_written": total_bytes,
                    "selected_member_crc": "zipfile streamed EOF validation", "test_extracted": False}


def validate_views(output, lists, modalities, expected_boxes, cv2, fixed_size=None):
    result = {}
    for split, stems in lists.items():
        for modality in modalities:
            image_dir, label_dir = output / modality / "images" / split, output / modality / "labels" / split
            require({p.stem for p in image_dir.glob("*.jpg")} == set(stems), f"output images mismatch {split}/{modality}")
            require({p.stem for p in label_dir.glob("*.txt")} == set(stems), f"output labels mismatch {split}/{modality}")
            rows = 0
            for index, stem in enumerate(stems, 1):
                image = cv2.imread(str(image_dir / (stem + ".jpg")), cv2.IMREAD_UNCHANGED)
                require(image is not None, f"unreadable output {split}/{modality}/{stem}")
                if fixed_size is not None:
                    require((image.shape[1], image.shape[0]) == fixed_size, "incorrect cropped dimensions")
                for line in (label_dir / (stem + ".txt")).read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    require(len(fields) == 5, "malformed YOLO label")
                    class_id = int(fields[0])
                    values = list(map(float, fields[1:]))
                    require(class_id in range(5 if fixed_size else 1), "invalid class id")
                    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "invalid normalized coordinates")
                    require(values[2] > 0 and values[3] > 0, "nonpositive box extent")
                    rows += 1
                if index % 2000 == 0:
                    event("output_validation_progress", split=split, modality=modality, done=index)
            require(rows == expected_boxes[f"{split}/{modality}"], f"output label count mismatch {split}/{modality}: {rows}")
            result[f"{split}/{modality}"] = {"images": len(stems), "labels": len(stems), "boxes": rows}
            event("output_validation_complete", split=split, modality=modality, boxes=rows)
    return result


def prepare_drone(drone, cv2):
    output = ROOT / "processed" / "dronevehicle" / "yolo" / "hbb_v1"
    output.mkdir(parents=True, exist_ok=False)
    historical = load_json(PAYLOAD / "dronevehicle_yolo_hbb_v1.json")
    lists = {split: [f"{i:05d}" for i in range(1, count + 1)]
             for split, count in (("train", 17990), ("val", 1469))}
    results, expected_boxes = {}, {}
    for split, stems in lists.items():
        for modality in ("rgb", "infrared"):
            image_dir, xml_dir = drone.SPLIT_SPECS[split][modality]
            images = drone._image_map(DRONE_SOURCE / split / image_dir)
            annotations = drone._xml_map(DRONE_SOURCE / split / xml_dir)
            require(set(images) == set(annotations) == set(stems), f"Drone source roster mismatch {split}/{modality}")
            require(all(path.suffix == ".jpg" for path in images.values()), "Drone expected original JPG extension")
            aggregate = {key: Counter() for key in ("source_geometry", "source_class_instances", "written_class_instances", "ignored", "dropped", "alias_usage")}
            totals = Counter()
            event("drone_convert_start", split=split, modality=modality, pairs=len(stems))
            for index, stem in enumerate(stems, 1):
                image_out = output / modality / "images" / split / (stem + ".jpg")
                label_out = output / modality / "labels" / split / (stem + ".txt")
                require(not image_out.exists() and not label_out.exists(), "refusing output overwrite")
                value = drone._process_image(images[stem], annotations[stem], image_out, label_out)
                for key in ("labels_written", "source_objects", "official_eval_candidates"):
                    totals[key] += value[key]
                for key in aggregate:
                    aggregate[key].update(value[key])
                if index % 500 == 0 or index == len(stems):
                    event("drone_convert_progress", split=split, modality=modality, done=index, total=len(stems))
            key = f"{split}/{modality}"
            observed = {**dict(totals), **{name: dict(value) for name, value in aggregate.items()}}
            expected = historical["by_split_modality"][key]
            for field, value in observed.items():
                require(value == expected[field], f"Drone historical receipt mismatch: {key} {field}")
            results[key] = observed
            expected_boxes[key] = expected["labels_written"]
    groups = drone.write_train_source_groups(DRONE_SOURCE, output)
    for modality, value in groups.items():
        require(value["rows_written"] == 17990 and value["missing_folder_count"] == 0, f"missing Drone {modality} source groups")
    saved_groups = (PAYLOAD / "drone_rgb_train_source_groups_original.tsv").read_text(encoding="utf-8-sig").splitlines()
    actual_groups = (output / "rgb_train_source_groups.tsv").read_text(encoding="utf-8").splitlines()
    require(actual_groups == saved_groups, "Drone RGB source-group list differs from original")
    validation = validate_views(output, lists, ("rgb", "infrared"), expected_boxes, cv2, (640, 512))
    receipt = {"version": "dronevehicle-yolo-hbb-v1-train-val-restoration", "source": str(DRONE_SOURCE),
               "output": str(output), "test_accessed": False, "historical_counters_exact": True,
               "by_split_modality": results, "source_groups": groups, "validation": validation,
               "byte_identity_with_94": "not asserted; original conversion functions and counter comparison only"}
    write_json_new(ROOT / "drone_receipt.json", receipt)
    return output, receipt


def prepare_llvip(llvip, cv2, lists):
    raw, extraction = extract_llvip(lists)
    output = ROOT / "processed" / "llvip" / "yolo" / "grouped_v1"
    output.mkdir(parents=True, exist_ok=False)
    manifest_dir = ROOT / "processed" / "llvip" / "splits" / "grouped_v1"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    historical = load_json(PAYLOAD / "llvip_yolo_grouped_v1.json")
    results, expected_boxes = {}, {}
    for split, stems in lists.items():
        rows, excluded, classes = 0, 0, Counter()
        with (manifest_dir / (split + ".tsv")).open("x", encoding="utf-8", newline="") as manifest:
            writer = csv.DictWriter(manifest, fieldnames=["stem", "sequence_prefix", "visible", "infrared", "annotation"], delimiter="\t")
            writer.writeheader()
            for index, stem in enumerate(stems, 1):
                annotation = raw / "Annotations" / (stem + ".xml")
                labels, dropped, names = llvip.yolo_labels(annotation)
                rows += len(labels)
                excluded += dropped
                classes.update(names)
                record = {"stem": stem, "sequence_prefix": stem[:2], "annotation": str(annotation)}
                for modality in ("visible", "infrared"):
                    source = raw / modality / "train" / (stem + ".jpg")
                    record[modality] = str(source)
                    llvip.link(source, output / modality / "images" / split / (stem + ".jpg"))
                    label_out = output / modality / "labels" / split / (stem + ".txt")
                    label_out.parent.mkdir(parents=True, exist_ok=True)
                    with label_out.open("x", encoding="utf-8") as handle:
                        handle.write("\n".join(labels) + ("\n" if labels else ""))
                writer.writerow(record)
                if index % 500 == 0 or index == len(stems):
                    event("llvip_convert_progress", split=split, done=index, total=len(stems))
        require(rows == historical["splits"][split]["output_objects_per_modality"], f"LLVIP {split} box receipt mismatch")
        require(excluded == historical["excluded_by_split"][split], f"LLVIP {split} exclusion mismatch")
        results[split] = {"pairs": len(stems), "output_boxes_per_modality": rows, "excluded_degenerate": excluded,
                          "class_instances_before_exclusion": dict(classes)}
        for modality in ("visible", "infrared"):
            expected_boxes[f"{split}/{modality}"] = rows
    validation = validate_views(output, lists, ("visible", "infrared"), expected_boxes, cv2)
    receipt = {"version": "llvip_yolo_grouped_v1-fit-dev-restoration", "archive": str(LLVIP_ZIP),
               "test_accessed": False, "extraction": extraction, "splits": results,
               "frozen_fit_dev_rosters_exact": True, "validation": validation,
               "byte_identity_with_94": "not asserted; selected ZIP CRC, original labels functions and historical counters checked"}
    write_json_new(ROOT / "llvip_receipt.json", receipt)
    return output, receipt


def main():
    # Exclusive attempt root is the ownership boundary for all writes and logs.
    require(sys.platform.startswith("linux"), "This prepared entry point executes on Linux server 90 only")
    require(not ROOT.exists() and not ROOT.is_symlink(), f"Refusing existing attempt: {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        event("start", root=str(ROOT), cpu_threads=1, no_new_hashes=True, test_data_policy="no reads/extraction/conversion")
        write_json_new(ROOT / "receipt_started.json", {"status": "STARTED", "script": str(Path(__file__).resolve()),
                       "payload": str(PAYLOAD), "cpu_threads": 1, "no_new_hashes": True})
        require(DRONE_SOURCE.is_dir(), f"missing Drone source {DRONE_SOURCE}")
        require(LLVIP_ZIP.is_file(), f"missing LLVIP ZIP {LLVIP_ZIP}")
        drone = load_module("frozen_drone", "prepare_dronevehicle_yolo.py")
        llvip = load_module("frozen_llvip", "prepare_llvip_yolo.py")
        mapping = load_module("frozen_mapping", "prepare_rgbt_experiment.py")
        cv2 = drone.cv2
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        lists = frozen_llvip_lists()
        event("frozen_lists_accepted", fit=len(lists["fit"]), dev=len(lists["dev"]))
        # Reject an incompatible ZIP before doing the lengthy Drone conversion.
        with zipfile.ZipFile(LLVIP_ZIP) as archive:
            selected_zip_members(archive, lists)
        drone_output, drone_receipt = prepare_drone(drone, cv2)
        llvip_output, llvip_receipt = prepare_llvip(llvip, cv2, lists)
        prepared = {}
        for dataset, source in (("dronevehicle", drone_output), ("llvip", llvip_output)):
            destination = ROOT / "prepared" / dataset
            require(not destination.exists(), "refusing prepared output overwrite")
            prepared[dataset] = mapping.prepare(dataset, source, destination)
            event("mapping_prepared", dataset=dataset, output=str(destination))
        receipt = {"status": "PREPARED", "root": str(ROOT), "elapsed_seconds": time.monotonic() - started,
                   "cpu_threads": 1, "test_accessed": False, "no_new_hashes": True,
                   "drone_receipt": str(ROOT / "drone_receipt.json"), "llvip_receipt": str(ROOT / "llvip_receipt.json"),
                   "prepared": {name: str(ROOT / "prepared" / name) for name in prepared},
                   "models_included": False, "training_started": False}
        write_json_new(ROOT / "receipt_success.json", receipt)
        event("complete", **receipt)
        return 0
    except BaseException as error:
        failure = {"status": "FAILED", "error_type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
                   "output_preserved": str(ROOT), "resume_supported": False}
        write_json_new(ROOT / "receipt_error.json", failure)
        event("failed", error_type=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
