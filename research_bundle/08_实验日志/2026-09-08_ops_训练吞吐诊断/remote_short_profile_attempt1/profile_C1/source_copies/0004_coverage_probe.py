"""Natural train-only 64-batch geometry-source coverage, without model/GPU.

The calibration loader is exported here to make the private data seed real.
It does not use the native builder's hard-coded generator. No resampling,
candidate oversampling, batch replacement, AP reading, or registration occurs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

FIXED_SEED = 20260907
FIXED_BATCHES = 64
HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "task_conditional_reference"
if not REFERENCE.is_dir():
    REFERENCE = HERE.parent / "rgbir_task_conditional_v1"


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def seed_natural_worker(worker_id):
    """DataLoader derives the initial seed from its private generator."""
    seed = torch.initial_seed()
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    random.seed(seed % (2 ** 32))
    np.random.seed(seed % (2 ** 32))


class NaturalTraceDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, int(seed)
        self._private_state = (random.Random(seed).getstate(), np.random.RandomState(seed % 2**32).get_state(),
                               torch.Generator().manual_seed(seed).get_state())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        worker = get_worker_info()
        before = _rng_state() if worker is None else None
        if before is not None:
            _restore_rng(self._private_state)
        try:
            row = self.dataset[index]
            # Metadata-only tap: no additional random draw, transform or label edit.
            info = dict(row.get("pair_info", {}))
            info["coverage"] = {"dataset_index": int(index),
                                "worker_id": -1 if worker is None else int(worker.id),
                                "torch_worker_seed": self.seed if worker is None else int(worker.seed),
                                "python_numpy_seed": self.seed % 2**32 if worker is None else int(worker.seed % 2**32)}
            row = dict(row, pair_info=info)
            return row
        finally:
            if before is not None:
                self._private_state = _rng_state()
                _restore_rng(before)

    def collate_fn(self, rows):
        batch = self.dataset.collate_fn(rows)
        if all("strong_" + key in batch for key in ("batch_idx", "cls", "bboxes")):
            batch["teacher_batch"] = {key: batch["strong_" + key] for key in ("batch_idx", "cls", "bboxes")}
        return batch


def make_natural_loader(dataset, batch_size, workers, seed=FIXED_SEED):
    """Exported loader for calibration and this coverage probe, not formal runs.

    A full dataset permutation is sampled without replacement. Worker seeds
    and index permutations use the private generator; the caller's CPU RNG is
    not consumed. For workers=0, transforms run under isolated private states.
    """
    if not isinstance(seed, int) or not isinstance(batch_size, int) or batch_size < 1 or workers < 0:
        raise ValueError("Invalid seed, batch_size or worker count")
    if len(dataset) < batch_size:
        raise ValueError("Calibration must not silently shrink the configured batch")
    generator = torch.Generator().manual_seed(seed)
    traced = NaturalTraceDataset(dataset, seed)
    loader = DataLoader(traced, batch_size=batch_size, shuffle=True, num_workers=int(workers),
                        collate_fn=traced.collate_fn, worker_init_fn=seed_natural_worker,
                        generator=generator, drop_last=False, pin_memory=False,
                        **({"prefetch_factor": 4} if workers else {}))
    loader.coverage_metadata = {
        "seed": seed, "generator_seed": seed, "worker_init": "seed_natural_worker",
        "shuffle": True, "replacement": False, "drop_last": False, "prefetch_factor": 4 if workers else None,
        "workers": int(workers), "batch_size": batch_size, "dataset_size": len(dataset),
        "main_rng_isolated_for_workers0": True, "loader_class": "torch.utils.data.DataLoader",
        "use": "fixed calibration/probe only; formal training retains matched legacy stream",
    }
    return loader


def build_natural_loader(cfg, seed=FIXED_SEED):
    """Build genuine paired train images/augmentations from the frozen recipe.

    Returns only the loader. img/strong_img are CPU uint8; teacher_batch labels
    are normalized and CPU. Caller moves/preprocesses exactly once. No model,
    validation loader, evaluator or CUDA context is created.
    """
    import yaml
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    for path in (REFERENCE, REFERENCE / "legacy_oev1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from tracked_pair_data import TrackedDualLabelRGBIRDataset
    if cfg.get("multi_scale", False) or cfg.get("rect", False) or cfg.get("fraction", 1.0) != 1.0:
        raise ValueError("Full natural train stream requires multi_scale=false, rect=false, fraction=1")
    overrides = {key: cfg[key] for key in ("imgsz", "batch", "nbs", "workers")}
    overrides.update(cfg["augmentation"])
    overrides.update(task="detect", mode="train", cache=False, rect=False, seed=int(seed), fraction=1.0)
    args = get_cfg(overrides=overrides)
    before = _rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        datasets, yaml_sources, manifests = [], [], []
        for key in ("student_data_yaml", "privileged_data_yaml"):
            source = Path(cfg["paths"][key])
            manifest = yaml.safe_load(source.read_text(encoding="utf-8"))
            if "test" in manifest:
                raise ValueError("Train/dev-only YAML required; sealed test reference found")
            data = check_det_dataset(str(source), autodownload=False)
            datasets.append(build_yolo_dataset(args, data["train"], cfg["batch"], data, mode="train", rect=False, stride=32))
            yaml_sources.append(str(source))
            manifests.append({"train": data["train"], "names": data["names"]})
        mapping_source = Path(cfg["paths"]["paired_train_mapping"])
        mapping = json.loads(mapping_source.read_text(encoding="utf-8"))
        paired = TrackedDualLabelRGBIRDataset(datasets[0], datasets[1], mapping,
                    max_teacher_cache=int(cfg["teacher_cache_images"]))
        loader = make_natural_loader(paired, int(cfg["batch"]), int(cfg["workers"]), seed)
        loader.coverage_metadata.update(dataset=cfg["dataset"], yaml_sources=yaml_sources,
            paired_mapping=str(mapping_source), train_manifests=manifests,
            augmentation=copy.deepcopy(cfg["augmentation"]), test_accessed=False,
            model_loaded=False, geometry_reference_directory=str(REFERENCE))
        return loader
    finally:
        _restore_rng(before)


def _path_key(path):
    return str(Path(str(path)).absolute().resolve())


def load_roster(path, dataset):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = document["roster"] if isinstance(document, dict) else document
    aliases = {"drone": "dronevehicle", "dronevehicle": "dronevehicle", "llvip": "llvip"}
    expected = aliases.get(str(dataset).lower(), str(dataset).lower())
    selected = [row for row in rows if aliases.get(str(row["dataset"]).lower(), str(row["dataset"]).lower()) == expected]
    if not selected:
        raise ValueError("Roster contains no rows for configured dataset")
    by_path = {}
    for row in selected:
        # Resolve real local server symlinks; never use basename-only matching
        # to authorize a source. Aliases are explicitly supplied by the roster.
        for name in ("rgb_path", "rgb_source"):
            if row.get(name):
                by_path.setdefault(_path_key(row[name]), []).append(row)
    return by_path, len(selected)


def summarize_batch(batch, batch_index, roster_paths):
    paths = batch["im_file"]
    infos = batch["pair_info"]
    if len(paths) != len(infos):
        raise ValueError("Image/augmentation metadata count mismatch")
    rows = []
    for image_index, (path, info) in enumerate(zip(paths, infos)):
        source = info.get("weak_source", path)
        keys = {_path_key(path), _path_key(source)}
        matched = [entry for key in keys for entry in roster_paths.get(key, [])]
        rgb_ids = batch["batch_idx"].reshape(-1).long() == image_index
        ir_ids = batch["teacher_batch"]["batch_idx"].reshape(-1).long() == image_index
        rows.append({"batch": batch_index, "position": image_index,
            "image": str(path), "rgb_source": str(source), "ir_source": info.get("strong_source"),
            "roster_hit": bool(matched), "roster_audit_ids": sorted({str(e.get("audit_id", "")) for e in matched}),
            "source_groups": sorted({str(e.get("group", "")) for e in matched}),
            "rgb_gt_count": int(rgb_ids.sum()), "ir_gt_count": int(ir_ids.sum()),
            "rgb_classes": batch["cls"][rgb_ids].reshape(-1).tolist(),
            "rgb_boxes_normalized_xywh": batch["bboxes"][rgb_ids].tolist(),
            "ir_classes": batch["teacher_batch"]["cls"][ir_ids].reshape(-1).tolist(),
            "ir_boxes_normalized_xywh": batch["teacher_batch"]["bboxes"][ir_ids].tolist(),
            "pair_info": info,
            "student_tensor_shape": list(batch["img"][image_index].shape),
            "teacher_tensor_shape": list(batch["strong_img"][image_index].shape),
            "student_dtype": str(batch["img"].dtype), "teacher_dtype": str(batch["strong_img"].dtype)})
    return rows


def run_coverage(loader, roster_paths, output, batches=FIXED_BATCHES):
    """Consume exactly the first natural batches; hits are only a hard upper bound."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    hits, images, source_ids, gt_hits, actual = [], set(), set(), [], 0
    iterator = iter(loader)
    try:
        with (output / "natural_batches.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    raise RuntimeError("Fewer than 64 natural batches; no repetition or substitution allowed")
                if len(batch["im_file"]) != loader.coverage_metadata["batch_size"]:
                    raise RuntimeError("Short batch in fixed calibration window")
                rows = summarize_batch(batch, index, roster_paths)
                hit = any(row["roster_hit"] for row in rows)
                hit_with_gt = any(row["roster_hit"] and row["rgb_gt_count"] and row["ir_gt_count"] for row in rows)
                hits.append(hit)
                gt_hits.append(hit_with_gt)
                for row in rows:
                    images.add(_path_key(row["rgb_source"]))
                    if row["roster_hit"]:
                        source_ids.add(_path_key(row["rgb_source"]))
                stream.write(json.dumps({"batch": index, "images": rows}, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                actual += 1
                print(json.dumps({"batch_completed": actual, "roster_hit_batches_so_far": sum(hits)}), flush=True)
                del batch, rows
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown:
            shutdown()
    receipt = dict(loader.coverage_metadata, status="COMPLETED", requested_batches=batches, actual_batches=actual,
        unique_source_images=len(images), unique_roster_source_images=len(source_ids),
        roster_hit_batch_indices=[i for i, value in enumerate(hits) if value],
        roster_hit_batch_upper_bound=sum(hits), roster_both_gt_batch_upper_bound=sum(gt_hits),
        signal_requirement=16, coverage_gate="ROSTER_CANNOT_REACH_16" if sum(gt_hits) < 16 else "GEOMETRY_AND_SIGNAL_STILL_REQUIRED",
        upper_bound_only=True, geometry_verified=False, actual_L_selected_not_measured=True,
        no_oversampling=True, no_batch_replacement=True, no_hashes_computed=True,
        seconds=time.time() - start)
    (output / "coverage_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # CLI only; importing this module from a CUDA calibrator must not hide GPU.
    if torch.cuda.is_initialized():
        raise RuntimeError("Coverage CLI must start without a CUDA context")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    roster, rows = load_roster(args.roster, cfg["dataset"])
    loader = build_natural_loader(cfg, FIXED_SEED)
    loader.coverage_metadata.update(roster_path=str(args.roster), roster_row_count=rows,
                                    roster_unique_resolved_paths=len(roster), config_path=str(args.config))
    print(json.dumps(run_coverage(loader, roster, args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
