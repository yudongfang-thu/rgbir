"""Paired data unchanged, plus a train-only wrong-IR view with exact RNG replay.

The parent loader executes its original RGB/paired-IR path. An entry observer
captures the student's actual pre-transform Python/NumPy/Torch-CPU state,
after get_image_and_label. The donor uses this same transform/state and then
restores the paired path's post-state. No donor labels are exposed to any KD
selector. Supported transforms are CPU-only and already validated by the
vendored parent; this loader never initializes or consumes CUDA RNG.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "legacy_oev1"))
sys.path.insert(0, str(HERE))
from paired_rgbir_data import PairingContractError, _canonical, _rng_state, _restore_rng
from tracked_pair_data import TrackedDualLabelRGBIRDataset

DONOR_SEED = 20260907
DONOR_PROTOCOL = "train_only_shape_stratified_single_cycle_derangement_v1"


def _train_path(path):
    for candidate in (Path(path), Path(path).resolve()):
        roles = {part.lower() for part in candidate.parts}
        if roles & {"test", "testing", "val", "validation", "dev", "dev_hold", "test2017"}:
            raise PairingContractError("Wrong-image donors must be train-only: " + str(path))
        if not roles & {"train", "fit"}:
            raise PairingContractError("A declared train/fit path is required: " + str(path))


def build_derangement(paired_mapping, shapes, seed=DONOR_SEED):
    """Pure deterministic mapping builder; shapes are donor original [H,W].

    A private NumPy generator picks a single cycle in each shape group. Every
    paired teacher appears once as a donor and no image donates to itself.
    Singleton shape groups fail explicitly rather than dropping examples.
    """
    if int(seed) != DONOR_SEED:
        raise PairingContractError("All training seeds share frozen donor seed 20260907")
    paired = {_canonical(k): _canonical(v) for k, v in paired_mapping.items()}
    if len(paired) != len(paired_mapping) or len(set(paired.values())) != len(paired):
        raise PairingContractError("Derangement requires one-to-one canonical paired paths")
    shape_map = {_canonical(k): tuple(v) for k, v in shapes.items()}
    groups = defaultdict(list)
    for weak, strong in sorted(paired.items()):
        _train_path(weak)
        _train_path(strong)
        if strong not in shape_map or len(shape_map[strong]) != 2:
            raise PairingContractError("Missing original donor image shape")
        groups[shape_map[strong]].append((weak, strong))
    generator = np.random.RandomState(DONOR_SEED)
    mapping, group_records = {}, []
    for shape in sorted(groups):
        rows = groups[shape]
        if len(rows) < 2:
            raise PairingContractError("Singleton donor shape group cannot be deranged: " + str(shape))
        order = generator.permutation(len(rows)).tolist()
        for position, current in enumerate(order):
            following = order[(position + 1) % len(order)]
            mapping[rows[current][0]] = rows[following][1]
        group_records.append({"original_shape": list(shape), "count": len(rows)})
    return {"protocol": DONOR_PROTOCOL, "split": "train", "donor_seed": DONOR_SEED,
            "training_seeds": [0, 42, 123], "same_mapping_for_all_training_seeds": True,
            "shape_stratified": True, "self_pairs": 0, "count": len(mapping),
            "shape_groups": group_records, "paired_mapping": paired, "mapping": mapping,
            "official_test_accessed": False}


def freeze_derangement(paired_mapping_path, output):
    """Read train image headers, then save one immutable manifest/TSV roster."""
    from PIL import Image
    paired_mapping_path, output = Path(paired_mapping_path), Path(output)
    paired = json.loads(paired_mapping_path.read_text())
    if not isinstance(paired, dict) or not all(isinstance(v, str) for v in paired.values()):
        raise PairingContractError("Input must be the actual RGB-to-paired-IR mapping")
    shapes = {}
    for weak, strong in paired.items():
        _train_path(weak)
        _train_path(strong)
        with Image.open(weak) as image:
            rgb_shape = (image.height, image.width)
        with Image.open(strong) as image:
            ir_shape = (image.height, image.width)
        if rgb_shape != ir_shape:
            raise PairingContractError("Paired original shapes already differ: " + weak)
        shapes[strong] = ir_shape
    manifest = build_derangement(paired, shapes)
    manifest["paired_mapping_source"] = str(paired_mapping_path)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "derangement.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "roster.tsv").open("w", encoding="utf-8") as handle:
        handle.write("rgb_source\tpaired_ir_source\twrong_ir_source\n")
        for weak in sorted(manifest["mapping"]):
            handle.write("\t".join((weak, manifest["paired_mapping"][weak], manifest["mapping"][weak])) + "\n")
    return manifest


class _EntryObserver:
    """Transparent call-through; observing RNG does not draw a random number."""
    def __init__(self, transform, source):
        self.transform, self.source, self.before = transform, source, None

    def __call__(self, labels):
        if self.before is None and _canonical(labels["im_file"]) == self.source:
            self.before = _rng_state()
        return self.transform(labels)


class ShuffledContentRGBIRDataset(TrackedDualLabelRGBIRDataset):
    """Keep paired output keys intact; append content_img/source/info only."""
    def __init__(self, base, teacher_base=None, strong_by_weak=None, *,
                 donor_roster=None, donor_by_weak=None, same_modal=False, max_teacher_cache=64):
        if same_modal:
            raise PairingContractError("Wrong-IR content loader is not a same-modal loader")
        super().__init__(base, teacher_base, strong_by_weak, same_modal=False,
                         max_teacher_cache=max_teacher_cache)
        if donor_roster is not None and donor_by_weak is not None:
            raise PairingContractError("Supply a frozen donor roster or its mapping, not both")
        if donor_roster is not None:
            manifest = json.loads(Path(donor_roster).read_text())
            if manifest.get("protocol") != DONOR_PROTOCOL or manifest.get("split") != "train" or manifest.get("donor_seed") != DONOR_SEED:
                raise PairingContractError("Wrong donor manifest protocol/split/seed")
            donor_by_weak = manifest["mapping"]
            if {_canonical(k): _canonical(v) for k, v in manifest["paired_mapping"].items()} != self._mapping:
                raise PairingContractError("Frozen donor roster has a different paired dataset")
        if not isinstance(donor_by_weak, Mapping):
            raise PairingContractError("An explicit frozen train-only donor mapping is required")
        self._donor_mapping = {_canonical(k): _canonical(v) for k, v in donor_by_weak.items()}
        weak_set = {_canonical(p) for p in self.base.im_files}
        paired_set = {self._mapping[p] for p in weak_set}
        if set(self._donor_mapping) != weak_set or set(self._donor_mapping.values()) != paired_set or len(set(self._donor_mapping.values())) != len(weak_set):
            raise PairingContractError("Donor mapping must permute precisely the paired train population")
        for weak, donor in self._donor_mapping.items():
            _train_path(weak)
            _train_path(donor)
            if donor == self._mapping[weak]:
                raise PairingContractError("Self-paired donor is prohibited")

    def __getitem__(self, index):
        source = _canonical(self.base.im_files[index])
        original_transform = self.base.transforms
        observer = _EntryObserver(original_transform, source)
        self.base.transforms = observer
        try:
            # Parent RGB and paired-IR implementation is called unchanged.
            output = super().__getitem__(index)
        finally:
            self.base.transforms = original_transform
        if observer.before is None:
            raise PairingContractError("Could not observe the actual student transform entry")
        after = _rng_state()
        paired_trace = copy.deepcopy(self.geometry_recorder)
        donor_path = self._donor_mapping[source]
        try:
            donor = self._teacher_record(donor_path)
            if tuple(donor["ori_shape"]) != tuple(output["ori_shape"]):
                raise PairingContractError("Wrong-IR original image size differs; no implicit transform substitution")
            # A standalone tap record must start at original -> resized once.
            self.geometry_recorder.pop(donor_path, None)
            _restore_rng(observer.before)
            content = dict(original_transform(donor))
            if content["img"].shape != output["strong_img"].shape:
                raise PairingContractError("Wrong-IR augmented shape differs")
            trace = self.geometry_recorder.get(donor_path)
            if trace is None or not np.allclose(trace["matrix"], output["pair_info"]["ir_matrix"], atol=1e-9, rtol=0):
                raise PairingContractError("Wrong-IR transform matrix differs from paired IR")
            if content["img"].device.type != "cpu":
                raise PairingContractError("Only CPU dataset transforms are supported")
            output["content_img"] = content["img"].clone()
            output["content_source"] = donor_path
            output["content_info"] = {"wrong_source": donor_path,
                "paired_source": output["pair_info"]["strong_source"],
                "original_shape": list(donor["ori_shape"]),
                "matrix": trace["matrix"].tolist(), "geometry_trace": copy.deepcopy(trace["events"]),
                "donor_seed": DONOR_SEED, "donor_labels_exposed": False,
                "rng_replay": "python_numpy_torch_cpu_from_actual_student_transform_entry"}
        finally:
            _restore_rng(after)
            # ParameterTap holds this dictionary by reference.
            self.geometry_recorder.clear()
            self.geometry_recorder.update(paired_trace)
        return output

    def collate_fn(self, batch):
        rows = [dict(row) for row in batch]
        content = [row.pop("content_img") for row in rows]
        sources = [row.pop("content_source") for row in rows]
        metadata = [row.pop("content_info") for row in rows]
        output = super().collate_fn(rows)
        output["content_img"] = torch.stack(content)
        output["content_source"] = sources
        output["content_info"] = metadata
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_derangement(args.paired_mapping, args.output)
    print(json.dumps({"status": "frozen", "count": manifest["count"], "donor_seed": DONOR_SEED,
                      "shape_groups": manifest["shape_groups"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
