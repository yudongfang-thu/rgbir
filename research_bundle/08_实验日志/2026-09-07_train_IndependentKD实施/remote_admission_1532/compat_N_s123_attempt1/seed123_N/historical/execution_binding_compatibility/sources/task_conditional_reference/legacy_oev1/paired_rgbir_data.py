"""Native-student RGB/IR pairs with independent teacher labels.

The student transform executes once, unchanged. For frozen no-mix/no-HSV
geometry, teacher annotations are independently transformed after replaying
Python/NumPy/Torch CPU RNG, then the student's post-transform RNG is restored.
This module does not edit the core Ultralytics or existing paired loader.
"""
from __future__ import annotations

import copy
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class PairingContractError(RuntimeError):
    pass


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _canonical(path):
    return str(Path(str(path)).resolve())


def _transform_members(transform):
    members = getattr(transform, "transforms", None)
    if members is not None and type(transform).__name__ == "Compose":
        for member in members:
            yield from _transform_members(member)
    else:
        yield transform


def validate_frozen_geometry(transform):
    """Reject transforms whose RNG replay or label semantics are unsupported."""
    for member in _transform_members(transform):
        name = type(member).__name__
        if name in {"Mosaic", "MixUp", "CutMix", "CopyPaste"}:
            if float(getattr(member, "p", -1)) != 0.0:
                raise PairingContractError(f"{name} must be disabled")
        elif name == "Albumentations":
            if getattr(member, "transform", None) is not None:
                raise PairingContractError("Albumentations must be unavailable/disabled; internal RNG is not replayed")
        elif name == "RandomHSV":
            if any(float(getattr(member, key, -1)) != 0.0 for key in ("hgain", "sgain", "vgain")):
                raise PairingContractError("HSV changes must be disabled")
        elif name == "RandomPerspective":
            if any(float(getattr(member, key, -1)) != 0.0 for key in ("degrees", "shear", "perspective")):
                raise PairingContractError("only translate/scale affine augmentation is supported")
        elif name == "RandomFlip":
            if getattr(member, "direction", None) not in {"horizontal", "vertical"}:
                raise PairingContractError("unrecognized flip direction")
        elif name == "Format":
            if any(bool(getattr(member, key, False)) for key in ("return_mask", "return_keypoint", "return_obb")):
                raise PairingContractError("only HBB detection labels are supported")
            if float(getattr(member, "bgr", 0)) != 0.0:
                raise PairingContractError("random BGR permutation must be disabled")
        elif name == "LetterBox":
            pass
        else:
            raise PairingContractError(f"unsupported transform: {name}")


def _name_order(dataset):
    names = getattr(dataset, "data", {}).get("names")
    if names is None:
        return None
    if isinstance(names, Mapping):
        return tuple(str(names[k]) for k in sorted(names, key=lambda x: int(x)))
    return tuple(str(v) for v in names)


class DualLabelRGBIRDataset(Dataset):
    """Wrap a native student dataset and a dedicated teacher-label dataset.

    Args:
        base: Native train YOLODataset. Its transforms/cache/labels are preserved.
        teacher_base: Separately constructed YOLODataset for the matching
            teacher split. Only get_image_and_label is called; its own transform
            is unused. Must have cache=False, rect=False, and not be base.
        strong_by_weak: Explicit image path mapping. Optional only when stems
            identify each teacher file uniquely.
        same_modal: Use the transformed student view and true student GT.
        max_teacher_cache: Hard limit of 1..64 full teacher records per worker.

    Output extra keys:
        strong_img [C,H,W] uint8, strong_cls [Nt,1],
        strong_bboxes [Nt,4] normalized xywh, strong_batch_idx [Nt].
    After collate, strong_batch_idx is each teacher target's image index.
    Teacher targets are independent annotations, not copies of student GT.
    Object correspondence is deliberately left to the declared loss policy.
    """
    def __init__(self, base: Any, teacher_base: Any | None = None,
                 strong_by_weak: Mapping[str, str] | None = None, *,
                 same_modal: bool = False, max_teacher_cache: int = 64):
        self.base = base
        self.teacher_base = teacher_base
        self.same_modal = bool(same_modal)
        self.max_teacher_cache = int(max_teacher_cache)
        if not 1 <= self.max_teacher_cache <= 64:
            raise PairingContractError("teacher cache capacity must be in [1,64]")
        if bool(getattr(base, "rect", False)):
            raise PairingContractError("rect training is outside the frozen contract")
        validate_frozen_geometry(base.transforms)
        self._cache = OrderedDict()
        if self.same_modal:
            self._teacher_index = {}
            self._mapping = {}
            return
        if teacher_base is None or teacher_base is base:
            raise PairingContractError("paired mode requires a separate teacher dataset")
        if getattr(teacher_base, "cache", None) not in (None, False):
            raise PairingContractError("teacher dataset must be constructed with cache=False")
        if bool(getattr(teacher_base, "rect", False)):
            raise PairingContractError("teacher dataset must use rect=False")
        if _name_order(base) != _name_order(teacher_base):
            raise PairingContractError("student/teacher class names or order differ")
        if int(getattr(base, "imgsz", 0)) != int(getattr(teacher_base, "imgsz", -1)):
            raise PairingContractError("student/teacher imgsz differ")
        if any(v is not None for v in getattr(teacher_base, "ims", [])):
            raise PairingContractError("teacher dataset must be fresh, without an existing image cache")
        # Its transform is unused. Disable only this dedicated dataset's native
        # image buffer so our bounded record LRU is the sole teacher cache.
        teacher_base.augment = False
        self._teacher_index = {_canonical(p): i for i, p in enumerate(teacher_base.im_files)}
        if len(self._teacher_index) != len(teacher_base.im_files):
            raise PairingContractError("duplicate teacher image paths")
        if strong_by_weak is None:
            by_stem = {}
            for path in teacher_base.im_files:
                stem = Path(path).stem
                if stem in by_stem:
                    raise PairingContractError("duplicate teacher stems; supply explicit mapping")
                by_stem[stem] = _canonical(path)
            self._mapping = {_canonical(p): by_stem.get(Path(p).stem, "") for p in base.im_files}
        else:
            self._mapping = {_canonical(k): _canonical(v) for k, v in strong_by_weak.items()}
        for weak in base.im_files:
            strong = self._mapping.get(_canonical(weak))
            if strong not in self._teacher_index:
                raise PairingContractError(f"missing valid teacher counterpart for {weak}")

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # Guard incomplete objects during deepcopy/unpickling.
        if name in {"base", "teacher_base", "_cache"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def _teacher_record(self, strong):
        if strong not in self._cache:
            record = self.teacher_base.get_image_and_label(self._teacher_index[strong])
            self._cache[strong] = copy.deepcopy(record)
            while len(self._cache) > self.max_teacher_cache:
                self._cache.popitem(last=False)
        self._cache.move_to_end(strong)
        return copy.deepcopy(self._cache[strong])

    def __getitem__(self, index):
        # This is exactly native YOLODataset.__getitem__: one fresh record then
        # one invocation of base.transforms. Additional work cannot advance RNG.
        weak_record = self.base.get_image_and_label(index)
        source = _canonical(weak_record["im_file"])
        before = _rng_state()
        weak = dict(self.base.transforms(weak_record))
        after = _rng_state()
        if self.same_modal:
            strong_path = source
            strong = weak
        else:
            strong_path = self._mapping[source]
            try:
                strong_record = self._teacher_record(strong_path)
                if tuple(strong_record["ori_shape"]) != tuple(weak.get("ori_shape", ())):
                    raise PairingContractError("paired original image sizes differ")
                _restore_rng(before)
                strong = dict(self.base.transforms(strong_record))
            finally:
                _restore_rng(after)
        if weak["img"].shape != strong["img"].shape:
            raise PairingContractError("paired image shapes differ after replay")
        output = dict(weak)
        output["strong_img"] = strong["img"].clone()
        output["strong_cls"] = strong["cls"].clone()
        output["strong_bboxes"] = strong["bboxes"].clone()
        output["strong_batch_idx"] = strong["batch_idx"].clone()
        output["pair_info"] = {"weak_source": source, "strong_source": strong_path,
                               "independent_teacher_gt": not self.same_modal}
        return output

    def collate_fn(self, batch):
        rows = [dict(row) for row in batch]
        extra = {key: [row.pop(key) for row in rows] for key in
                 ("strong_img", "strong_cls", "strong_bboxes", "strong_batch_idx", "pair_info")}
        output = self.base.collate_fn(rows)
        output["strong_img"] = torch.stack(extra["strong_img"])
        output["strong_cls"] = torch.cat(extra["strong_cls"], dim=0)
        output["strong_bboxes"] = torch.cat(extra["strong_bboxes"], dim=0)
        output["strong_batch_idx"] = torch.cat(
            [torch.full_like(idx, i) for i, idx in enumerate(extra["strong_batch_idx"])], dim=0)
        output["pair_info"] = extra["pair_info"]
        return output

