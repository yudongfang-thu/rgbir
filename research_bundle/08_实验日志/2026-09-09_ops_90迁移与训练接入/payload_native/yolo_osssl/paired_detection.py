"""Paired SAR/EO loading with synchronized geometric augmentation."""

from __future__ import annotations

import copy
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedDetectionError(RuntimeError):
    pass


def _rng_state() -> tuple[Any, Any, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _restore_rng(state: tuple[Any, Any, torch.Tensor]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _tensor(record: Mapping[str, Any], key: str) -> torch.Tensor:
    value = record.get(key)
    if not isinstance(value, torch.Tensor):
        raise PairedDetectionError(f"transformed record lacks tensor {key}")
    return value


def _same_targets(sar: Mapping[str, Any], eo: Mapping[str, Any]) -> bool:
    return all(torch.equal(_tensor(sar, key), _tensor(eo, key)) for key in ("cls", "bboxes", "batch_idx"))


def _iter_mix_transforms(pipeline: Any) -> list[Any]:
    """Ultralytics mix transforms (Mosaic/MixUp/CutMix) inside a pipeline.

    Identified by their ``dataset`` + ``get_indexes`` contract so the rebind
    stays testable without importing ultralytics.
    """

    members = getattr(pipeline, "transforms", None)
    if members is None:
        members = [pipeline]
    return [t for t in members if hasattr(t, "dataset") and hasattr(t, "get_indexes")]


def _rebind_mix_dataset(pipeline: Any, view: Any) -> None:
    for mix in _iter_mix_transforms(pipeline):
        mix.dataset = view


def _with_mix_dataset(transform: Any, view: Any, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` with mix transforms sampling tiles from ``view``, then restore."""

    mixes = _iter_mix_transforms(transform)
    if view is None or not mixes:
        return fn()
    originals = [(mix, mix.dataset) for mix in mixes]
    try:
        for mix, _ in originals:
            mix.dataset = view
        return fn()
    finally:
        for mix, original in originals:
            mix.dataset = original


class _PairedMixView:
    """Serves modality-counterpart records for mix-transform tile sampling.

    Mosaic/MixUp pull extra tiles through ``dataset.get_image_and_label``;
    the counterpart replay must sample counterpart images of the same
    indices, not the base-modality images the pipeline object points at.
    """

    def __init__(self, paired: "PairedDetectionDataset", mapping: Mapping[str, str] | None):
        self.paired = paired
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.paired.base)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        paired = self.paired
        record = dict(paired.base.get_image_and_label(index))
        source = str(Path(str(record["im_file"])).resolve())
        counterpart = paired.eo_by_sar.get(source) if self.mapping is None else self.mapping.get(source)
        if counterpart is None:
            raise PairedDetectionError(f"no counterpart image for {source}")
        image, original, resized = paired.cached_image(counterpart)
        record.update({"im_file": counterpart, "img": image, "ori_shape": original, "resized_shape": resized})
        record["ratio_pad"] = (resized[0] / original[0], resized[1] / original[1])
        return record


def transform_paired_record(
    sar_record: Mapping[str, Any],
    eo_record: Mapping[str, Any],
    transform: Callable[[dict[str, Any]], Mapping[str, Any]],
    donor_record: Mapping[str, Any] | None = None,
    paired_view: Any = None,
    donor_view: Any = None,
):
    before = _rng_state()
    sar = dict(transform(copy.deepcopy(dict(sar_record))))
    after = _rng_state()
    _restore_rng(before)
    eo = _with_mix_dataset(
        transform, paired_view, lambda: dict(transform(copy.deepcopy(dict(eo_record))))
    )
    donor = None
    if donor_record is not None:
        _restore_rng(before)
        donor = _with_mix_dataset(
            transform, donor_view, lambda: dict(transform(copy.deepcopy(dict(donor_record))))
        )
    _restore_rng(after)
    if not _same_targets(sar, eo):
        raise PairedDetectionError("SAR and EO augmentations produced different targets")
    eo_image = _tensor(eo, "img")
    if eo_image.shape != _tensor(sar, "img").shape:
        raise PairedDetectionError("SAR and EO image shapes differ after augmentation")
    pair_info = {"sar_source": str(sar_record.get("im_file", "")), "eo_source": str(eo_record.get("im_file", ""))}
    if donor is None:
        return sar, eo_image, pair_info
    if not _same_targets(sar, donor):
        raise PairedDetectionError("shuffled EO augmentation produced different targets")
    donor_image = _tensor(donor, "img")
    pair_info["shuffled_eo_source"] = str(donor_record.get("im_file", ""))
    return sar, eo_image, donor_image, pair_info


def _load_image(base: Any, path: str) -> tuple[Any, tuple[int, int], tuple[int, int]]:
    from ultralytics.utils.patches import imread

    image = imread(path, flags=getattr(base, "cv2_flag", 1))
    if image is None:
        raise FileNotFoundError(f"paired image is missing: {path}")
    original = tuple(int(value) for value in image.shape[:2])
    size = int(getattr(base, "imgsz"))
    ratio = size / max(original)
    if ratio != 1:
        import cv2

        width = min(math.ceil(original[1] * ratio), size)
        height = min(math.ceil(original[0] * ratio), size)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        image = image[..., None]
    return image, original, tuple(int(value) for value in image.shape[:2])


class PairedDetectionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        base: Any,
        eo_by_sar: Mapping[str, str],
        shuffled_eo_by_sar: Mapping[str, str] | None = None,
        group_by_sar: Mapping[str, str] | None = None,
    ) -> None:
        self.base = base
        self.eo_by_sar = {str(Path(key).resolve()): str(Path(value).resolve()) for key, value in eo_by_sar.items()}
        self.shuffled_eo_by_sar = {
            str(Path(key).resolve()): str(Path(value).resolve()) for key, value in (shuffled_eo_by_sar or {}).items()
        }
        self.group_by_sar = {str(Path(key).resolve()): str(value) for key, value in (group_by_sar or {}).items()}
        self._cache: OrderedDict[str, tuple[Any, tuple[int, int], tuple[int, int]]] = OrderedDict()
        self._eo_mix_view = _PairedMixView(self, None)
        self._donor_mix_view = (
            _PairedMixView(self, self.shuffled_eo_by_sar) if self.shuffled_eo_by_sar else None
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def cached_image(self, path: str) -> tuple[Any, tuple[int, int], tuple[int, int]]:
        if path not in self._cache:
            self._cache[path] = _load_image(self.base, path)
        return self._cache[path]

    def _image_record(self, sar_record: Mapping[str, Any], path: str) -> dict[str, Any]:
        record = copy.deepcopy(dict(sar_record))
        if path not in self._cache:
            self._cache[path] = _load_image(self.base, path)
        image, original, resized = self._cache[path]
        if tuple(sar_record.get("ori_shape", ())) != original or tuple(sar_record.get("resized_shape", ())) != resized:
            raise PairedDetectionError("paired SAR and EO source sizes differ")
        record.update({"im_file": path, "img": image.copy(), "ori_shape": original, "resized_shape": resized})
        return record

    def __getitem__(self, index: int) -> dict[str, Any]:
        sar_record = dict(self.base.get_image_and_label(index))
        source = str(Path(str(sar_record["im_file"])).resolve())
        eo_path = self.eo_by_sar.get(source)
        if eo_path is None:
            raise PairedDetectionError(f"no EO pair for {source}")
        eo_record = self._image_record(sar_record, eo_path)
        donor_path = self.shuffled_eo_by_sar.get(source)
        donor_record = self._image_record(sar_record, donor_path) if donor_path else None
        transformed = transform_paired_record(
            sar_record,
            eo_record,
            self.base.transforms,
            donor_record,
            paired_view=self._eo_mix_view,
            donor_view=self._donor_mix_view,
        )
        if donor_record is None:
            sar, eo_image, pair_info = transformed
        else:
            sar, eo_image, donor_image, pair_info = transformed
            sar["shuffled_eo_img"] = donor_image
        if source in self.group_by_sar:
            pair_info["group_id"] = self.group_by_sar[source]
        sar["eo_img"] = eo_image
        sar["pair_info"] = pair_info
        return sar

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        eo_images = [row.pop("eo_img") for row in batch]
        pair_info = [row.pop("pair_info") for row in batch]
        has_donor = all("shuffled_eo_img" in row for row in batch)
        donors = [row.pop("shuffled_eo_img") for row in batch] if has_donor else None
        output = self.base.collate_fn(batch)
        output["eo_img"] = torch.stack(eo_images)
        output["pair_info"] = pair_info
        if donors is not None:
            output["shuffled_eo_img"] = torch.stack(donors)
        return output


class GroupDetectionDataset(Dataset[dict[str, Any]]):
    """Attach fit-group identity without loading an EO counterpart."""

    def __init__(self, base: Any, group_by_sar: Mapping[str, str]) -> None:
        self.base = base
        self.group_by_sar = {str(Path(key).resolve()): str(value) for key, value in group_by_sar.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.base[index])
        source = str(Path(str(record["im_file"])).resolve())
        try:
            group_id = self.group_by_sar[source]
        except KeyError as exc:
            raise PairedDetectionError(f"no fit group for {source}") from exc
        record["pair_info"] = {"sar_source": source, "group_id": group_id}
        return record

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pair_info = [row.pop("pair_info") for row in batch]
        output = self.base.collate_fn(batch)
        output["pair_info"] = pair_info
        return output
