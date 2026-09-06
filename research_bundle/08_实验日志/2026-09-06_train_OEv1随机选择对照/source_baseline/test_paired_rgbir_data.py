"""CPU contract checks against the pinned real Ultralytics transforms."""
from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.augment import Compose, Format, RandomFlip, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances

from paired_rgbir_data import DualLabelRGBIRDataset, _rng_state, _restore_rng


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class TinyDataset:
    def __init__(self, modality="rgb", count=5):
        self.modality = modality
        self.im_files = [f"/synthetic/{modality}/{i}.png" for i in range(count)]
        self.imgsz = 128
        self.rect = False
        self.cache = None
        self.augment = True
        self.ims = [None] * count
        self.data = {"names": {0: "car", 1: "truck"}}
        self.transforms = Compose([
            RandomPerspective(degrees=0, translate=0.1, scale=0.5, shear=0,
                              perspective=0, size=(128, 128)),
            RandomFlip(p=0.5, direction="horizontal"),
            Format(bbox_format="xywh", normalize=True, batch_idx=True),
        ])
        self.calls = 0

    def __len__(self):
        return len(self.im_files)

    def get_image_and_label(self, index):
        self.calls += 1
        image = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3).copy()
        if self.modality == "ir":
            image = 255 - image
            boxes = np.array([[.5, .5, .20, .20], [.65, .65, .15, .15]], dtype=np.float32)
            classes = np.array([[1], [0]], dtype=np.float32)
        else:
            boxes = np.array([[.5, .5, .20, .20]], dtype=np.float32)
            classes = np.array([[0]], dtype=np.float32)
        return {"im_file": self.im_files[index], "img": image,
                "ori_shape": (128, 128), "resized_shape": (128, 128),
                "ratio_pad": (1.0, 1.0), "cls": classes,
                "instances": Instances(boxes, segments=np.zeros((0, 0, 2), dtype=np.float32),
                                       bbox_format="xywh", normalized=True)}

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))

    @staticmethod
    def collate_fn(batch):
        return YOLODataset.collate_fn(batch)


class PairedContracts(unittest.TestCase):
    def test_native_student_and_rng_equivalence_independent_labels(self):
        for value in range(24):
            native = TinyDataset()
            adapter = DualLabelRGBIRDataset(TinyDataset(), TinyDataset("ir"))
            seed(value)
            expected = native[0]
            expected_draws = (random.random(), float(np.random.random()), torch.rand(4))
            seed(value)
            actual = adapter[0]
            actual_draws = (random.random(), float(np.random.random()), torch.rand(4))
            for key in ("img", "cls", "bboxes", "batch_idx"):
                self.assertTrue(torch.equal(expected[key], actual[key]), f"student mismatch {key}, seed={value}")
            self.assertEqual(expected_draws[:2], actual_draws[:2])
            self.assertTrue(torch.equal(expected_draws[2], actual_draws[2]))
            # Teacher classes/count are its own annotations, not student copies.
            self.assertEqual(actual["cls"].tolist(), [[0.0]])
            self.assertEqual(actual["strong_cls"].tolist(), [[1.0], [0.0]])
            # The shared geometric box is identical after actual affine+flip.
            self.assertTrue(torch.equal(actual["bboxes"][0], actual["strong_bboxes"][0]))
            self.assertFalse(torch.equal(actual["img"], actual["strong_img"]))

    def test_collate_preserves_separate_counts_and_batch_ids(self):
        adapter = DualLabelRGBIRDataset(TinyDataset(), TinyDataset("ir"))
        seed(12)
        batch = adapter.collate_fn([adapter[0], adapter[1]])
        self.assertEqual(batch["cls"].shape[0], 2)
        self.assertEqual(batch["strong_cls"].shape[0], 4)
        self.assertEqual(batch["batch_idx"].tolist(), [0.0, 1.0])
        self.assertEqual(batch["strong_batch_idx"].tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(tuple(batch["strong_img"].shape), (2, 3, 128, 128))

    def test_same_modal_uses_exact_student_view_and_gt(self):
        adapter = DualLabelRGBIRDataset(TinyDataset(), same_modal=True)
        seed(77)
        record = adapter[0]
        for key in ("img", "cls", "bboxes", "batch_idx"):
            self.assertTrue(torch.equal(record[key], record["strong_" + key]))
        self.assertFalse(record["pair_info"]["independent_teacher_gt"])

    def test_teacher_cache_is_bounded_lru(self):
        teacher = TinyDataset("ir")
        adapter = DualLabelRGBIRDataset(TinyDataset(), teacher, max_teacher_cache=3)
        seed(22)
        for index in (0, 1, 2, 0, 3):
            adapter[index]
            self.assertLessEqual(len(adapter._cache), 3)
        self.assertEqual(teacher.calls, 4)
        self.assertEqual([Path(p).stem for p in adapter._cache], ["2", "0", "3"])
        self.assertFalse(teacher.augment)

    def test_independent_crop_drop_keeps_correct_teacher_class(self):
        base = TinyDataset()
        affine = base.transforms.transforms[0]
        affine._compute_affine_matrix = lambda image, size: (
            np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], dtype=np.float32), 1.0)
        adapter = DualLabelRGBIRDataset(base, TinyDataset("ir"))
        seed(3)
        record = adapter[0]
        # Shared middle box survives clipping; teacher's second box is outside.
        self.assertEqual(record["cls"].tolist(), [[0.0]])
        self.assertEqual(record["strong_cls"].tolist(), [[1.0]])
        self.assertTrue(torch.equal(record["bboxes"], record["strong_bboxes"]))

    def test_teacher_failure_restores_student_rng(self):
        native = TinyDataset()
        adapter = DualLabelRGBIRDataset(TinyDataset(), TinyDataset("ir"))
        seed(10)
        native[0]
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        def fail(_):
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("synthetic read failure")
        adapter._teacher_record = fail
        seed(10)
        with self.assertRaisesRegex(RuntimeError, "synthetic read failure"):
            adapter[0]
        actual = (random.random(), float(np.random.random()), torch.rand(2))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PairedContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"status": "passed" if result.wasSuccessful() else "failed",
                      "tests": result.testsRun, "failures": len(result.failures),
                      "errors": len(result.errors), "cuda_used": False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
