"""CPU data-flow, donor-policy and optional pinned native-transform checks."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "legacy_oev1"))
from paired_rgbir_data import PairingContractError, _canonical
from tracked_pair_data import TrackedDualLabelRGBIRDataset
from shuffled_pair_data import (ShuffledContentRGBIRDataset, DONOR_SEED,
                                build_derangement, freeze_derangement)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def draws():
    return random.random(), float(np.random.random()), torch.rand(4)


class RandomFlip:
    direction = "horizontal"
    def get_params(self, row):
        # Exercise all CPU RNG streams used by supported data transforms.
        flip = random.random() < .5
        np.random.random()
        torch.rand(1)
        return {"flip": flip, "direction": self.direction}

    def __call__(self, row):
        params = self.get_params(row)
        if params["flip"]:
            row["img"] = np.ascontiguousarray(row["img"][:, ::-1])
            row["bboxes"][:, 0] = 1 - row["bboxes"][:, 0]
        return row


class Format:
    def __call__(self, row):
        row["img"] = torch.from_numpy(np.ascontiguousarray(row["img"].transpose(2, 0, 1)))
        row["cls"] = torch.as_tensor(row["cls"]).clone()
        row["bboxes"] = torch.as_tensor(row["bboxes"]).clone()
        row["batch_idx"] = torch.zeros(len(row["cls"]))
        return row


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, row):
        for transform in self.transforms:
            row = transform(row)
        return row


class TinyDataset:
    def __init__(self, modality="rgb", count=4):
        self.modality = modality
        self.im_files = ["/synthetic/{}/images/train/{}.png".format(modality, i) for i in range(count)]
        self.imgsz, self.rect, self.cache, self.augment = 32, False, None, True
        self.ims = [None] * count
        self.data = {"names": {0: "car", 1: "truck"}}
        self.transforms = Compose([RandomFlip(), Format()])
        self.calls = 0

    def __len__(self):
        return len(self.im_files)

    def get_image_and_label(self, index):
        self.calls += 1
        # Deliberately consume RNG before transform entry. Capturing before
        # super().__getitem__ instead would produce incorrect donor geometry.
        draws()
        image = ((np.arange(32*32*3).reshape(32, 32, 3) + index*17) % 256).astype(np.uint8)
        if self.modality == "ir":
            image = 255 - image
        return {"im_file": self.im_files[index], "ori_shape": (32, 32), "img": image,
                "cls": np.array([[0], [1]] if self.modality == "ir" else [[0]], dtype=np.float32),
                "bboxes": np.array([[.4, .4, .2, .2], [.65, .6, .2, .2]] if self.modality == "ir" else [[.4, .4, .2, .2]], dtype=np.float32)}

    @staticmethod
    def collate_fn(rows):
        output = {"img": torch.stack([r["img"] for r in rows]),
                  "im_file": [r["im_file"] for r in rows]}
        for key in ("cls", "bboxes"):
            output[key] = torch.cat([r[key] for r in rows])
        output["batch_idx"] = torch.cat([torch.full_like(r["batch_idx"], i) for i, r in enumerate(rows)])
        return output


def mapping_for(base, teacher):
    paired = dict(zip(base.im_files, teacher.im_files))
    shapes = {path: (32, 32) for path in teacher.im_files}
    return paired, build_derangement(paired, shapes)["mapping"]


def make_pair(factory=TinyDataset, shuffled=False, cache=64):
    base, teacher = factory(), factory("ir")
    paired, donor = mapping_for(base, teacher)
    if shuffled:
        return ShuffledContentRGBIRDataset(base, teacher, paired, donor_by_weak=donor, max_teacher_cache=cache)
    return TrackedDualLabelRGBIRDataset(base, teacher, paired, max_teacher_cache=cache)


class ShuffledTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_draws_equal(self, a, b):
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))

    def test_original_pair_output_and_rng_exact_across_seeds(self):
        for value in range(12):
            paired, shuffled = make_pair(), make_pair(shuffled=True)
            seed(value)
            expected = paired[0]
            expected_draws = draws()
            seed(value)
            actual = shuffled[0]
            actual_draws = draws()
            for key in ("img", "cls", "bboxes", "batch_idx", "strong_img", "strong_cls", "strong_bboxes", "strong_batch_idx"):
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
            self.assertEqual(expected["pair_info"], actual["pair_info"])
            self.assert_draws_equal(expected_draws, actual_draws)
            self.assertNotEqual(actual["content_source"], actual["pair_info"]["strong_source"])
            self.assertFalse(torch.equal(actual["content_img"], actual["strong_img"]))
            np.testing.assert_array_equal(actual["content_info"]["matrix"], actual["pair_info"]["ir_matrix"])
            self.assertFalse(any(key in actual for key in ("content_cls", "content_bboxes", "content_batch_idx")))

    def test_collate_preserves_independent_gt_and_adds_only_content(self):
        dataset = make_pair(shuffled=True)
        seed(42)
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["content_img"].shape), (2, 3, 32, 32))
        self.assertEqual(batch["strong_cls"].shape[0], 4)
        self.assertEqual(batch["strong_batch_idx"].tolist(), [0., 0., 1., 1.])
        self.assertEqual(len(batch["content_source"]), 2)
        self.assertEqual(batch["cls"].shape[0], 2)

    def test_wrong_read_failure_restores_paired_rng_and_trace(self):
        paired, shuffled = make_pair(), make_pair(shuffled=True)
        seed(5)
        paired[0]
        expected = draws()
        original = shuffled._teacher_record
        paired_source = shuffled._mapping[_canonical(shuffled.base.im_files[0])]
        def fail_wrong(source):
            if source != paired_source:
                draws()
                raise RuntimeError("donor read failed")
            return original(source)
        shuffled._teacher_record = fail_wrong
        seed(5)
        with self.assertRaisesRegex(RuntimeError, "donor read failed"):
            shuffled[0]
        self.assert_draws_equal(expected, draws())
        self.assertEqual(set(shuffled.geometry_recorder), {_canonical(shuffled.base.im_files[0]), paired_source})

    def test_size_mismatch_rejected_and_pair_cache_stays_bounded(self):
        dataset = make_pair(shuffled=True, cache=2)
        for index in range(4):
            dataset[index]
            self.assertLessEqual(len(dataset._cache), 2)
        original = dataset._teacher_record
        wrong_source = dataset._donor_mapping[_canonical(dataset.base.im_files[0])]
        def bad_shape(source):
            row = original(source)
            if source == wrong_source:
                row["ori_shape"] = (16, 32)
            return row
        dataset._teacher_record = bad_shape
        with self.assertRaisesRegex(PairingContractError, "original image size differs"):
            dataset[0]

    def test_derangement_fixed_bijective_shape_preserving_no_rng(self):
        base, teacher = TinyDataset(count=8), TinyDataset("ir", count=8)
        paired = dict(zip(base.im_files, teacher.im_files))
        shapes = {path: (32, 32) if i < 4 else (64, 64) for i, path in enumerate(teacher.im_files)}
        seed(123)
        a = build_derangement(paired, shapes)
        actual_draws = draws()
        seed(123)
        expected_draws = draws()
        b = build_derangement(paired, shapes)
        self.assertEqual(a, b)
        self.assert_draws_equal(actual_draws, expected_draws)
        self.assertEqual(set(a["mapping"].values()), set(map(_canonical, paired.values())))
        for weak, donor in a["mapping"].items():
            original = a["paired_mapping"][weak]
            self.assertNotEqual(original, donor)
            self.assertEqual(shapes[next(p for p in shapes if _canonical(p) == original)], shapes[next(p for p in shapes if _canonical(p) == donor)])
        with self.assertRaises(PairingContractError):
            build_derangement(paired, shapes, seed=0)
        shapes[teacher.im_files[0]] = (17, 17)
        with self.assertRaisesRegex(PairingContractError, "Singleton"):
            build_derangement(paired, shapes)

    def test_self_pair_and_nontraining_mapping_rejected(self):
        base, teacher = TinyDataset(), TinyDataset("ir")
        paired, _ = mapping_for(base, teacher)
        with self.assertRaisesRegex(PairingContractError, "Self-paired"):
            ShuffledContentRGBIRDataset(base, teacher, paired, donor_by_weak=paired)
        invalid = {"/data/rgb/images/train/0.png": "/data/ir/images/val/0.png",
                   "/data/rgb/images/train/1.png": "/data/ir/images/val/1.png"}
        with self.assertRaisesRegex(PairingContractError, "train-only"):
            build_derangement(invalid, {v: (32, 32) for v in invalid.values()})

    def test_freeze_tool_reads_headers_and_preserves_existing_output(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paired = {}
            for i in range(4):
                rgb, ir = root/"rgb/images/train"/(str(i)+".png"), root/"ir/images/train"/(str(i)+".png")
                rgb.parent.mkdir(parents=True, exist_ok=True)
                ir.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32)).save(rgb)
                Image.new("RGB", (32, 32)).save(ir)
                paired[str(rgb)] = str(ir)
            source = root/"paired.json"
            source.write_text(json.dumps(paired))
            result = freeze_derangement(source, root/"frozen")
            self.assertEqual(result["count"], 4)
            self.assertTrue((root/"frozen/roster.tsv").is_file())
            with self.assertRaises(FileExistsError):
                freeze_derangement(source, root/"frozen")


@unittest.skipUnless(importlib.util.find_spec("ultralytics") is not None, "Pinned Ultralytics unavailable locally; run these CPU checks on 94")
class NativeTransformChecks(unittest.TestCase):
    def test_real_affine_flip_format_preserves_pair_and_rng(self):
        from ultralytics.data.augment import Compose as NativeCompose, Format as NativeFormat, RandomFlip as NativeFlip, RandomPerspective
        from ultralytics.utils.instance import Instances
        class NativeTiny(TinyDataset):
            def __init__(self, modality="rgb"):
                super().__init__(modality)
                self.transforms = NativeCompose([
                    RandomPerspective(degrees=0, translate=.1, scale=.5, shear=0, perspective=0, size=(32, 32)),
                    NativeFlip(p=.5, direction="horizontal"),
                    NativeFormat(bbox_format="xywh", normalize=True, batch_idx=True)])
            def get_image_and_label(self, index):
                row = super().get_image_and_label(index)
                boxes = row.pop("bboxes")
                row.update(resized_shape=(32, 32), ratio_pad=(1., 1.),
                           instances=Instances(boxes, segments=np.zeros((0, 0, 2), dtype=np.float32), bbox_format="xywh", normalized=True))
                return row
        for value in (0, 42, 123, 20260907):
            paired, shuffled = make_pair(NativeTiny), make_pair(NativeTiny, shuffled=True)
            seed(value)
            expected = paired[0]
            expected_draws = draws()
            seed(value)
            actual = shuffled[0]
            actual_draws = draws()
            for key in ("img", "cls", "bboxes", "batch_idx", "strong_img", "strong_cls", "strong_bboxes", "strong_batch_idx"):
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
            self.assertEqual(expected["pair_info"], actual["pair_info"])
            self.assertEqual(expected_draws[:2], actual_draws[:2])
            self.assertTrue(torch.equal(expected_draws[2], actual_draws[2]))
            np.testing.assert_allclose(actual["content_info"]["matrix"], actual["pair_info"]["ir_matrix"], atol=1e-9, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
