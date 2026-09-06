"""CPU scientific-operator checks; run `python test_object_evidence_loss.py`.

No training data or GPU is required. These checks establish implementation
properties, not method effectiveness. The enclosing trainer must separately
verify native-loss parity and exactly-once lambda / batch-size scaling.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

import object_evidence_loss as module
from object_evidence_loss import EvidenceConfig, object_evidence_loss


def labels(boxes=((28 / 64, 28 / 64, 24 / 64, 24 / 64),), classes=None):
    return {"batch_idx": torch.zeros(len(boxes), dtype=torch.long),
            "cls": torch.tensor(classes or [0] * len(boxes)).reshape(-1, 1),
            "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)}


def raw(foreground: float, own_labels=None, requires_grad=False):
    """64x64 fixture: 24px-square pre-NMS boxes, spatially varying class logits."""
    own_labels = labels() if own_labels is None else own_labels
    sizes = (8, 4, 2)
    n = sum(s * s for s in sizes)
    scores = torch.zeros(1, 2, n)
    scores[:, 1] = -5
    feats = [torch.zeros(1, 2, s, s) for s in sizes]
    dfl = torch.full((1, 4, 8, n), -40.0)
    offset = 0
    for size, stride in zip(sizes, (8, 16, 32)):
        # Expectation 12px/stride, encoded by neighboring distance bins.
        distance = 12 / stride
        lo, hi = int(distance), int(distance) + 1
        upper = distance - lo
        dfl[:, :, lo, offset:offset+size*size] = torch.tensor(1-upper).log()
        dfl[:, :, hi, offset:offset+size*size] = torch.tensor(upper).log() if upper else -40
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
        xy = torch.stack((x.flatten()+.5, y.flatten()+.5), -1) * stride
        for cls, box in zip(own_labels["cls"].flatten(), own_labels["bboxes"]):
            corners = torch.cat((box[:2]-box[2:]/2, box[:2]+box[2:]/2)) * 64
            inside = module._inside(corners[None], xy)[0]
            scores[0, int(cls), offset:offset+size*size][inside] = foreground
        offset += size * size
    return {"scores": scores.requires_grad_(requires_grad),
            "boxes": dfl.reshape(1, 32, n).requires_grad_(requires_grad), "feats": feats}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = EvidenceConfig(input_size=64)
        self.batch = labels()
        self.batch["teacher_batch"] = labels()

    def test_gradient_sign_and_detached_teacher_reference(self):
        s, t, r = raw(0, requires_grad=True), raw(4, requires_grad=True), raw(1, requires_grad=True)
        loss, stats = object_evidence_loss(s, t, r, self.batch, config=self.cfg)
        self.assertEqual(stats["selected_count"], 1)
        self.assertGreater(float(loss), 0)
        loss.backward()
        centers, _, ranges, _ = module._layout(s, self.cfg, (8, 16, 32))
        gt = torch.tensor([[16., 16., 40., 40.]])
        fg = module._inside(gt, centers)[0]
        annulus = module._inside(torch.tensor([[4., 4., 52., 52.]]), centers)[0] & ~fg
        active = torch.arange(len(centers)) < ranges[1][1]
        self.assertTrue(bool((s["scores"].grad[0, 0, fg & active] < 0).all()))
        self.assertTrue(bool((s["scores"].grad[0, 0, annulus & active] > 0).all()))
        self.assertTrue(bool((s["scores"].grad[0, 1] == 0).all()))
        self.assertTrue(bool((s["scores"].grad[..., ranges[2][0]:] == 0).all()))
        self.assertIsNone(t["scores"].grad)
        self.assertIsNone(t["boxes"].grad)
        self.assertIsNone(r["scores"].grad)
        self.assertIsNone(r["boxes"].grad)
        self.assertIsNone(s["boxes"].grad)
        json.dumps(stats, allow_nan=False)

    def test_additive_logit_offset_invariance(self):
        s, t, r = raw(0), raw(4), raw(1)
        baseline, stats = object_evidence_loss(s, t, r, self.batch, config=self.cfg)
        # Constant offsets cancel per class and level; no calibration matching.
        for model, shift in ((s, 3.), (t, 2.), (r, 1.)):
            model["scores"] = model["scores"] + shift
        shifted, shifted_stats = object_evidence_loss(s, t, r, self.batch, config=self.cfg)
        self.assertAlmostEqual(float(baseline), float(shifted), places=6)
        self.assertEqual(stats["selected_object_ids"], shifted_stats["selected_object_ids"])

    def test_independent_teacher_geometry(self):
        teacher_gt = labels(((32 / 64, 28 / 64, 24 / 64, 24 / 64),))
        batch = deepcopy(self.batch)
        batch["teacher_batch"] = teacher_gt
        s, t, r = raw(0), raw(4, teacher_gt), raw(1)
        _, independent = object_evidence_loss(s, t, r, batch, config=self.cfg)
        _, copied_rgb = object_evidence_loss(s, t, r, self.batch, config=self.cfg)
        self.assertEqual(independent["common_count"], 1)
        self.assertAlmostEqual(independent["teacher_evidence_base_mean"], 2., places=6)
        self.assertLess(copied_rgb["teacher_evidence_base_mean"], independent["teacher_evidence_base_mean"])

    def test_background_excludes_other_classes_and_unmatched_gt(self):
        s = raw(0)
        centers, _, ranges, _ = module._layout(s, self.cfg, (8, 16, 32))
        target = torch.tensor([[16., 16., 40., 40.]])
        other = torch.tensor([[40., 16., 56., 40.]])
        all_boxes = torch.cat((target, other))
        contaminant = module._inside(other, centers)[0]
        s["scores"][0, 0, contaminant] = 12.
        excluded, valid = module._evidence(s["scores"][0], torch.tensor([0]), target, all_boxes, centers, ranges, self.cfg)
        leaked, _ = module._evidence(s["scores"][0], torch.tensor([0]), target, target, centers, ranges, self.cfg)
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.equal(excluded, torch.zeros_like(excluded)))
        self.assertTrue(bool((leaked < 0).all()))

    def test_threshold_aware_matching_maximizes_cardinality(self):
        boxes, classes = torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)
        with patch.object(module, "_iou", return_value=torch.tensor([[.9, .49], [.49, 0.]])):
            a, b = module._match_objects(boxes, classes, boxes, classes, .5)
        self.assertEqual(list(zip(a.tolist(), b.tolist())), [(0, 0)])
        with patch.object(module, "_iou", return_value=torch.tensor([[1., .5], [.5, 0.]])):
            a, b = module._match_objects(boxes, classes, boxes, classes, .5)
        self.assertEqual(list(zip(a.tolist(), b.tolist())), [(0, 1), (1, 0)])

    def test_teacher_top_class_is_required(self):
        t = raw(4)
        t["scores"][:, 1] = 5.
        loss, stats = object_evidence_loss(raw(0, requires_grad=True), t, raw(1), self.batch, config=self.cfg)
        self.assertEqual(stats["base_count"], 1)
        self.assertEqual(stats["teacher_correct_base_count"], 0)
        self.assertEqual(stats["selected_count"], 0)
        self.assertEqual(float(loss), 0.)

    def test_zero_eligible_is_differentiable_and_finite(self):
        s = raw(0, requires_grad=True)
        loss, stats = object_evidence_loss(s, raw(0), raw(1), self.batch, config=self.cfg)
        loss.backward()
        self.assertEqual(stats["eligible_count"], 0)
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.equal(s["scores"].grad, torch.zeros_like(s["scores"])))
        empty = labels(())
        empty["teacher_batch"] = labels(())
        loss, empty_stats = object_evidence_loss(raw(0, requires_grad=True), raw(4), raw(1), empty, config=self.cfg)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(empty_stats["common_count"], 0)
        json.dumps(empty_stats, allow_nan=False)

    def test_weight_zero_preserves_native_gradient(self):
        s = raw(0, requires_grad=True)
        native = s["scores"].square().mean()
        expected = torch.autograd.grad(native, s["scores"], retain_graph=True)[0]
        loss, stats = object_evidence_loss(s, raw(4), raw(1), self.batch, config=self.cfg, arm="weight0")
        actual = torch.autograd.grad(native + loss, s["scores"])[0]
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(stats["selected_count"], 1)
        self.assertEqual(float(loss), 0.)

    def test_invalid_regions_give_finite_zero(self):
        tiny = labels(((28 / 64, 28 / 64, 1 / 64, 1 / 64),))
        tiny["teacher_batch"] = deepcopy(tiny)
        loss, stats = object_evidence_loss(raw(0, requires_grad=True), raw(4), raw(1), tiny, config=self.cfg)
        self.assertEqual(stats["common_count"], 1)
        self.assertEqual(stats["valid_region_count"], 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        json.dumps(stats, allow_nan=False)

    def test_fixed_base_normalizer_before_quality_filter(self):
        targets = labels(((20 / 64, 28 / 64, 24 / 64, 24 / 64), (44 / 64, 28 / 64, 24 / 64, 24 / 64)))
        targets["teacher_batch"] = deepcopy(targets)
        s, t, r = raw(0, targets), raw(4, targets), raw(1, targets)
        centers, _, _, _ = module._layout(s, self.cfg, (8, 16, 32))
        second = module._inside(torch.tensor([[32., 16., 56., 40.]]), centers)[0]
        t["scores"][0, 0, second] = -1.
        loss, stats = object_evidence_loss(s, t, r, targets, config=self.cfg)
        self.assertEqual(stats["base_count"], 2)
        self.assertEqual(stats["eligible_count"], 1)
        self.assertEqual(stats["selected_count"], 1)
        self.assertEqual(stats["normalizer"], 2)
        self.assertEqual(stats["nominal_dose"], .5)
        self.assertAlmostEqual(float(loss), .75, places=6)

    def test_same_modal_never_reads_ir_gt(self):
        batch = deepcopy(self.batch)
        batch["teacher_batch"] = labels(())
        _, stats = object_evidence_loss(raw(0), raw(4), raw(1), batch, config=self.cfg, arm="same_modal")
        self.assertEqual(stats["teacher_gt_count"], 1)
        self.assertEqual(stats["selected_count"], 1)

    def test_random_selection_equal_count_dose_and_private_rng(self):
        q = torch.tensor([1., 3., 3., 0., 0., 2.])
        base = torch.ones(6, dtype=torch.bool)
        eligible = q > 0
        selected = module._choose(q, eligible, base, "paired", .5, 9)
        self.assertEqual(selected.tolist(), [1, 2])  # stable tie policy
        before = torch.random.get_rng_state().clone()
        random_a = module._choose(q, eligible, base, "paired_random", .5, 11)
        random_b = module._choose(q, eligible, base, "paired_random", .5, 11)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(random_a, random_b))
        self.assertEqual(len(random_a), len(selected))
        self.assertEqual(len(random_a) / int(base.sum()), len(selected) / int(base.sum()))

    def test_unfrozen_controls_fail_fast(self):
        for arm in ("shuffled", "gt_control"):
            with self.assertRaises(NotImplementedError):
                object_evidence_loss(raw(0), raw(4), raw(1), self.batch, config=self.cfg, arm=arm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
