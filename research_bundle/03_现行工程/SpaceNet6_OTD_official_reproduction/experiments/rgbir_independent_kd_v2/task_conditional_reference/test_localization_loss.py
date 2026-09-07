"""CPU selector/math contracts; no detector, dataset, GPU or gain claims.

Run from any directory: python test_localization_loss.py
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.nn import functional as F

from localization_loss import (
    LocalizationConfig, build_localization_selection, dfl_view,
    gt_dfl_distribution, localization_kd, localization_loss, native_candidate_mask,
)


def labels(boxes=((16., 16., 40., 40.),), classes=None, indices=None):
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {"bboxes": torch.cat(((b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]), -1) / 64,
            "batch_idx": torch.tensor(indices if indices is not None else [0] * len(b)),
            "cls": torch.tensor(classes if classes is not None else [0] * len(b)).reshape(-1, 1)}


def encode_distance(raw, batch_index, anchor_index, distances):
    dfl = raw["boxes"].reshape(raw["boxes"].shape[0], 4, 8, -1)
    for side, distance in enumerate(distances):
        lo = int(distance)
        frac = distance - lo
        dfl[batch_index, side, :, anchor_index] = -40
        dfl[batch_index, side, lo, anchor_index] = torch.tensor(1-frac).log()
        if frac:
            dfl[batch_index, side, lo+1, anchor_index] = torch.tensor(frac).log()


def raw(batch_size=1, distance=1., requires_grad=False):
    n = 64 + 16 + 4
    result = {"scores": torch.full((batch_size, 2, n), -20.),
              "boxes": torch.full((batch_size, 32, n), -40.),
              "feats": [torch.zeros(batch_size, 2, n, n) for n in (8, 4, 2)]}
    result["boxes"].reshape(batch_size, 4, 8, -1)[:, :, 1, :] = 0
    for bi in range(batch_size):
        result["scores"][bi, 0, 27] = 4
        encode_distance(result, bi, 27, [distance] * 4)
    result["scores"].requires_grad_(requires_grad)
    result["boxes"].requires_grad_(requires_grad)
    return result


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = LocalizationConfig(input_size=64)
        self.batch = labels()
        self.batch["teacher_batch"] = labels()

    def compute(self, s=None, t=None, r=None, batch=None, **kwargs):
        return localization_loss(s or raw(requires_grad=True), t or raw(distance=1.5, requires_grad=True),
                                 r or raw(requires_grad=True), batch or self.batch, config=self.cfg, **kwargs)

    def test_layout_side_bin_anchor_and_analytic_kl(self):
        logits = torch.arange(2*4*3*5, dtype=torch.float32).reshape(2, 12, 5)
        out = dfl_view(logits)
        for b in range(2):
            for a in range(5):
                for side in range(4):
                    for bin_id in range(3):
                        self.assertEqual(float(out[b, a, side, bin_id]), float(logits[b, side*3+bin_id, a]))
        s = torch.tensor([[[0., 1., 2.], [2., 0., 1.], [1., 2., 0.], [0., 2., 1.]]], requires_grad=True)
        t = torch.tensor([[[1., 0., 3.], [1., 2., 0.], [3., 0., 1.], [2., 1., 0.]]], requires_grad=True)
        pt = F.softmax(t / 2, -1)
        loss = localization_kd(s, pt, 3, 2)
        expected = ((pt.detach() * (F.log_softmax(t.detach()/2, -1) - F.log_softmax(s/2, -1))).sum() / 3)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))
        loss.backward()
        self.assertTrue(torch.allclose(s.grad, (F.softmax(s.detach()/2, -1)-pt.detach()) / 6, atol=1e-7))
        self.assertIsNone(t.grad)

    def test_selected_anchor_and_teacher_reference_detach(self):
        s, t, r = raw(requires_grad=True), raw(distance=1.5, requires_grad=True), raw(requires_grad=True)
        loss, stats = self.compute(s, t, r, return_records=True)
        self.assertEqual(stats["base_count"], 1)
        self.assertEqual(stats["eligible_count"], 1)
        self.assertEqual(stats["selected_anchors"], [(0, 27, 0)])
        self.assertGreater(float(loss), 0)
        loss.backward()
        nonzero = s["boxes"].grad.abs().sum((0, 1)) > 0
        self.assertEqual(torch.nonzero(nonzero).flatten().tolist(), [27])
        self.assertIsNone(s["scores"].grad)
        for other in (t, r):
            self.assertIsNone(other["boxes"].grad)
            self.assertIsNone(other["scores"].grad)
        self.assertEqual(stats["geometry_status"], "UNVERIFIED_DIAGNOSTIC")
        json.dumps(stats, allow_nan=False)

    def test_student_values_never_choose_anchors_or_gates(self):
        s = raw()
        _, first = self.compute(s=s, return_records=True)
        s["boxes"] = s["boxes"].flip(1) * .1 + 7
        s["scores"] = -s["scores"]
        _, changed = self.compute(s=s, return_records=True)
        self.assertEqual(first["selected_anchors"], changed["selected_anchors"])
        self.assertEqual(first["base_records"], changed["base_records"])

    def test_zero_geometry_has_differentiable_zero(self):
        s = raw(requires_grad=True)
        loss, stats = self.compute(s=s, geometry_eligible=torch.tensor([False]), geometry_verified=True)
        self.assertEqual(stats["base_count"], 0)
        self.assertEqual(stats["normalizer"], 1)
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.equal(s["boxes"].grad, torch.zeros_like(s["boxes"])))
        with self.assertRaises(ValueError):
            self.compute(geometry_verified=True)

    def test_anchor_specific_geometry_mask(self):
        mask = torch.zeros(1, 84, dtype=torch.bool)
        mask[0, 27] = True
        _, selected = self.compute(geometry_eligible=mask, geometry_verified=True)
        self.assertEqual(selected["selected_count"], 1)
        mask[0, 27] = False
        _, rejected = self.compute(geometry_eligible=mask, geometry_verified=True)
        self.assertEqual(rejected["selected_count"], 0)

    def test_fixed_base_denominator_includes_bad_teacher(self):
        b = labels(((16, 16, 40, 40), (16, 16, 40, 40)), indices=[0, 1])
        b["teacher_batch"] = deepcopy(b)
        t = raw(2, 1.5)
        t["scores"][1, 1, 27] = 5  # same base, unreliable teacher top class
        two, stats = self.compute(s=raw(2), t=t, r=raw(2), batch=b)
        one, _ = self.compute()
        self.assertEqual(stats["base_count"], 2)
        self.assertEqual(stats["selected_count"], 1)
        self.assertEqual(stats["normalizer"], 2)
        self.assertEqual(stats["nominal_dose"], .5)
        self.assertAlmostEqual(float(two), float(one)/2, places=6)

    def test_gt_temperature_preserves_native_two_bin_optimum(self):
        d = torch.tensor([[4.7, 0., 3., 6.99]])
        qt = gt_dfl_distribution(d, 8, 2)
        q = gt_dfl_distribution(d, 8, 1)
        restored = qt.square() / qt.square().sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(q, restored, atol=1e-7))
        self.assertTrue(torch.equal(q == 0, qt == 0))
        expectation = (restored * torch.arange(8)).sum(-1)
        self.assertTrue(torch.allclose(expectation, d, atol=1e-6))
        self.assertEqual(float(qt[0, 1, 0]), 1.)
        self.assertEqual(float(qt[0, 2, 3]), 1.)
        before = d.clone()
        gt_dfl_distribution(d, 8, 2)
        self.assertTrue(torch.equal(before, d))
        for invalid in (-.001, 7.0):
            with self.assertRaises(ValueError):
                gt_dfl_distribution(torch.full((1, 4), invalid), 8)

    def test_gt_control_uses_identical_mask_anchor_denominator(self):
        _, teacher = self.compute(return_records=True)
        loss, control = self.compute(mode="gt", return_records=True)
        for key in ("base_count", "selected_count", "selected_anchors", "normalizer", "nominal_dose", "base_records"):
            self.assertEqual(teacher[key], control[key])
        self.assertGreater(float(loss), 0)

    def test_small_unpaired_other_class_gt_blocks_native_competition(self):
        centers = torch.tensor([[28., 28.]])
        boxes = torch.tensor([[16., 16., 40., 40.], [30., 27., 32., 29.]])
        # Anchor is outside original tiny GT but inside its native expansion.
        self.assertLess(float(centers[0, 0]), float(boxes[1, 0]))
        candidate = native_candidate_mask(boxes, centers, (8, 16, 32))
        self.assertTrue(bool(candidate.all()))
        b = labels(boxes.tolist(), classes=[0, 1])
        b["teacher_batch"] = labels()
        _, stats = self.compute(batch=b)
        self.assertEqual(stats["common_count"], 1)
        self.assertEqual(stats["base_count"], 0)

    def test_strict_pair_filter_follows_half_iou_matching(self):
        b = deepcopy(self.batch)
        b["teacher_batch"] = labels(((20., 16., 44., 40.),))
        _, stats = self.compute(batch=b)
        self.assertEqual(stats["common_count"], 1)
        self.assertEqual(stats["pair_iou_count"], 0)

    def test_unclamped_support_precedes_reference_candidates(self):
        b = labels(((0., 0., 64., 64.),))
        b["teacher_batch"] = deepcopy(b)
        mask = torch.zeros(1, 84, dtype=torch.bool)
        mask[0, 0] = True  # center=4,4, far edge distance 7.5 exceeds 6.99
        _, stats = self.compute(batch=b, geometry_eligible=mask)
        self.assertEqual(stats["inside_count"], 1)
        self.assertEqual(stats["support_count"], 0)

    def test_teacher_must_be_good_at_reference_selected_anchor(self):
        t = raw(distance=1.)
        t["scores"][0, 0, 28] = 6
        encode_distance(t, 0, 28, [2.5, 1.5, .5, 1.5])  # exact GT at different anchor
        _, stats = self.compute(t=t)
        self.assertEqual(stats["base_count"], 1)
        self.assertEqual(stats["selected_count"], 0)

    def test_tie_break_confidence_then_iou_then_global_anchor(self):
        r = raw()
        r["scores"][0, 0, 28] = 4
        encode_distance(r, 0, 28, [2., 1., 0., 1.])  # same decoded R box as anchor27
        selection = build_localization_selection(raw(distance=1.5), r, self.batch, config=self.cfg)
        self.assertEqual(selection.anchor_indices.tolist(), [27])
        r["scores"][0, 0, 28] = 4.1
        selection = build_localization_selection(raw(distance=1.5), r, self.batch, config=self.cfg)
        self.assertEqual(selection.anchor_indices.tolist(), [28])

    def test_random_same_k_dose_and_private_rng(self):
        b = labels(((16, 16, 40, 40), (16, 16, 40, 40)), indices=[0, 1])
        b["teacher_batch"] = deepcopy(b)
        t = raw(2, 1.5)
        t["scores"][1, 1, 27] = 5
        before = torch.random.get_rng_state().clone()
        _, a = self.compute(s=raw(2), t=t, r=raw(2), batch=b, mode="random", seed=17)
        _, c = self.compute(s=raw(2), t=t, r=raw(2), batch=b, mode="random", seed=17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(a["selected_anchors"], c["selected_anchors"])
        self.assertEqual(a["selected_count"], 1)
        self.assertEqual(a["normalizer"], 2)

    def test_weight_zero_and_equal_distributions(self):
        s = raw(requires_grad=True)
        native = s["boxes"].square().mean()
        expected = torch.autograd.grad(native, s["boxes"], retain_graph=True)[0]
        zero, stats = self.compute(s=s, mode="weight0")
        actual = torch.autograd.grad(native + zero, s["boxes"])[0]
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(stats["selected_count"], 1)
        t = raw(distance=1.5)
        same, _ = self.compute(s=deepcopy(t), t=t)
        self.assertLess(abs(float(same)), 1e-6)

    def test_invalid_layout_nonfinite_and_empty_labels(self):
        t = raw()
        t["boxes"] = t["boxes"][:, :28]
        with self.assertRaises(ValueError):
            self.compute(t=t)
        t = raw()
        t["scores"][0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.compute(t=t, geometry_eligible=torch.tensor([False]))
        b = labels(())
        b["teacher_batch"] = labels(())
        loss, stats = self.compute(batch=b, return_records=True)
        self.assertEqual(float(loss), 0)
        self.assertEqual(stats["common_count"], 0)
        json.dumps(stats, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
