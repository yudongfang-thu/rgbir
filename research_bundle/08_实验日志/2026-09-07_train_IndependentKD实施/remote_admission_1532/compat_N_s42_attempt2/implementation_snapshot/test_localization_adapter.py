"""Synthetic CPU checks; synthetic point fixtures are never geometry evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization_adapter import LocalizationAdapter, LocalizationConfig
from geometry_support import GeometryContract, PairedGeometrySupport, build_paired_geometry_mask
from localization_loss import localization_loss


def labels(box=(16., 16., 40., 40.)):
    b = torch.tensor([box])
    return dict(bboxes=torch.cat(((b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]), -1) / 64,
                cls=torch.zeros((1, 1), dtype=torch.long), batch_idx=torch.zeros(1, dtype=torch.long))


def raw(distance=1., grad=False):
    output = dict(scores=torch.full((1, 2, 84), -20.), boxes=torch.full((1, 32, 84), -40.),
                  feats=[torch.zeros(1, 2, side, side) for side in (8, 4, 2)])
    output["boxes"].reshape(1, 4, 8, 84)[:, :, 1, :] = 0
    output["scores"][0, 0, 27] = 4
    dfl = output["boxes"].reshape(1, 4, 8, 84)
    for side in range(4):
        lo, fraction = int(distance), distance - int(distance)
        dfl[0, side, :, 27] = -40
        dfl[0, side, lo, 27] = np.log(1 - fraction)
        if fraction:
            dfl[0, side, lo + 1, 27] = np.log(fraction)
    output["boxes"].requires_grad_(grad)
    output["scores"].requires_grad_(grad)
    return output


def fixture(points=None, uncertainty=0.):
    points = points or [[4, 4], [32, 4], [60, 4], [60, 32], [60, 60], [32, 60], [4, 60], [4, 32]]
    data = dict(mode="verified_identity_grid", verified=True, entries=[dict(status="accepted",
        source="independent_physical_correspondences", image_paths=["synthetic_fixture_rgb.png"],
        original_shape=[64, 64], rgb_points=points, ir_points=points,
        annotator="SYNTHETIC_TEST_ONLY", independent_review_required=True, review_status="accepted",
        uncertainty_raw_px=uncertainty)])
    metadata = dict(rgb_matrix=np.eye(3).tolist(), ir_matrix=np.eye(3).tolist(),
                    original_shape=[64, 64], output_shape=[64, 64])
    batch = labels()
    batch.update(teacher_batch=labels(), im_file=["synthetic_fixture_rgb.png"], pair_info=[metadata])
    return GeometryContract(data), batch


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.contract, self.batch = fixture()
        self.cfg = LocalizationConfig(input_size=64)
        self.adapter = LocalizationAdapter(self.contract, self.cfg)

    def test_standalone_L_direct_gradient_and_detach(self):
        s, t, r = raw(grad=True), raw(1.5, True), raw(grad=True)
        loss, stats = self.adapter.compute(s, t, r, self.batch, return_records=True)
        loss.backward()
        self.assertEqual(stats["base_count"], 1)
        self.assertEqual(stats["selected_anchors"], [(0, 27, 0)])
        self.assertFalse(stats["classification_kd_evaluated"])
        self.assertIsNone(s["scores"].grad)
        self.assertGreater(s["boxes"].grad.abs().sum(), 0)
        self.assertEqual(torch.nonzero(s["boxes"].grad.abs().sum((0, 1))).flatten().tolist(), [27])
        self.assertIsNone(t["boxes"].grad)
        self.assertIsNone(r["scores"].grad)
        self.assertIsNone(r["boxes"].grad)
        json.dumps(stats, allow_nan=False)

    def test_GT_identical_selection_normalizer_and_target_identity(self):
        a, stats_a = self.adapter.compute(raw(grad=True), raw(1.5), raw(), self.batch, task="L1", return_records=True)
        b, stats_b = self.adapter.compute(raw(grad=True), raw(1.5), raw(), self.batch, task="L_GT", return_records=True)
        for key in ("base_count", "selected_count", "normalizer", "selected_anchors", "base_records"):
            self.assertEqual(stats_a[key], stats_b[key])
        self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-6))

    def test_exact_old_loss_when_extra_geometry_does_not_reject(self):
        s, t, r = raw(grad=True), raw(1.5), raw()
        mask, _ = build_paired_geometry_mask(r, self.batch, config=self.cfg, contract=self.contract)
        new, ns = self.adapter.compute(s, t, r, self.batch)
        old, os = localization_loss(s, t, r, self.batch, config=self.cfg, geometry_eligible=mask, geometry_verified=True)
        self.assertTrue(torch.equal(new, old))
        self.assertEqual(ns["selected_anchors"], os["selected_anchors"])
        self.assertEqual(ns["normalizer"], os["normalizer"])

    def test_student_values_do_not_change_selection(self):
        _, a = self.adapter.compute(raw(grad=True), raw(1.5), raw(), self.batch)
        changed = raw(2.8, True)
        _, b = self.adapter.compute(changed, raw(1.5), raw(), self.batch)
        self.assertEqual(a["selected_anchors"], b["selected_anchors"])

    def test_unverified_rejected_or_explicit_zero_diagnostic(self):
        with self.assertRaises(ValueError):
            LocalizationAdapter(None, self.cfg)
        s = raw(grad=True)
        diagnostic = LocalizationAdapter(None, self.cfg, require_verified=False)
        loss, stats = diagnostic.compute(s, raw(1.5), raw(), self.batch)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats["base_count"], 0)
        self.assertTrue(s["boxes"].grad.eq(0).all())
        self.assertFalse(stats["geometry_verified"])

    def test_no_joint_or_classification_arm(self):
        for task in ("C1", "CL", "CGT", "N"):
            with self.assertRaises(ValueError):
                self.adapter.compute(raw(), raw(), raw(), self.batch, task=task)


class PairGeometryTests(unittest.TestCase):
    def test_invalid_annotation_uncertainty_rejected(self):
        contract, _ = fixture(uncertainty=-1.)
        with self.assertRaises(ValueError):
            PairedGeometrySupport(contract)

    def test_IR_box_must_be_fully_covered(self):
        points = [[4, 4], [32, 4], [40, 4], [40, 32], [40, 60], [32, 60], [4, 60], [4, 32]]
        contract, batch = fixture(points)
        reason, _ = PairedGeometrySupport(contract).pair_decision(batch["im_file"][0], [16, 16, 40, 40],
            [16, 16, 41, 40], batch["pair_info"][0], 8)
        self.assertEqual(reason, "ir:uncovered_region")

    def test_boundary_uncertainty_neighborhood_is_covered(self):
        points = [[16, 4], [32, 4], [60, 4], [60, 32], [60, 60], [32, 60], [16, 60], [16, 32]]
        contract, batch = fixture(points, 1.)
        support = PairedGeometrySupport(contract)
        self.assertEqual(contract._one(batch["im_file"][0], [16, 16, 40, 40], batch["pair_info"][0], 8), "accepted")
        self.assertEqual(support.pair_decision(batch["im_file"][0], [16, 16, 40, 40], [16, 16, 40, 40], batch["pair_info"][0], 8)[0],
                         "rgb:uncovered_boundary_neighborhood")

    def test_anchor_must_be_inside_both_targets(self):
        contract, batch = fixture()
        mask, reason, _ = PairedGeometrySupport(contract).anchor_mask(batch["im_file"][0], [16, 16, 40, 40],
            [18, 16, 42, 40], batch["pair_info"][0], 8, [[17, 28], [28, 28], [50, 50]])
        self.assertEqual(reason, "accepted")
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_different_grid_unseen_path_and_nan_rejected(self):
        contract, batch = fixture()
        support = PairedGeometrySupport(contract)
        args = (batch["im_file"][0], [16, 16, 40, 40], [16, 16, 40, 40])
        metadata = copy.deepcopy(batch["pair_info"][0])
        metadata["ir_matrix"][0][2] = 1
        self.assertIn("different_augmented_grids", support.pair_decision(*args, metadata, 8)[0])
        self.assertIn("uncovered_image", support.pair_decision("same_folder_other.png", *args[1:], batch["pair_info"][0], 8)[0])
        self.assertEqual(support.pair_decision(args[0], [float("nan"), 1, 3, 4], args[2], metadata, 8)[0], "invalid_paired_box")

    def test_same_flip_scale_grids_remain_valid(self):
        contract, batch = fixture()
        metadata = copy.deepcopy(batch["pair_info"][0])
        metadata.update(rgb_matrix=[[-1, 0, 64], [0, 1, 0], [0, 0, 1]],
                        ir_matrix=[[-1, 0, 64], [0, 1, 0], [0, 0, 1]])
        reason, _ = PairedGeometrySupport(contract).pair_decision(batch["im_file"][0], [24, 16, 48, 40], [24, 16, 48, 40], metadata, 8)
        self.assertEqual(reason, "accepted")


if __name__ == "__main__":
    unittest.main()
