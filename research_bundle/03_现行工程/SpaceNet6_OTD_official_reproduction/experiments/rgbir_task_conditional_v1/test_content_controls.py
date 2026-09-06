"""CPU equivalence and intervention checks for content-only control kernels."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "legacy_oev1"))
import object_evidence_loss as original
from content_controls import (object_evidence_content_loss, localization_content_loss,
                              same_modal_evidence_loss, rgb_only_labels)
from localization_loss import (LocalizationConfig, build_localization_selection, localization_loss)
from test_localization_loss import raw as loc_raw, labels


def evidence_raw(foreground, own_labels=None, requires_grad=True):
    own_labels = labels() if own_labels is None else own_labels
    result = loc_raw(distance=1.5)
    result["scores"] = torch.zeros_like(result["scores"])
    result["scores"][:, 1] = -5.
    centers, _, _, _ = original._layout(result, original.EvidenceConfig(input_size=64), (8, 16, 32))
    for cls, box in zip(own_labels["cls"].flatten(), own_labels["bboxes"]):
        corners = torch.cat((box[:2]-box[2:]/2, box[:2]+box[2:]/2)) * 64
        inside = original._inside(corners[None], centers)[0]
        result["scores"][0, int(cls), inside] = foreground
    result["scores"].requires_grad_(requires_grad)
    result["boxes"].requires_grad_(requires_grad)
    return result


class ContentControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = original.EvidenceConfig(input_size=64)
        self.lcfg = LocalizationConfig(input_size=64)
        self.batch = labels()
        self.batch["teacher_batch"] = labels()

    def test_identical_c_content_exact_loss_gradient_and_all_stats(self):
        for arm in ("paired", "paired_random", "paired_uniform", "weight0", "same_modal"):
            s, t, r = evidence_raw(0), evidence_raw(4), evidence_raw(1)
            source = deepcopy(s)
            before = torch.random.get_rng_state().clone()
            old_loss, old_stats = original.object_evidence_loss(source, t, r, self.batch,
                                                               config=self.cfg, arm=arm, seed=11)
            new_loss, new_stats = object_evidence_content_loss(s, t, r, self.batch,
                content_teacher=deepcopy(t), config=self.cfg, arm=arm, seed=11)
            self.assertTrue(torch.equal(old_loss, new_loss), arm)
            self.assertEqual(old_stats, new_stats, arm)
            old_loss.backward()
            new_loss.backward()
            self.assertTrue(torch.equal(source["scores"].grad, s["scores"].grad), arm)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()), arm)
            self.assertIsNone(t["scores"].grad)
            self.assertIsNone(r["scores"].grad)

    def test_content_changes_only_target_not_e_k_quality_or_rng(self):
        s, t, r = evidence_raw(0), evidence_raw(4), evidence_raw(1)
        wrong = evidence_raw(1, requires_grad=False)
        wrong["scores"][:, 1] = 20  # grossly wrong top class is not a new gate
        wrong["scores"].requires_grad_(True)
        before = torch.random.get_rng_state().clone()
        paired_loss, paired = object_evidence_content_loss(s, t, r, self.batch, config=self.cfg)
        wrong_loss, replaced = object_evidence_content_loss(s, t, r, self.batch,
                                                            config=self.cfg, content_teacher=wrong)
        self.assertNotEqual(float(paired_loss), float(wrong_loss))
        target_only = {"loss_unweighted", "target_binary_entropy_selected",
                       "target_gap_to_positive_selected", "target_clipped_count"}
        for key in paired:
            if key not in target_only:
                self.assertEqual(paired[key], replaced[key], key)
        self.assertEqual(replaced["selected_count"], 1)
        self.assertEqual(replaced["normalizer"], 1)
        wrong_loss.backward()
        self.assertIsNone(wrong["scores"].grad)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_no_content_labels_or_regions_can_enter_selection(self):
        b = deepcopy(self.batch)
        b["teacher_batch"] = labels(((20., 16., 44., 40.),))
        t = evidence_raw(4, b["teacher_batch"])
        wrong = evidence_raw(1, b["teacher_batch"])
        s, r = evidence_raw(0), evidence_raw(1)
        a, stats_a = object_evidence_content_loss(s, t, r, b, config=self.cfg, content_teacher=wrong)
        b["content_teacher_batch"] = {"bboxes": "poisoned wrong-image GT"}
        z, stats_z = object_evidence_content_loss(s, t, r, b, config=self.cfg, content_teacher=wrong)
        self.assertTrue(torch.equal(a, z))
        self.assertEqual(stats_a, stats_z)

    def test_c_no_common_object_is_exact_differentiable_zero(self):
        b = labels(())
        b["teacher_batch"] = labels(())
        s = evidence_raw(0)
        loss, stats = object_evidence_content_loss(s, evidence_raw(4), evidence_raw(1), b,
                                                  config=self.cfg, content_teacher=evidence_raw(2))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertEqual(stats["selected_count"], 0)
        self.assertTrue(torch.equal(s["scores"].grad, torch.zeros_like(s["scores"])))

    def test_l_identical_content_exact_loss_gradient_and_stats(self):
        s, t, r = loc_raw(requires_grad=True), loc_raw(distance=1.5), loc_raw()
        old_s = deepcopy(s)
        old_loss, old_stats = localization_loss(old_s, t, r, self.batch, config=self.lcfg)
        selection = build_localization_selection(t, r, self.batch, config=self.lcfg)
        loss, stats = localization_content_loss(s, deepcopy(t), selection)
        self.assertTrue(torch.equal(loss, old_loss))
        self.assertEqual(stats, old_stats)
        loss.backward()
        old_loss.backward()
        self.assertTrue(torch.equal(s["boxes"].grad, old_s["boxes"].grad))
        self.assertEqual(selection.stats["loss_unweighted"], 0.)

    def test_l_content_cannot_regate_or_reassign_even_if_wrong_class(self):
        s, t, r = loc_raw(requires_grad=True), loc_raw(distance=1.5), loc_raw()
        selection = build_localization_selection(t, r, self.batch, config=self.lcfg)
        old_selection = deepcopy(selection)
        wrong = loc_raw(distance=.5)
        wrong["scores"][:, 1] = 20
        wrong["boxes"].requires_grad_(True)
        before = torch.random.get_rng_state().clone()
        loss, stats = localization_content_loss(s, wrong, selection)
        self.assertEqual(stats["selected_anchors"], [(0, 27, 0)])
        self.assertEqual(stats["base_count"], 1)
        self.assertEqual(stats["selected_count"], 1)
        self.assertTrue(torch.equal(selection.quality_gate, old_selection.quality_gate))
        self.assertEqual(selection.stats, old_selection.stats)
        loss.backward()
        self.assertIsNone(wrong["boxes"].grad)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_l_content_preserves_zero_and_frozen_temperature(self):
        selection = build_localization_selection(loc_raw(distance=1.5), loc_raw(), self.batch,
            config=self.lcfg, geometry_eligible=torch.tensor([False]), geometry_verified=True)
        s = loc_raw(requires_grad=True)
        loss, stats = localization_content_loss(s, loc_raw(), selection)
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.equal(s["boxes"].grad, torch.zeros_like(s["boxes"])))
        self.assertEqual(stats["selected_count"], 0)
        with self.assertRaises(ValueError):
            localization_content_loss(s, loc_raw(), selection, temperature=1)

    def test_same_modal_reads_rgb_allowlist_and_retains_legacy_signal(self):
        class RGBOnlyBatch(dict):
            def __getitem__(self, key):
                if key in ("teacher_batch", "ir_batch", "strong_img"):
                    raise AssertionError("Same-modal control touched IR")
                return super().__getitem__(key)
        b = RGBOnlyBatch(labels())
        b["teacher_batch"] = {"poison": "IR GT must not be read"}
        self.assertEqual(set(rgb_only_labels(b)), {"batch_idx", "cls", "bboxes"})
        loss, stats = same_modal_evidence_loss(evidence_raw(0), evidence_raw(4), evidence_raw(1), b, config=self.cfg)
        expected, old = original.object_evidence_loss(evidence_raw(0), evidence_raw(4), evidence_raw(1),
                                                     labels(), config=self.cfg, arm="same_modal")
        self.assertTrue(torch.equal(loss, expected))
        self.assertEqual(stats, old)
        self.assertGreater(stats["selected_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
