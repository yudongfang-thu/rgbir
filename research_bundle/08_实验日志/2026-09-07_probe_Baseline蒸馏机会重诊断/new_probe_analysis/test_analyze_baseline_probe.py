"""Known-truth mathematical/denominator checks; all fixtures are synthetic."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_baseline_probe as a


def obj(identifier="a", cls=0, ncls=0, niou=.7, nconf=.8, tcls=0, tiou=.8, tconf=.9, pair=.9):
    def model(c, i, p):
        return {"assigned": {"class": c, "confidence": p, "iou": .99},
                "iou_to_rgb_gt": i,
                "regions": {k: {"valid": True} for k in ("P3", "P4")},
                "same_anchor": {"pred_class": c, "confidence": p, "iou_to_rgb_gt": i}}
    return {"object_id": identifier, "class": cls, "split": "train", "source_group": "g", "scale_bin": "small",
            "mean_rgb_luma": 80, "is_background": False, "paired_gt_iou": pair,
            "gt_box_input": [0, 0, 32, 32], "anchor_center": [16, 16], "anchor_stride": 8,
            "anchor_has_reference_candidate": True, "N42": model(ncls, niou, nconf), "T42": model(tcls, tiou, tconf)}


class ProbeKnownTruth(unittest.TestCase):
    def test_background_excluded_from_gt_opportunity_denominator(self):
        bg = obj("bg"); bg["is_background"] = True
        q = a.opportunity_summary([obj(), bg], [True, True])
        self.assertEqual(q["all_rgb_gt"], 1)

    def test_repairs_use_rgb_iou_not_teacher_native_ir_iou(self):
        r = obj(nconf=.1, tiou=.1)
        q = a.opportunity_summary([r], [True])
        self.assertEqual(q["repair"]["n"], 0)
        self.assertEqual(r["T42"]["assigned"]["iou"], .99)

    def test_pair_threshold_excludes_unassociated_teacher(self):
        q = a.opportunity_summary([obj(nconf=.1, pair=.49)], [True])
        self.assertEqual(q["repair"]["n"], 0)

    def test_low_confidence_repair_is_not_class_repair(self):
        q = a.opportunity_summary([obj(nconf=.1)], [True])
        self.assertEqual(q["repair"]["n"], 1)
        self.assertEqual(q["factor_specific_repair"]["low_confidence_fix_at_correct_class_localization"]["n"], 1)
        self.assertEqual(q["factor_specific_repair"]["class_fix_at_localized_native"]["n"], 0)

    def test_class_and_localization_can_overlap(self):
        q = a.opportunity_summary([obj(ncls=1, niou=.2)], [True])
        self.assertEqual(q["native_states"]["class_and_localization"]["n"], 1)
        self.assertEqual(q["native_overlapping_error_flags"]["class_ok"]["n"], 1)
        self.assertEqual(q["native_overlapping_error_flags"]["localization_ok"]["n"], 1)

    def test_missing_teacher_is_replacement_harm_and_unpaired_not_harm(self):
        r = obj(); r["T42"]["assigned"] = None
        self.assertEqual(a.opportunity_summary([r], [True])["harm_if_teacher_replaces_native"]["n"], 1)
        r["paired_gt_iou"] = None
        self.assertEqual(a.opportunity_summary([r], [True])["harm_if_teacher_replaces_native"]["n"], 0)

    def test_same_anchor_and_assigned_are_separate(self):
        r = obj(); r["T42"]["same_anchor"]["iou_to_rgb_gt"] = .1
        self.assertEqual(a.opportunity_summary([r], [True], True)["harm_if_teacher_replaces_native"]["n"], 1)
        self.assertEqual(a.opportunity_summary([r], [True], False)["harm_if_teacher_replaces_native"]["n"], 0)

    def test_same_modal_object_comparison_does_not_require_ir_pair(self):
        r = obj(nconf=.1, pair=.1); r["N0"] = r["T42"]
        result = a.opportunity_summary([r], [True], teacher_model="N0", require_pair=False)
        self.assertEqual(result["repair"]["n"], 1)
        self.assertEqual(result["paired_gt"]["n"], 0)
        self.assertEqual(result["comparison_eligible_gt"]["n"], 1)

    def test_train_only_standardization(self):
        x = np.array([[1., 5.], [3., 5.], [1000., -9999.]])
        z, mean, std = a.standardize_fit(x, np.array([1, 1, 0], bool))
        np.testing.assert_allclose(mean, [2, 5])
        np.testing.assert_allclose(std, [1, 1])
        np.testing.assert_allclose(z[:2].mean(0), [0, 0])

    def test_mean_objective_ridge_invariant_to_train_duplication(self):
        x = np.array([[-2.], [-1.], [1.], [2.], [0.4]])
        y = np.array([0, 0, 1, 1, 1])
        train = np.array([1, 1, 1, 1, 0], bool)
        p1, _ = a.ridge_fit_predict(x, y, train, 2)
        x2 = np.concatenate([np.repeat(x[:4], 3, axis=0), x[4:]])
        y2 = np.concatenate([np.repeat(y[:4], 3), y[4:]])
        p2, _ = a.ridge_fit_predict(x2, y2, np.arange(13) < 12, 2)
        self.assertEqual(p1[-1], p2[-1])
        # Direct analytic mean-normalized solution must produce the same score.
        z, _, _ = a.standardize_fit(x, train)
        target = np.eye(2)[y[train]]
        w = np.linalg.solve(z[train].T @ z[train] / 4 + np.eye(1), z[train].T @ (target - target.mean(0)) / 4)
        np.testing.assert_array_equal((z @ w + target.mean(0)).argmax(1), p1)

    def test_fixed_projection_reproducible_and_label_free(self):
        x = np.arange(20).reshape(4, 5)
        np.testing.assert_array_equal(a.fixed_projection(x), a.fixed_projection(x))
        self.assertEqual(a.fixed_projection(x).shape, (4, 128))

    def test_shuffle_never_crosses_split_and_retains_joint_alignment(self):
        x = np.arange(30).reshape(10, 3)
        train = np.arange(10) < 6
        shuffled, ix = a.shuffled_within_split(x, train, ~train)
        np.testing.assert_array_equal(np.sort(ix[:6]), np.arange(6))
        np.testing.assert_array_equal(np.sort(ix[6:]), np.arange(6, 10))
        _, ix2 = a.shuffled_within_split(x + 1000, train, ~train)
        np.testing.assert_array_equal(ix, ix2)
        np.testing.assert_array_equal(shuffled, x[ix])

    def test_accuracy_and_macro_recall_not_interchanged(self):
        m = a.classification_metrics(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), 3)
        self.assertEqual(m["accuracy"], .75)
        self.assertEqual(m["balanced_accuracy"], .5)
        self.assertTrue(np.isnan(m["per_class"]["2"]["recall"]))

    def test_empty_classification_subset(self):
        m = a.classification_metrics(np.array([], int), np.array([], int), 3)
        self.assertIsNone(m["accuracy"])
        self.assertIsNone(m["balanced_accuracy"])

    def test_dfl_interpolated_targets_and_exact_final_bin(self):
        q, support = a.dfl_targets(np.array([[0, 1.5, 14.25, 15]]))
        self.assertTrue(support[0])
        np.testing.assert_allclose(q.sum(-1), np.ones((1, 4)))
        self.assertEqual(q[0, 1, 1], .5)
        self.assertEqual(q[0, 1, 2], .5)
        self.assertEqual(q[0, 3, 15], 1)

    def test_dfl_out_of_support_never_clamped(self):
        q, support = a.dfl_targets(np.array([[-.01, 1, 1, 1], [1, 1, 15.01, 1]]))
        self.assertFalse(support.any())
        self.assertTrue(np.isnan(q).all())

    def test_dfl_gtce_uniform_is_log16(self):
        q, _ = a.dfl_targets(np.array([[1.5] * 4]))
        z = np.zeros((1, 4, 16))
        ce_n, ce_t, kl, cos = a.dfl_values(z, z, q)
        np.testing.assert_allclose(ce_n, np.log(16))
        np.testing.assert_allclose(kl, 0, atol=1e-12)
        self.assertTrue(np.isnan(cos[0]))

    def test_native_dfl_support_excludes_mathematical_bin15(self):
        r = obj(); r["gt_box_input"] = [-104, -104, 136, 136]
        logits = {k: np.zeros((1, 4, 16)) for k in ("N42_dfl", "T42_dfl")}
        with tempfile.TemporaryDirectory() as tmp:
            result = a.dfl_analysis([r], logits, {}, Path(tmp))
        self.assertEqual(result["native_support_upper"], 14.99)
        self.assertEqual(result["splits"]["train"]["all_supported_gt"]["n"], 0)
        self.assertEqual(result["splits"]["train"]["out_of_support_gt"], 1)

    def test_dfl_gradient_cosine_direction(self):
        np.testing.assert_allclose(a.gradient_cosine(np.array([[1, 2], [1, 2]]), np.array([[1, 2], [-1, -2]])), [1, -1])

    def test_temperature2_kd_raw_logit_gradient_finite_difference(self):
        rng = np.random.default_rng(17)
        native, teacher = rng.normal(size=(2, 4, 16)), rng.normal(size=(2, 4, 16))
        p_t = a.softmax(teacher / 2)
        def loss(z):
            return 4 * (p_t * (a.log_softmax(teacher / 2) - a.log_softmax(z / 2))).sum()
        eps = 1e-5
        plus, minus = native.copy(), native.copy()
        plus[0, 2, 7] += eps; minus[0, 2, 7] -= eps
        numeric = (loss(plus) - loss(minus)) / (2 * eps)
        analytic = 2 * (a.softmax(native / 2) - p_t)[0, 2, 7]
        self.assertAlmostEqual(numeric, analytic, places=8)

    def test_dfl_gt_raw_logit_gradient_finite_difference(self):
        q, _ = a.dfl_targets(np.array([[1.5, 2.3, 4, 5]]))
        z = np.random.default_rng(3).normal(size=(1, 4, 16))
        def loss(x):
            return -(q * a.log_softmax(x)).sum()
        eps = 1e-5
        plus, minus = z.copy(), z.copy()
        plus[0, 1, 2] += eps; minus[0, 1, 2] -= eps
        self.assertAlmostEqual((loss(plus) - loss(minus)) / (2 * eps), (a.softmax(z) - q)[0, 1, 2], places=8)

    def test_dfl_background_pair_and_reference_denominators(self):
        rows = [obj(str(i)) for i in range(4)]
        rows[0]["split"] = "val"
        rows[1]["split"] = "val"; rows[1]["anchor_has_reference_candidate"] = False
        rows[2]["split"] = "val"; rows[2]["is_background"] = True
        rows[3]["split"] = "val"; rows[3]["paired_gt_iou"] = .1
        logits = {k: np.zeros((4, 4, 16)) for k in ("N42_dfl", "T42_dfl")}
        with tempfile.TemporaryDirectory() as tmp:
            q = a.dfl_analysis(rows, logits, {}, Path(tmp))["splits"]["val"]
        self.assertEqual(q["all_rgb_gt"], 3)
        self.assertEqual(q["paired_supported_gt"]["n"], 2)
        self.assertEqual(q["paired_supported_gt_with_native_reference_candidate"]["n"], 1)

    def test_end_to_end_synthetic_export_and_no_overwrite(self):
        rng = np.random.default_rng(5)
        rows = [obj(str(i), cls=i % 3) for i in range(18)]
        for i, r in enumerate(rows):
            r["split"] = "train" if i < 12 else "val"
            r["is_background"] = i % 3 == 2
            r["N0"] = r["N42"].copy()
        for i in (0, 12):
            rows[i]["T42"]["regions"]["P4"]["valid"] = False
        logits = {m + "_cls": rng.normal(size=(18, 2)) for m in ("N42", "T42", "N0")}
        logits.update({m + "_dfl": rng.normal(size=(18, 4, 16)) for m in ("N42", "T42")})
        logits["N42_region_cls"] = rng.normal(size=(18, 2, 2))
        logits["T42_region_cls"] = rng.normal(size=(18, 2, 2))
        logits["N0_region_cls"] = rng.normal(size=(18, 2, 2))
        feats = {m + "_" + level: rng.normal(size=(18, 4)) for m in ("N42", "T42", "N0") for level in ("P3", "P4", "anchor_P3", "anchor_P4")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "objects.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            np.savez(root / "features.npz", **feats)
            np.savez(root / "logits.npz", **logits)
            (root / "summary.json").write_text(json.dumps({"input_box_format": "xyxy", "reg_max": 16, "training_recipes_equal": False}), encoding="utf-8")
            out = a.analyze(root, root / "analysis")
            self.assertEqual(out["classification_probe"]["classes"], 3)
            self.assertEqual(out["classification_probe"]["val_rows"], 6)
            self.assertEqual(out["object_opportunity"]["val"]["assigned"]["all_rgb_gt"], 4)
            self.assertTrue((root / "analysis" / "summary.json").is_file())
            self.assertIn("N_logits+T_anchor_feature_P3", out["classification_probe"]["arms"])
            self.assertIn("N_logits+T_logits", out["classification_probe"]["arms"])
            self.assertIn("N_logits+N_anchor_feature+T_logits_P3", out["classification_probe"]["arms"])
            self.assertIn("N_region_logits_P3+T_region_logits", out["classification_probe"]["arms"])
            self.assertEqual(len(out["classification_probe"]["carrier_comparisons"]), 9)
            self.assertEqual(out["classification_probe"]["arms"]["N_logits_ROI_valid"]["train_n"], 11)
            self.assertEqual(out["classification_probe"]["arms"]["N_logits_ROI_valid"]["val_n"], 5)
            self.assertEqual(out["classification_probe"]["arms"]["N_logits+T_anchor_feature_P3"]["val_n"], 6)
            self.assertFalse(out["input_metadata"]["training_recipes_equal"])
            with np.load(root / "analysis" / "probe_predictions.npz") as predictions:
                valid = predictions["roi_common_valid"]
                self.assertTrue(valid[predictions["roi_shuffle_source_row"][valid]].all())
            with self.assertRaises(FileExistsError):
                a.analyze(root, root / "analysis")


if __name__ == "__main__":
    unittest.main(verbosity=2)
