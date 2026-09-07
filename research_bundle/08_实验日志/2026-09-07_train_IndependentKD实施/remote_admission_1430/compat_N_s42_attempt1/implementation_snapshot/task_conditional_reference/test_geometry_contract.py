"""Synthetic tests validate contracts; synthetic points are never audit evidence."""
import copy
import unittest

import numpy as np

from geometry_contract import GeometryContract, correspondence_summary


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = [[5, 5], [50, 5], [95, 5], [95, 50], [95, 95], [50, 95], [5, 95], [5, 50]]
        self.data = dict(mode="verified_identity_grid", verified=True, entries=[dict(
            status="accepted", source="independent_physical_correspondences",
            image_paths=["sample_rgb.png"], original_shape=[100, 100],
            rgb_points=self.points, ir_points=self.points, annotator="synthetic_fixture",
            independent_review_required=True, review_status="accepted", uncertainty_raw_px=0.0)])
        self.meta = [dict(rgb_matrix=np.eye(3).tolist(), ir_matrix=np.eye(3).tolist(),
                          original_shape=[100, 100])]

    def decision(self, data=None, box=None, meta=None, stride=8, path="sample_rgb.png"):
        return GeometryContract(data or self.data).object_decisions(
            [path], np.array([box or [20, 20, 60, 60]]), np.array([0]), meta or self.meta, stride)

    def test_known_region_identity(self):
        self.assertEqual(self.decision()[1], ["accepted"])

    def test_unverified_never_admitted(self):
        data = copy.deepcopy(self.data)
        data["verified"] = False
        self.assertFalse(self.decision(data)[0][0])

    def test_unseen_image_never_extrapolated(self):
        self.assertEqual(self.decision(path="same_folder_other_frame.png")[1], ["uncovered_image"])

    def test_path_aliases_canonicalized(self):
        self.assertEqual(self.decision(path="unused/../sample_rgb.png")[1], ["accepted"])

    def test_region_requires_entire_box(self):
        self.assertEqual(self.decision(box=[0, 10, 60, 60])[1], ["uncovered_region"])

    def test_missing_transform(self):
        self.assertEqual(self.decision(meta=[{}])[1], ["missing_transform"])

    def test_scale_translate_flip(self):
        a = [[-2, 0, 210], [0, 2, -10], [0, 0, 1]]
        meta = [dict(rgb_matrix=a, ir_matrix=a, original_shape=[100, 100])]
        self.assertEqual(self.decision(box=[90, 30, 170, 110], meta=meta)[1], ["accepted"])

    def test_different_augmented_grid_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta[0]["ir_matrix"][0][2] = 1
        self.assertEqual(self.decision(meta=meta)[1], ["different_augmented_grids"])

    def test_error_propagates_scale(self):
        data = copy.deepcopy(self.data)
        data["entries"][0]["ir_points"] = (np.array(self.points) + [1.1, 0]).tolist()
        self.assertEqual(self.decision(data)[1], ["accepted"])
        a = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        meta = [dict(rgb_matrix=a, ir_matrix=a, original_shape=[100, 100])]
        self.assertEqual(self.decision(data, box=[40, 40, 120, 120], meta=meta)[1], ["p95_exceeds_tolerance"])

    def test_small_object_tolerance(self):
        data = copy.deepcopy(self.data)
        data["entries"][0]["ir_points"] = (np.array(self.points)+[1.1, 0]).tolist()
        self.assertEqual(self.decision(data, box=[20, 20, 30, 30])[1], ["p95_exceeds_tolerance"])

    def test_uncertainty_not_ignored(self):
        data = copy.deepcopy(self.data)
        data["entries"][0]["uncertainty_raw_px"] = 2.1
        self.assertEqual(self.decision(data)[1], ["p95_exceeds_tolerance"])

    def test_insufficient_points_or_review_rejected(self):
        data = copy.deepcopy(self.data)
        data["entries"][0]["review_status"] = "pending"
        with self.assertRaises(ValueError):
            GeometryContract(data)
        data = copy.deepcopy(self.data)
        data["entries"][0]["rgb_points"] = self.points[:3]
        data["entries"][0]["ir_points"] = self.points[:3]
        with self.assertRaises(ValueError):
            GeometryContract(data)

    def test_statistics_are_physical_residuals(self):
        summary = correspondence_summary(self.points, np.array(self.points)+[3, 4], [100, 100])
        self.assertEqual(summary["p95_error_raw_px"], 5)
        self.assertEqual(summary["max_error_raw_px"], 5)
        self.assertEqual(summary["rgb_quadrants"], 4)

    def test_empty_objects(self):
        mask, reasons = GeometryContract(self.data).object_decisions(
            ["sample_rgb.png"], np.empty((0, 4)), np.empty(0), self.meta, 8)
        self.assertEqual(mask.shape, (0,))
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
