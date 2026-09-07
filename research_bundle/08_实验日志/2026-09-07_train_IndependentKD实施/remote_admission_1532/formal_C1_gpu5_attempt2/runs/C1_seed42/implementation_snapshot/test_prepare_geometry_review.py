"""Synthetic train-only candidate selection checks; no visual evidence created."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_geometry_review import generate_candidates, write_package


def inputs(root, natural=True):
    root = Path(root)
    rows = [{"rgb_path": str(root / ("rgb/%03d.jpg" % i)), "ir_path": str(root / ("ir/%03d.jpg" % i)),
             "stem": "%03d" % i, "source_group": str(i % 4), "split": "train"} for i in range(80)]
    roster = root / "roster.json"
    roster.write_text(json.dumps({"dataset": "dronevehicle", "split": "train", "roster": rows}))
    d2 = root / "d2.jsonl"
    d2.write_text("\n".join(json.dumps({"image_id": row["rgb_path"], "selected": True, "quality_gate": True,
                   "object_id": row["stem"] + ":0", "scale_bin": "small" if i % 2 else "medium"}) for i, row in enumerate(rows[40:])))
    spec = {"dataset": "dronevehicle", "train_roster": str(roster), "d2_jsonl": str(d2)}
    if natural:
        batchfile = root / "natural.jsonl"
        batchfile.write_text("\n".join(json.dumps({"batch": i, "images": [{"image": row["rgb_path"], "rgb_source": row["rgb_path"]}]}) for i, row in enumerate(rows[:64])))
        receipt = root / "coverage.json"
        receipt.write_text(json.dumps({"status": "COMPLETED", "actual_batches": 64, "generator_seed": 20260907, "dataset": "dronevehicle"}))
        spec.update(natural_batches=str(batchfile), coverage_receipt=str(receipt))
    return spec


class CandidateTests(unittest.TestCase):
    def test_bound24_review5_natural_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_candidates([inputs(tmp)])
            self.assertEqual(len(result["roster"]), 24)
            self.assertEqual(result["independent_reviews_required"], 5)
            self.assertEqual(sum(row["independent_review_required"] for row in result["roster"]), 5)
            self.assertTrue(all(row["natural_hit"] for row in result["roster"]))
            self.assertFalse(result["geometry_verified"])
            self.assertEqual(len({row["rgb_source"] for row in result["roster"]}), 24)
            self.assertEqual(len({row["source_group"] for row in result["roster"]}), 4)
            self.assertEqual({row["representative_scale"] for row in result["roster"]}, {"small", "medium"})

    def test_missing_natural_is_draft_and_repeat_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = inputs(tmp, natural=False)
            a, b = generate_candidates([spec]), generate_candidates([spec])
            self.assertEqual(a, b)
            self.assertEqual(a["status"], "DRAFT_AWAIT_NATURAL_STREAM")

    def test_validation_split_cannot_enter(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = inputs(tmp)
            path = Path(spec["train_roster"])
            data = json.loads(path.read_text()); data["split"] = "val"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): generate_candidates([spec])

    def test_same_basename_outside_declared_train_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = inputs(tmp)
            Path(spec["d2_jsonl"]).write_text(json.dumps({"image_id": str(Path(tmp) / "val/040.jpg"), "selected": True, "quality_gate": True}))
            with self.assertRaises(ValueError): generate_candidates([spec])

    def test_incomplete_natural_and_overbudget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = inputs(tmp)
            path = Path(spec["coverage_receipt"]); data = json.loads(path.read_text()); data["actual_batches"] = 63
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): generate_candidates([spec])
            with self.assertRaises(ValueError): generate_candidates([], limit=25)

    def test_visual_feed_hides_static_model_content_and_no_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_candidates([inputs(tmp)])
            out = Path(tmp) / "review"
            write_package(result, out)
            visual = json.loads((out / "visual_review_roster.json").read_text())
            self.assertNotIn("static_object_ids", visual["roster"][0])
            self.assertNotIn("representative_scale", visual["roster"][0])
            template = json.loads((out / "primary_observations_template.json").read_text())
            self.assertTrue(all(row["points"] == [] for row in template["observations"]))
            self.assertFalse(template["formal_geometry_verified"])
            with self.assertRaises(FileExistsError): write_package(result, out)


if __name__ == "__main__":
    unittest.main()
