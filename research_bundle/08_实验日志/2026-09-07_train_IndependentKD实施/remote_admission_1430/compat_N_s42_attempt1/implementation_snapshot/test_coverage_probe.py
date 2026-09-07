"""CPU tests for natural stream, seed ownership, source coverage upper bounds."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_probe import make_natural_loader, run_coverage, _rng_state


class TinyPaired(Dataset):
    def __len__(self):
        return 80

    def __getitem__(self, index):
        image = torch.tensor([random.randint(0, 254), np.random.randint(0, 255), torch.randint(0, 255, ()).item()], dtype=torch.uint8).reshape(3, 1, 1)
        return dict(img=image, strong_img=image.clone(), im_file=str(Path("tiny") / (str(index) + ".jpg")),
                    pair_info={"weak_source": str(Path("tiny") / (str(index) + ".jpg")), "strong_source": "ir/" + str(index)},
                    bboxes=torch.tensor([[.5, .5, .2, .2]]), cls=torch.zeros((1, 1)))

    @staticmethod
    def collate_fn(rows):
        return dict(img=torch.stack([r["img"] for r in rows]), strong_img=torch.stack([r["strong_img"] for r in rows]),
                    im_file=[r["im_file"] for r in rows], pair_info=[r["pair_info"] for r in rows],
                    cls=torch.cat([r["cls"] for r in rows]), bboxes=torch.cat([r["bboxes"] for r in rows]),
                    batch_idx=torch.arange(len(rows)), strong_cls=torch.cat([r["cls"] for r in rows]),
                    strong_bboxes=torch.cat([r["bboxes"] for r in rows]), strong_batch_idx=torch.arange(len(rows)))


def first_batches(seed=20260907, workers=0):
    loader = make_natural_loader(TinyPaired(), 4, workers, seed)
    result = []
    iterator = iter(loader)
    try:
        for _ in range(4):
            b = next(iterator)
            result.append((b["im_file"], b["img"].clone(), b["pair_info"]))
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    return result


class CoverageTests(unittest.TestCase):
    def test_same_seed_same_indices_and_augments(self):
        first, second = first_batches(), first_batches()
        for a, b in zip(first, second):
            self.assertEqual(a[0], b[0])
            self.assertTrue(torch.equal(a[1], b[1]))
            self.assertEqual(a[2], b[2])
        ids = [v for batch in first for v in batch[0]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_seed_changes_real_stream(self):
        self.assertNotEqual(first_batches(20260907)[0][0], first_batches(20260908)[0][0])

    def test_caller_rng_not_consumed_workers0(self):
        before = _rng_state()
        first_batches()
        after = _rng_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1][1], after[1][1])
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_workers_are_seeded_and_reproducible(self):
        first, second = first_batches(workers=2), first_batches(workers=2)
        for a, b in zip(first, second):
            self.assertEqual(a[0], b[0])
            self.assertTrue(torch.equal(a[1], b[1]))
            self.assertEqual(a[2], b[2])
        seeds = {r["coverage"]["torch_worker_seed"] for batch in first for r in batch[2]}
        self.assertEqual(len(seeds), 2)

    def test_exact_hit_upper_bound_and_no_false_geometric_acceptance(self):
        loader = make_natural_loader(TinyPaired(), 4, 0)
        first = first_batches()
        wanted = str(Path(first[0][0][0]).absolute().resolve())
        roster = {wanted: [{"audit_id": "synthetic", "group": "g"}]}
        with tempfile.TemporaryDirectory() as root:
            receipt = run_coverage(loader, roster, Path(root) / "fresh", batches=4)
            self.assertEqual(receipt["roster_hit_batch_upper_bound"], 1)
            self.assertFalse(receipt["geometry_verified"])
            self.assertTrue(receipt["actual_L_selected_not_measured"])
            rows = [json.loads(s) for s in (Path(root) / "fresh/natural_batches.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["images"][0]["student_dtype"], "torch.uint8")
            with self.assertRaises(FileExistsError):
                run_coverage(loader, roster, Path(root) / "fresh", batches=4)


if __name__ == "__main__":
    unittest.main()
