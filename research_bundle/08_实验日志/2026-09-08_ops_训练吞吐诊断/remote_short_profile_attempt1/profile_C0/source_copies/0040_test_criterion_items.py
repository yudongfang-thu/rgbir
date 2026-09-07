"""CPU fixtures for the actual pinned loss.py dict(zip(loss_names, loss.detach()))."""
import unittest
import torch

from criterion_items import clone_items, compare_items


class CriterionItemsTests(unittest.TestCase):
    def native_items(self):
        # Matches pinned v8DetectionLoss.loss_names / get_assigned_targets_and_loss;
        # the numbers are synthetic fixtures, not claimed training observations.
        loss = torch.tensor([1.25, 2.5, 0.75], requires_grad=True)
        return loss, dict(zip(("box_loss", "cls_loss", "dfl_loss"), loss.detach()))

    def test_actual_native_dict_structure(self):
        loss, items = self.native_items()
        saved = clone_items(items)
        report = compare_items(items, saved)
        self.assertTrue(report["exact"])
        self.assertEqual(set(report["children"]), {"box_loss", "cls_loss", "dfl_loss"})
        self.assertTrue(all(value.ndim == 0 and not value.requires_grad for value in saved.values()))
        with torch.no_grad():
            loss[1] = 4.0
        self.assertEqual(float(saved["cls_loss"]), 2.5)
        self.assertFalse(compare_items(items, saved)["exact"])

    def test_nested_dict_list_tuple_and_tensor(self):
        _, items = self.native_items()
        value = dict(native=items, nested=[(torch.tensor([2.0], requires_grad=True), None), [3, "tag"]])
        saved = clone_items(value)
        self.assertTrue(compare_items(value, saved)["exact"])
        self.assertIsInstance(saved["nested"], list)
        self.assertIsInstance(saved["nested"][0], tuple)
        self.assertFalse(saved["nested"][0][0].requires_grad)
        saved["nested"][0][0][0] = 3.0
        self.assertEqual(float(value["nested"][0][0][0].detach()), 2.0)
        self.assertFalse(compare_items(value, saved)["exact"])

    def test_keys_dtype_shape_and_container_mismatches(self):
        _, items = self.native_items()
        saved = clone_items(items)
        saved["extra_loss"] = torch.tensor(0.0)
        self.assertFalse(compare_items(items, saved)["exact"])
        self.assertFalse(compare_items(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))["exact"])
        self.assertFalse(compare_items(torch.tensor([1.0]), torch.tensor([[1.0]]))["exact"])
        self.assertFalse(compare_items([torch.tensor(1.0)], (torch.tensor(1.0),))["exact"])

    def test_clone_does_not_mutate_original_prediction_graph(self):
        prediction = torch.tensor([2.0, 3.0], requires_grad=True)
        loss = prediction.square()
        items = {"box_loss": loss[0], "cls_loss": loss[1]}
        snapshot = clone_items(items)
        gradient = torch.autograd.grad(loss.sum(), prediction)[0]
        self.assertTrue(torch.equal(gradient, torch.tensor([4.0, 6.0])))
        self.assertTrue(compare_items(items, snapshot)["exact"])
        self.assertIsNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
