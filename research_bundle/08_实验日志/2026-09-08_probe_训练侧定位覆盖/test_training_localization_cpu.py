"""Small CPU truths for native matching and frozen C denominator accounting."""
import argparse
import json
from pathlib import Path
import unittest
import torch
from analyze_training_localization import bucket, coverage
from native_cached_match import load_native, match_row


class Truths(unittest.TestCase):
    def test_independent_rebinding_at_high_iou(self):
        # Native .5 selects original prediction 0; .75 must independently bind 1.
        r = dict(pred_boxes=[[0, 0, 6, 10], [0, 0, 9, 10]], pred_classes=[0, 0],
                 pred_confidence=[.9, .8], gt_boxes=[[0, 0, 10, 10]], gt_classes=[0])
        a = match_row(r, None, .5, NATIVE); b = match_row(r, None, .75, NATIVE)
        self.assertEqual(a['gt_matches'][0]['prediction_id'], 0)
        self.assertEqual(b['gt_matches'][0]['prediction_id'], 1)

    def test_empty_and_wrong_class(self):
        r = dict(pred_boxes=[], pred_classes=[], pred_confidence=[], gt_boxes=[[0, 0, 10, 10]], gt_classes=[0])
        self.assertEqual(match_row(r, None, .75, NATIVE)['fn'], 1)
        r.update(pred_boxes=[[0, 0, 10, 10]], pred_classes=[1], pred_confidence=[.9])
        self.assertEqual(match_row(r, None, .5, NATIVE)['tp'], 0)

    def test_bucket_and_frozen_C_denominator(self):
        a = dict(frame_id='a', stable_rgb_gt_id='a:0', C_gates=dict(base=True, eligible=True, selected=True),
                 matches={'S': {'0.5': {}, '0.75': None}, 'T': {'0.5': {}, '0.75': {}}})
        b = dict(a, frame_id='b', stable_rgb_gt_id='b:0', C_gates=dict(base=True, eligible=False, selected=False))
        self.assertEqual(bucket(a), 'both05_onlyT075')
        c = coverage([a, b], 8)
        self.assertEqual(c['fraction_all_gt'], .25)
        self.assertEqual(c['selected']['fraction_bucket'], .5)
        self.assertEqual(c['selected']['unique_images'], 1)
        self.assertIsNone(coverage([], 8)['selected']['fraction_bucket'])

    def test_strict_confidence_and_original_prediction_id(self):
        r = dict(pred_boxes=[[0, 0, 10, 10], [0, 0, 10, 10]], pred_classes=[0, 0],
                 pred_confidence=[.25, .8], gt_boxes=[[0, 0, 10, 10]], gt_classes=[0])
        x = match_row(r, .25, .75, NATIVE)
        self.assertEqual(x['kept_prediction_ids'], [1])
        self.assertEqual(x['gt_matches'][0]['prediction_id'], 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--native-source-dir', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    NATIVE = load_native(args.native_source_dir); torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors), CUDA_initialized=torch.cuda.is_initialized(),
                   new_hash_computed=False, native_source_dir=str(args.native_source_dir))
    args.output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
