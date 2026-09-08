"""Small known-truth CPU/gradient checks; synthetic labels are not physical evidence."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).absolute().parent
parser = argparse.ArgumentParser()
parser.add_argument('--pinned', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path[:0] = [str(args.pinned), str(args.pinned/'task_conditional_reference'), str(HERE)]
import localization_box_v2 as l2


def labels(boxes=((16., 16., 40., 40.),), classes=None, indices=None):
    b = torch.tensor(boxes).reshape(-1, 4)
    return dict(bboxes=torch.cat(((b[:, :2]+b[:, 2:])/2, b[:, 2:]-b[:, :2]), -1)/64,
        cls=torch.tensor(classes if classes is not None else [0]*len(b)).reshape(-1, 1),
        batch_idx=torch.tensor(indices if indices is not None else [0]*len(b)))


def set_box(raw, anchor, xyxy, bi=0):
    center = torch.tensor(((anchor%8+.5)*8, (anchor//8+.5)*8))
    b = torch.tensor(xyxy)
    dist = torch.cat((center-b[:2], b[2:]-center))/8
    assert bool(((dist >= 0) & (dist < 7)).all())
    view = raw['boxes'].reshape(raw['boxes'].shape[0], 4, 8, 84)
    for side, d in enumerate(dist.tolist()):
        lo, frac = int(d), d-int(d)
        view[bi, side, :, anchor] = -40.
        view[bi, side, lo, anchor] = math.log(1-frac)
        if frac:
            view[bi, side, lo+1, anchor] = math.log(frac)


def raw(anchor=27, box=(20., 20., 36., 36.), grad=False):
    result = dict(scores=torch.full((1, 2, 84), -20.), boxes=torch.full((1, 32, 84), -40.),
                  feats=[torch.zeros(1, 2, x, x) for x in (8, 4, 2)])
    result['boxes'].reshape(1, 4, 8, 84)[:, :, 1, :] = 0.
    set_box(result, anchor, box)
    result['scores'][0, 0, anchor] = 4.
    for key in ('scores', 'boxes'):
        result[key].requires_grad_(grad)
    return result


def fixture():
    batch = labels()
    batch['teacher_batch'] = labels()
    return raw(grad=True), raw(18, (16.,16.,40.,40.), True), raw(grad=True), batch


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_transform_and_unclipped_coordinates(self):
        ir = torch.tensor([[10.,20.,30.,60.]])
        rgb = torch.tensor([[20.,40.,60.,100.]])
        pred = torch.tensor([[8.,24.,34.,56.]])
        result = l2.map_teacher_box(pred, ir, rgb)
        self.assertTrue(torch.allclose(result, torch.tensor([[16.,46.,68.,94.]])))
        self.assertTrue(torch.allclose(l2.object_relative(result,rgb), l2.object_relative(pred,ir)))

    def test_native_decode_exact_and_direct_gradient(self):
        s, t, r, b = fixture()
        original, native = l2._helpers()
        c, st, _, _ = l2._layout(s, original, (8,16,32))
        selected = l2.decode_selected(s, torch.tensor([0]), torch.tensor([27]), c, st)
        self.assertTrue(torch.equal(selected.detach(),original._decode_boxes(s,c,st)[0,27:28]))
        loss, stats = l2.compute(s,t,r,b)
        loss.backward()
        self.assertEqual(stats['selected_anchors'], [(0,27,18,0,0)])
        self.assertEqual(stats['base_count'],1)
        self.assertGreater(float(s['boxes'].grad.abs().sum()),0.)
        self.assertEqual(torch.nonzero(s['boxes'].grad.abs().sum((0,1))).flatten().tolist(),[27])
        for tensor in (s['scores'], t['scores'], t['boxes'], r['scores'],r['boxes']):
            self.assertIsNone(tensor.grad)
        json.dumps(stats, allow_nan=False)

    def test_same_mask_gt_control_and_known_loss(self):
        s,t,r,b=fixture()
        a, sa=l2.compute(s,t,r,b,variant='L2-box')
        g, sg=l2.compute(s,t,r,b,variant='L2-GT')
        self.assertTrue(torch.equal(a,g))
        self.assertAlmostEqual(float(a), 1/6-.05, places=6)
        for k in ('base_records','selected_anchors','selected_object_ids','normalizer','base_count'):
            self.assertEqual(sa[k],sg[k])

    def test_nonperfect_teacher_has_distinct_gt_target_same_set(self):
        s,t,r,b=fixture()
        t=raw(18,(18.,18.,40.,40.))
        a,sa=l2.compute(s,t,r,b,variant='L2-box')
        g,sg=l2.compute(s,t,r,b,variant='L2-GT')
        self.assertNotEqual(float(a),float(g))
        self.assertEqual(sa['base_records'],sg['base_records'])

    def test_student_values_do_not_change_fixed_selection(self):
        s,t,r,b=fixture()
        _,a=l2.compute(s,t,r,b)
        _,z=l2.compute(raw(box=(18.,18.,39.,39.),grad=True),t,r,b)
        self.assertEqual(a['base_records'],z['base_records'])

    def test_reference_coarse_denominator_precedes_reliability(self):
        s,t,r,b=fixture()
        one,_=l2.compute(s,t,r,b)
        for v in (s,t,r):
            for key in ('scores','boxes'):
                v[key]=v[key].detach().repeat(2,1,1).requires_grad_(key=='boxes' and v is s)
            v['feats']=[x.repeat(2,1,1,1) for x in v['feats']]
        r['scores'][1,0,27]=math.log(.1/.9)
        b=labels(((16.,16.,40.,40.),)*2,indices=[0,1]);b['teacher_batch']=copy.deepcopy(b)
        two,stats=l2.compute(s,t,r,b)
        self.assertEqual((stats['base_count'],stats['selected_count'],stats['normalizer']),(2,1,2))
        self.assertAlmostEqual(float(two),float(one)/2)

    def test_all_rgb_gt_ambiguity_rejected(self):
        s,t,r,b=fixture()
        b=labels(((16.,16.,40.,40.),)*2, classes=[0,1]);b['teacher_batch']=labels()
        loss,stats=l2.compute(s,t,r,b)
        self.assertEqual(stats['base_count'],0)
        loss.backward();self.assertTrue(s['boxes'].grad.eq(0).all())

    def test_all_ir_gt_ambiguity_rejected_without_shrinking_base(self):
        s,t,r,b=fixture()
        b['teacher_batch']=labels(((16.,16.,40.,40.),)*2, classes=[0,1])
        _,stats=l2.compute(s,t,r,b)
        self.assertEqual((stats['base_count'],stats['selected_count']),(1,0))

    def test_match_and_label_pair_gate(self):
        s,t,r,b=fixture();b['teacher_batch']=labels(((20.,16.,44.,40.),))
        _,stats=l2.compute(s,t,r,b)
        self.assertEqual((stats['common_count'],stats['pair_iou_count'],stats['base_count']),(1,0,0))

    def test_R_support_preserved(self):
        s,t,r,b=fixture()
        # One-bin-width support cannot fit this GT at any native anchor.
        for v in (s,t,r):
            v['boxes']=torch.zeros(1,8,84,requires_grad=v is s)
        _,stats=l2.compute(s,t,r,b)
        self.assertEqual(stats['base_count'],0)

    def test_bad_labels_and_nonfinite_fail_closed(self):
        s,t,r,b=fixture();b['bboxes'][0,2]=0.
        with self.assertRaises(ValueError):l2.compute(s,t,r,b)
        s,t,r,b=fixture();t['boxes']=t['boxes'].detach();t['boxes'][0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):l2.compute(s,t,r,b)
        with self.assertRaises(ValueError):l2.object_relative(torch.tensor([[0.,0.,0.,1.]]),torch.tensor([[0.,0.,1.,1.]]))

    def test_empty_and_unsupported_variant(self):
        s,t,r,b=fixture();b=labels(());b['teacher_batch']=labels(())
        loss,stats=l2.compute(s,t,r,b);loss.backward()
        self.assertEqual(float(loss),0.);self.assertEqual(stats['normalizer'],1)
        self.assertTrue(s['boxes'].grad.eq(0).all())
        with self.assertRaises(ValueError):l2.compute(s,t,r,b,variant='L1')
        self.assertFalse(stats['L1_geometry_admitted'])

    def test_deterministic_T_iou_then_conf_tie(self):
        s,t,r,b=fixture();t=raw(18,(16.,16.,40.,40.))
        set_box(t,27,(16.,16.,40.,40.));t['scores'][0,0,27]=4.
        _,stats=l2.compute(s,t,r,b)
        self.assertEqual(stats['selected_anchors'][0][2],18)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    source=HERE/'localization_box_v2.py';st=source.stat()
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),known_truth_only=True,
        new_AP_read=False,gpu_or_ssh=False,new_hash_computed=False,
        source=dict(path=str(source),bytes=st.st_size,mtime_ns=st.st_mtime_ns),
        pinned=str(args.pinned),scope='CPU_OPERATOR_IMPLEMENTATION_ONLY',L1_admitted=False)
    with args.receipt.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
