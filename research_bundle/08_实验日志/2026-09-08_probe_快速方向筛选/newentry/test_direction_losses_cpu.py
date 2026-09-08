"""Deterministic CPU-only truths; imports the saved pinned classification code."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PINNED = WORKSPACE/'2026-09-07_train_IndependentKD实施/remote_admission_1532/formal_C1_gpu5_attempt2/runs/C1_seed42/implementation_snapshot'
sys.path[:0] = [str(HERE/'release'), str(PINNED)]
import direction_losses as d
from classification_logit import _classification_terms


def cls_selection(labels=(0, 0, 1), selected=(True, False, True), nc=3):
    m = len(labels)
    gen = torch.Generator().manual_seed(731)
    return NS(student_delta=torch.randn(m, 2, nc, generator=gen, requires_grad=True),
              teacher_delta=torch.randn(m, 2, nc, generator=gen, requires_grad=True),
              labels=torch.tensor(labels, dtype=torch.long), selected=torch.tensor(selected, dtype=torch.bool),
              valid_levels=torch.ones(m, 2, dtype=torch.bool), config=NS(levels=(0, 1)))


def feature_fixture():
    gen = torch.Generator().manual_seed(731)
    # Interleaved GT rows ensure base identities cannot be interpreted as local per-image indices.
    rgb = dict(batch_idx=torch.tensor([1, 0, 0]), cls=torch.tensor([[1.], [0.], [0.]]),
               bboxes=torch.tensor([[.5,.5,.7,.6], [.35,.4,.4,.5], [.7,.6,.35,.45]]))
    ir = dict(batch_idx=torch.tensor([0, 1, 0]), cls=torch.tensor([[0.], [1.], [0.]]),
              bboxes=torch.tensor([[.3,.6,.4,.45], [.6,.5,.6,.7], [.7,.35,.45,.5]]))
    batch = dict(rgb, img=torch.zeros(2, 3, 12, 18), strong_img=torch.zeros(2, 3, 16, 20), teacher_batch=ir)
    s = NS(labels=torch.tensor([0, 0, 1]), selected=torch.tensor([True, True, False]),
           valid_levels=torch.tensor([[True, False], [True, True], [True, True]]),
           config=NS(levels=(0, 1)), base_object_ids=((0, 1, 0), (0, 2, 2), (1, 0, 1)), source_batch=batch)
    student = dict(feats=[torch.randn(2, c, h, w, generator=gen, requires_grad=True)
                         for c,h,w in [(4,6,9), (5,3,5), (6,2,3)]])
    teacher = dict(feats=[torch.randn(2, c, h, w, generator=gen, requires_grad=True)
                         for c,h,w in [(6,8,10), (7,4,5), (8,2,3)]])
    return s, student, teacher, batch


class Truths(unittest.TestCase):
    def test_c2_one_base_class_is_c1_bitwise_and_gradient(self):
        for nc in (1, 5):
            s = cls_selection((0,0,0), (True,False,True), nc)
            c1 = _classification_terms(s.student_delta, s.teacher_delta, s.valid_levels, s.selected, s.labels)['loss']
            c2, stats = d.class_balanced_loss(NS(learning=s))
            self.assertTrue(torch.equal(c1, c2))
            a = torch.autograd.grad(c1, s.student_delta, retain_graph=True)[0]
            b = torch.autograd.grad(c2, s.student_delta)[0]
            self.assertTrue(torch.equal(a,b)); self.assertEqual(stats['present_base_class_count'],1)

    def test_c2_unselected_base_class_and_object_stay_in_denominator(self):
        s = cls_selection((0,0,1), (True,False,False))
        t = _classification_terms(s.student_delta,s.teacher_delta,s.valid_levels,s.selected,s.labels)
        obj0 = (t['target'][0]+.25*t['non_target'][0]).mean()*4
        actual, stats = d.class_balanced_loss(s)
        self.assertTrue(torch.equal(actual,obj0/2/2))
        self.assertEqual(stats['base_class_counts'],[2,1,0])
        actual.backward(); self.assertIsNone(s.teacher_delta.grad)
        self.assertGreater(float(s.student_delta.grad[0].abs().sum()),0)
        self.assertEqual(float(s.student_delta.grad[1:].abs().sum()),0)

    def test_c2_manual_class_mean_mixed_valid_levels(self):
        s = cls_selection(); s.valid_levels[0,1]=False
        t = _classification_terms(s.student_delta,s.teacher_delta,s.valid_levels,s.selected,s.labels)
        obj=(t['target']+.25*t['non_target'])*s.valid_levels
        obj=obj.sum(1)/s.valid_levels.sum(1)*4*s.selected
        expected=(obj[:2].sum()/2+obj[2])/2
        actual,_=d.class_balanced_loss(s)
        torch.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)

    def test_c2_empty_and_no_selected_connected_zero(self):
        for labels,sel in (((),()),((0,1),(False,False))):
            s=cls_selection(labels,sel)
            loss,_=d.class_balanced_loss(s); self.assertEqual(float(loss),0)
            loss.backward(); self.assertIsNotNone(s.student_delta.grad); self.assertIsNone(s.teacher_delta.grad)

    def test_roi_actual_canvas_and_cell_centers(self):
        # Feature pixel (x,y) samples at physical image center (2*x+1,2*y+1).
        yy,xx=torch.meshgrid(torch.arange(6.),torch.arange(9.))
        feat=torch.stack((2*xx+1,2*yy+1))[None]
        boxes=torch.tensor([[3.,2.,15.,10.]])
        tok=d._sample_roi_tokens(feat,boxes,(12,18))[0]
        ex=torch.tensor([5.,9.,13.]).repeat(3)
        ey=torch.tensor([2+8/6,6.,10-8/6]).repeat_interleave(3)
        torch.testing.assert_allclose(tok,torch.stack((ex,ey),1),atol=2e-6,rtol=1e-6)

    def test_multi_roi_grid_equivalent_to_separate_sampling_no_map_repeat(self):
        feat=torch.arange(3*6*9,dtype=torch.float32).reshape(1,3,6,9)
        boxes=torch.tensor([[2.,1.,10.,8.],[6.,4.,16.,11.]])
        original=F.grid_sample; calls=[]
        def record(x,g,**kw):
            calls.append((x.shape,g.shape,kw['align_corners']))
            return original(x,g,**kw)
        with patch.object(d.F,'grid_sample',side_effect=record):
            combined=d._sample_roi_tokens(feat,boxes,(12,18))
        separate=torch.cat([d._sample_roi_tokens(feat,b[None],(12,18)) for b in boxes])
        self.assertTrue(torch.equal(combined,separate))
        self.assertEqual(calls,[(torch.Size([1,3,6,9]),torch.Size([1,6,3,2]),False)])

    def test_relations_invariant_to_channel_orthogonal_mapping(self):
        tokens=torch.randn(2,9,4,generator=torch.Generator().manual_seed(3))
        transformed=tokens[:,:,[2,0,3,1]]*torch.tensor([1.,-1.,1.,-1.])
        torch.testing.assert_allclose(d._relation(tokens),d._relation(transformed),atol=2e-7,rtol=1e-6)
        padded=torch.cat((tokens,torch.zeros(2,9,3)),dim=-1)
        torch.testing.assert_allclose(d._relation(tokens),d._relation(padded),atol=2e-7,rtol=1e-6)

    def test_feature_manual_global_gt_mapping_scale_base_and_teacher_detach(self):
        s,student,teacher,batch=feature_fixture()
        actual,stats=d.feature_relation_loss(NS(learning=s),student,teacher,batch)
        rb=d._pixel_boxes(batch,(12,18),torch.device('cpu'))[2]
        tb=d._pixel_boxes(batch['teacher_batch'],(16,20),torch.device('cpu'))[2]
        mask=~torch.eye(9,dtype=torch.bool); expected=actual.detach()*0
        for row in (0,1):
            bi,rg,tg=s.base_object_ids[row]; losses=[]
            for li in (0,1):
                if not s.valid_levels[row,li]: continue
                a=d._sample_roi_tokens(student['feats'][li][bi:bi+1],rb[rg:rg+1],(12,18))
                b=d._sample_roi_tokens(teacher['feats'][li][bi:bi+1],tb[tg:tg+1],(16,20))
                losses.append(((d._relation(a)-d._relation(b))[:,mask]**2).mean())
            expected=expected+torch.stack(losses).mean()/3
        torch.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
        self.assertEqual(stats['sampled_roi_levels'],3); self.assertEqual(stats['sample_calls'],4)
        actual.backward()
        for feat in teacher['feats']: self.assertIsNone(feat.grad)
        for feat in student['feats'][:2]:
            self.assertGreater(float(feat.grad[0].abs().sum()),0)
            self.assertEqual(float(feat.grad[1].abs().sum()),0)
        self.assertIsNone(student['feats'][2].grad)

    def test_feature_identical_modalities_zero(self):
        s,student,teacher,batch=feature_fixture()
        batch['strong_img']=batch['img'].clone(); batch['teacher_batch']={k:batch[k].clone() for k in ('batch_idx','cls','bboxes')}
        s.base_object_ids=((0,1,1),(0,2,2),(1,0,0))
        teacher={'feats':[f.detach().clone().requires_grad_() for f in student['feats']]}
        loss,_=d.feature_relation_loss(s,student,teacher,batch)
        self.assertEqual(float(loss),0);loss.backward()
        self.assertIsNone(teacher['feats'][0].grad)

    def test_feature_empty_or_unselected_connected_zero(self):
        for empty in (False,True):
            s,student,teacher,batch=feature_fixture(); s.selected[:]=False
            if empty:
                s.labels=s.labels[:0];s.selected=s.selected[:0];s.valid_levels=s.valid_levels[:0]
            loss,stats=d.feature_relation_loss(s,student,teacher,batch)
            self.assertEqual(float(loss),0);self.assertEqual(stats['sample_calls'],0)
            loss.backward(); self.assertIsNotNone(student['feats'][0].grad)
            self.assertEqual(float(student['feats'][0].grad.abs().sum()),0)
            self.assertIsNone(teacher['feats'][0].grad)

    def test_wrong_image_gt_and_selected_invalid_level_rejected(self):
        s,student,teacher,batch=feature_fixture();s.base_object_ids=((0,0,0),)+s.base_object_ids[1:]
        with self.assertRaisesRegex(ValueError,'different image'):d.feature_relation_loss(s,student,teacher,batch)
        s,student,teacher,batch=feature_fixture();s.valid_levels[0]=False
        with self.assertRaisesRegex(ValueError,'lacks a valid level'):d.feature_relation_loss(s,student,teacher,batch)

    def test_inputs_not_mutated(self):
        s,student,teacher,batch=feature_fixture()
        vals=student['feats']+teacher['feats']+[s.labels,s.selected,s.valid_levels,batch['bboxes'],batch['teacher_batch']['bboxes']]
        before=[v.detach().clone() for v in vals]
        d.feature_relation_loss(s,student,teacher,batch)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,vals)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),torch=torch.__version__,device='cpu',
        pinned_classification_source=str(PINNED/'classification_logit.py'),
        new_hash_computed=False,gpu_execution=False,production_admission=False,
        failure_details=[{'test':str(t),'traceback':e} for t,e in result.failures+result.errors])
    with args.output.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,ensure_ascii=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
