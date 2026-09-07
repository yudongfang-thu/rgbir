"""CPU mathematical/selector regression tests, not a real-training receipt."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection_adapter import (original, EvidenceConfig, build_classification_selection,
                               pool_relative_logits)
from classification_logit import (classification_relative_kd, classification_logit_loss,
                                  classification_loss_components, classification_loss_from_selection)


def labels(boxes=((16., 16., 40., 40.),), classes=None, indices=None):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {"bboxes": torch.cat(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]), -1)/64,
            "batch_idx": torch.tensor(indices if indices is not None else [0]*len(boxes), dtype=torch.long),
            "cls": torch.tensor(classes if classes is not None else [0]*len(boxes), dtype=torch.long).reshape(-1, 1)}


def evidence_raw(foreground, own_labels=None, classes=2, batch_size=1, requires_grad=True):
    own_labels = labels() if own_labels is None else own_labels
    n = 64 + 16 + 4
    raw = {"scores": torch.zeros(batch_size, classes, n),
           "boxes": torch.full((batch_size, 32, n), -40.),
           "feats": [torch.zeros(batch_size, 2, h, h) for h in (8, 4, 2)]}
    raw["scores"][:, 1:] = -5.
    dfl = raw["boxes"].reshape(batch_size, 4, 8, n)
    dfl[:, :, 1] = 0.
    dfl[:, :, :, 27] = -40.
    dfl[:, :, 1:3, 27] = math.log(.5)
    cfg = EvidenceConfig(input_size=64)
    centers, _, _, _ = original._layout(raw, cfg, (8, 16, 32))
    for bi, cls, box in zip(own_labels["batch_idx"], own_labels["cls"].flatten(), own_labels["bboxes"]):
        corners = torch.cat((box[:2]-box[2:]/2, box[:2]+box[2:]/2))*64
        inside = original._inside(corners[None], centers)[0]
        raw["scores"][int(bi), int(cls), inside] = foreground
    raw["scores"].requires_grad_(requires_grad)
    raw["boxes"].requires_grad_(requires_grad)
    return raw


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cfg = EvidenceConfig(input_size=64)
        self.batch = labels()
        self.batch["teacher_batch"] = labels()

    def assertClose(self, a, b, atol=1e-6):
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=1e-6), (a, b))

    def sample(self, classes=5):
        generator = torch.Generator().manual_seed(17)
        s = torch.randn(3, 2, classes, generator=generator, requires_grad=True)
        t = torch.randn(3, 2, classes, generator=generator, requires_grad=True)
        valid = torch.tensor([[True, True], [True, False], [True, True]])
        selected = torch.tensor([True, False, True])
        y = torch.tensor([0, 0, 0], dtype=torch.long)
        return s, t, valid, selected, y

    def test_c0_loss_gradient_and_complete_stats_exact(self):
        s, t, r = evidence_raw(0), evidence_raw(4), evidence_raw(1)
        old_s = deepcopy(s)
        before = torch.get_rng_state().clone()
        old_loss, old_stats = original.object_evidence_loss(old_s, t, r, self.batch, config=self.cfg)
        selection = build_classification_selection(s, t, r, self.batch, config=self.cfg)
        self.assertTrue(torch.equal(old_loss, selection.c0_loss))
        self.assertEqual(old_stats, selection.c0_stats)
        old_loss.backward()
        selection.c0_loss.backward()
        self.assertTrue(torch.equal(s["scores"].grad, old_s["scores"].grad))
        self.assertIsNone(s["boxes"].grad)
        self.assertIsNone(t["scores"].grad)
        self.assertIsNone(r["scores"].grad)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(selection.student_delta.shape, (1, 2, 2))

    def test_matches_compact_base_and_selected_mapping(self):
        # First matched object has no foreground center, middle one is valid,
        # third RGB object is unmatched. M must be one, not 2 or 3.
        b = labels(((1.,1.,2.,2.), (16.,16.,40.,40.), (45.,45.,60.,60.)), classes=[0,0,1])
        b["teacher_batch"] = labels(((1.,1.,2.,2.), (16.,16.,40.,40.)), classes=[0,0])
        s, t, r = evidence_raw(0,b), evidence_raw(4,b["teacher_batch"]), evidence_raw(1,b)
        sel = build_classification_selection(s,t,r,b,config=self.cfg)
        old, stats = original.object_evidence_loss(s,t,r,b,config=self.cfg)
        self.assertEqual(sel.base_count, 1)
        self.assertEqual(sel.base_to_matched.tolist(), [1])
        self.assertEqual(sel.matched_to_base.tolist(), [-1,0])
        self.assertEqual(sel.selected_matched_indices.tolist(), [1])
        self.assertEqual(sel.base_object_ids, ((0,1,1),))
        self.assertEqual(sel.c0_stats, stats)
        self.assertTrue(torch.equal(old,sel.c0_loss))

    def test_base_includes_teacher_rejected_object(self):
        # R's full box candidate qualifies for both same-class RGB boxes;
        # teacher only sees a confidently correct candidate for the first.
        b = labels(((16.,16.,40.,40.), (40.,16.,64.,40.)))
        b["teacher_batch"] = deepcopy(b)
        s,t,r = evidence_raw(0,b), evidence_raw(4,labels()), evidence_raw(1,b)
        with torch.no_grad():
            # Explicit R candidates for both objects, but T only first.
            for raw in (s,t,r):
                raw["boxes"].reshape(1,4,8,-1)[:,:,:,30] = -40
                raw["boxes"].reshape(1,4,8,-1)[:,:,1:3,30] = math.log(.5)
            r["scores"][0,0,30] = 2.
            t["scores"][0,0,t["scores"][0,0] == 0] = -20
        sel = build_classification_selection(s,t,r,b,config=self.cfg)
        old,stats = original.object_evidence_loss(s,t,r,b,config=self.cfg)
        self.assertEqual(sel.c0_stats,stats)
        self.assertTrue(torch.equal(old,sel.c0_loss))
        self.assertEqual(sel.base_count,stats["base_count"])
        self.assertEqual(sel.base_count,2)
        self.assertEqual(stats["teacher_correct_base_count"],1)
        self.assertEqual(int(sel.selected.sum()),1)

    def test_exact_quality_tie_keeps_original_gt_order(self):
        b=labels(((16.,16.,40.,40.),(40.,16.,64.,40.)))
        b["teacher_batch"]=deepcopy(b)
        s,t,r=evidence_raw(0,b),evidence_raw(4,b),evidence_raw(1,b)
        sel=build_classification_selection(s,t,r,b,config=self.cfg)
        self.assertTrue(torch.equal(sel.quality[:1],sel.quality[1:]))
        self.assertEqual(sel.c0_stats["eligible_count"],2)
        self.assertEqual(sel.selected_matched_indices.tolist(),[0])

    def test_one_valid_level_preserves_original_selection(self):
        b=labels(((16.,16.,26.,26.),))
        b["teacher_batch"]=deepcopy(b)
        s,t,r=evidence_raw(0,b),evidence_raw(4,b),evidence_raw(1,b)
        with torch.no_grad():
            dfl=t["boxes"].reshape(1,4,8,-1)
            dfl[:,:,:,18]=-40
            dfl[:,:,0,18]=math.log(.375)
            dfl[:,:,1,18]=math.log(.625)
        sel=build_classification_selection(s,t,r,b,config=self.cfg)
        old,stats=original.object_evidence_loss(s,t,r,b,config=self.cfg)
        self.assertEqual(sel.valid_levels.tolist(),[[True,False]])
        self.assertEqual(sel.c0_stats,stats)
        self.assertTrue(torch.equal(old,sel.c0_loss))

    def test_student_changes_cannot_change_selection(self):
        s,t,r = evidence_raw(0),evidence_raw(4),evidence_raw(1)
        sel = build_classification_selection(s,t,r,self.batch,config=self.cfg)
        with torch.no_grad():
            s["scores"].mul_(-10).add_(70)
            s["boxes"].mul_(-.1)
        changed = build_classification_selection(s,t,r,self.batch,config=self.cfg)
        for field in ("base_to_matched", "matched_to_base", "selected", "quality", "valid_levels"):
            self.assertTrue(torch.equal(getattr(sel,field),getattr(changed,field)),field)

    def test_independent_teacher_regions_and_background_all_gt(self):
        b=labels()
        b["teacher_batch"]=labels(((22.,16.,46.,40.),(46.,20.,60.,36.)), classes=[0,1])
        sel=build_classification_selection(evidence_raw(0,b),evidence_raw(4,b["teacher_batch"]),evidence_raw(1,b),b,config=self.cfg)
        self.assertEqual(sel.base_count,1)
        region=sel.regions[0]
        self.assertFalse(torch.equal(region.rgb_foreground,region.teacher_foreground))
        raw=evidence_raw(0)
        centers,_,_,_=original._layout(raw,self.cfg,(8,16,32))
        _,_,tboxes=original._labels(b["teacher_batch"],1,2,centers.device,(64,64))
        all_t=original._inside(tboxes,centers[region.anchor_start:region.anchor_end]).any(0)
        self.assertFalse(bool((region.teacher_background & all_t[None]).any()))

    def test_no_matched_object_and_zero_selected_attach_score_gradient(self):
        for empty in (True,False):
            b=labels(()) if empty else labels()
            b["teacher_batch"]=labels(()) if empty else labels()
            s,t,r=evidence_raw(0),evidence_raw(0),evidence_raw(1)
            sel=build_classification_selection(s,t,r,b,config=self.cfg)
            loss,stats=classification_loss_from_selection(sel)
            loss.backward()
            self.assertEqual(float(loss),0)
            self.assertEqual(stats["selected_count"],0)
            self.assertTrue(torch.equal(s["scores"].grad,torch.zeros_like(s["scores"])))

    def test_pool_shift_invariance_and_both_region_gradients(self):
        z=torch.arange(18,dtype=torch.float32).reshape(3,6).requires_grad_()
        fg=torch.tensor([[True,True,False,False,False,False]])
        bg=~fg
        a,v=pool_relative_logits(z,fg,bg)
        b,_=pool_relative_logits(z+torch.tensor([[10.],[-3.],[.75]]),fg,bg)
        self.assertClose(a,b)
        a.sum().backward()
        self.assertTrue(bool((z.grad[:,:2]>0).all()))
        self.assertTrue(bool((z.grad[:,2:]<0).all()))

    def test_invalid_and_empty_pool_regions(self):
        z=torch.randn(3,6,requires_grad=True)
        fg=torch.zeros(1,6,dtype=torch.bool)
        out,v=pool_relative_logits(z,fg,~fg)
        self.assertFalse(bool(v.any()))
        out.sum().backward()
        self.assertTrue(bool(z.grad.eq(0).all()))
        out,v=pool_relative_logits(z,fg[:0],(~fg)[:0])
        self.assertEqual(out.shape,(0,3))

    def test_binary_kl_equal_zero_and_teacher_detached(self):
        s,t,v,k,y=self.sample()
        self.assertLess(abs(float(classification_relative_kd(s,s.detach(),v,k,y))),1e-6)
        loss=classification_relative_kd(s,t,v,k,y)
        loss.backward()
        self.assertGreater(float(s.grad.abs().sum()),0)
        self.assertIsNone(t.grad)
        self.assertTrue(bool(s.grad[1].eq(0).all()))

    def test_analytic_temperature_and_kl_direction(self):
        s=torch.tensor([[[2.]]],requires_grad=True)
        t=torch.tensor([[[0.]]])
        loss=classification_relative_kd(s,t,torch.tensor([[True]]),torch.tensor([True]),torch.tensor([0]))
        expected=4*(.5*math.log(.5)-.5*F.logsigmoid(s/2)+.5*math.log(.5)-.5*F.logsigmoid(-s/2)).sum()
        self.assertClose(loss,expected)
        loss.backward()
        self.assertClose(s.grad,2*(torch.sigmoid(s.detach()/2)-.5))

    def test_target_coefficient_offtarget_average_not_class_sum(self):
        s=torch.zeros(1,1,5,requires_grad=True)
        t=torch.ones_like(s)
        v,k,y=torch.tensor([[True]]),torch.tensor([True]),torch.tensor([0])
        full=classification_relative_kd(s,t,v,k,y)
        target=classification_relative_kd(s,t,v,k,y,off_target_weight=0)
        self.assertClose(full,target*1.25)

    def test_y_only_offtarget_teacher_intervention(self):
        s,t,v,k,y=self.sample()
        changed=t.detach().clone(); changed[:,:,1:]+=4
        a=classification_relative_kd(s,t,v,k,y)
        b=classification_relative_kd(s,changed,v,k,y)
        ay=classification_relative_kd(s,t,v,k,y,off_target_weight=0)
        by=classification_relative_kd(s,changed,v,k,y,off_target_weight=0)
        self.assertNotEqual(float(a),float(b))
        self.assertTrue(torch.equal(ay,by))
        ay.backward()
        self.assertTrue(bool(s.grad[:,:,1:].eq(0).all()))

    def test_single_class_c1_is_c1_y(self):
        s,t,v,k,y=self.sample(1)
        a=classification_relative_kd(s,t,v,k,y)
        b=classification_relative_kd(s,t,v,k,y,off_target_weight=0)
        self.assertTrue(torch.equal(a,b))
        a.backward(); self.assertGreater(float(s.grad.abs().sum()),0)

    def test_preteacher_denominator(self):
        s,t,v,k,y=self.sample()
        a=classification_relative_kd(s,t,v,k,y)
        b=classification_relative_kd(torch.cat([s,s[:1]]),torch.cat([t,t[:1]]),
            torch.cat([v,v[:1]]),torch.cat([k,torch.tensor([False])]),torch.cat([y,y[:1]]))
        self.assertClose(b,a*3/4)

    def test_scale_divergence_precedes_scale_average(self):
        s=torch.tensor([[[4.],[-4.]]],requires_grad=True)
        t=torch.zeros_like(s)
        v,k,y=torch.tensor([[True,True]]),torch.tensor([True]),torch.tensor([0])
        full=classification_relative_kd(s,t,v,k,y)
        averaged=classification_relative_kd(s.mean(1,keepdim=True),t.mean(1,keepdim=True),v[:,:1],k,y)
        self.assertGreater(float(full),1)
        self.assertEqual(float(averaged),0)

    def test_invalid_scale_is_ignored_in_gradient_and_normalization(self):
        s=torch.tensor([[[4.],[100.]]],requires_grad=True)
        t=torch.tensor([[[1.],[-100.]]])
        v,k,y=torch.tensor([[True,False]]),torch.tensor([True]),torch.tensor([0])
        full=classification_relative_kd(s,t,v,k,y)
        one=classification_relative_kd(s[:,:1],t[:,:1],v[:,:1],k,y)
        self.assertClose(full,one)
        full.backward(); self.assertEqual(float(s.grad[0,1,0]),0)

    def test_teacher_clip_before_temperature_student_not_clipped(self):
        s=torch.tensor([[[-1000.],[1000.]]],requires_grad=True)
        t=torch.tensor([[[1000.],[-1000.]]],requires_grad=True)
        v,k,y=torch.tensor([[True,True]]),torch.tensor([True]),torch.tensor([0])
        full=classification_relative_kd(s,t,v,k,y)
        clipped=classification_relative_kd(s,t.detach().clamp(-16,16),v,k,y)
        self.assertTrue(torch.equal(full,clipped))
        self.assertTrue(bool(torch.isfinite(full)))
        full.backward()
        self.assertTrue(bool(torch.isfinite(s.grad).all()))
        self.assertTrue(bool(s.grad.abs().gt(.9).all()))
        self.assertIsNone(t.grad)

    def test_raw_gt_channel_pool_matches_legacy_per_scale(self):
        s,t,r=evidence_raw(.3),evidence_raw(4),evidence_raw(1)
        sel=build_classification_selection(s,t,r,self.batch,config=self.cfg)
        centers,_,ranges,size=original._layout(s,self.cfg,(8,16,32))
        _,classes,boxes=original._labels(self.batch,1,2,s["scores"].device,size)
        expected,_=original._evidence(s["scores"][0],classes,boxes,boxes,centers,ranges,self.cfg)
        actual=sel.student_delta.gather(-1,sel.labels[:,None,None].expand(-1,2,1)).squeeze(-1)/2
        self.assertTrue(torch.equal(actual,expected))

    def test_cache_rejects_another_batch_and_mutated_source(self):
        s,t,r=evidence_raw(0),evidence_raw(4),evidence_raw(1)
        sel=build_classification_selection(s,t,r,self.batch,config=self.cfg)
        classification_logit_loss(s,t,r,self.batch,selection=sel)
        with self.assertRaises(ValueError):
            classification_logit_loss(s,t,r,deepcopy(self.batch),selection=sel)
        with self.assertRaises(ValueError):
            classification_logit_loss(s,t,r,self.batch,selection=sel,config=replace(self.cfg,rho=.9))
        with torch.no_grad(): s["scores"].add_(1)
        with self.assertRaises(ValueError):
            classification_logit_loss(s,t,r,self.batch,selection=sel)

    def test_nonzero_c1_and_components_json_serializable(self):
        s,t,r=evidence_raw(0),evidence_raw(4),evidence_raw(1)
        sel=build_classification_selection(s,t,r,self.batch,config=self.cfg)
        components=classification_loss_components(sel)
        loss,stats=classification_loss_from_selection(sel,return_records=True)
        self.assertTrue(torch.equal(loss,components["target_loss"]+components["off_target_loss"]))
        self.assertGreater(float(loss),0)
        json.dumps(stats,allow_nan=False)
        loss.backward()
        self.assertIsNone(t["scores"].grad)
        self.assertIsNone(r["scores"].grad)

    def test_y_only_statistics_exclude_unused_teacher_channels(self):
        s,t,r=evidence_raw(0),evidence_raw(4),evidence_raw(1)
        sel=build_classification_selection(s,t,r,self.batch,config=self.cfg)
        delta=sel.teacher_delta.clone()
        delta[:,:,0]=1
        delta[:,:,1]=100
        sel=replace(sel,teacher_delta=delta)
        _,full=classification_loss_from_selection(sel)
        _,target=classification_loss_from_selection(sel,off_target_weight=0)
        self.assertEqual(full["target_clipped_count"],2)
        self.assertEqual(target["target_clipped_count"],0)
        self.assertEqual(target["target_clipping_population"],2)
        self.assertEqual(target["supervised_class_channels"],1)

    def test_overlap_statistics_only_use_selected_valid_levels(self):
        s,t,r=evidence_raw(0),evidence_raw(4),evidence_raw(1)
        sel=build_classification_selection(s,t,r,self.batch,config=self.cfg)
        regions=(replace(sel.regions[0],rgb_other_gt_fraction=torch.tensor([0.])),
                 replace(sel.regions[1],rgb_other_gt_fraction=torch.tensor([1.])))
        sel=replace(sel,valid_levels=torch.tensor([[True,False]]),regions=regions)
        _,stats=classification_loss_from_selection(sel)
        self.assertEqual(stats["rgb_other_gt_foreground_fraction_selected_level_mean"],0)

    def test_deleting_non_target_can_increase_shared_gradient_norm(self):
        # Shared parameter induces opposite target/non-target gradients.
        w=torch.tensor(0.,requires_grad=True)
        s=torch.stack([w,w*8]).reshape(1,1,2)
        t=torch.tensor([[[4.,-4.]]])
        v,k,y=torch.tensor([[True]]),torch.tensor([True]),torch.tensor([0])
        full=classification_relative_kd(s,t,v,k,y,off_target_weight=.1)
        target=classification_relative_kd(s,t,v,k,y,off_target_weight=0)
        gf=torch.autograd.grad(full,w,retain_graph=True)[0]
        gy=torch.autograd.grad(target,w)[0]
        self.assertLess(float(gf.abs()),float(gy.abs()))

    def test_reject_bad_shape_label_mask_and_nonfinite(self):
        s,t,v,k,y=self.sample()
        with self.assertRaises(ValueError): classification_relative_kd(s,t,v,k,y+7)
        with self.assertRaises(ValueError): classification_relative_kd(s,t,v, k.float(),y)
        bad=v.clone();bad[0]=False
        with self.assertRaises(ValueError): classification_relative_kd(s,t,bad,k,y)
        bad=t.detach().clone();bad[0,0,0]=float("nan")
        with self.assertRaises(ValueError): classification_relative_kd(s,bad,v,k,y)
        with self.assertRaises(ValueError): classification_relative_kd(s,t,v,k,y,temperature=0)

    def test_small_actual_batch_does_not_scale_inside_kernel(self):
        b=labels(indices=[1]); b["teacher_batch"]=labels(indices=[1])
        s,t,r=(evidence_raw(value,b if i!=1 else b["teacher_batch"],batch_size=2)
               for i,value in enumerate((0,4,1)))
        sel=build_classification_selection(s,t,r,b,config=self.cfg)
        loss,stats=classification_loss_from_selection(sel)
        self.assertEqual(stats["base_count"],1)
        b1=labels();b1["teacher_batch"]=labels()
        one,_=classification_logit_loss(evidence_raw(0),evidence_raw(4),evidence_raw(1),b1,config=self.cfg)
        self.assertTrue(torch.equal(loss,one))


if __name__ == "__main__":
    unittest.main(verbosity=2)
