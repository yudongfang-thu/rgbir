"""Portable CPU truths for the frozen readout, no real model or experiment raw."""
import argparse,json,unittest
from pathlib import Path
import torch
from dfl_export import anchor_geometry,distances,build_requests
from run_dfl_probe import setup_with_validated_amp
from types import SimpleNamespace

def obj(selected=False):
    return dict(image_index=0,frame_id='f',stable_rgb_gt_id='rgb0',stable_ir_gt_id='ir0',bucket='both05_onlyT075',
        rgb_gt_xyxy=[0,0,10,10],ir_gt_xyxy=[0,0,10,10],C_selected=True,
        historical_L2_record=dict(reference_anchor=10,teacher_anchor=30),historical_L2_gates=dict(selected=selected),
        native_iou50_matches={'S':dict(anchor_index=10),'R':dict(anchor_index=10),'T':dict(anchor_index=20)})

class Truths(unittest.TestCase):
    def test_amp_bypass_restored_success_and_failure(self):
        original=lambda model:False
        module=SimpleNamespace(check_amp=original)
        trainer=SimpleNamespace(model=object(),amp=None)
        def setup():trainer.amp=module.check_amp(trainer.model)
        trainer._setup_train=setup
        self.assertTrue(setup_with_validated_amp(trainer,True,True,module)['original_binding_restored'])
        self.assertIs(module.check_amp,original)
        def failed():module.check_amp(trainer.model);raise RuntimeError('setup failure')
        trainer._setup_train=failed
        with self.assertRaises(RuntimeError):setup_with_validated_amp(trainer,True,True,module)
        self.assertIs(module.check_amp,original)
        with self.assertRaises(ValueError):setup_with_validated_amp(trainer,True,False,module)
    def test_same_mean_different_distribution(self):
        x=torch.zeros(16);y=torch.full((16,),-8.);y[7:9]=8.
        p,q=x.softmax(0),y.softmax(0);bins=torch.arange(16).float()
        self.assertAlmostEqual(float((p*bins).sum()),float((q*bins).sum()),places=5)
        self.assertFalse(torch.equal(p,q));self.assertGreater(float(-(p*p.log()).sum()),float(-(q*q.log()).sum()))
    def test_layout_boundaries(self):
        feats=[[32,64,80,80],[32,128,40,40],[32,256,20,20]]
        self.assertEqual(anchor_geometry(6399,feats,[8,16,32])['level'],0)
        self.assertEqual(anchor_geometry(6400,feats,[8,16,32])['center_xy'],[8.,8.])
        self.assertEqual(anchor_geometry(8000,feats,[8,16,32])['level'],2)
        with self.assertRaises(ValueError):anchor_geometry(8400,feats,[8,16,32])
    def test_unclamped_support(self):
        x=distances([0,0,240,240],[0,0],16)
        self.assertEqual(x['unclamped_gt_distance_bins'],[0.,0.,15.,15.]);self.assertEqual(x['gt_distance_in_range_per_side'],[True,True,False,False])
        self.assertLess(distances([2,0,4,4],[0,0],1)['unclamped_gt_distance_bins'][0],0)
    def test_roles_deduplicate_but_do_not_fill_missing(self):
        o=obj();objects,keys=build_requests([o]);self.assertEqual(len(keys),4)
        self.assertIsNone(objects[0]['roles']['T']['historical_selected_T'])
        o['historical_L2_record']=None;o['native_iou50_matches']['S']=None
        objects,keys=build_requests([o]);self.assertIsNone(objects[0]['roles']['S']['own_native_iou50']);self.assertIsNone(objects[0]['roles']['T']['same_historical_R_index'])
    def test_selected_teacher_and_empty(self):
        objects,keys=build_requests([obj(True)]);self.assertIn(('T',0,30),keys)
        self.assertEqual(build_requests([]),([],[]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),GPU_used=False,real_model_forward=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not r.wasSuccessful())
