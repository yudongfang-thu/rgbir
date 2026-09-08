import argparse
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import analyze_dev_cache as a
from native_cached_match import load_native,match_row,call_with_pairs


def row():
    return dict(image='frame',canvas_shape=[20,20],original_shape=[20,20],gt_boxes=[[0.,0.,10.,10.]],gt_classes=[0.],
        pred_boxes=[[0.,0.,8.,10.],[0.,0.,9.,10.],[0.,0.,10.,10.]],pred_classes=[0.,0.,0.],pred_confidence=[.1,.8,.25])


class Truths(unittest.TestCase):
    def test_original_prediction_order_and_confidence_boundary(self):
        r=row();original=copy.deepcopy(r)
        low=match_row(r,None,.5,NATIVE);high=match_row(r,.25,.5,NATIVE)
        self.assertEqual(low['gt_matches'][0]['prediction_id'],0)
        self.assertEqual(high['gt_matches'][0]['prediction_id'],1)
        self.assertEqual(high['kept_prediction_ids'],[1]);self.assertEqual(r,original)
    def test_duplicate_predictions_and_overlapping_gt(self):
        r=row();r['gt_boxes']*=2;r['gt_classes']*=2
        m=match_row(r,None,.5,NATIVE)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,2,1))
        self.assertEqual(len(set(m['prediction_matches'].values())),m['tp'])
    def test_wrong_class_empty_gt_and_empty_predictions(self):
        for what in ('class','gt','pred'):
            r=row()
            if what=='class':r['gt_classes']=[1.]
            if what=='gt':r['gt_boxes']=[];r['gt_classes']=[]
            if what=='pred':r['pred_boxes']=[];r['pred_classes']=[];r['pred_confidence']=[]
            self.assertEqual(match_row(r,None,.5,NATIVE)['tp'],0)
    def test_native_independent_iou_can_change_gt_nonmonotonically(self):
        V,B,_=NATIVE;matrix=torch.tensor([[.6,.6,.4],[.6,.4,.9],[.4,.9,.6]])
        class MatrixValidator(V):
            def _process_batch(self,pred,gt):
                return {'tp':self.match_predictions(pred['cls'],gt['cls'],matrix).numpy()}
        pred=gt={'cls':torch.zeros(3)};results=[];prior=sys.getprofile()
        for threshold in (.5,.75):
            v=object.__new__(MatrixValidator);v.iouv=torch.tensor([threshold]);v.niou=1
            pairs,tp=call_with_pairs(v,B,pred,gt);results.append(set(pairs[:,0]))
        self.assertEqual(results,[{0,2},{1,2}]);self.assertIs(sys.getprofile(),prior)
    def test_coordinate_checks_allow_native_outside_canvas(self):
        r=row();r['pred_boxes'][0]=[-1.,-1.,8.,10.]
        self.assertEqual(a.validate_row(r),1)
        for what in ('nan','inverted','gtoutside'):
            q=copy.deepcopy(r)
            if what=='nan':q['pred_boxes'][0][0]=float('nan')
            if what=='inverted':q['pred_boxes'][0][2]=-2.
            if what=='gtoutside':q['gt_boxes'][0][0]=-1.
            with self.assertRaises(ValueError):a.validate_row(q)
    def test_joint_and_bucket_known_truth(self):
        records=[];metrics={g:{n:dict(tp=0,fp=0,fn=0,predictions=0) for n in ('N','T')} for g in a.GROUPS}
        for i,(n,t) in enumerate(((True,True),(False,True),(True,False),(False,False))):
            conditions={}
            for g in a.GROUPS:
                conditions[g]=dict(N=dict(correct=n),T=dict(correct=t),bucket=a.bucket(n,t))
                for name,ok in (('N',n),('T',t)):
                    metrics[g][name]['tp']+=ok;metrics[g][name]['fn']+=not ok
            records.append(dict(pair_key='p'+str(i%2),groups=conditions))
        v=a.build_summary(records,metrics,2,{})
        for g in v['groups'].values():self.assertEqual([g['buckets'][b]['objects'] for b in a.BUCKETS],[1,1,1,1])
        self.assertEqual(sum(x['objects'] for x in v['gt025_joint_iou50_iou75']),4)
        self.assertEqual(sum(x['objects'] for x in v['low_to_gt025_transitions']['iou50']),4)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native-source-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();NATIVE=load_native(args.native_source_dir);torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    with args.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),GPU_used=False,checkpoint_loaded=False,
        new_hash_computed=False,real_full_dev_analysis_run=False,native_source=str(args.native_source_dir)),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
