"""Synthetic fixed-frame cross-GT truths, independent of experiment results."""
import argparse,copy,json,unittest
from pathlib import Path
import torch
from analyze_cross_gt import CPU,load_native,make_record,cohort,stratum,STRATA,count
IOU=load_native(CPU/'frozen_sources')[2]

def fixture(changed=False):
    ngt=[0,0,10,10];tgt=[2,0,12,10];nbox=[0,0,7.2,10];tbox=[2,0,9,10] if changed else tgt
    own=lambda gt,box:float(IOU(torch.tensor([gt],dtype=torch.float32),torch.tensor([box],dtype=torch.float32))[0,0])
    pair=dict(pair_key='frame',N_gt_id='n::0',T_gt_id='t::0',N_gt_row=0,T_gt_row=0,N_gt_box=ngt,T_gt_box=tgt,gt_class=0,pair_iou=2/3)
    witnesses={n:dict(prediction_id=0,filtered_prediction_id=0,iou=own(gt,box),confidence=.9) for n,gt,box in [('N',ngt,nbox),('T',tgt,tbox)]}
    high=dict(prediction_id=1 if changed else 0,filtered_prediction_id=1 if changed else 0,iou=own(tgt,tgt),confidence=.9)
    pair['groups']={'gt025_iou50':{n:dict(correct=True,witness=witnesses[n]) for n in ('N','T')},'gt025_iou75':dict(N=dict(correct=False,witness=None),T=dict(correct=True,witness=high))}
    preds={}
    for n,box in [('N',nbox),('T',tbox)]:
        preds['frame',n,0]=dict(prediction_id=0,box=box,pred_class=0,confidence=.9,groups={'gt025_iou50':dict(included=True,gt_id=pair[n+'_gt_id']), 'gt025_iou75':dict(included=True,gt_id=pair[n+'_gt_id'] if n=='T' and not changed else None)})
    if changed:preds['frame','T',1]=dict(prediction_id=1,box=tgt,pred_class=0,confidence=.9,groups={'gt025_iou75':dict(included=True,gt_id='t::0')})
    frame=dict(pair_key='frame',canvas_shape=[544,672],original_shape=[512,640])
    return pair,frame,preds

class Truths(unittest.TestCase):
    def test_gt_offset_reverses_teacher_own_advantage(self):
        r=make_record(*fixture(),IOU)
        self.assertGreater(r['delta_T_own_minus_N_own'],0)
        self.assertLess(r['delta_T_minus_N_on_RGB_GT'],-.05)
        c=count([r]);self.assertEqual(c['teacher_own_positive_but_cross_nonpositive'],1)
        self.assertEqual(c['T_on_RGB_ge075'],0)
    def test_iou75_change_does_not_replace_fixed_box(self):
        r=make_record(*fixture(True),IOU)
        self.assertEqual(r['T']['prediction_id'],0);self.assertEqual(r['T']['native_iou75_match']['prediction_id'],1)
        self.assertEqual(r['T']['iou75_identity_status'],'changed')
        self.assertLess(r['T_fixed_to_IR_GT_iou'],.75)
        self.assertEqual(r['T']['native_iou75_match']['iou'],1.)
    def test_fixed_strata_boundary(self):
        self.assertEqual(stratum(.5),STRATA[0]);self.assertEqual(stratum(.7999),STRATA[0]);self.assertEqual(stratum(.8),STRATA[1]);self.assertEqual(stratum(1.),STRATA[1])
        with self.assertRaises(ValueError):stratum(.49)
    def test_reverse_cohort_and_exclusion(self):
        p,_,_=fixture();p['groups']['gt025_iou75']['N']['correct']=True;p['groups']['gt025_iou75']['T']['correct']=False
        self.assertEqual(cohort(p),'N_own75_only')
        p['groups']['gt025_iou50']['T']['correct']=False;self.assertIsNone(cohort(p))
    def test_strict_difference_counts_and_zero(self):
        r=make_record(*fixture(),IOU);values=[]
        for d in [.1,.05,0.,-.05,-.1]:
            x=copy.deepcopy(r);x['delta_T_minus_N_on_RGB_GT']=d;values.append(x)
        c=count(values);self.assertEqual((c['delta_positive'],c['delta_gt005'],c['delta_negative'],c['delta_lt_negative005'],c['delta_zero']),(2,1,2,1,1))
    def test_missing_or_foreign_witness_rejected(self):
        p,f,preds=fixture();preds['frame','T',0]['groups']['gt025_iou50']['gt_id']='another_gt'
        with self.assertRaises(ValueError):make_record(p,f,preds,IOU)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),real_experiment_input_read=False,new_GPU=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not r.wasSuccessful())
