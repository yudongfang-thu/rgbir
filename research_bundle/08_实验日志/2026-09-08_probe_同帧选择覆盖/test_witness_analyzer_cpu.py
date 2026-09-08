import argparse
import copy
import json
from pathlib import Path
import unittest
import witness_analyzer as a


def fixture():
    previous=[];current=[]
    for i in range(80):
        old=dict(frame_id='frame'+str(i%32),stable_rgb_gt_id='rgb::'+str(i),stable_ir_gt_id='ir::'+str(i),
            rgb_global_row=i,ir_global_row=i,states={m:dict(correct=not (i in ((0,2) if m in ('S','R') else (1,2)))) for m in ('S','T','R')},
            gates=dict(selected=i<32,matched=True,teacher_correct_own=True))
        new=copy.deepcopy(old);new['detector']={}
        for m in ('S','T','R'):
            w=dict(anchor_index=i,confidence=.7,**{'class':0},box=[0.,0.,10.,10.],iou=.7,other_gt_overlaps=[])
            post=not (i in ((0,2,3) if m in ('S','R') else (1,2)))
            new['detector'][m]=dict(dense_any_correct=True,dense_correct_count=2,dense_witness=copy.deepcopy(w),
                native_pre_nms_any_correct=True,native_pre_nms_correct_count=2,native_pre_nms_witness=copy.deepcopy(w),
                native_postnms_matched=post,native_witness=dict(w,prediction_index=i) if post else None,
                own_gt_scope='ir' if m=='T' else 'rgb',global_gt_row=i,stable_gt_id=old['stable_ir_gt_id' if m=='T' else 'stable_rgb_gt_id'])
        previous.append(old);current.append(new)
    receipt=dict(status=a.SCOPE+'_COMPLETED',scope=a.SCOPE,dataset='llvip',seed=42,batches=1,frames=32,objects_n=80,
        training=0,backward=0,optimizer_updates=0,ema_updates=0,actual_batch_size=32,previous_objects_exact=True,
        first_batch_stream_exact=True,student_full_state_unchanged=True,auxiliary_gradients_absent=True,new_hash_computed=False,
        official_test_accessed=False,full_dev_evaluated=False,formal_paper_gain_claim=False,global_population_coverage_claim=False,
        raw_forward_counts=dict(student=1,teacher=1,reference=1))
    contract=dict(scope=a.SCOPE,objects_n=80,native_tp_and_gt_pairs_exact=True,
        native_profile=dict(conf=.25,iou=.7,max_det=300,agnostic_nms=False,multi_label=True,matching_iou=.5),
        native_match_provenance=dict(native_process_batch='/native/val.py',native_match_predictions='/native/validator.py',captured_matches_exact=True),
        dense_conf_operator='>=',native_conf_operator='>',decode_diagnostics={})
    return previous,current,receipt,contract


class Truths(unittest.TestCase):
    def test_known_confusions_and_native_coverage(self):
        p,c,r,k=fixture();v=a.summarize(p,c,r,k)
        self.assertEqual(v['modalities']['S']['correct_counts'],dict(old_assigned=78,dense_any_correct=80,native_pre_nms_any_correct=80,native_postnms_matched=77))
        bins=v['coverage_by_definition']['native_postnms_matched']
        self.assertEqual((bins['T_correct_S_error']['objects'],bins['T_correct_S_error']['selected']),(2,2))
        self.assertEqual((bins['S_correct_T_error']['objects'],bins['S_correct_T_error']['selected']),(1,1))
        self.assertEqual(bins['both_correct']['objects'],76)
        for model in v['modalities'].values():
            for cm in model['confusion_tables']:self.assertEqual(sum(x['n'] for x in cm['cells']),80)
    def test_exact_source_scope_and_native_profile_reject(self):
        for case in ('source','scope','profile','captured','teacher_gate'):
            p,c,r,k=fixture()
            if case=='source':c[0]['gates']['selected']=False
            if case=='scope':k['scope']='OLD_PROXY'
            if case=='profile':k['native_profile']['multi_label']=False
            if case=='captured':k['native_match_provenance']['captured_matches_exact']=False
            if case=='teacher_gate':p[0]['gates']['teacher_correct_own']=c[0]['gates']['teacher_correct_own']=False
            with self.assertRaises(ValueError):a.summarize(p,c,r,k)
    def test_witness_consistency_native_one_to_one_and_threshold(self):
        for case in ('count','reuse','boundary'):
            p,c,r,k=fixture()
            if case=='count':c[4]['detector']['S']['dense_correct_count']=0
            if case=='reuse':c[36]['detector']['S']['native_witness']['prediction_index']=4
            if case=='boundary':c[4]['detector']['S']['native_witness']['confidence']=.25
            with self.assertRaises(ValueError):a.summarize(p,c,r,k)
        p,c,r,k=fixture();c[4]['detector']['S']['dense_witness']['confidence']=.25
        a.summarize(p,c,r,k)
    def test_unknown_teacher_and_zero_bucket_denominators(self):
        p,c,r,k=fixture()
        for i in range(80):
            for row in (p[i],c[i]):
                row['ir_global_row']=None;row['stable_ir_gt_id']=None;row['states']['T']=None
                row['gates']['matched']=False;row['gates']['teacher_correct_own']=None
            c[i]['detector']['T']=None
        v=a.summarize(p,c,r,k)
        bins=v['coverage_by_definition']['native_postnms_matched']
        self.assertEqual(bins['teacher_unknown']['objects'],80)
        self.assertIsNone(bins['T_correct_S_error']['selected_over_bucket'])
        self.assertEqual(v['modalities']['T']['confusion_tables'][0]['unknown_gt_n'],80)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    with args.receipt.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),GPU_used=False,real_witness_results_read=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
