import argparse
import copy
import json
from pathlib import Path
import unittest
import analyzer as a


def state(correct=True,error='low_confidence'):
    name='correct' if correct else error
    return dict(state=name,candidate=name!='no_candidate',confidence_ok=name not in ('no_candidate','low_confidence'),
        class_ok=name not in ('no_candidate','class_and_localization','class_only'),
        localization_ok=name not in ('no_candidate','class_and_localization','localization_only'),correct=correct)


def fixture():
    frames=[dict(frame_id='f'+str(i),image_index=i,image='/rgb/'+str(i)+'.jpg') for i in range(32)]
    records=[];rg=[];ir=[];base=0
    for i in range(8):
        image_index=i%2;fid='f'+str(image_index);sid='/rgb/'+str(image_index)+'::native_gt:'+str(i)
        iid='/ir/'+str(image_index)+'::native_gt:'+str(i)
        st=dict(S=state(i in (1,6)),R=state(i==6),T=state(i!=1),T_to_RGB=state(i not in (0,1)))
        valid=i!=2;ref=i!=3;teacher=i!=4;q={0:3.,1:4.,2:1.,3:1.,4:1.,5:0.,6:2.,7:0.}[i]
        g=dict(matched=i!=7,valid_levels=[valid,False],region_valid=valid,reference_candidate=ref,
            teacher_correct_own=teacher,quality_q=q,quality_positive=q>0,base=valid and ref,
            eligible=valid and ref and teacher and q>0,selected=i in (0,1),matched_row=i,
            base_row=None,selected_rank={0:2,1:1}.get(i))
        if i==7:
            st['T']=st['T_to_RGB']=None;g={k:(False if k=='matched' else None) for k in g}
        else:
            if g['base']:g['base_row']=base;base+=int(g['base'])
            ir.append(dict(global_gt_row=i,image_index=image_index,stable_gt_id=iid))
        records.append(dict(frame_id=fid,image_index=image_index,image=frames[image_index]['image'],
            rgb_global_row=i,stable_rgb_gt_id=sid,ir_global_row=i if i!=7 else None,stable_ir_gt_id=iid if i!=7 else None,
            states=st,gates=g))
        rg.append(dict(global_gt_row=i,image_index=image_index,stable_gt_id=sid))
    counts=a.selector_counts(records);counts['teacher_gt_count']=len(ir)
    receipt=dict(status=a.SCOPE+'_COMPLETED',scope=a.SCOPE,dataset='llvip',seed=42,frames=32,batches=1,objects_n=len(records),
        training=0,backward=0,optimizer_updates=0,ema_updates=0,new_hash_computed=False,official_test_accessed=False,
        full_dev_evaluated=False,formal_paper_gain_claim=False,global_population_coverage_claim=False,actual_batch_size=32,
        student_full_state_unchanged=True,auxiliary_gradients_absent=True,optimizer_state_empty=True,
        raw_forward_counts=dict(student=1,teacher=1,reference=1),first_batch_stream_exact=True,
        identity_contract=dict(status='STABLE_GT_IDENTITY_VERIFIED',rgb_rows=rg,ir_rows=ir),selector_counts=counts)
    return records,receipt,frames


class Truths(unittest.TestCase):
    def test_known_buckets_and_first_loss(self):
        rows,receipt,frames=fixture();v=a.summarize(rows,receipt,frames)
        self.assertEqual(v['actual_selector_counts']['base_count'],5)
        self.assertEqual(v['actual_selector_counts']['selected_count'],2)
        self.assertEqual(v['empty_frames'],30)
        primary=v['teacher_own_gt_primary'];self.assertEqual(primary['T_correct_S_error']['objects'],5)
        self.assertEqual(primary['S_correct_T_error']['objects'],1)
        self.assertEqual(primary['teacher_unpaired']['objects'],1)
        f=primary['T_correct_S_error']['actual_gate_funnel']
        self.assertEqual([r['lost_n'] for r in f],[0,1,1,0,1,1,0,0])
        self.assertEqual(f[-1]['retained_n'],1)
        self.assertEqual(v['teacher_box_to_RGB_secondary']['T_correct_S_error']['objects'],4)
        self.assertEqual(v['assigned_teacher_state_vs_actual_any_candidate_gate'][1]['objects'],1)
    def test_no_objects_and_zero_denominators(self):
        _,r,frames=fixture();r['objects_n']=0;r['identity_contract'].update(rgb_rows=[],ir_rows=[])
        r['selector_counts']=dict(a.selector_counts([]),teacher_gt_count=0)
        v=a.summarize([],r,frames)
        self.assertEqual(v['empty_frames'],32);self.assertEqual(v['actual_selector_counts']['normalizer'],1)
        self.assertIsNone(v['teacher_own_gt_primary']['T_correct_S_error']['fraction_all_rgb'])
    def test_bad_identity_scope_and_actual_count_reject(self):
        for kind in ('duplicate','mapping','scope','count','teacher_unknown'):
            rows,r,f=fixture()
            if kind=='duplicate':rows[1]['stable_rgb_gt_id']=rows[0]['stable_rgb_gt_id']
            if kind=='mapping':r['identity_contract']['rgb_rows'][0]['stable_gt_id']='wrong'
            if kind=='scope':r['scope']='OLD_DEV200'
            if kind=='count':r['selector_counts']['reference_candidate_count']-=1
            if kind=='teacher_unknown':rows[-1]['states']['T']=state(False)
            with self.assertRaises(ValueError):a.summarize(rows,r,f)
        contract=dict(scope=a.SCOPE,objects_n=8,exact_selector_crosscheck=True,full_diagnostics=True,
            stable_identity_contract_status='STABLE_GT_IDENTITY_VERIFIED',evidence_config=dict(rho=.5,levels=[0,1],
                match_iou=.5,reference_conf=.05,reference_iou=.1,teacher_conf=.25,teacher_iou=.5,
                minimum_foreground=1,minimum_background=4,background_scale=2.),
            state_definition=dict(coarse_confidence=.05,coarse_iou=.1,correct_confidence=.25,correct_iou=.5,state_priority=list(a.STATES)))
        a.validate_export_contract(contract,8)
        contract['exact_selector_crosscheck']=False
        with self.assertRaises(ValueError):a.validate_export_contract(contract,8)
    def test_global_quota_and_stable_tie(self):
        rows,r,f=fixture()
        rows[0]['gates']['quality_q']=4.
        # q ties retain matched row order, not former ranks or per-image quota.
        with self.assertRaises(ValueError):a.summarize(rows,r,f)
        rows[0]['gates']['selected_rank']=1;rows[1]['gates']['selected_rank']=2
        a.summarize(rows,r,f)
        rows[6]['gates']['selected']=True;rows[6]['gates']['selected_rank']=3
        r['selector_counts']['selected_count']=3
        with self.assertRaises(ValueError):a.summarize(rows,r,f)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    with args.receipt.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),GPU_used=False,real_probe_results_read=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
