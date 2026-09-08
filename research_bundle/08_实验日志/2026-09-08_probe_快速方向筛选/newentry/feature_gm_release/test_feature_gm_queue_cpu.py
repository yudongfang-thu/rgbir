"""Synthetic queue contract tests: no dispatch, GPU, SSH or checkpoint loads."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
import run_feature_gm_queue as q


def candidate():
    return q.config(Path(__file__).parent/'configs/drone_F-rel-GM_s42_FT3.yaml')


def control(cfg,arm):
    c=copy.deepcopy(cfg);c.update(scope=q.CONTROL_SCOPE,arm=arm,kd_coefficient=0. if arm=='N' else .1,
        classification_coefficient=0. if arm=='N' else .1)
    c['direction_screen'].update(scope=q.CONTROL_SCOPE,endpoint=q.CONTROL_ENDPOINT)
    return c


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);q.write_new(path,value)


def stream(path):
    rows=[dict(batch=i,im_file=['image']*32,cls=[0],bboxes=[[.5,.5,.2,.2]],batch_idx=[0],
               teacher_cls=[0],teacher_bboxes=[[.5,.5,.2,.2]],teacher_batch_idx=[0]) for i in range(1,31)]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')


def receipt(cfg,stage,cp,directory):
    r=dict(status={'canary':'DIRECTION_CANARY_COMPLETED','train':'DIRECTION_TRAINING_COMPLETED',
        'eval':'DIRECTION_EVALUATION_COMPLETED'}[stage],scope=cfg['scope'],
        endpoint=q.ENDPOINT if cfg['arm']==q.ARM else q.CONTROL_ENDPOINT,dataset='drone',arm=cfg['arm'],seed=42,
        single_seed=True,formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False,
        bn_running_buffers_unchanged=True,bn_affine_trainable=True,bn_buffer_count=243,successful_updates=24,
        epochs_configured=3,last_epoch=3,batches=192,epochs=3,full_dev_images=1469,full_dev_gt_objects=22462,
        config_copy=str(cp),training_model=cfg['model'],training_configuration=str(cp),
        resources=dict(per_gpu_peak_vram_mib={'4':1536},peak_rss_mib=4096),
        gpu_allocated_peak_mib=1024,gpu_reserved_peak_mib=1280)
    for k in ('model','teacher','reference','kd_coefficient','classification_coefficient','localization_coefficient'):r[k]=cfg[k]
    p=directory/'weights/last.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture, not a checkpoint')
    r['checkpoint']=q.file_stat(p)
    return r


def initial(directory,cfg):
    save(directory/'initialization_check.json',dict(status='PASS_FULL_STATE_WARM_START',
        initial_checkpoint=q.file_stat(Path(cfg['model']).resolve()),head_included=True,
        fresh_optimizer=True,fresh_ema=True,teacher_reference_isolated=True))
    stream(directory/'sample_stream.jsonl')


def fixture(root):
    main=root/'main';out=root/'output';(out/'queue').mkdir(parents=True)
    cfg=candidate();p=root/'initial.pt';p.write_bytes(b'fixture only');cfg['model']=str(p.resolve())
    cp=out/'frozen_configs/drone_F-rel-GM_s42_FT3.yaml';cp.parent.mkdir();cp.write_text(yaml.safe_dump(cfg))
    for arm in ('N','C1'):
        cc=control(cfg,arm);ccp=main/'effective_configs'/('drone_'+arm+'_s42_FT3.yaml')
        ccp.parent.mkdir(parents=True,exist_ok=True);ccp.write_text(yaml.safe_dump(cc))
        ca=main/'canaries/drone'/arm;tr=main/'runs/drone'/arm;ev=main/'evaluations/drone'/arm
        car=receipt(cc,'canary',ccp,ca);trr=receipt(cc,'train',ccp,tr);evr=receipt(cc,'eval',ccp,ev)
        evr['checkpoint']=trr['checkpoint']
        evr['training_completion']=str((tr/'completion_receipt.json').resolve())
        save(ca/'canary.json',car);save(tr/'completion_receipt.json',trr);save(ev/'direction_evaluation_receipt.json',evr)
        initial(ca,cc);initial(tr,cc)
    ca=out/'canaries'/q.ARM
    save(ca/'canary.json',receipt(cfg,'canary',cp,ca));initial(ca,cfg)
    return main,out,cfg,cp


class Checks(unittest.TestCase):
    def test_frozen_actual_config_and_dose(self):
        cfg=candidate();q.validate_candidate(cfg)
        cfg['kd_coefficient']=1.
        with self.assertRaises(ValueError):q.validate_candidate(cfg)

    def test_cross_scope_common_changes_reject(self):
        cfg=candidate();c=control(cfg,'C1');q.compare_common(cfg,c,'C1')
        for key,value in [('lr0',.001),('freeze_bn_running_statistics',False),('model','other'),('paths',{})]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):q.compare_common(cfg,bad,'C1')

    def test_wait_gate_success_only_no_lease_wait(self):
        r=dict(status='CONFIDENCE_MATRIX_COMPLETED',scope='LLVIP_CONFIDENCE_FT3',seed=42,epochs=3,
            formal_e200_complete=False,new_hash_computed=False)
        q.preceding_success(r)
        r['status']='CONFIDENCE_MATRIX_FAILED'
        with self.assertRaises(ValueError):q.preceding_success(r)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'completion.json';save(p.parent/'failure.json',{})
            with self.assertRaises(ValueError):q.wait_for_confidence(p)

    def test_canary_budget_rounding_and_ceiling(self):
        c=dict(resources=dict(per_gpu_peak_vram_mib={'4':1536},peak_rss_mib=4096),
            gpu_allocated_peak_mib=1024,gpu_reserved_peak_mib=1280)
        b=q.training_reservation(c);self.assertEqual((b['vram_mib'],b['rss_mib']),(1792,6144))
        c['resources']['peak_rss_mib']=32768
        with self.assertRaises(ValueError):q.training_reservation(c)

    def test_interfaces_venv_and_native_binding(self):
        release,out=Path('/release'),Path('/output')
        self.assertEqual(q.make_job(release,out,'canary')['command'][0],str(q.PY))
        self.assertIn('--canary',q.make_job(release,out,'canary')['command'])
        tr=q.make_job(release,out,'train',dict(vram_mib=1792,rss_mib=6144))
        self.assertIn('--canary-receipt',tr['command'])
        ev=q.make_job(release,out,'eval',native_config='/native.yaml')
        self.assertIn('--native-profile-binding',ev['command']);self.assertEqual((ev['vram_mib'],ev['rss_mib']),(2048,8192))
        with self.assertRaises(ValueError):q.make_job(release,out,'calibration')

    def test_projection_two_controls_and_first30_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            main,out,cfg,cp=fixture(Path(d));p,rows=q.matched_controls(main,out,cfg,cp)
            self.assertEqual(set(p['controls']),{'N','C1'});self.assertEqual(len(rows),30)
            s=main/'canaries/drone/C1/sample_stream.jsonl';r=q.first_thirty(s);r[4]['bboxes'][0][0]=.51
            s.write_text('\n'.join(json.dumps(x) for x in r))
            with self.assertRaisesRegex(ValueError,'first30'):q.matched_controls(main,out,cfg,cp)

    def test_initial_stat_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            main,out,cfg,cp=fixture(Path(d));Path(cfg['model']).write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError,'stat changed'):q.matched_controls(main,out,cfg,cp)

    def test_endpoints_have_distinct_scopes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=candidate();r=receipt(cfg,'eval',Path('/config'),Path(d));q.check_terminal('eval',q.ARM,r)
            r['scope']=q.CONTROL_SCOPE
            with self.assertRaises(ValueError):q.check_terminal('eval',q.ARM,r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    q.write_new(a.receipt,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),synthetic_only=True,gpu_or_ssh=False,
        new_hash_computed=False,actual_feature_AP_read=False,source=q.file_stat(Path(q.__file__))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
