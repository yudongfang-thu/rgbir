"""Small synthetic checks; never dispatches a job or starts a GPU."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import run_confidence_queue as q


def configs():
    shared=dict(dataset='llvip',scope=q.SCOPE,seed=42,epochs=3,imgsz=640,batch=32,nbs=64,workers=4,
        amp=True,optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,expected_train_images=2048,
        expected_val_images=2406,expected_nc=1,freeze_bn_running_statistics=True,localization_coefficient=0.,
        model='/init.pt',teacher='/ir.pt',reference='/ref.pt',paths={},auxiliary_data_identity={},augmentation={},
        evidence={},torch_version='fixture',ultralytics_version='fixture')
    return {a:dict(copy.deepcopy(shared),arm=a,kd_coefficient=0. if a=='N' else .1,
                   classification_coefficient=0. if a=='N' else .1) for a in q.ARMS}


class Checks(unittest.TestCase):
    def test_fixed_config_and_wrong_dose(self):
        c=configs();q.validate_configs(c)
        c['C0']['kd_coefficient']=.09
        with self.assertRaises(ValueError):q.validate_configs(c)
        c=configs();c['C0']['teacher']='/different.pt'
        with self.assertRaises(ValueError):q.validate_configs(c)

    def test_terminal_main_only_and_blocked_still_terminal(self):
        for status in ('DIRECTION_MATRIX_COMPLETED','DIRECTION_MATRIX_COMPLETED_WITH_BLOCKED_ARMS'):
            self.assertEqual(q.main_queue_terminal(dict(status=status,seed=42,epochs=3,new_hash_computed=False)),status)
        with self.assertRaises(ValueError):q.main_queue_terminal(dict(status='RUNNING'))
        with self.assertRaisesRegex(ValueError,'shared implementation review'):
            q.main_queue_terminal(dict(status='DIRECTION_MATRIX_PARTIAL_FAILURE',seed=42,epochs=3,new_hash_computed=False))

    def test_measured_reservation_rounding_and_ceiling(self):
        c=dict(resources=dict(per_gpu_peak_vram_mib={'4':1536},peak_rss_mib=4096),
               gpu_allocated_peak_mib=1024,gpu_reserved_peak_mib=1280)
        r=q.training_reservation(c);self.assertEqual((r['vram_mib'],r['rss_mib']),(1792,6144))
        c['resources']['per_gpu_peak_vram_mib']['4']=8192
        with self.assertRaises(ValueError):q.training_reservation(c)

    def test_job_interface_and_no_calibration(self):
        release=Path('/release');out=Path('/output')
        canary=q.make_job(release,out,'N','canary')
        self.assertIn('--canary',canary['command']);self.assertEqual(canary['vram_mib'],8192)
        train=q.make_job(release,out,'C0','train',dict(vram_mib=1792,rss_mib=6144))
        self.assertIn('--canary-receipt',train['command']);self.assertEqual(train['vram_mib'],1792)
        ev=q.make_job(release,out,'C0','eval')
        self.assertEqual(ev['command'][0],str(q.PY));self.assertIn('--native-config',ev['command'])
        self.assertNotIn('--native-profile-binding',ev['command'])
        self.assertEqual(ev['expected_receipt'],str(out/'evaluations/C0/direction_evaluation_receipt.json'))
        self.assertEqual((ev['vram_mib'],ev['rss_mib']),(2048,8192))
        with self.assertRaises(ValueError):q.make_job(release,out,'C0','calibration')

    def test_first30_checks_full_batch_and_content(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stream.jsonl'
            rows=[dict(batch=i,im_file=['image']*32,cls=[0],bboxes=[[.5,.5,.2,.2]],teacher_cls=[0],teacher_bboxes=[[.5,.5,.2,.2]]) for i in range(1,31)]
            p.write_text('\n'.join(json.dumps(r) for r in rows));self.assertEqual(len(q.first_thirty(p)),30)
            rows[5]['batch']=30;p.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):q.first_thirty(p)

    def test_eval_scope_and_population_rejected(self):
        r=dict(status='DIRECTION_EVALUATION_COMPLETED',scope=q.SCOPE,endpoint=q.ENDPOINT,dataset='llvip',arm='C0',seed=42,
            single_seed=True,formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False,
            epochs=3,full_dev_images=2406,full_dev_gt_objects=7879)
        q.check_terminal('eval','C0',r)
        r['scope']='DIRECTION_FT3_BNFROZEN'
        with self.assertRaises(ValueError):q.check_terminal('eval','C0',r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    q.write_new(args.receipt,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),synthetic_only=True,gpu_or_ssh=False,
        new_hash_computed=False,actual_AP_read=False,source=q.file_stat(Path(q.__file__))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
