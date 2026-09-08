"""Synthetic CPU contracts only; no model or actual AP inputs."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import yaml

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('direction_eval_under_test', HERE/'evaluate_direction.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)


def config(dataset='drone', arm='N'):
    n, objects, nc = ev.POPULATIONS[dataset]
    return dict(scope=ev.SCOPE, direction_screen=dict(scope=ev.SCOPE), dataset=dataset, arm=arm,
        seed=42, epochs=3, batch=32, workers=4, imgsz=640, lr0=.0001, lrf=1., warmup_epochs=0.,
        expected_val_images=n, expected_nc=nc, expected_train_images=2048, freeze_bn_running_statistics=True,
        kd_coefficient=0., model='/initial.pt', teacher='/teacher.pt', reference='/reference.pt',
        classification_coefficient=0., localization_coefficient=0., torch_version='fixture', ultralytics_version='fixture',
        paths=dict(student_data_yaml='/subset.yaml'), auxiliary_data_identity=dict(student_data_yaml='/full.yaml'))


class Profile:
    def __init__(self, count): self.count=count; self.drift=False; self.validations=0
    def dev_roster(self, path):
        rows=['/fixture/%05d.jpg'%i for i in range(self.count)]
        if path=='/subset.yaml' and self.drift: rows[0]='/fixture/changed.jpg'
        return rows
    def validate_evaluation_profile_binding(self, *args):
        self.validations+=1
        return dict(actual_effective_kwargs=dict(ev.EFFECTIVE))


def projection(dataset='drone'):
    cfg=config(dataset); native=copy.deepcopy(cfg)
    native['dataset']='dronevehicle' if dataset=='drone' else 'llvip'
    native['paths']['student_data_yaml']='/full.yaml'
    return cfg,native,Path('/native.yaml'),Path('/binding.json') if dataset=='drone' else None,Profile(cfg['expected_val_images']),Path('/pinned')


def values(nc):
    result=dict(AP50=.6,AP75=.4,mAP50_95=.3,precision=.7,recall=.8)
    result['per_class']=[dict(class_id=i,name=str(i),AP50=.6,AP75=.4,mAP50_95=.3) for i in range(nc)]
    return result


class Tests(unittest.TestCase):
    def test_all_fixed_arms_and_datasets_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.yaml'
            for ds in ev.POPULATIONS:
                for arm in ev.DATASET_ARMS[ds]:
                    p.write_text(yaml.safe_dump(config(ds,arm)))
                    self.assertEqual(ev.load_config(p)['arm'],arm)

    def test_bad_training_schedule_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.yaml'
            for field,value in (('epochs',8),('lr0',.001),('lrf',.1),('scope','HOURLY_SCREEN_FT'),('expected_val_images',1)):
                cfg=config();cfg[field]=value;p.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):ev.load_config(p)

    def test_drone_prior_binding_and_full_roster(self):
        args=projection();roster,identity=ev.evaluation_projection(*args)
        self.assertEqual(len(roster),1469);self.assertTrue(identity['prior_binding_validated'])
        self.assertEqual(args[4].validations,1)

    def test_llvip_new_actual_profile_without_fake_prior(self):
        args=projection('llvip');roster,identity=ev.evaluation_projection(*args)
        self.assertEqual(len(roster),2406);self.assertFalse(identity['prior_binding_validated'])
        self.assertFalse(identity['native_vs_capture_parity_rerun'])
        self.assertFalse(identity['accepted_endpoint_claim']);self.assertEqual(args[4].validations,0)

    def test_wrong_binding_policy_rejected(self):
        args=list(projection());args[3]=None
        with self.assertRaises(ValueError):ev.evaluation_projection(*args)
        args=list(projection('llvip'));args[3]=Path('/drone_binding.json')
        with self.assertRaises(ValueError):ev.evaluation_projection(*args)

    def test_roster_and_full_identity_drift_rejected(self):
        args=list(projection());args[4].drift=True
        with self.assertRaises(ValueError):ev.evaluation_projection(*args)
        args=list(projection());args[0]['auxiliary_data_identity']['student_data_yaml']='/different.yaml'
        with self.assertRaises(ValueError):ev.evaluation_projection(*args)

    def test_cross_dataset_projection_rejected(self):
        args=list(projection('llvip'));args[1]['dataset']='dronevehicle'
        with self.assertRaises(ValueError):ev.evaluation_projection(*args)

    def test_metric_fraction_classes_and_macro(self):
        for nc in (1,5):ev.validate_values(values(nc),nc)
        for field,value in (('AP50',60.),('AP75',float('nan')),('precision',True)):
            data=values(1);data[field]=value
            with self.assertRaises(ValueError):ev.validate_values(data,1)
        data=values(5);data['per_class'][0]['AP75']=.3
        with self.assertRaises(ValueError):ev.validate_values(data,5)

    def test_duplicate_or_missing_class_rejected(self):
        data=values(5);data['per_class'][0]['class_id']=1
        with self.assertRaises(ValueError):ev.validate_values(data,5)
        with self.assertRaises(ValueError):ev.validate_values(values(1),5)

    def test_completion_and_last_stat(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'weights').mkdir();checkpoint=run/'weights/last.pt';checkpoint.write_bytes(b'synthetic-only')
            cfg=config()
            receipt=dict(status='DIRECTION_TRAINING_COMPLETED',scope=ev.SCOPE,endpoint=ev.ENDPOINT,
                single_seed=True,seed=42,last_epoch=3,epochs_configured=3,batches=192,
                formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False,
                bn_running_buffers_unchanged=True,config_copy=str(run/'direction_config.yaml'),
                checkpoint=ev.stat(checkpoint),**{k:cfg[k] for k in ('arm','dataset','model','teacher','reference',
                    'classification_coefficient','localization_coefficient','kd_coefficient')})
            (run/'direction_config.yaml').write_text(yaml.safe_dump(cfg))
            p=run/'completion_receipt.json';p.write_text(json.dumps(receipt))
            result,_=ev.training_completion(cfg,checkpoint)
            self.assertEqual(result['checkpoint'],ev.stat(checkpoint))
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):ev.training_completion(cfg,checkpoint)

    def test_best_and_incomplete_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'weights').mkdir();p=run/'weights/last.pt';p.write_bytes(b'synthetic')
            with self.assertRaises(ValueError):ev.training_completion(config(),run/'weights/best.pt')
            (run/'completion_receipt.json').write_text(json.dumps(dict(status='FAILED')))
            with self.assertRaises(ValueError):ev.training_completion(config(),p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),known_truth_only=True,new_AP_read=False,
        gpu_or_ssh=False,new_hash_computed=False,torch_imported='torch' in sys.modules,
        source=ev.stat(HERE/'evaluate_direction.py'),scope='CPU_EVALUATOR_ENTRY_ONLY')
    ev.write_new(args.receipt,receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)
