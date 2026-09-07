"""DRAFT contract truth/negative tests; no runtime, torch, GPU or training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import yaml

import generate_configs as generation
import screen_common as common
import evaluate_short_screen as evaluation
import train_short_screen as training

HERE=Path(__file__).resolve().parent


class ContractTests(unittest.TestCase):
    def configs(self):
        return {arm:common.load_config(HERE/'configs'/('drone_'+arm+'_s42_E20_DRAFT.yaml')) for arm in common.COEFFICIENTS}

    def test_three_actual_configs_and_unmodified_recipe(self):
        base=yaml.safe_load((HERE/'configs'/'original_formal_C1_s42.yaml').read_text(encoding='utf-8'))
        configs=self.configs()
        for arm,cfg in configs.items():
            self.assertEqual(cfg['classification_coefficient'],common.COEFFICIENTS[arm])
            self.assertEqual({k:v for k,v in cfg.items() if k not in generation.CHANGED_FIELDS},
                             {k:v for k,v in base.items() if k not in generation.CHANGED_FIELDS})
            self.assertEqual(cfg['epochs'],20);self.assertFalse(cfg['formal_training_authorized'])
            self.assertFalse(cfg['short_screen']['queue_authorized'])
        self.assertEqual(configs['C1']['short_screen']['candidate']['kind'],'UNRESOLVED')

    def test_independent_lr_horizon_and_warmup(self):
        cfg=self.configs()['C1']
        self.assertEqual(cfg['short_screen']['scheduler_horizon_epochs'],20)
        self.assertEqual(cfg['warmup_epochs'],3.);self.assertFalse(cfg['cos_lr']);self.assertEqual(cfg['close_mosaic'],0)

    def test_reject_e200_and_changed_lambda_batch_seed(self):
        cfg=self.configs()['C1']
        for key,value in [('epochs',200),('classification_coefficient',.1),('batch',16),('seed',0),('amp',1)]:
            with self.assertRaises(ValueError):common.validate_config(dict(cfg,**{key:value}))

    def test_no_admission_rejects_before_runtime(self):
        with self.assertRaises(ValueError):
            common.require_admission(None,self.configs()['C1'],Path('unused'),Path('unused'),'training')
        self.assertNotIn('torch',sys.modules);self.assertNotIn('runtime',sys.modules)

    def test_draft_receipt_cannot_admit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'draft.json';common.write_new(path,dict(status='DRAFT',queue_authorized=False))
            with self.assertRaises(ValueError):common.require_admission(path,self.configs()['N'],Path('unused'),Path('unused'),'training')

    def test_original_candidate_is_direct_original_builder(self):
        class Fake:
            @staticmethod
            def build_trainer():return 1
        fn,deps=training.candidate_builder(Fake,None,None,dict(kind='original'))
        self.assertIs(fn,Fake.build_trainer);self.assertFalse(deps)
        with self.assertRaises(ValueError):training.candidate_builder(Fake,None,None,dict(kind='UNRESOLVED'))

    def test_criterion_factory_binds_both_api_and_private_criterion(self):
        class Base:pass
        def builder():return IndependentCriterion
        builder.__globals__['IndependentCriterion']=Base
        fake_training=types.SimpleNamespace(build_trainer=builder)
        fake_criterion=types.SimpleNamespace(IndependentCriterion=Base)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'thin.py'
            source.write_text('def make_api(adapter, classification):\n    return (adapter, classification)\n'
                'def make_criterion_type(criterion_module, api):\n    return type("Private", (criterion_module.IndependentCriterion,), {"api":api})\n',encoding='utf-8')
            candidate=dict(kind='criterion_factory',source=str(source))
            fn,_=training.candidate_builder(fake_training,fake_criterion,'selector',candidate,'classification')
            self.assertIs(builder(),Base);self.assertTrue(issubclass(fn(),Base));self.assertEqual(fn().api,('selector','classification'))
            with self.assertRaises(ValueError):
                training.candidate_builder(fake_training,fake_criterion,'selector',dict(candidate,kind='selector',export='make_api'),'classification')

    def completion(self,root,status='SHORT_SCREEN_TRAINING_COMPLETED'):
        cfg=self.configs()['C1'];(root/'weights').mkdir();checkpoint=root/'weights'/'last.pt';checkpoint.write_bytes(b'CPU fixture only')
        r=dict(status=status,scope='SHORT_SCREEN',single_seed=True,last_epoch=20,epochs_configured=20,
            endpoint=common.ENDPOINT,formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False,
            checkpoint=common.stat(checkpoint),**{k:cfg[k] for k in ('arm','seed','dataset','model','teacher','reference',
                'classification_coefficient','localization_coefficient')})
        common.write_new(root/'short_training_receipt.json',r)
        return cfg,checkpoint

    def test_short_completion_uses_own_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,checkpoint=self.completion(root)
            _,actual=evaluation.short_completion(root,cfg);self.assertEqual(actual,checkpoint)

    def test_formal_training_completed_cannot_impersonate_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_=self.completion(root,status='training_completed')
            with self.assertRaises(ValueError):evaluation.short_completion(root,cfg)

    def test_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,checkpoint=self.completion(root);checkpoint.write_bytes(b'changed CPU fixture')
            with self.assertRaises(ValueError):evaluation.short_completion(root,cfg)

    def test_native_effective_kwargs_frozen(self):
        self.assertEqual(evaluation.EFFECTIVE,dict(imgsz=640,batch=32,workers=4,quantize=None,conf=.001,iou=.7,
            max_det=300,agnostic_nms=False,single_cls=False,rect=True,augment=False,half=False))

    def test_source_copy_is_byte_identity_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.py';source.write_bytes(b'line1\r\nline2\r\n');out=root/'out';out.mkdir()
            common.copy_sources(out,[source]);manifest=common.read(out/'source_manifest.json')
            self.assertFalse(manifest['new_hash_computed'])
            self.assertEqual(Path(manifest['files'][0]['copy']).read_bytes(),source.read_bytes())
            with self.assertRaises(FileExistsError):common.copy_sources(out,[source])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if len(sys.argv)>1:
        common.write_new(Path(sys.argv[1]),dict(status='PASS_DRAFT_CPU_CONTRACTS' if result.wasSuccessful() else 'FAIL',
            tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            runtime_imported='runtime' in sys.modules,torch_imported='torch' in sys.modules,
            execution_authorized=False,new_hash_computed=False))
    if not result.wasSuccessful():raise SystemExit(1)
