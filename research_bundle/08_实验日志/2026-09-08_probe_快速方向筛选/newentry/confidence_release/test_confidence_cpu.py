"""Bounded source and CPU truth checks; no actual data, weights or GPU."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import time
import unittest
import yaml
import torch
import confidence_common as common
import confidence_criterion as criterion
import train_confidence as train
import evaluate_confidence as evaluation

HERE=Path(__file__).resolve().parent
PINNED=HERE.parents[2]/'2026-09-07_train_IndependentKD实施/remote_admission_1532/formal_C1_gpu5_attempt2/runs/C1_seed42/implementation_snapshot/task_conditional_reference/legacy_oev1/train_object_evidence.py'


def original_class(call_records):
    tree=ast.parse(PINNED.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('EvidenceCriterion','combine_loss')]
    def evidence(student,teacher,reference,batch,**kwargs):
        call_records.append(kwargs)
        return (student['scores']-teacher['scores']).square().mean(),dict(selected_count=1,loss_unweighted=1.)
    env=dict(torch=torch,time=time,EvidenceConfig=lambda **kw:NS(**kw),raw_prediction=lambda x:x,
             object_evidence_loss=evidence,append_json=lambda *args:None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PINNED),'exec'),env)
    return env['EvidenceCriterion']


class Aux(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2),requires_grad=False)
    def forward(self,img):return dict(scores=self.weight.expand(2,1,4))


class Truths(unittest.TestCase):
    def test_configs_fixed_original_dose_evidence_and_no_new_losses(self):
        source=yaml.safe_load((HERE.parent/'release/configs/llvip_N_s42_FT3.yaml').read_text(encoding='utf-8'))
        for arm,w in (('N',0.),('C0',.1)):
            cfg=common.load_config(HERE/'configs'/('llvip_'+arm+'_s42_FT3.yaml'))
            for key in ('evidence','paths','auxiliary_data_identity','model','teacher','reference','augmentation'):
                self.assertEqual(cfg[key],source[key])
            self.assertEqual((cfg['kd_weight'],cfg['kd_coefficient'],cfg['classification_coefficient']),(w,w,w))
            self.assertEqual((cfg['epochs'],cfg['lr0'],cfg['lrf'],cfg['warmup_epochs']),(3,.0001,1.,0.))
            self.assertNotIn('classification',cfg);self.assertNotIn('localization',cfg)

    def test_wrong_dose_scope_or_dataset_rejected(self):
        good=common.load_config(HERE/'configs/llvip_C0_s42_FT3.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.yaml'
            for key,value in (('kd_weight',.2),('classification_coefficient',.2),('scope','old'),('dataset','drone'),('localization_coefficient',.1)):
                cfg=copy.deepcopy(good);cfg[key]=value;p.write_text(yaml.safe_dump(cfg),encoding='utf-8')
                with self.assertRaises(ValueError):common.load_config(p)

    def test_inherited_original_call_math_zero_and_nonzero_dose(self):
        for arm,w in (('weight0',0.),('paired',.1)):
            calls=[];original=original_class(calls);wrapped=criterion.make_type(original)
            self.assertIs(wrapped.__mro__[1],original)
            cfg=dict(evidence={},seed=42,kd_weight=.1,log_every_batches=100,canary_execution=True)
            native=lambda pred,batch:(pred['scores'].square().mean(),torch.tensor([0.]))
            trainer=NS(model=NS(stride=torch.tensor([8,16,32])),epoch=0,real_updates=0,wall_started=time.time(),save_dir=Path('/unused'))
            c=wrapped(native,Aux(),Aux(),cfg,arm,trainer,False)
            scores=torch.arange(8,dtype=torch.float32).reshape(2,1,4).div(8).requires_grad_()
            batch=dict(img=torch.zeros(2,3,8,8),strong_img=torch.zeros(2,3,8,8),teacher_batch={})
            total,_=c(dict(scores=scores),batch)
            expected=scores.square().mean()+2*w*(scores-.2).square().mean()
            self.assertTrue(torch.equal(total,expected));total.backward()
            self.assertEqual(c.selected_total,1);self.assertTrue(c.gradient_checks[0]['weight0_exact_loss_gradient'])
            self.assertEqual(calls[0]['arm'],'paired');self.assertEqual(calls[0]['seed'],43)
            self.assertTrue(c.sanity);self.assertIsNone(c.teacher.weight.grad)
            self.assertIsNone(c.reference.weight.grad)

    def test_full_run_first_only_sanity_and_private_globals(self):
        class Original:
            def __call__(self,p,b):self.calls+=1;return self.sanity
        obj=criterion.make_type(Original)();obj.cfg={};obj.calls=0
        self.assertTrue(obj(None,None));self.assertFalse(obj(None,None))
        def f():return binding
        copied=train.cloned(f,binding=123)
        self.assertEqual(copied(),123);self.assertNotIn('binding',f.__globals__)

    def test_bn_running_frozen_affine_trainable(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        trainer=NS(model=model);trainer._model_train=lambda:model.train()
        before=train.bn_state(model);train.install_bn_freeze(trainer)()
        self.assertFalse(model[1].training);self.assertTrue(model[1].weight.requires_grad)
        model(torch.randn(4,3)).sum().backward()
        self.assertIsNotNone(model[1].weight.grad)
        self.assertTrue(all(torch.equal(v,train.bn_state(model)[k]) for k,v in before.items()))

    def test_eval_scope_two_arms_and_source_no_candidate(self):
        self.assertEqual(evaluation.SCOPE,'LLVIP_CONFIDENCE_FT3')
        self.assertEqual(evaluation.ENDPOINT,'LLVIP_CONFIDENCE_FT3_LAST_EMA')
        self.assertEqual(evaluation.DATASET_ARMS,{'llvip':('N','C0')})
        for arm in ('N','C0'):evaluation.load_config(HERE/'configs'/('llvip_'+arm+'_s42_FT3.yaml'))
        source=(HERE/'train_confidence.py').read_text(encoding='utf-8')
        self.assertNotIn('selected_only',source);self.assertNotIn('independent_criterion',source)
        self.assertIn('runtime.ORIGINAL_CRITERION',source)
        self.assertIn('max_steps=None',source)
        for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),pinned_original_criterion=str(PINNED),
        kernel_stub_scope='Original criterion AST executed with explicit toy evidence kernel; original loss math not reimplemented or replaced in production',
        gpu_execution=False,new_hash_computed=False,actual_AP_read=False,
        failures_detail=[dict(test=str(t),traceback=e) for t,e in result.failures+result.errors])
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
