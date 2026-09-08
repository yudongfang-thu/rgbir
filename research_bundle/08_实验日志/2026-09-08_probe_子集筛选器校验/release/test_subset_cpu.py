"""Bounded CPU checks for new population/identity/flow/budget only; no forward."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import sys
import torch
import screen_common as common
import train_short_screen as train
import evaluate_short_screen as evaluate

HERE=Path(__file__).parent

class Checks(unittest.TestCase):
    def cfg(self,arm='N'):return common.load_config(HERE/'configs'/('drone_'+arm+'_s42_E8.yaml'))
    def test_01_population_arm_and_recipe(self):
        n,c=self.cfg(),self.cfg('C0');self.assertEqual(common.common_identity(n),common.common_identity(c))
        for key,value in [('arm','C1'),('expected_train_images',17990),('epochs',3),('lr0',.0001),('bn_running_statistics','frozen')]:
            bad=copy.deepcopy(n);bad[key]=value
            with self.assertRaises(ValueError):common.validate_config(bad)
        bad=copy.deepcopy(c);bad['classification_coefficient']=.10000001
        with self.assertRaises(ValueError):common.validate_config(bad)
    def test_02_canary_attempt_budget(self):
        self.assertEqual(common.canary_decision(24,29,29),'continue')
        self.assertEqual(common.canary_decision(24,30,30),'completed')
        self.assertEqual(common.canary_decision(24,48,48),'completed')
        for counters in [(23,48,48),(24,49,49),(31,30,30)]:
            with self.assertRaises((ValueError,RuntimeError)):common.canary_decision(*counters)
    def test_03_flow_all_elements_and_dual_labels(self):
        p=dict(metadata={'im_file':['a'],'pair_info':[{'weak_source':'a','strong_source':'b'}]},
            tensors={k:torch.tensor([[1,2]],dtype=torch.uint8 if 'img' in k else torch.float32) for k in common.TENSOR_KEYS})
        common.assert_flow_equal(p,copy.deepcopy(p),torch)
        for key in common.TENSOR_KEYS:
            q=copy.deepcopy(p);q['tensors'][key][0,1]+=1
            with self.assertRaises(AssertionError):common.assert_flow_equal(p,q,torch)
        q=copy.deepcopy(p);q['metadata']['pair_info'][0]['strong_source']='different'
        with self.assertRaises(AssertionError):common.assert_flow_equal(p,q,torch)
        q=copy.deepcopy(p);q['tensors']['cls']=q['tensors']['cls'].double()
        with self.assertRaises(AssertionError):common.assert_flow_equal(p,q,torch)
    def test_04_full_initialized_state_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'reference';out=root/'N';out.mkdir()
            model=torch.nn.Linear(2,3);writer=common.FlowAudit(folder,True,self.cfg(),out,torch)
            result=writer.initial(model,self.cfg())
            self.assertTrue(result['state_reference_written_and_verified'])
            self.assertIsNone(result['cross_arm_initial_state_exact'])
            reader=common.FlowAudit(folder,False,self.cfg('C0'),out,torch)
            self.assertTrue(reader.initial(model,self.cfg('C0'))['cross_arm_initial_state_exact'])
            with torch.no_grad():model.bias[0]+=1
            with self.assertRaises(AssertionError):reader.initial(model,self.cfg('C0'))
    def test_05_private_binding_preserves_original_builder(self):
        calls=[]
        class Criterion:
            def __call__(self,p,b):return 'original_math'
        def frozen(p,d,n):calls.append((p,d,n));return 'frozen'
        def old_builder(c,config,out,arm,max_steps):
            return dict(arm=arm,weight=c['kd_weight'],criterion=EvidenceCriterion,
                data=DualLabelRGBIRDataset,teacher=load_frozen(c['teacher'],c['paths']['privileged_data_yaml'],'names'))
        old_builder=train.cloned(old_builder,EvidenceCriterion=Criterion,DualLabelRGBIRDataset='ORIGINAL',load_frozen=frozen)
        legacy=types.SimpleNamespace(build_trainer=old_builder,load_frozen=frozen,EvidenceCriterion=Criterion,DualLabelRGBIRDataset='ORIGINAL')
        # Executes the actual pinned small build_trainer body via AST extraction;
        # never imports Torch/CUDA training dependencies or fake loss arithmetic.
        import ast
        source=Path(REFERENCE)/'train_independent.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='build_trainer')
        module=ast.Module(body=[node],type_ignores=[]);ns=dict(legacy=legacy,IndependentCriterion=Criterion,ORIGINAL_CRITERION=Criterion,
            ORIGINAL_DATASET='ORIGINAL',TrackedDualLabelRGBIRDataset='TRACKED')
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
        runtime=types.SimpleNamespace(legacy=legacy,TrackedDualLabelRGBIRDataset='TRACKED')
        training=types.SimpleNamespace(build_trainer=ns['build_trainer'])
        criterion=types.SimpleNamespace(IndependentCriterion=Criterion)
        for arm,old_arm in [('N','weight0'),('C0','paired')]:
            cfg=self.cfg(arm);build=train.private_builder(runtime,training,criterion,cfg,False)
            result=build(cfg,'config','output',arm=arm,max_steps=None,historical=False)
            self.assertEqual((result['arm'],result['weight'],result['data']),(old_arm,.1,'TRACKED'))
            self.assertEqual(calls[-1][1],cfg['auxiliary_data_identity']['privileged_data_yaml'])
        self.assertIs(legacy.EvidenceCriterion,Criterion);self.assertEqual(legacy.DualLabelRGBIRDataset,'ORIGINAL')
    def test_06_amp_scope_restoration(self):
        model=object();trainer=types.SimpleNamespace(model=model,amp=False)
        original=lambda *_:False;module=types.SimpleNamespace(check_amp=original)
        def setup():trainer.amp=module.check_amp(model)
        result=common.setup_with_validated_amp(trainer,setup,module)
        self.assertTrue(result['actual_autocast']);self.assertIs(module.check_amp,original)
        def failure():module.check_amp(model);raise RuntimeError('setup failed')
        with self.assertRaises(RuntimeError):common.setup_with_validated_amp(trainer,failure,module)
        self.assertIs(module.check_amp,original)
    def test_07_deadline_explicit_incomplete(self):
        for value in (0,-1,2701,float('inf')):
            with self.assertRaises(ValueError):common.Deadline(value)
        d=common.Deadline(1);d.started-=2
        with self.assertRaises(common.ExecutionDeadline):d.check()
    def test_08_full_dev_projection(self):
        cfg=self.cfg();native=copy.deepcopy(cfg);native['paths']=dict(cfg['paths'],student_data_yaml=cfg['auxiliary_data_identity']['student_data_yaml'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);native_path=p/'native.yaml';native_path.write_text('native');binding=p/'binding.json';binding.write_text('{}')
            cfg['native_contract_config']=str(native_path)
            profile=types.SimpleNamespace(validate_evaluation_profile_binding=lambda *_:{'actual':True},dev_roster=lambda _:list(map(str,range(1469))))
            _,projection=evaluate.evaluation_projection(cfg,native,native_path,profile,binding,p)
            self.assertTrue(projection['subset_dev_roster_exact_to_full'])
            bad=copy.deepcopy(native);bad['expected_nc']=1
            with self.assertRaises(ValueError):evaluate.evaluation_projection(cfg,bad,native_path,profile,binding,p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();global REFERENCE;REFERENCE=args.reference_dir
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],
        cuda_initialized=torch.cuda.is_initialized(),torch_version=torch.__version__,new_hash_computed=False,
        scope='new_subset_contract_private_binding_flow_budget_only_no_model_forward')
    common.write_new(args.output,record)
    if not result.wasSuccessful() or record['cuda_initialized']:raise SystemExit(1)
if __name__=='__main__':main()
