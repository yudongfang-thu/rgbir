"""CPU truth tests only. Does not import runtime/Ultralytics or initialize CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

import compare_24_updates as m


class ComparisonTests(unittest.TestCase):
    def test_exact_nested_state(self):
        x=dict(t=torch.ones(3),arr=np.arange(3),nested=[None,True,3,(.25,'a')])
        r=m.compare_tree(x,copy.deepcopy(x))
        self.assertTrue(r['bitwise_exact']);self.assertTrue(r['numeric_agreement'])

    def test_signed_zero_is_not_bitwise_exact(self):
        r=m.compare_tree(torch.tensor([0.]),torch.tensor([-0.]))
        self.assertFalse(r['bitwise_exact']);self.assertTrue(r['numeric_agreement'])

    def test_close_and_outside_are_distinct(self):
        a=torch.tensor([0.],dtype=torch.float64)
        near=m.compare_tree(a,torch.tensor([5e-7],dtype=torch.float64))
        far=m.compare_tree(a,torch.tensor([2e-6],dtype=torch.float64))
        self.assertFalse(near['bitwise_exact']);self.assertTrue(near['numeric_agreement'])
        self.assertFalse(far['numeric_agreement']);self.assertEqual(far['finite_outside_tolerance'],1)

    def test_tolerance_uses_old_as_reference_in_float64(self):
        a=torch.tensor([1.],dtype=torch.float64)
        self.assertTrue(m.compare_tree(a,a+1.099e-5)['numeric_agreement'])
        self.assertFalse(m.compare_tree(a,a+1.101e-5)['numeric_agreement'])

    def test_nonfinite_is_not_usable_gradient(self):
        a=torch.tensor([float('nan'),float('inf'),-float('inf')])
        r=m.compare_tree(a,a.clone())
        self.assertTrue(r['bitwise_exact']);self.assertTrue(r['numeric_agreement'])
        self.assertFalse(r['all_finite']);self.assertEqual(r['nonfinite_elements'],3)
        self.assertFalse(m.compare_tree(a,torch.zeros(3))['numeric_agreement'])

    def test_nan_payload_is_not_repaired(self):
        a=torch.from_numpy(np.array([0x7fc00001],dtype=np.uint32).view(np.float32))
        b=torch.from_numpy(np.array([0x7fc00002],dtype=np.uint32).view(np.float32))
        self.assertFalse(m.compare_tree(a,b)['numeric_agreement'])

    def test_shape_dtype_none_and_counter_strict(self):
        for a,b in [(torch.ones(1),torch.ones(1,dtype=torch.float64)),
                    (torch.ones(1),torch.ones(1,1)),(None,torch.zeros(1)),(1,True),(1,2)]:
            self.assertFalse(m.compare_tree(a,b)['numeric_agreement'])

    def test_clone_function_preserves_original_binding(self):
        original=lambda: ORIGINAL_SENTINEL
        original.__globals__['ORIGINAL_SENTINEL']='old'
        new=m.clone_function(original,ORIGINAL_SENTINEL='new')
        self.assertEqual(original(),'old');self.assertEqual(new(),'new')

    def test_clone_preserves_zero_argument_super_class_closure(self):
        class Base:
            def __call__(self):return 3
        class Original(Base):
            def __call__(self):return super().__call__()+CLOSURE_OFFSET
        Original.__call__.__globals__['CLOSURE_OFFSET']=1
        cloned=m.clone_function(Original.__call__,CLOSURE_OFFSET=2)
        Private=type('Private',(Original,),{'__call__':cloned})
        self.assertEqual(Original()(),4);self.assertEqual(Private()(),5)
        self.assertIs(cloned.__closure__,Original.__call__.__closure__)

    def test_frozen_coefficient_and_no_short_recipe(self):
        cfg=dict(arm='C1',source='paired',seed=42,epochs=200,imgsz=640,batch=32,nbs=64,
                 workers=4,amp=True,classification_coefficient=m.COEFFICIENT,
                 localization_coefficient=0.,model='/fixed/yolo11n.pt')
        m.validate_cfg(cfg)
        for key,value in [('classification_coefficient',.1),('epochs',20),('amp',1),('seed',0),('model','last.pt')]:
            with self.assertRaises(ValueError):m.validate_cfg(dict(cfg,**{key:value}))

    def make_runs(self, root, change=None):
        for worker in ('old','block16'):
            path=root/worker;path.mkdir();(path/'sources').mkdir()
            (path/'sources'/'0000_source.py').write_text('fixed source',encoding='utf-8')
            (path/'sources'/'input_config.yaml').write_text('fixed config',encoding='utf-8')
            m.write_json(path/'source_manifest.json',dict(new_hash_computed=False,
                files=[dict(path='/source.py',copy='/remote/'+worker+'/sources/0000_source.py',byte_identity=True)],
                models={'model':dict(path='/fixed/yolo11n.pt',bytes=123,mtime_ns=456)}))
            r=dict(updates=24,attempts=24,optimizer_calls=24,amp_skips=0,batches=24,
                   selected_calls=24,frozen_auxiliaries_unchanged=True)
            m.write_json(path/'worker_receipt.json',dict(status='COMPLETED_24_UPDATE_DIAGNOSTIC',
                worker=worker,result=r,new_hash_computed=False,configuration={'coefficient':m.COEFFICIENT}))
            torch.save({'param':torch.ones(1)},path/'initial.pt')
            torch.save({'img':torch.ones(1),'strong_img':torch.zeros(1)},path/'first_batch.pt')
            torch.save(dict(control=r,gradient_checks=[dict(kd_score_gradient_l2=.1)]),path/'final.pt')
            for i in range(1,25):
                torch.save(dict(metadata=dict(gt=torch.ones(1),rng=torch.arange(3)),parent_rng=(1,)),path/('batch_%04d.pt'%i))
                torch.save(dict(identity=dict(selected=torch.tensor([True])),floating=dict(delta=torch.zeros(1))),path/('selection_%04d.pt'%i))
                state=dict(before=dict(gradients_scaled={'p':torch.ones(1)},losses={'total':torch.ones(1)},control={'batch':i}),
                    after=dict(student={'p':torch.zeros(1)},optimizer={'buffer':torch.zeros(1)},ema={'p':torch.zeros(1)},
                        applied_gradients={'p':torch.ones(1)},control={'updates':i}))
                if change=='parameter' and worker=='block16' and i==24:state['after']['student']['p']+=2e-6
                torch.save(state,path/('attempt_%03d.pt'%i))
        if change=='intermediate':
            p=root/'block16'/'selection_0024.pt';r=m.load_pt(p);r['floating']['delta']+=2e-6;torch.save(r,p)
        if change=='identity':
            p=root/'block16'/'selection_0024.pt';r=m.load_pt(p);r['identity']['selected'][0]=False;torch.save(r,p)

    def run_comparison(self, change=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_runs(root,change)
            return m.compare_runs(root/'old',root/'block16',root/'out')

    def test_complete_comparison_closes(self):
        r=self.run_comparison()
        self.assertTrue(r['trajectory_bitwise_exact']);self.assertTrue(r['trajectory_numeric_agreement'])
        self.assertFalse(r['long_training_switch_admitted'])

    def test_intermediate_tolerance_failure_is_not_hidden_by_exact_trajectory(self):
        r=self.run_comparison('intermediate')
        self.assertTrue(r['trajectory_bitwise_exact'])
        self.assertFalse(r['categories']['selection_floating']['numeric_agreement'])

    def test_parameter_failure_cannot_pass_numeric_trajectory(self):
        r=self.run_comparison('parameter')
        self.assertFalse(r['trajectory_numeric_agreement'])
        self.assertTrue(r['initial_flow_selection_and_control_exact'])

    def test_selection_identity_change_is_hard_contract_difference(self):
        r=self.run_comparison('identity')
        self.assertFalse(r['initial_flow_selection_and_control_exact'])
        self.assertFalse(r['trajectory_numeric_agreement'])

    def test_missing_attempt_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_runs(root)
            (root/'block16'/'attempt_024.pt').unlink()
            with self.assertRaises(AssertionError):m.compare_runs(root/'old',root/'block16',root/'out')

    def test_cuda_not_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests))
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),torch=str(torch.__version__),
        cuda_initialized=torch.cuda.is_initialized(),scope='CPU comparison/helper truth tests; no real trainer or GPU execution',
        new_hash_computed=False)
    if len(sys.argv)>1:
        target=Path(sys.argv[1])
        if target.exists():raise FileExistsError(target)
        m.write_json(target,receipt)
    if not result.wasSuccessful():raise SystemExit(1)
