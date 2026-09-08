"""Bounded CPU checks of actual source functions, with synthetic trainer objects."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

HERE=Path(__file__).absolute().parent
SOURCE=HERE.parent/'newentry/release'


def function(filename,name,namespace):
    tree=ast.parse((SOURCE/filename).read_text(encoding='utf-8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),filename,'exec'),namespace)
    return namespace[name]


def criterion(coefficient,empty=False,llvip=False):
    parameter=torch.nn.Parameter(torch.tensor([2.,3.]))
    names=[('fixture',parameter)]
    legacy=SimpleNamespace(raw_prediction=lambda x:x,append_json=lambda *args:None)
    ns=dict(torch=torch,ORIGINAL_CRITERION=object,legacy=legacy,
            shared_parameter_set=lambda model:([0,1],names))
    cls=function('direction_criterion.py','make_type',ns)(None)
    obj=cls();obj.calls=0;obj.selected_total=0;obj.gradient_checks=[]
    obj.cfg=dict(arm='N' if coefficient==0 else 'C1',dataset='llvip' if llvip else 'drone',
                 kd_coefficient=coefficient,canary_execution=True)
    obj.trainer=SimpleNamespace(model=SimpleNamespace(stride=[8,16,32]),epoch=0,save_dir=HERE)
    obj.native=lambda prediction,batch:(torch.stack((parameter.square().sum(),parameter.sum())),{})
    obj.teacher=torch.nn.Identity();obj.reference=torch.nn.Identity()
    requests=[]
    def losses(student,teacher,reference,batch,arms):
        requests.extend(arms)
        loss=parameter[:0].sum() if empty else (parameter-1).square().sum()
        return {a:(loss,dict(selected_count=0 if empty else 1)) for a in arms}
    obj.losses=losses
    return obj,parameter,requests


class Checks(unittest.TestCase):
    def test_actual_scalar_composition_and_gradient(self):
        c,p,_=criterion(.17);batch=dict(img=torch.zeros(2,1),strong_img=torch.zeros(2,1))
        total,_=c({},batch)
        expected=p.square().sum()+p.sum()+2*.17*(p-1).square().sum()
        self.assertTrue(torch.equal(total,expected))
        g=torch.autograd.grad(total,p,retain_graph=True)[0]
        exact=torch.autograd.grad(expected,p)[0]
        self.assertTrue(torch.equal(g,exact))
        self.assertEqual(c.last_stats['actual_B'],2)
        self.assertEqual(c.last_stats['coefficient'],.17)

    def test_zero_dose_native_loss_and_gradient_exact(self):
        c,p,_=criterion(0.);total,_=c({},dict(img=torch.zeros(2,1),strong_img=torch.zeros(2,1)))
        native=p.square().sum()+p.sum()
        self.assertTrue(torch.equal(total,native))
        self.assertTrue(torch.equal(torch.autograd.grad(total,p,retain_graph=True)[0],torch.autograd.grad(native,p)[0]))

    def test_llvip_N_empty_uses_L2_but_is_not_blocked(self):
        c,p,requests=criterion(0.,empty=True,llvip=True)
        total,_=c({},dict(img=torch.zeros(2,1),strong_img=torch.zeros(2,1)))
        self.assertEqual(requests,['L2-box']);self.assertEqual(c.selected_total,0)
        self.assertEqual(c.gradient_checks[0]['kd_gradient_l2'],0.)
        self.assertTrue(torch.equal(total,p.square().sum()+p.sum()))

    def test_fixed_cosine_counts_missing_side_norm(self):
        cosine=function('calibrate_direction.py','cosine',{})
        self.assertAlmostEqual(cosine([torch.tensor(3.),torch.tensor(4.)],[torch.tensor(3.),None]),.6)
        self.assertIsNone(cosine([None],[torch.tensor(1.)]))

    def test_actual_bn_wrapper_keeps_buffers_and_affine_gradient(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        trainer=SimpleNamespace(model=model,_model_train=lambda:model.train())
        install=function('train_direction.py','install_bn_freeze',{})
        install(trainer)();before={k:v.clone() for k,v in model[1].named_buffers()}
        model(torch.tensor([[1.,2.,3.],[4.,2.,1.]])).square().sum().backward()
        self.assertFalse(model[1].training)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model[1].named_buffers()))
        self.assertIsNotNone(model[1].weight.grad);self.assertTrue(model[1].weight.requires_grad)


if __name__=='__main__':
    torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    sources=[]
    for name in ('train_direction.py','direction_criterion.py','calibrate_direction.py'):
        p=SOURCE/name;s=p.stat();sources.append(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    with (HERE/'cpu_receipt.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),sources=sources,
            known_truth_only=True,new_AP_read=False,new_hash_computed=False,gpu_or_ssh=False,
            scope='BOUNDED_SOURCE_AND_CPU_INTEGRATION_REVIEW'),f,ensure_ascii=False,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
