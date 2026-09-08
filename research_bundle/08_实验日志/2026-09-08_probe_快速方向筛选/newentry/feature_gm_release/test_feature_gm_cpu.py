"""Bounded pre-AP CPU checks; no new calibration or GPU execution."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
import torch
import yaml
import feature_gm_common as common
import evaluate_feature_gm as ev

HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'release'
COEF=14.438521129817886


class Base:
    def __init__(self,native,teacher,reference,cfg,arm,trainer,sanity):
        self.native,self.teacher,self.reference=native,teacher,reference
        self.cfg,self.trainer=cfg,trainer
        self.calls=0;self.selected_total=0;self.gradient_checks=[];self.evidence_cfg=NS()


class Aux(torch.nn.Module):
    def forward(self,img):return dict(scores=torch.zeros(2,1))


class API:
    def __init__(self):self.calls=[]
    def build(self,student,teacher,reference,batch,**kw):
        self.calls.append(kw);return NS(student=student)
    def loss(self,payload):return payload.student['scores'].square().mean(),dict(selected_count=1)


def load_make_type(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_type')
    env=dict(torch=torch,ORIGINAL_CRITERION=Base,
        legacy=NS(raw_prediction=lambda p:p,append_json=lambda *args:None),
        shared_parameter_set=lambda model:([],list(model.named_parameters())))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    return env['make_type']


class Truths(unittest.TestCase):
    def test_feature_operator_exact_copy_and_only_arm_alias_in_criterion(self):
        self.assertEqual((HERE/'direction_losses.py').read_bytes(),(OLD/'direction_losses.py').read_bytes())
        old=(OLD/'direction_criterion.py').read_text(encoding='utf-8')
        new=(HERE/'feature_gm_criterion.py').read_text(encoding='utf-8')
        line="            if arm=='F-rel-GM':request='F-rel'  # Explicit alias; identical frozen operator.\n"
        self.assertEqual(new.replace(line,''),old)

    def test_actual_n_control_body_loss_gradients_and_selection_call_equal(self):
        results=[]
        for source in (OLD/'direction_criterion.py',HERE/'feature_gm_criterion.py'):
            model=torch.nn.Linear(2,1,bias=False);model.weight.data.fill_(.3);model.stride=torch.tensor([8,16,32])
            api=API();factory=load_make_type(source)
            trainer=NS(model=model,epoch=0,save_dir=Path('/unused'))
            cfg=dict(arm='N',dataset='drone',seed=42,kd_coefficient=0.,canary_execution=False)
            native=lambda pred,batch:(pred['scores'].square().sum(),torch.tensor([0.]))
            c=factory(api)(native,Aux(),Aux(),cfg,'weight0',trainer,False)
            scores=model(torch.tensor([[.1,.2],[.3,.4]]));prediction=dict(scores=scores)
            batch=dict(img=torch.zeros(2,3,8,8),strong_img=torch.zeros(2,3,8,8))
            loss,items=c(prediction,batch)
            grad=torch.autograd.grad(loss,model.weight)[0]
            self.assertTrue(torch.equal(loss,scores.square().sum()))
            results.append((loss.detach(),grad,api.calls,c.selected_total,c.gradient_checks))
        self.assertTrue(torch.equal(results[0][0],results[1][0]));self.assertTrue(torch.equal(results[0][1],results[1][1]))
        # EvidenceConfig objects are fixture instances; other selection arguments are exact.
        for row in results:
            row[2][0].pop('config')
        self.assertEqual(results[0][2:],results[1][2:])

    def test_config_training_recipe_matches_main_f_and_only_fixed_dose(self):
        cfg=common.load_config(HERE/'configs/drone_F-rel-GM_s42_FT3.yaml')
        old=yaml.safe_load((OLD/'configs/drone_F-rel_s42_FT3.yaml').read_text(encoding='utf-8'))
        changed={'arm','scope','method_id','method_identity','description','kd_coefficient','classification_coefficient','protocol_status','direction_screen'}
        for key in old:
            if key not in changed:self.assertEqual(cfg[key],old[key],key)
        self.assertEqual(cfg['kd_coefficient'],COEF);self.assertEqual(cfg['classification_coefficient'],COEF)
        self.assertEqual(cfg['localization_coefficient'],0.)
        self.assertEqual(cfg['direction_screen']['matched_control_scope'],'DIRECTION_FT3_BNFROZEN')

    def test_fixed_coefficient_scope_and_no_localization_rejection(self):
        original=common.load_config(HERE/'configs/drone_F-rel-GM_s42_FT3.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.yaml'
            for key,value in (('kd_coefficient',1.),('classification_coefficient',14.),('scope','old'),('dataset','llvip'),('localization_coefficient',.1)):
                cfg=copy.deepcopy(original);cfg[key]=value;p.write_text(yaml.safe_dump(cfg),encoding='utf-8')
                with self.assertRaises(ValueError):common.load_config(p)
            cfg=copy.deepcopy(original);cfg.update(arm='N',kd_coefficient=0.,classification_coefficient=0.)
            p.write_text(yaml.safe_dump(cfg),encoding='utf-8');self.assertEqual(common.load_config(p)['kd_coefficient'],0.)

    def test_preserved_blocked_calibration_and_finite_actual_ratio(self):
        original=json.loads((HERE/'original_blocked_calibration_receipt.json').read_text(encoding='utf-8'))
        r=json.loads((HERE/'CALIBRATION_RATIO_READOUT.json').read_text(encoding='utf-8'))
        self.assertIsNone(original['coefficients']['F-rel']);self.assertIn('F-rel',original['blocked'])
        self.assertEqual(r['fixed_coefficient'],COEF);self.assertEqual(len(r['rows']),8)
        self.assertTrue(r['finite_all_eight']);self.assertFalse(r['new_calibration'])
        self.assertAlmostEqual(r['actual_ratio_median'],1.008674928623853)
        self.assertNotEqual(r['actual_ratio_median'],1.)
        for x in r['rows']:self.assertAlmostEqual(x['fixed_B_lambda_F_norm'],COEF*x['unit_B_F_norm'])

    def test_eval_scope_and_full_population_and_compile(self):
        self.assertEqual(ev.SCOPE,'FEATURE_RELATION_GM_FT3')
        self.assertEqual(ev.ENDPOINT,'FEATURE_RELATION_GM_FT3_LAST_EMA')
        cfg=ev.load_config(HERE/'configs/drone_F-rel-GM_s42_FT3.yaml')
        self.assertEqual((cfg['expected_val_images'],cfg['expected_nc']),(1469,5))
        self.assertEqual(ev.POPULATIONS,{'drone':(1469,22462,5)})
        for source in HERE.glob('*.py'):ast.parse(source.read_text(encoding='utf-8'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        gpu_used=False,feature_AP_read=False,new_calibration=False,new_hash_computed=False,
        N_body_proof='Actual make_type source executed with explicit toy API/native dependencies; loss/parameter gradient/selection call and diagnostics exact; full real flow checked by queue canary',
        details=[dict(test=str(t),traceback=e) for t,e in result.failures+result.errors])
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
