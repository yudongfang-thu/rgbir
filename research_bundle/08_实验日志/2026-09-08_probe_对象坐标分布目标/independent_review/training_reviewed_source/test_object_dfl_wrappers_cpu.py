"""Bounded CPU contracts, using synthetic values; no GPU or AP inputs."""
import argparse,copy,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import object_dfl_common as common
import calibrate_object_dfl as calibration
import evaluate_object_dfl as evaluation
import train_object_dfl as training

def cfg(arm='N'):
    return yaml.safe_load((HERE/'configs'/('llvip_'+arm+'_s42_FT3.yaml')).read_text(encoding='utf-8'))

class Profile:
    drift=False
    def dev_roster(self,path):
        result=['/fixture/%04d.jpg'%i for i in range(2406)]
        if self.drift and path=='/subset.yaml':result[0]='/changed.jpg'
        return result

class Contracts(unittest.TestCase):
    def test_three_real_template_configs(self):
        for arm in ('N','L3-DFL','L3-GT'):
            path=HERE/'configs'/('llvip_'+arm+'_s42_FT3.yaml')
            self.assertEqual(common.load_config(path)['arm'],arm)
            self.assertEqual(evaluation.load_config(path)['arm'],arm)
        n=cfg()
        for arm in ('L3-DFL','L3-GT'):
            for key in ('model','teacher','reference','paths','augmentation','auxiliary_data_identity','object_dfl'):
                self.assertEqual(n[key],cfg(arm)[key])

    def test_old_arm_scope_or_temperature_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml'
            for key,value in [('arm','L2-box'),('dataset','drone'),('scope','DIRECTION_FT3_BNFROZEN'),('epochs',8)]:
                c=cfg();c[key]=value;p.write_text(yaml.safe_dump(c))
                with self.assertRaises(ValueError):common.load_config(p)
                with self.assertRaises(ValueError):evaluation.load_config(p)
            c=cfg();c['object_dfl']['temperature']=1.;p.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):common.load_config(p)

    def test_one_coefficient_and_zero_native(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml';c=cfg();c['kd_coefficient']=.1;p.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):common.load_config(p)
            c=cfg('L3-DFL');c['localization_coefficient']=.5;p.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):common.load_config(p)

    def test_dfl_calibrates_shared_gt_without_using_gt_ratios(self):
        values,blocked,details=calibration.coefficient_plan({'L3-DFL':[.1,.3,.5,.7],'L3-GT':[100]*8})
        self.assertEqual(values,{'N':0.,'L3-DFL':.4,'L3-GT':.4});self.assertFalse(blocked)
        values,_,_=calibration.coefficient_plan({'L3-DFL':[2]*4,'L3-GT':[]})
        self.assertEqual(values['L3-DFL'],1.);self.assertEqual(values['L3-GT'],1.)

    def test_insufficient_dfl_and_invalid_ratio(self):
        values,blocked,_=calibration.coefficient_plan({'L3-DFL':[.1]*3,'L3-GT':[.1]*8})
        self.assertIsNone(values['L3-DFL']);self.assertIsNone(values['L3-GT']);self.assertEqual(len(blocked),2)
        with self.assertRaises(ValueError):calibration.coefficient_plan({'L3-DFL':[float('inf')]*4,'L3-GT':[]})

    def test_actual_resource_ceiling_includes_margin(self):
        r=dict(per_gpu_peak_vram_mib={'7':7000},peak_rss_mib=24000)
        a=common.calibration_resource_check(r,6000,7600)
        self.assertEqual(a['status'],'PASS');self.assertEqual(a['measured_with_margin_vram_mib'],8192)
        self.assertEqual(common.calibration_resource_check(r,6000,7900)['status'],'RESOURCE_LIMIT_EXCEEDED')
        r['peak_rss_mib']=32000
        self.assertEqual(common.calibration_resource_check(r,6000,7000)['status'],'RESOURCE_LIMIT_EXCEEDED')
        with self.assertRaises(ValueError):common.calibration_resource_check(r,0,7000)

    def test_amp_scoped_binding_restored_success_and_failure(self):
        original=lambda model:False;module=types.SimpleNamespace(check_amp=original)
        trainer=types.SimpleNamespace(model=object(),amp=None)
        def setup():trainer.amp=module.check_amp(trainer.model)
        row=common.setup_with_validated_amp(trainer,setup,True,True,module)
        self.assertEqual(row['calls'],1);self.assertIs(module.check_amp,original)
        def failing():module.check_amp(trainer.model);raise RuntimeError('synthetic setup failure')
        with self.assertRaises(RuntimeError):common.setup_with_validated_amp(trainer,failing,True,True,module)
        self.assertIs(module.check_amp,original)

    def test_amp_prior_full_identity_and_version(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);c=cfg()
            for key in ('model','teacher','reference'):
                p=d/(key+'.fixture');p.write_bytes(key.encode());c[key]=str(p)
            prior=dict(status='RAW_DFL_SINGLE_BATCH_COMPLETED',scope='RAW_DFL_SINGLE_BATCH',actual_amp=True,
                dataset='llvip',seed=42,initialization={k:common.stat(c[k]) for k in ('model','teacher','reference')},current_forward_id='synthetic')
            p=d/'prior.json';p.write_text(json.dumps(prior));q=d/'prior.yaml';q.write_text(yaml.safe_dump(c))
            self.assertTrue(common.validate_amp_prior(c,p,q)['actual_amp'])
            c['torch_version']='changed'
            with self.assertRaises(ValueError):common.validate_amp_prior(c,p,q)

    def test_full_dev_projection_and_no_drone_binding(self):
        c=cfg();c['paths']['student_data_yaml']='/subset.yaml';c['auxiliary_data_identity']['student_data_yaml']='/full.yaml'
        n=copy.deepcopy(c);n['paths']['student_data_yaml']='/full.yaml';profile=Profile()
        args=[c,n,Path('/native.yaml'),None,profile,Path('/pinned')]
        roster,projection=evaluation.evaluation_projection(*args)
        self.assertEqual(len(roster),2406);self.assertTrue(projection['subset_dev_roster_exact_to_full'])
        profile.drift=True
        with self.assertRaises(ValueError):evaluation.evaluation_projection(*args)
        profile.drift=False;args[3]=Path('/drone_binding.json')
        with self.assertRaises(ValueError):evaluation.evaluation_projection(*args)

    def test_eval_completion_new_scope_and_exact_config_binding(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'weights').mkdir();p=d/'weights/last.pt';p.write_bytes(b'synthetic-only')
            c=cfg();config=d/'direction_config.yaml';config.write_text(yaml.safe_dump(c))
            row=dict(status='OBJECT_DFL_TRAINING_COMPLETED',scope=evaluation.SCOPE,endpoint=evaluation.ENDPOINT,
                single_seed=True,seed=42,last_epoch=3,epochs_configured=3,batches=192,formal_e200_complete=False,
                new_hash_computed=False,official_test_accessed=False,bn_running_buffers_unchanged=True,
                config_copy=str(config),checkpoint=evaluation.stat(p),**{k:c[k] for k in
                    ('arm','dataset','model','teacher','reference','method_identity','classification_coefficient','localization_coefficient','kd_coefficient')})
            receipt=d/'completion_receipt.json';receipt.write_text(json.dumps(row))
            evaluation.training_completion(c,p,config)
            row['scope']='DIRECTION_FT3_BNFROZEN';receipt.write_text(json.dumps(row))
            with self.assertRaises(ValueError):evaluation.training_completion(c,p,config)

    def test_fraction_metric_contract(self):
        row=dict(AP50=.6,AP75=.4,mAP50_95=.3,precision=.7,recall=.8,
            per_class=[dict(class_id=0,name='person',AP50=.6,AP75=.4,mAP50_95=.3)])
        evaluation.validate_values(row,1);row['AP50']=60
        with self.assertRaises(ValueError):evaluation.validate_values(row,1)

    def test_cloned_binding_does_not_mutate_original(self):
        def function():return len([1,2])
        cloned=training.cloned(function,len=lambda x:99)
        self.assertEqual(cloned(),99);self.assertEqual(function(),2)


class CriterionTruth(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;self.saved={n:sys.modules.get(n) for n in ('runtime','gradient_observation','l3_distribution_loss')}
        runtime=types.ModuleType('runtime')
        runtime.ORIGINAL_CRITERION=object
        runtime.legacy=types.SimpleNamespace(raw_prediction=lambda prediction:prediction,append_json=lambda *a:None)
        observation=types.ModuleType('gradient_observation')
        observation.shared_parameter_set=lambda model:([0,1],[('p',model.p)])
        loss_module=types.ModuleType('l3_distribution_loss');self.requests=[]
        def compute(s,t,r,b,strides,variant):
            self.requests.append(variant)
            loss=s['boxes'].square().mean() if variant=='L3-DFL' else (s['boxes']-1).square().mean()
            return loss,dict(base_count=2,normalizer=2,selected_count=1,
                selected_anchors=[(0,3,3,0,0)],selected_object_ids=[(0,0,0)])
        loss_module.compute=compute
        sys.modules.update(runtime=runtime,gradient_observation=observation,l3_distribution_loss=loss_module)
        spec=importlib.util.spec_from_file_location('_object_dfl_criterion_truth',HERE/'object_dfl_criterion.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def tearDown(self):
        for n,v in self.saved.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v

    def criterion(self,arm,dose):
        torch=self.torch
        model=torch.nn.Module();model.p=torch.nn.Parameter(torch.tensor([2.,3.]));model.stride=(8,16,32)
        class Auxiliary(torch.nn.Module):
            def forward(self,x):return {'boxes':x}
        c=self.module.make_type(None)();c.teacher=Auxiliary();c.reference=Auxiliary()
        c.cfg=dict(arm=arm,kd_coefficient=dose,method_identity=common.METHOD_IDENTITY,canary_execution=True)
        c.trainer=types.SimpleNamespace(model=model,save_dir=Path('/unused'),epoch=0)
        c.native=lambda prediction,batch:(prediction['boxes'].sum().reshape(1),torch.tensor([7.]))
        c.calls=0;c.selected_total=0;c.gradient_checks=[]
        return c,model,dict(img=torch.zeros(4,1),strong_img=torch.zeros(4,1))

    def test_native_plus_B_lambda_once_and_nonzero_shared_grad(self):
        c,m,b=self.criterion('L3-DFL',.25);total,_=c({'boxes':m.p},b)
        self.assertEqual(float(total),11.5);total.backward()
        self.assertTrue(self.torch.equal(m.p.grad,self.torch.tensor([3.,4.])))
        self.assertGreater(c.gradient_checks[0]['kd_gradient_l2'],0)
        self.assertEqual(self.requests,['L3-DFL'])

    def test_N_keeps_auxiliary_selection_but_native_gradient(self):
        c,m,b=self.criterion('N',0.);total,_=c({'boxes':m.p},b);total.backward()
        self.assertEqual(float(total),5.);self.assertTrue(self.torch.equal(m.p.grad,self.torch.ones(2)))
        self.assertEqual(self.requests,['L3-DFL']);self.assertEqual(c.selected_total,1)

    def test_same_batch_DFL_GT_selection_binding(self):
        c,m,b=self.criterion('L3-GT',.1);result=c.losses({'boxes':m.p},{},{},b,['L3-DFL','L3-GT'])
        self.assertEqual(result['L3-DFL'][1],result['L3-GT'][1])
        with self.assertRaises(ValueError):c.losses({'boxes':m.p},{},{},b,['L2-box'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(t) for t in (Contracts,CriterionTruth))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),test_scope='synthetic_CPU_wrapper_contracts',
        torch_version=str(torch.__version__),cuda_initialized=torch.cuda.is_initialized(),new_AP_read=False,
        new_GPU=False,new_hash_computed=False,operator_truths_are_separate=True)
    common.write_new(args.output,receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)
