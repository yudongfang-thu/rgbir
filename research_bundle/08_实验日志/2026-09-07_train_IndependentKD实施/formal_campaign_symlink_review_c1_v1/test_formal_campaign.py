"""Synthetic CPU scheduling tests; no real profiles, screen, CUDA or SSH."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import yaml
import formal_campaign as fc


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'dispatch').mkdir()
        self.resources=dict(vram_mib=8000,rss_mib=40000)
        self.rows=[fc.build_jobs('/synthetic/release',self.root,'/synthetic/python',s,'/synthetic/C1_'+str(s)+'.yaml',
            '/synthetic/canary_profile','/synthetic/eval_profile',self.resources,self.resources,'train-key','eval-key') for s in fc.ORDER]
        self.manifest=dict(schema=fc.SCHEMA,seed_order=list(fc.ORDER),seeds=self.rows,release='/synthetic/release',repo='/synthetic/repo',input_snapshots=[])
        self.state=dict(paused=False,pause_reason=None,seeds={str(s):dict(status='NOT_STARTED') for s in fc.ORDER})
        fc.write_new(self.root/'manifest.json',self.manifest);fc.write_new(self.root/'state.json',self.state)
        self.lock_patch=patch.object(fc,'lock',side_effect=lambda *a,**kw:nullcontext())
        self.lock_patch.start();self.addCleanup(self.lock_patch.stop)

    def launch(self,row,stage='train'):
        path=self.root/'dispatch'/(row[stage]['id']+'_events.jsonl')
        with path.open('a') as stream:stream.write(json.dumps(dict(status='LAUNCHED',pid=123))+'\n')

    def train_complete(self,row):
        path=Path(row['train']['result_receipt']);path.parent.mkdir(parents=True,exist_ok=True)
        fc.write_new(path,dict(status='training_completed',arm='C1',source='paired',seed=row['seed'],last_epoch=200,epochs_configured=200))

    def training_measurement(self,row):
        fc.write_new(self.root/'dispatch'/(row['train']['id']+'_resource_profile.json'),dict(status='COMPLETED',
            measurement_valid=True,stage='train',result_receipt=row['train']['result_receipt']))

    def evaluation_complete(self,row):
        self.launch(row,'evaluation')
        path=Path(row['evaluation']['result_receipt']);path.parent.mkdir(parents=True,exist_ok=True)
        fc.write_new(path,dict(status='completed',arm='C1',seed=row['seed'],endpoint='fixed_budget_last_ema',official_test_accessed=False,mAP50_95=-999))
        fc.write_new(self.root/'dispatch'/(row['evaluation']['id']+'_resource_profile.json'),dict(status='COMPLETED',measurement_valid=True))

    def test_jobs_are_separate_train_then_eval_with_no_gpu_or_short_budget(self):
        row=self.rows[0]
        self.assertEqual(row['screen'],'ikdv2_C1_42')
        self.assertEqual(row['train']['stage'],'train');self.assertTrue(row['train']['formal'])
        self.assertEqual(row['evaluation']['stage'],'evaluation')
        self.assertIn('train_independent.py',' '.join(row['train']['command']))
        self.assertIn('evaluate_independent.py',' '.join(row['evaluation']['command']))
        for job in (row['train'],row['evaluation']):
            self.assertNotIn('--max-steps',job['command'])
            self.assertFalse(set(job)&{'gpu','gpus','device','candidate_gpus'})
            self.assertEqual(job['requires_profile'], '/synthetic/canary_profile' if job['stage']=='train' else '/synthetic/eval_profile')

    def test_next_seed_waits_for_actual_launched_not_screen_started(self):
        self.assertEqual(fc.next_seed_to_launch(self.root,self.manifest,self.state),42)
        self.state['seeds']['42']['status']='SCREEN_REQUESTED'
        self.assertIsNone(fc.next_seed_to_launch(self.root,self.manifest,self.state))
        self.launch(self.rows[0]);self.assertEqual(fc.next_seed_to_launch(self.root,self.manifest,self.state),0)
        self.state['seeds']['0']['status']='TRAIN_WAITING'
        self.assertIsNone(fc.next_seed_to_launch(self.root,self.manifest,self.state))
        self.launch(self.rows[1]);self.assertEqual(fc.next_seed_to_launch(self.root,self.manifest,self.state),123)

    def test_pending_evaluation_blocks_until_its_actual_resource_launch(self):
        self.launch(self.rows[0]);self.train_complete(self.rows[0]);self.state['seeds']['42']['status']='EVAL_PENDING'
        self.assertEqual(fc.pending_evaluations(self.root,self.manifest),[42])
        self.assertIsNone(fc.next_seed_to_launch(self.root,self.manifest,self.state))
        self.launch(self.rows[0],'evaluation')
        self.assertEqual(fc.pending_evaluations(self.root,self.manifest),[])
        self.assertEqual(fc.next_seed_to_launch(self.root,self.manifest,self.state),0)

    def test_partial_completion_publication_defers_training(self):
        path=Path(self.rows[0]['train']['result_receipt']);path.parent.mkdir(parents=True)
        path.write_text('{')
        self.assertEqual(fc.pending_evaluations(self.root,self.manifest),[42])

    def test_gate_rechecks_new_pending_eval_on_every_atomic_acquire(self):
        class Unavailable(Exception):pass
        original_calls=[]
        dispatch=types.SimpleNamespace(atomic_acquire=lambda *a,**kw:original_calls.append(a) or 'REAL_LEASE')
        guard_module=types.SimpleNamespace(ResourceUnavailable=Unavailable)
        self.launch(self.rows[0])
        fc.install_campaign_gate(dispatch,self.root,self.manifest,0)
        self.assertEqual(dispatch.atomic_acquire(None,guard_module,self.rows[1]['train'],{}),'REAL_LEASE')
        self.train_complete(self.rows[0])
        with self.assertRaises(Unavailable):dispatch.atomic_acquire(None,guard_module,self.rows[1]['train'],{})
        self.assertEqual(len(original_calls),1)
        self.assertEqual(dispatch.atomic_acquire(None,guard_module,self.rows[0]['evaluation'],{}),'REAL_LEASE')

    def test_pause_withdraws_unlaunched_train_but_preserves_eval_duty(self):
        class Unavailable(Exception):pass
        calls=[];dispatch=types.SimpleNamespace(atomic_acquire=lambda *a,**kw:calls.append(a) or 'LEASE')
        guard_module=types.SimpleNamespace(ResourceUnavailable=Unavailable)
        fc.pause(self.root,42,'synthetic technical failure')
        state=fc.read(self.root/'state.json');self.assertTrue(state['paused'])
        self.assertIsNone(fc.next_seed_to_launch(self.root,self.manifest,state))
        fc.install_campaign_gate(dispatch,self.root,self.manifest,0)
        with self.assertRaises(fc.AdmissionPaused):dispatch.atomic_acquire(None,guard_module,self.rows[1]['train'],{})
        self.assertEqual(dispatch.atomic_acquire(None,guard_module,self.rows[0]['evaluation'],{}),'LEASE')

    def test_completion_never_uses_ap_for_stopping_or_selection(self):
        self.train_complete(self.rows[0]);self.evaluation_complete(self.rows[0])
        self.assertTrue(fc.complete_training(self.rows[0]))
        self.assertTrue(fc.complete_evaluation(self.root,self.rows[0]))

    def test_profile_requires_real_measured_stage_and_sufficient_reservation(self):
        path=self.root/'fixture-profile.json'
        value=dict(schema='rgbir-independent-resource-profile-v1',status='COMPLETED',measurement_valid=True,stage='canary',
            reservation=self.resources,resources={'per_gpu_peak_vram_mib':{'2':7500},'peak_rss_mib':35000})
        fc.write_new(path,value)
        self.assertEqual(fc.profile_resources(path,'canary')[1],dict(vram_mib=9548,rss_mib=39096))
        for change in ({'measurement_valid':False},{'stage':'calibration'},{'reservation':dict(vram_mib=7000,rss_mib=40000)}):
            path.write_text(json.dumps(dict(value,**change)))
            with self.subTest(change=change),self.assertRaises(ValueError):fc.profile_resources(path,'canary')

    def test_formal_eval_reservation_uses_actual_peak_not_bootstrap_cap(self):
        path=self.root/'fixture-eval-profile.json'
        value=dict(schema='rgbir-independent-resource-profile-v1',status='COMPLETED',measurement_valid=True,stage='evaluation_profile',
            reservation=dict(vram_mib=16000,rss_mib=49152),resources={'per_gpu_peak_vram_mib':{'5':1370},'peak_rss_mib':6260})
        fc.write_new(path,value)
        self.assertEqual(fc.profile_resources(path,'evaluation_profile')[1],dict(vram_mib=1626,rss_mib=10356))
        value['resources']['peak_rss_mib']=100.2;value['resources']['per_gpu_peak_vram_mib']['5']=1370.2
        path.write_text(json.dumps(value))
        self.assertEqual(fc.profile_resources(path,'evaluation_profile')[1],dict(vram_mib=1627,rss_mib=8192))

    def test_actual_config_discovery_does_not_fabricate_missing_seed_or_readiness(self):
        folder=self.root/'configs';folder.mkdir()
        for seed in fc.ORDER:
            (folder/f'C1_{seed}.yaml').write_text(yaml.safe_dump(dict(arm='C1',source='paired',seed=seed,epochs=200,
                protocol_status='FROZEN',formal_training_authorized=True)))
        self.assertEqual(set(fc.discover_configs(folder)),set(fc.ORDER))
        (folder/'duplicate.yaml').write_bytes((folder/'C1_42.yaml').read_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate'):fc.discover_configs(folder)

    def test_initialize_preserves_venv_entry_when_symlink_resolve_targets_base(self):
        # Simulate symlink resolution on Windows, where unprivileged symlink
        # creation is unavailable. This exercises both actual initialize outputs.
        alias=self.root/'venv'/'bin'/'python';alias.parent.mkdir(parents=True);alias.write_text('synthetic alias')
        base=self.root/'base-python';base.write_text('synthetic base')
        configs=self.root/'frozen-configs';configs.mkdir()
        for seed in fc.ORDER:
            (configs/f'C1_{seed}.yaml').write_text(yaml.safe_dump(dict(arm='C1',source='paired',seed=seed,epochs=200,
                protocol_status='FROZEN',formal_training_authorized=True)))
        paths=[]
        for stage in ('canary','evaluation_profile'):
            path=self.root/(stage+'.json');paths.append(path)
            fc.write_new(path,dict(schema='rgbir-independent-resource-profile-v1',status='COMPLETED',measurement_valid=True,
                stage=stage,profile_key=stage,reservation=self.resources,
                resources={'per_gpu_peak_vram_mib':{'2':7500},'peak_rss_mib':35000}))
        args=types.SimpleNamespace(campaign=self.root/'prepared',release=self.root/'synthetic-release',repo=self.root/'synthetic-repo',
            formal_config_dir=configs,python=alias,canary_profile=paths[0],evaluation_profile=paths[1])
        fixture_modules=types.SimpleNamespace(require_valid_config=lambda *a,**kw:None,
            check_readiness=lambda *a:None,measured_reservation=lambda *a:None)
        real_resolve=Path.resolve
        def symlink_resolve(path,*args,**kwargs):
            return base if path==alias else real_resolve(path,*args,**kwargs)
        with patch.object(Path,'resolve',new=symlink_resolve),patch.object(fc,'require_data_path',side_effect=lambda p:p),\
                patch.object(fc,'module_from_release',return_value=fixture_modules):
            manifest=fc.initialize(args)
        self.assertEqual(manifest['python'],str(alias.absolute()))
        for row in manifest['seeds']:
            self.assertEqual(row['train']['command'][0],str(alias.absolute()))
            self.assertEqual(row['evaluation']['command'][0],str(alias.absolute()))

    def test_worker_calls_real_dispatch_api_train_then_evaluation(self):
        calls=[];row=self.rows[0]
        def run_job(job,*args):
            calls.append(job['stage'])
            if job['stage']=='train':self.train_complete(row);self.training_measurement(row)
            else:self.evaluation_complete(row)
        dispatch=types.SimpleNamespace(POLL_SECONDS=30,atomic_acquire=lambda *a:None,load_guard=lambda p:object(),run_job=run_job)
        with patch.object(fc,'module_from_release',return_value=dispatch):fc.worker(self.root,42)
        self.assertEqual(calls,['train','evaluation'])
        self.assertEqual(fc.read(self.root/'state.json')['seeds']['42']['status'],'DONE')

    def test_worker_recovery_keeps_completed_training_and_only_evaluates(self):
        row=self.rows[0];self.train_complete(row);self.training_measurement(row);calls=[]
        def run_job(job,*args):calls.append(job['stage']);self.evaluation_complete(row)
        dispatch=types.SimpleNamespace(POLL_SECONDS=30,atomic_acquire=lambda *a:None,load_guard=lambda p:object(),run_job=run_job)
        with patch.object(fc,'module_from_release',return_value=dispatch):fc.worker(self.root,42)
        self.assertEqual(calls,['evaluation'])

    def test_completed_training_without_guard_success_cannot_silently_evaluate(self):
        row=self.rows[0];self.train_complete(row)
        dispatch=types.SimpleNamespace(POLL_SECONDS=30,atomic_acquire=lambda *a:None,load_guard=lambda p:object(),
            run_job=lambda *a:self.fail('must not evaluate a technically unresolved train'))
        with patch.object(fc,'module_from_release',return_value=dispatch),self.assertRaisesRegex(RuntimeError,'resource receipt'):
            fc.worker(self.root,42)
        self.assertTrue(fc.read(self.root/'state.json')['paused'])

    def test_partial_evaluation_publication_does_not_crash_observer(self):
        row=self.rows[0];self.train_complete(row);self.evaluation_complete(row)
        Path(row['evaluation']['result_receipt']).write_text('{')
        self.assertFalse(fc.complete_evaluation(self.root,row))
        self.assertEqual(fc.pending_evaluations(self.root,self.manifest),[])

    def test_technical_failure_keeps_raw_attempt_and_pauses_future_launches(self):
        row=self.rows[0];Path(row['run']).mkdir(parents=True);raw=Path(row['run'])/'failure.txt';raw.write_text('original evidence')
        dispatch=types.SimpleNamespace(POLL_SECONDS=30,atomic_acquire=lambda *a:None,load_guard=lambda p:object(),run_job=lambda *a:self.fail('must not relaunch'))
        with patch.object(fc,'module_from_release',return_value=dispatch),self.assertRaisesRegex(RuntimeError,'Incomplete prior'):
            fc.worker(self.root,42)
        self.assertEqual(raw.read_text(),'original evidence');self.assertTrue(fc.read(self.root/'state.json')['paused'])
        self.assertTrue(list(self.root.glob('worker_42_failure_*.json')))


if __name__=='__main__':unittest.main(verbosity=2)
