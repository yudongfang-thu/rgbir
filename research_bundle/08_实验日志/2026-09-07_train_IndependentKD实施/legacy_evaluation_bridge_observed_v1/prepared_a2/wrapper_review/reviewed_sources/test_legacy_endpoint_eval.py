"""CPU tests. All temporary checkpoints/receipts/objects are SYNTHETIC fixtures.

Real historical snapshots/profile are read only for metadata tests. No inference,
GPU context, source acceptance, or new AP evaluation is performed by this suite.
"""
import gzip
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import yaml
import legacy_checkpoint_evaluate as ev
import prepare_queue as queue

HERE=Path(__file__).resolve().parent
LOG=HERE.parent
WORKSPACE=LOG.parents[1]
RELEASE=WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
SNAPSHOT=LOG/'snapshots/2026-09-07T140740.823638_0800/raw'
PROFILE=LOG/'remote_admission_1532/admission_queue_attempt5/evaluator_profile_a2_resource_profile.json'


class LegacyEndpointEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer=ev.frozen_module(RELEASE,'analyze_independent')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def original_config(self,name):
        path=SNAPSHOT/name
        return self.analyzer._bound_configs(path,ev.read(path/'run_evidence/run_receipt.json'),'run_evidence')[0]

    def test_six_real_metadata_endpoints_and_actual_seed_not_yaml_default(self):
        for arm,prefix in (('N','N'),('C0','C')):
            for seed in (0,42,123):
                with self.subTest(arm=arm,seed=seed):
                    cfg=self.original_config(prefix+str(seed));before=json.dumps(cfg,sort_keys=True)
                    derived=ev.derived_config(cfg,arm,seed)
                    result=ev.load_legacy_metadata(SNAPSHOT/(prefix+str(seed)),arm,seed,derived,self.analyzer)
                    self.assertEqual(result['record']['status'],'complete')
                    self.assertEqual(cfg['seed'],42)
                    self.assertEqual(derived['seed'],seed)
                    self.assertEqual(derived['arm'],ev.ARMS[arm])
                    self.assertEqual(derived['kd_weight'],cfg['kd_weight'])
                    self.assertEqual(json.dumps(cfg,sort_keys=True),before)

    def test_native_config_change_and_wrong_arm_do_not_pass_legacy_identity(self):
        cfg=ev.derived_config(self.original_config('N42'),'N',42)
        with self.assertRaisesRegex(ValueError,'differs from original bound'):
            ev.load_legacy_metadata(SNAPSHOT/'N42','N',42,dict(cfg,batch=16),self.analyzer)
        with self.assertRaisesRegex(ValueError,'Invalid actual legacy endpoint'):
            ev.load_legacy_metadata(SNAPSHOT/'N42','C0',42,cfg,self.analyzer)

    def test_changed_old_metric_cannot_bypass_actual_snapshot_binding(self):
        path=self.root/'SYNTHETIC_mutated_copy';shutil.copytree(SNAPSHOT/'N42',path)
        cfg=ev.derived_config(self.original_config('N42'),'N',42)
        value=ev.read(path/'evaluation_val.json');value['AP50']+=.001
        (path/'evaluation_val.json').write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'bound metric snapshot does not match'):
            ev.load_legacy_metadata(path,'N',42,cfg,self.analyzer)

    def test_fixed_last_checkpoint_physical_path_is_checked(self):
        old=self.root/'old';last=old/'weights/last.pt';last.parent.mkdir(parents=True)
        last.write_bytes(b'SYNTHETIC NOT A MODEL')
        self.assertEqual(ev.checkpoint_identity(old,{'checkpoint':str(last)}),last)
        with self.assertRaises(ValueError):ev.checkpoint_identity(old,{'checkpoint':str(old/'weights/best.pt')})

    def test_new_output_cannot_overwrite_or_be_inside_original(self):
        old=self.root/'original';old.mkdir()
        self.assertEqual(ev.validate_output(self.root/'new_attempt',old),self.root/'new_attempt')
        for target in (old,old/'anything',self.root):
            with self.subTest(target=target),self.assertRaises(ValueError):ev.validate_output(target,old)
        existing=self.root/'attempt';existing.mkdir()
        with self.assertRaises(FileExistsError):ev.validate_output(existing,old)

    def test_original_evidence_and_checkpoint_change_are_detected(self):
        checkpoint=self.root/'SYNTHETIC_last.pt';checkpoint.write_bytes(b'fixture')
        source=self.root/'old_metric.json';source.write_text('{}',encoding='utf-8')
        copy=self.root/'copy.json';copy.write_bytes(source.read_bytes())
        origin=dict(original_checkpoint=str(checkpoint),checkpoint_stat_before=ev.file_state(checkpoint),
                    input_files=[dict(original=str(source),copy=str(copy))])
        ev.verify_origins(origin)
        source.write_text('{"changed":true}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'legacy evidence changed'):ev.verify_origins(origin)
        checkpoint.write_bytes(b'fixture modified')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):ev.verify_origins(origin)

    def test_historical_comparison_exact_mismatch_nonfinite_and_units(self):
        old=dict({k:.25 for k in ev.METRICS},metric_units='fraction_0_to_1')
        self.assertEqual(ev.historical_comparison(old,dict(old))['status'],'EXACT')
        result=ev.historical_comparison(old,dict(old,AP75=.250000001))
        self.assertEqual(result['status'],'MISMATCH_REVIEW_REQUIRED')
        self.assertFalse(result['old_results_overwritten'])
        with self.assertRaises(ValueError):ev.historical_comparison(old,dict(old,AP50=float('nan')))
        with self.assertRaises(ValueError):ev.historical_comparison(old,dict(old,AP50=True))
        with self.assertRaises(ValueError):ev.historical_comparison(dict(old,metric_units='percent'),old)

    def test_all_five_classes_cannot_be_omitted_or_duplicated(self):
        rows=[dict(class_id=i,AP50=.1,AP75=.1,mAP50_95=.1) for i in range(5)]
        ev.require_all_classes({'per_class':rows},5)
        for wrong in (rows[:4],rows[:4]+[rows[0]],rows[:4]+[dict(rows[4],AP75=float('inf'))]):
            with self.assertRaises(ValueError):ev.require_all_classes({'per_class':wrong},5)

    def test_actual_measured_reservations_and_no_bootstrap_substitution(self):
        profile=ev.read(PROFILE)
        self.assertEqual(queue.measured_resources(profile),{'vram_mib':1626,'rss_mib':10356})
        with self.assertRaises(ValueError):queue.measured_resources(dict(profile,stage='canary'))
        with self.assertRaises(ValueError):queue.measured_resources(dict(profile,measurement_valid=False))
        with self.assertRaises(ValueError):queue.measured_resources(dict(profile,resources={'per_gpu_peak_vram_mib':{},'peak_rss_mib':0}))

    def test_six_prepared_jobs_dynamic_gpu_new_outputs_and_no_old_random(self):
        manifest=ev.read(HERE/'candidate_bundle_v1/queue_candidate.json')
        self.assertEqual(manifest['status'],'PREPARED_NOT_RUN_NOT_REVIEWED')
        self.assertEqual(len(manifest['jobs']),6)
        for job,entry in zip(manifest['jobs'],manifest['entries']):
            self.assertEqual(job['stage'],'evaluation');self.assertFalse(job['formal'])
            self.assertEqual(job['vram_mib'],1626);self.assertEqual(job['rss_mib'],10356)
            self.assertFalse(any(k in job for k in ('device','gpu','gpus','candidate_gpus')))
            self.assertIn('--review-receipt',job['command'])
            self.assertFalse(entry['new_evaluation_attempt'].startswith(entry['original_run']+'/'))
            self.assertIn(entry['normalized_method_arm'],('N','C0'))

    def test_absent_or_draft_review_does_not_authorize_execution(self):
        path=self.root/'SYNTHETIC_review.json';ev.write_new(path,dict(status='DRAFT',schema=ev.REVIEW_SCHEMA,reviewer='fixture'))
        with self.assertRaisesRegex(ValueError,'independent source review'):ev.require_review(path)
        path.write_text(json.dumps(dict(status='ACCEPTED',schema=ev.REVIEW_SCHEMA,reviewer='fixture',source_files=[])),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'bind all'):ev.require_review(path)

    def test_review_binds_actual_four_source_bytes_and_rejects_mutation(self):
        rows=[]
        for name in ev.REVIEW_FILES:
            destination=self.root/name;destination.write_bytes((HERE/name).read_bytes())
            rows.append(dict(relative=name,accepted_copy=str(destination)))
        review=self.root/'SYNTHETIC_review_fixture.json'
        ev.write_new(review,dict(schema=ev.REVIEW_SCHEMA,status='ACCEPTED',reviewer='SYNTHETIC_CPU_TEST_ONLY',source_files=rows))
        self.assertEqual(ev.require_review(review)['reviewer'],'SYNTHETIC_CPU_TEST_ONLY')
        destination=self.root/ev.REVIEW_FILES[0];destination.write_bytes(destination.read_bytes()+b'\n# fixture change\n')
        with self.assertRaisesRegex(ValueError,'source bytes differ'):ev.require_review(review)

    def test_origin_manifest_is_readonly_evidence_not_new_training(self):
        # save_origins itself uses actual old source metadata but a synthetic
        # temporary checkpoint, and writes only this fixture output directory.
        cfg=ev.derived_config(self.original_config('N42'),'N',42)
        metadata=ev.load_legacy_metadata(SNAPSHOT/'N42','N',42,cfg,self.analyzer)
        # Local historical downloads intentionally omit the large train mapping.
        # This copy-layout fixture supplies only actually downloaded files;
        # production origin_files still requires every original server reference.
        metadata=copy.deepcopy(metadata)
        for group,paths in metadata['train_receipt']['source_snapshots'].items():
            metadata['train_receipt']['source_snapshots'][group]=[p for p in paths if (SNAPSHOT/'N42/run_evidence'/p).is_file()]
        checkpoint=self.root/'SYNTHETIC_last.pt';checkpoint.write_bytes(b'fixture')
        out=self.root/'new';out.mkdir()
        origin=ev.save_origins(SNAPSHOT/'N42',metadata,out,checkpoint)
        self.assertFalse(origin['new_training_receipt_created'])
        self.assertFalse((out/'completion_receipt.json').exists())
        self.assertFalse((out/'run_evidence').exists())
        self.assertTrue((out/'origin_evidence/run_evidence/run_receipt.json').is_file())
        ev.verify_origins(origin)

    def test_single_metric_difference_retains_observation_but_no_completed_result(self):
        checkpoint=self.root/'SYNTHETIC_last.pt';checkpoint.write_bytes(b'fixture')
        origin=dict(original_checkpoint=str(checkpoint),checkpoint_stat_before=ev.file_state(checkpoint),input_files=[])
        output=self.root/'mismatch';output.mkdir()
        old=dict({k:.25 for k in ev.METRICS},metric_units='fraction_0_to_1')
        observed=dict(old,AP75=.251,status='evaluated_pending_historical_check')
        with self.assertRaisesRegex(ValueError,'five-metric mismatch'):
            ev.persist_checked_result(output,observed,old,origin)
        self.assertTrue((output/'evaluation_observed.json').is_file())
        self.assertEqual(ev.read(output/'historical_metric_comparison.json')['status'],'MISMATCH_REVIEW_REQUIRED')
        self.assertFalse((output/'evaluation_val.json').exists())
        self.assertFalse((output/'reevaluation_receipt.json').exists())
        exact=self.root/'exact';exact.mkdir()
        target,value=ev.persist_checked_result(exact,dict(old),old,origin)
        self.assertTrue(target.is_file());self.assertEqual(value['status'],'completed')

    def test_new_eval_layout_is_readable_by_existing_object_analyzer(self):
        # Synthetic full-population fixture validates only on-disk interface.
        spec=importlib.util.spec_from_file_location('_legacy_diag_object_reader',LOG/'object_error_analysis.py')
        reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
        output=self.root/'SYNTHETIC_eval_layout';output.mkdir()
        cfg=ev.derived_config(self.original_config('N42'),'N',42)
        roster=[str(output/('synthetic_'+str(i)+'.jpg')) for i in range(1469)]
        objects=output/'objects.jsonl.gz'
        with gzip.open(objects,'xt',encoding='utf-8') as stream:
            for image in roster:
                stream.write(json.dumps(dict(image=image,canvas_shape=[640,640],original_shape=[640,640],
                    gt_boxes=[],gt_classes=[],pred_boxes=[],pred_classes=[],pred_confidence=[]))+'\n')
        contract=output/'evaluation_contract.json'
        ev.write_new(contract,dict(schema='rgbir-evaluation-contract-v1',roster=roster,expected_val_images=1469,
            observed_images=1469,endpoint='fixed_budget_last_ema',official_test_accessed=False))
        metric=output/'evaluation_val.json'
        raw=dict(status='completed',dataset='dronevehicle',seed=42,arm='weight0',source='paired',
            method_id=cfg['method_id'],checkpoint='/SYNTHETIC/last.pt',endpoint='fixed_budget_last_ema',
            split='val',official_test_accessed=False,objects=str(objects),evaluation_contract=str(contract))
        ev.write_new(metric,raw)
        effective=output/'evaluation_config.yaml';effective.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        folder=output/'eval_evidence';folder.mkdir()
        for path in (metric,objects,contract,effective):ev.copy_new(path,folder/path.name)
        ev.write_new(folder/'run_receipt.json',dict(terminal_status='COMPLETED',run_kind='eval',data_role='development_val',
            dataset='dronevehicle',seed=42,inputs={k:raw[k] for k in ('method_id','arm','source','checkpoint','endpoint')},
            metric_snapshots=[metric.name,objects.name],source_snapshots={'config':[effective.name,contract.name]}))
        loaded=reader.load_evaluation(metric)
        self.assertEqual(loaded['nc'],5);self.assertEqual(len(loaded['rows']),1469)
        self.assertFalse((output/'run_evidence').exists())
        objects.write_bytes(b'SYNTHETIC corruption')
        with self.assertRaisesRegex(ValueError,'bytes not present'):reader.load_evaluation(metric)


if __name__=='__main__':unittest.main(verbosity=2)
