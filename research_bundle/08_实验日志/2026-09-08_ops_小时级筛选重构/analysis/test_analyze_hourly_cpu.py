"""Known-truth inputs only. Does not read any new experiment AP."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('hourly_analysis_under_test', HERE/'analyze_hourly.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    evaluations, trainings = {}, {}
    for j, arm in enumerate(a.ARMS):
        pc = []
        changes = {'N':[0]*5, 'C0':[.01]*5, 'C1':[.005,-.005,.02,0,-.01]}[arm]
        for i, (name, delta) in enumerate(zip(a.CLASSES, changes)):
            base = .2 + .1*i + delta
            pc.append(dict(class_id=i, name=name, mAP50_95=base, AP50=base+.2, AP75=base+.1))
        shared = dict(arm=arm, seed=42, dataset='dronevehicle', scope='HOURLY_SCREEN_FT', single_seed=True,
            endpoint=a.ENDPOINT, formal_e200_complete=False, formal_paper_gain_claim=False,
            new_hash_computed=False, official_test_accessed=False,
            checkpoint=dict(path='/fixture/runs/'+arm+'/weights/last.pt', bytes=100, mtime_ns=1),
            canary_receipt='/fixture/canaries/'+arm+'/canary.json')
        evaluations[arm] = dict(shared, status='HOURLY_SCREEN_EVALUATION_COMPLETED',
            epochs=3, independent_lr_horizon=3, full_dev_images=1469, full_dev_gt_objects=22462,
            metric_units='fraction_0_to_1', native_profile_binding='/fixed/binding.json', training_model='/N42.pt',
            per_class=pc, precision=.6, recall=.7, seconds=10.,
            **{k:sum(p[k] for p in pc)/5 for k in a.AP})
        trainings[arm] = dict(copy.deepcopy(shared), status='HOURLY_SCREEN_TRAINING_COMPLETED',
            model='/N42.pt', teacher='/IR42.pt', reference='/RGB42.pt', epochs_configured=3,
            classification_coefficient=a.COEFFICIENTS[arm], localization_coefficient=0.,
            last_epoch=3, batches=192, optimizer_updates=95, attempts=96, amp_skips=1,
            ema_updates=96, successful_updates=95, seconds=100.+10*j)
    completed = []
    for index, (arm, stage) in enumerate(a.SEQUENCE):
        suffix = {'canary':'/canaries/'+arm+'/canary.json',
                  'train':'/runs/'+arm+'/hourly_training_receipt.json',
                  'eval':'/evaluations/'+arm+'/hourly_evaluation_receipt.json'}[stage]
        completed.append(dict(id=str(index), arm=arm, stage=stage, receipt='/fixture'+suffix))
    queue = dict(status='HOURLY_SCREEN_MATRIX_COMPLETED', single_seed=True, seed=42, epochs=3,
        train_images=2048, batches_per_arm=192, new_hash_computed=False, formal_e200_complete=False,
        formal_paper_gain_claim=False, ap_adaptive=False, completed=completed, seconds=400.)
    return evaluations, trainings, queue


class Tests(unittest.TestCase):
    def test_known_macro_and_signed_pp(self):
        result = a.analyze(*fixture())
        self.assertAlmostEqual(result['values']['N']['percent']['mAP50_95'], 40.)
        for key, expected in [('C0-N',1.),('C1-N',.2),('C1-C0',-.8)]:
            self.assertAlmostEqual(result['differences'][key]['pp']['mAP50_95'], expected)
        self.assertEqual(result['timing']['training_seconds'], 330.)
        self.assertEqual(result['timing']['evaluation_seconds'], 30.)
        self.assertEqual(result['timing']['other_including_canaries_seconds'], 40.)

    def test_nonuniform_class_alignment_and_delta_mean(self):
        ev, tr, queue = fixture(); original = a.analyze(ev, tr, queue)
        ev['C1']['per_class'].reverse()
        result = a.analyze(ev, tr, queue)
        self.assertEqual(result, original)
        for delta in result['differences'].values():
            for metric in a.AP:
                self.assertAlmostEqual(sum(p[metric] for p in delta['per_class_pp'])/5, delta['pp'][metric])

    def test_no_sd_or_scientific_upgrade(self):
        result = a.analyze(*fixture())
        self.assertEqual(result['scope'], 'DESCRIPTIVE_SINGLE_SEED_FT3_ONLY')
        self.assertEqual(result['n_seeds'], 1)
        self.assertIsNone(result['sample_sd']); self.assertIsNone(result['significance_test'])
        for key in ('formal_gain_claim','automatic_expansion','E200_decision','C0_review_released','new_hash_computed'):
            self.assertIs(result[key], False)

    def reject(self, mutate):
        ev, tr, queue = fixture(); mutate(ev, tr, queue)
        with self.assertRaises((ValueError, KeyError, TypeError)): a.analyze(ev, tr, queue)

    def test_incomplete_or_E8_input_rejected(self):
        self.reject(lambda ev,tr,q:ev.pop('C1'))
        self.reject(lambda ev,tr,q:ev['C1'].update(status='FAILED'))
        self.reject(lambda ev,tr,q:ev['C1'].update(epochs=8))
        self.reject(lambda ev,tr,q:tr['C1'].update(batches=4504))

    def test_invalid_units_nonfinite_and_boolean_rejected(self):
        self.reject(lambda ev,tr,q:ev['C1'].update(metric_units='percent'))
        for value in (float('nan'), float('inf'), True, -0.1, 1.01):
            self.reject(lambda ev,tr,q,value=value:ev['C1'].update(AP50=value))
            self.reject(lambda ev,tr,q,value=value:ev['C1']['per_class'][0].update(AP75=value))

    def test_class_duplicate_mean_and_names_rejected(self):
        self.reject(lambda ev,tr,q:ev['C1']['per_class'][0].update(class_id=1))
        self.reject(lambda ev,tr,q:ev['C1']['per_class'][0].update(name='different'))
        self.reject(lambda ev,tr,q:ev['C1']['per_class'][0].update(AP50=.9))

    def test_identity_and_population_rejected(self):
        self.reject(lambda ev,tr,q:ev['C1'].update(seed=0))
        self.reject(lambda ev,tr,q:ev['C1'].update(full_dev_gt_objects=22461))
        self.reject(lambda ev,tr,q:ev['C1'].update(native_profile_binding='/different'))
        self.reject(lambda ev,tr,q:tr['C1'].update(model='/C0.pt'))
        self.reject(lambda ev,tr,q:tr['C1']['checkpoint'].update(mtime_ns=2))
        self.reject(lambda ev,tr,q:tr['C1'].update(canary_receipt='/different'))

    def test_counter_closure_and_no_single_arm_replenishment(self):
        self.reject(lambda ev,tr,q:tr['C1'].update(optimizer_updates=96))
        self.reject(lambda ev,tr,q:tr['C1'].update(ema_updates=95))
        self.reject(lambda ev,tr,q:tr['C1'].update(attempts=True))
        ev,tr,q=fixture();tr['C1'].update(optimizer_updates=94,successful_updates=94,amp_skips=2)
        result=a.analyze(ev,tr,q)
        self.assertEqual(result['training_counters']['C1']['amp_skips'],2)

    def test_fixed_arm_coefficients(self):
        self.reject(lambda ev,tr,q:tr['N'].update(classification_coefficient=.1))
        self.reject(lambda ev,tr,q:tr['C0'].update(classification_coefficient=0.))
        self.reject(lambda ev,tr,q:tr['C1'].update(classification_coefficient=.1))
        self.reject(lambda ev,tr,q:tr['C1'].update(localization_coefficient=.01))

    def test_forbidden_claim_and_test_flags_rejected(self):
        for key in ('formal_e200_complete','formal_paper_gain_claim','official_test_accessed','new_hash_computed'):
            self.reject(lambda ev,tr,q,key=key:ev['C1'].update({key:True}))
            self.reject(lambda ev,tr,q,key=key:tr['C1'].update({key:True}))

    def test_incomplete_reordered_duplicate_queue_rejected(self):
        self.reject(lambda ev,tr,q:q['completed'].pop())
        self.reject(lambda ev,tr,q:q['completed'].reverse())
        self.reject(lambda ev,tr,q:q['completed'][1].update(id='0'))
        self.reject(lambda ev,tr,q:q.update(status='FAILED'))
        self.reject(lambda ev,tr,q:q.update(ap_adaptive=True))

    def test_time_boundaries_and_invalid_time(self):
        self.reject(lambda ev,tr,q:q.update(seconds=300.))
        self.reject(lambda ev,tr,q:tr['C1'].update(seconds=float('nan')))
        self.reject(lambda ev,tr,q:ev['C1'].update(seconds=0.))

    def test_read_only_load_and_roster_drift(self):
        ev, tr, queue=fixture()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for arm in a.ARMS:
                train=root/'runs'/arm; train.mkdir(parents=True)
                val=root/'evaluations'/arm; val.mkdir(parents=True)
                (train/'hourly_training_receipt.json').write_text(json.dumps(tr[arm]))
                (val/'hourly_evaluation_receipt.json').write_text(json.dumps(ev[arm]))
                (val/'development_roster.txt').write_text(''.join('/image/{}.jpg\n'.format(i) for i in range(1469)))
            (root/'queue').mkdir(); (root/'queue/completion.json').write_text(json.dumps(queue))
            result, sources=a.load_campaign(root)
            self.assertEqual(len(sources),10)
            self.assertEqual(result,a.analyze(ev,tr,queue))
            p=root/'evaluations/C1/development_roster.txt'
            p.write_text(p.read_text().replace('/image/0.jpg','/changed/0.jpg'))
            with self.assertRaises(ValueError):a.load_campaign(root)

    def test_markdown_uses_pp_and_explicit_time_scope(self):
        text=a.comparison_markdown(a.analyze(*fixture()))
        self.assertIn('|C1-C0|-0.800000|',text)
        self.assertIn('400.000000',text)
        self.assertIn('canary',text)
        self.assertIn('REVIEW_REQUIRED',text)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    import sys
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),known_truth_only=True,new_AP_read=False,
        torch_imported='torch' in sys.modules,gpu_or_ssh=False,new_hash_computed=False,
        source=a.source_stat(HERE/'analyze_hourly.py'))
    with (HERE/'CPU_CHECKS_v2.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
