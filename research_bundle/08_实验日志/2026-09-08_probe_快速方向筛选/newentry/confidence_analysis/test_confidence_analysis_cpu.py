"""Four synthetic known truths; never reads new AP or launches GPU."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import analyze_confidence as a


def receipt(arm):
    metrics = dict(mAP50_95=.3, AP50=.7, AP75=.25, precision=.75, recall=.6)
    subset = dict(student_data_yaml='/fixed/subset/visible.yaml', privileged_data_yaml='/fixed/subset/infrared.yaml',
                  paired_train_mapping='/fixed/subset/pairs.json')
    return dict(status='DIRECTION_EVALUATION_COMPLETED', scope=a.SCOPE, endpoint=a.ENDPOINT,
        dataset='llvip', arm=arm, seed=42, single_seed=True, epochs=3, independent_lr_horizon=3,
        expected_train_images=2048, full_dev_images=2406, full_dev_gt_objects=7879,
        observed_images=2406, gt_objects_captured=7879, metric_units='fraction_0_to_1',
        formal_e200_complete=False, formal_paper_gain_claim=False, accepted_endpoint_claim=False,
        official_test_accessed=False, new_hash_computed=False,
        kd_coefficient=0. if arm == 'N' else .1, **metrics,
        per_class=[dict(class_id=0, name='person', **{k:metrics[k] for k in a.AP})],
        training_model='/fixed/visible42/last.pt', training_configuration='/new/'+arm+'/config.yaml',
        training_completion='/new/'+arm+'/completion.json', training_subset_identity=subset,
        evaluation_identity_projection=dict(dataset='llvip', subset_dev_roster_exact_to_full=True,
            expected_dev_images=2406, expected_dev_gt_objects=7879, actual_evaluation_data_yaml='/full/visible.yaml',
            training_data_yaml=subset['student_data_yaml'], training_model='/fixed/visible42/last.pt'),
        bn_training_evidence=dict(bn_running_buffers_unchanged=True, bn_affine_trainable=True, bn_buffer_count=243), seconds=15.)


class Tests(unittest.TestCase):
    def test_known_raw_fraction_and_signed_pp(self):
        n, c = receipt('N'), receipt('C0')
        c.update(mAP50_95=.301, AP50=.699, AP75=.25, recall=.61)
        c['per_class'][0].update(**{k:c[k] for k in a.AP})
        result = a.summarize({'N':n,'C0':c})
        self.assertEqual(result['raw_fraction']['C0']['mAP50_95'],.301)
        self.assertAlmostEqual(result['comparison']['delta_pp']['mAP50_95'],.1)
        self.assertAlmostEqual(result['comparison']['delta_pp']['AP50'],-.1)
        self.assertAlmostEqual(result['comparison']['delta_pp']['recall'],1.)
        self.assertEqual(result['comparison']['direction']['AP75'],'zero')
        self.assertIsNone(result['standard_deviation'])

    def test_old_N_scope_and_identity_mismatches_rejected(self):
        mutations = [lambda r:r.update(scope='DIRECTION_FT3_BNFROZEN'), lambda r:r.update(seed=0),
            lambda r:r.update(expected_train_images=9619), lambda r:r.update(training_model='/other/last.pt'),
            lambda r:r['training_subset_identity'].update(paired_train_mapping='/changed/pairs.json'),
            lambda r:r['evaluation_identity_projection'].update(actual_evaluation_data_yaml='/other/full.yaml'),
            lambda r:r.update(kd_coefficient=.2), lambda r:r.update(full_dev_images=200)]
        for mutate in mutations:
            n,c=receipt('N'),receipt('C0');mutate(c)
            with self.assertRaises(ValueError):a.summarize({'N':n,'C0':c})
        with self.assertRaises(ValueError):a.summarize({'C0':receipt('C0')})

    def test_fraction_person_macro_and_real_BN_contract(self):
        mutations=[lambda r:r.update(mAP50_95=float('nan')),lambda r:r.update(recall=61.),
            lambda r:r.update(AP50=True),lambda r:r['per_class'][0].update(name='car'),
            lambda r:r['per_class'][0].update(AP75=.9),lambda r:r.update(observed_images=True),
            lambda r:r['bn_training_evidence'].update(bn_running_buffers_unchanged=False),
            lambda r:r.update(training_subset_identity={})]
        for mutate in mutations:
            r=receipt('C0');mutate(r)
            with self.assertRaises(ValueError):a.validate(r,'C0')

    def test_fixed_paths_collect_and_conflicting_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for arm in a.ARMS:
                folder=p/'evaluations'/arm;folder.mkdir(parents=True)
                (folder/'direction_evaluation_receipt.json').write_text(json.dumps(receipt(arm)),encoding='utf-8')
            result=a.collect(p)
            self.assertEqual(result['status'],'COMPLETE_CONFIDENCE_PAIR_READOUT')
            self.assertEqual(len(result['input_sources']),2)
            (p/'evaluations/C0/direction_evaluation_failure.json').write_text('{}')
            with self.assertRaises(ValueError):a.collect(p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    value=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),synthetic_known_truth_only=True,
        actual_new_AP_read=False,GPU_used=False,new_hash_computed=False)
    with args.receipt.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
