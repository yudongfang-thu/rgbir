import copy
import json
from pathlib import Path
import tempfile
import unittest

from analyze_independent import (ERROR_CONTRACT, analyze_records, harm_review, paired_comparison,
                                  passes_gain, summarize, to_percent, validate_endpoint,
                                  verify_analyzer_acceptance, load_endpoint, _legacy_loader)


def record(arm, seed, ap, source='paired', dataset='drone'):
    return {'arm': arm, 'source': source, 'seed': seed, 'dataset': dataset,
            'protocol_id': 'frozen-' + dataset, 'status': 'complete', 'receipt_validation': 'passed',
            'official_test_accessed': False, 'endpoint': 'fixed_budget_last_ema',
            'frozen_recipe': {'epochs': 200, 'workers': 4, 'augmentation': {'scale': 0.5}},
            'evaluator_contract': {'version': 'fixed-dev-v1', 'half': False},
            'roster': ['image-a', 'image-b'], 'expected_val_images': 2,
            'metrics_percent': {'mAP50_95': ap, 'AP50': 70 + ap - 50,
                                'AP75': 60 + ap - 50, 'precision': 80, 'recall': 75},
            'per_class_percent': {'0': ap, '1': ap},
            'error_analysis': {'contract': dict(ERROR_CONTRACT), 'background_fp_per_image': 0.2},
            'localization_diagnostics_consistent': True}


def group(arm, values, source='paired', dataset='drone'):
    return [record(arm, seed, ap, source, dataset) for seed, ap in zip((0, 42, 123), values)]


def base():
    return group('N', [50, 50, 50]) + group('C0', [50.2, 50.2, 50.2])


def file_fixture(root):
    """Known synthetic receipt chain, never a real experimental AP endpoint."""
    root.mkdir(exist_ok=True)
    (root / 'run_evidence').mkdir(); (root / 'eval_evidence').mkdir()
    checkpoint = str(root).replace('\\', '/') + '/weights/last.pt'
    complete = {'status': 'training_completed', 'last_epoch': 200, 'epochs_configured': 200,
                'official_test_accessed': False, 'seed': 42, 'arm': 'C1', 'checkpoint': checkpoint}
    metric = {'AP50': .7, 'AP75': .6, 'mAP50_95': .5, 'precision': .8, 'recall': .75,
              'metric_units': 'fraction_0_to_1', 'official_test_accessed': False,
              'split': 'val', 'endpoint': 'fixed_budget_last_ema', 'seed': 42,
              'arm': 'C1', 'source': 'paired', 'method_id': 'truth-fixture', 'checkpoint': checkpoint}
    recipe = {key: 1 for key in _legacy_loader().RECIPE_FIELDS}
    recipe.update(epochs=200, workers=4, model='generic.pt', augmentation={'scale': .5},
                  paths={'student_data_yaml': 'rgb.yaml'})
    def write(relative, data):
        (root / relative).write_text(json.dumps(data), encoding='utf-8')
    write('completion_receipt.json', complete); write('evaluation_val.json', metric)
    write('run_evidence/metric.json', complete); write('eval_evidence/metric.json', metric)
    write('run_evidence/config.yaml', recipe)
    for folder, kind, role in (('run_evidence', 'train', 'development_train'), ('eval_evidence', 'eval', 'development_val')):
        receipt = {'terminal_status': 'COMPLETED', 'run_kind': kind, 'data_role': role,
                   'seed': 42, 'dataset': 'drone', 'inputs': {'arm': 'C1', 'method_id': 'truth-fixture'},
                   'resources': {'lease_id': 'synthetic-fixture-lease'}, 'metric_snapshots': ['metric.json'],
                   'source_snapshots': {'config': ['config.yaml']} if kind == 'train' else {'split_roster': ['roster.txt']}}
        write(folder + '/run_receipt.json', receipt)
    (root / 'evaluation_val_roster.txt').write_text('image-a\nimage-b\n')
    (root / 'eval_evidence/roster.txt').write_text('image-a\nimage-b\n')
    return {'arm': 'C1', 'source': 'paired', 'seed': 42, 'source_arm': 'C1', 'path': str(root),
            'protocol_id': 'truth-fixture', 'evaluator_contract': {'version': 'test'}, 'expected_val_images': 2}


class AnalyzerTests(unittest.TestCase):
    def test_units(self):
        self.assertAlmostEqual(to_percent(0.543, 'fraction_0_to_1'), 54.3)
        self.assertEqual(to_percent(54.3, 'percent_0_to_100'), 54.3)
        for value, units in ((True, 'fraction_0_to_1'), (1.1, 'fraction_0_to_1'),
                             (float('nan'), 'percent_0_to_100'), (54, 'unknown')):
            with self.assertRaises(ValueError): to_percent(value, units)

    def test_sample_sd_not_population_sd(self):
        s = summarize([1, 2, 3])
        self.assertEqual(s['mean'], 2)
        self.assertEqual(s['sample_sd'], 1)
        self.assertIsNone(summarize([1])['sample_sd'])

    def test_true_paired_deltas_and_boundary(self):
        rows = group('N', [40, 50, 60]) + group('C1', [40.1, 50.2, 60.3])
        p = paired_comparison(rows, 'C1', 'N')
        self.assertAlmostEqual(p['stats_pp']['mAP50_95']['mean'], 0.2)
        self.assertAlmostEqual(p['stats_pp']['mAP50_95']['sample_sd'], 0.1)
        self.assertTrue(passes_gain(p))
        self.assertTrue(passes_gain(paired_comparison(group('N', [50]*3)+group('C1', [50.1]*3), 'C1', 'N')))

    def test_negative_seed_blocks_practical_upgrade(self):
        rows = base() + group('C1', [50.6, 50.1, 50.6])
        result = analyze_records(rows)['datasets']['drone']['classification']
        self.assertEqual(result['proposed_decision'], 'KEEP_C0')

    def test_missing_seed_and_invalid_receipt_do_not_upgrade(self):
        rows = base() + group('C1', [51, 51, 51])
        rows[-1]['receipt_validation'] = 'failed'
        result = analyze_records(rows)['datasets']['drone']['classification']
        self.assertEqual(result['proposed_decision'], 'AWAIT_C1_ENDPOINTS')
        self.assertFalse(passes_gain(paired_comparison(rows[:-1], 'C1', 'N')))

    def test_cross_seed_recipe_change_withholds_aggregate(self):
        rows = group('N', [50]*3) + group('C1', [51]*3)
        for r in rows:
            if r['seed'] == 123: r['frozen_recipe']['workers'] = 8
        p = paired_comparison(rows, 'C1', 'N')
        self.assertFalse(p['complete_three_seed_pairing'])
        self.assertIsNone(p['stats_pp'])

    def test_evaluator_roster_endpoint_and_test_flag_rejected(self):
        for key, value in (('evaluator_contract', None), ('roster', ['image-a', 'image-a']),
                           ('expected_val_images', 3), ('endpoint', 'best'), ('official_test_accessed', True)):
            r = record('C1', 42, 51); r[key] = value
            self.assertFalse(validate_endpoint(r)['valid'], key)
        rows = group('N', [50]*3) + group('C1', [51]*3)
        rows[-1]['evaluator_contract']['half'] = True
        self.assertFalse(paired_comparison(rows, 'C1', 'N')['complete_three_seed_pairing'])

    def test_duplicate_attempts_not_outcome_selected(self):
        rows = base() + [copy.deepcopy(base()[0])]
        with self.assertRaises(ValueError): analyze_records(rows)

    def test_unaccepted_analyzer_cannot_expand(self):
        result = analyze_records(base() + group('C1', [50.5]*3), True)
        c = result['datasets']['drone']['classification']
        self.assertEqual(c['proposed_decision'], 'PROMOTE_C1')
        self.assertFalse(c['auto_expansion_eligible'])
        self.assertEqual(result['analyzer_acceptance'], 'NOT_ACCEPTED')

    def test_simple_target_only_fallback(self):
        rows = base() + group('C1', [50.5]*3) + group('C1_y', [50.6]*3)
        c = analyze_records(rows, True, True)['datasets']['drone']['classification']
        self.assertEqual(c['proposed_decision'], 'PROMOTE_C1_Y')
        self.assertTrue(c['auto_expansion_eligible'])
        self.assertFalse(c['non_target_content_supported'])

    def test_fallback_requires_y_gain_over_c0(self):
        rows = base() + group('C1', [50.24]*3) + group('C1_y', [50.25]*3)
        c = analyze_records(rows)['datasets']['drone']['classification']
        self.assertEqual(c['proposed_decision'], 'KEEP_C0')

    def test_non_target_content_has_positive_three_seed_requirement(self):
        rows = base() + group('C1', [50.6]*3) + group('C1_y', [50.5]*3)
        c = analyze_records(rows)['datasets']['drone']['classification']
        self.assertEqual(c['proposed_decision'], 'PROMOTE_C1')
        self.assertTrue(c['non_target_content_supported'])

    def test_harm_requires_review_never_early_stop(self):
        rows = base() + group('C1', [50.6]*3)
        for r in rows:
            if r['arm'] == 'C1':
                r['per_class_percent']['1'] = 49
                r['error_analysis']['background_fp_per_image'] = 0.3
        c = analyze_records(rows, True, True)['datasets']['drone']['classification']
        self.assertEqual(c['expansion_status'], 'REVIEW_REQUIRED')
        self.assertFalse(c['auto_expansion_eligible'])
        self.assertFalse(c['harm']['stop_training'])
        self.assertEqual(len(c['harm']['triggers']), 2)

    def test_missing_fixed_threshold_diagnostics_cannot_be_replaced_by_recall(self):
        rows = base() + group('C1', [50.6]*3)
        rows[-1].pop('error_analysis')
        self.assertEqual(harm_review(rows, 'C1')['status'], 'INCOMPLETE')

    def test_l_needs_independent_content_and_consistency(self):
        rows = group('N', [50]*3) + group('L1', [50.3]*3) + group('L_GT', [50.4]*3)
        loc = analyze_records(rows, True, True)['datasets']['drone']['localization']
        self.assertEqual(loc['proposed_decision'], 'L1_EFFECTIVE')
        self.assertFalse(loc['teacher_content_supported'])
        rows[-4]['localization_diagnostics_consistent'] = False
        loc = analyze_records(rows, True, True)['datasets']['drone']['localization']
        self.assertEqual(loc['expansion_status'], 'AWAIT_LOCALIZATION_DIAGNOSTICS')

    def test_gt_and_y_do_not_complete_four_arms(self):
        rows = base() + group('C1', [50.6]*3) + group('C1_y', [50.5]*3)
        out = analyze_records(rows)['datasets']['drone']['four_arm_attribution']['C1']
        self.assertFalse(out['complete_four_arms'])
        rows += group('C1', [50.1]*3, 'shuffled') + group('C1', [50.2]*3, 'same_modal')
        self.assertTrue(analyze_records(rows)['datasets']['drone']['four_arm_attribution']['C1']['complete_four_arms'])

    def test_llvip_never_borrows_drone_n(self):
        rows = group('N', [50]*3, dataset='drone') + group('L1', [60]*3, dataset='llvip')
        loc = analyze_records(rows)['datasets']['llvip']['localization']
        self.assertEqual(loc['proposed_decision'], 'AWAIT_L1_ENDPOINTS')

    def test_acceptance_requires_reviewed_source_copies(self):
        self.assertFalse(verify_analyzer_acceptance(None))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'review.json'
            p.write_text(json.dumps({'status': 'ACCEPTED', 'reviewer': 'external'}))
            self.assertFalse(verify_analyzer_acceptance(p))

    def test_file_loader_verifies_complete_receipt_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'; spec = file_fixture(root)
            actual = load_endpoint(spec, tmp)
            self.assertTrue(validate_endpoint(actual)['valid'])
            self.assertEqual(actual['metrics_percent']['mAP50_95'], 50)

    def test_file_loader_rejects_failed_or_tampered_receipt(self):
        for target, key, value in (('run_evidence/run_receipt.json', 'terminal_status', 'FAILED'),
                                    ('evaluation_val.json', 'mAP50_95', .99)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'run'; spec = file_fixture(root)
                path = root / target; data = json.loads(path.read_text()); data[key] = value
                path.write_text(json.dumps(data))
                actual = load_endpoint(spec, tmp)
                self.assertEqual(actual['status'], 'invalid')
                self.assertFalse(validate_endpoint(actual)['valid'])

    def test_manifest_cannot_relabel_paired_as_same_modal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'; spec = file_fixture(root)
            spec['source'] = 'same_modal'
            self.assertEqual(load_endpoint(spec, tmp)['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
