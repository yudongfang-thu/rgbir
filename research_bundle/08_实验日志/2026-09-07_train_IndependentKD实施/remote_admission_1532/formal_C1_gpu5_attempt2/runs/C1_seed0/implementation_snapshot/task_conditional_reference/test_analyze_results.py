"""Known-truth CPU fixtures; no GPU, datasets, or external results required."""
import json
import math
import tempfile
import unittest
from pathlib import Path
import yaml

import analyze_results as ar


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(root, arm, seed, percent, units='fraction_0_to_1'):
    path = root / f'{arm}{seed}'
    checkpoint = f'/remote/{arm}{seed}/weights/last.pt'
    complete = {'status': 'training_completed', 'last_epoch': 200, 'epochs_configured': 200,
                'seed': seed, 'arm': arm, 'checkpoint': checkpoint, 'official_test_accessed': False}
    metric = {'seed': seed, 'arm': arm, 'checkpoint': checkpoint, 'official_test_accessed': False,
              'method_id': 'FIXTURE', 'endpoint': 'fixed_budget_last_ema', 'split': 'val', 'metric_units': units}
    metric.update({name: percent / 100 if units == 'fraction_0_to_1' else percent for name in ar.METRICS})
    write_json(path / 'completion_receipt.json', complete)
    write_json(path / 'evaluation_val.json', metric)
    config = {'model': '/pretrain/yolo11n.pt', 'imgsz': 640, 'epochs': 200,
              'batch': 32, 'nbs': 64, 'workers': 4, 'optimizer': 'SGD',
              'lr0': .01, 'lrf': .01, 'momentum': .937, 'weight_decay': .0005,
              'warmup_epochs': 3., 'warmup_momentum': .8, 'warmup_bias_lr': .1,
              'cos_lr': False, 'close_mosaic': 0, 'patience': 0, 'amp': True,
              'deterministic': True, 'torch_version': '2.10.0+cu128',
              'ultralytics_version': '8.4.115', 'augmentation': {'mosaic': 0., 'scale': .5},
              'paths': {'student_data_yaml': '/data/rgb.yaml'}, 'seed': 42,
              'teacher': '/teacher/a.pt', 'kd_weight': .1, 'method_id': 'fixture'}
    (path / 'protocol_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    (path / 'evaluation_val_roster.txt').write_text('/val/a.jpg\n/val/b.jpg\n')
    for kind, original, folder, role in [('train', complete, 'run_evidence', 'development_train'),
                                        ('eval', metric, 'eval_evidence', 'development_val')]:
        receipt = {'terminal_status': 'COMPLETED', 'run_kind': kind, 'data_role': role,
                   'seed': seed, 'dataset': 'dronevehicle', 'inputs': {'arm': arm, 'method_id': 'FIXTURE'},
                   'resources': {'lease_id': 'fixture'}, 'metric_snapshots': ['metric_snapshot/bound.json'],
                   'source_snapshots': {'split_roster': ['source_snapshot/roster.txt'],
                                        'config': ['source_snapshot/config.yaml']}}
        write_json(path / folder / 'metric_snapshot/bound.json', original)
        write_json(path / folder / 'run_receipt.json', receipt)
        (path / folder / 'source_snapshot').mkdir(exist_ok=True)
        (path / folder / 'source_snapshot/roster.txt').write_text('/val/a.jpg\n/val/b.jpg\n')
        (path / folder / 'source_snapshot/config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    return {'arm': arm, 'seed': seed, 'path': str(path), 'protocol_id': 'frozen', 'source_arm': arm}


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def analyze(self, specs, **kwargs):
        return ar.analyze_manifest({'runs': specs, **kwargs}, self.root)

    def test_fraction_and_percent_are_identical(self):
        a = fixture(self.root, 'C', 42, 54.5)
        b = fixture(self.root, 'N', 42, 54.5, units='percent_0_to_100')
        out = self.analyze([a, b])
        self.assertAlmostEqual(out['comparisons']['C_minus_N']['pairs'][0]['deltas_pp']['mAP50_95'], 0)

    def test_unknown_units_and_out_of_range_are_rejected(self):
        for value, units in [(0.5, None), (54, 'fraction_0_to_1'), (math.nan, 'fraction_0_to_1')]:
            with self.assertRaises(ValueError):
                ar.to_percent(value, units)

    def test_paired_statistics_use_fixed_seed_order_and_sample_sd(self):
        specs = []
        for seed, n, c in [(123, 53., 51.), (0, 51., 52.), (42, 52., 55.)]:
            specs += [fixture(self.root, 'C', seed, c), fixture(self.root, 'N', seed, n)]
        result = self.analyze(specs)['comparisons']['C_minus_N']
        self.assertEqual([x['seed'] for x in result['pairs']], [0, 42, 123])
        stats = result['stats_pp']['mAP50_95']
        self.assertAlmostEqual(stats['mean'], 2 / 3)
        self.assertAlmostEqual(stats['sample_sd'], math.sqrt(19 / 3))
        self.assertEqual((stats['positive_seeds'], stats['negative_seeds']), (2, 1))
        self.assertTrue(result['complete_three_seed_pairing'])
        self.assertEqual(result['claim_supported'], 'not_automatically_assessed')

    def test_missing_seed_does_not_become_three_seed_claim(self):
        out = self.analyze([fixture(self.root, 'C', 42, 55), fixture(self.root, 'N', 42, 54)])
        comp = out['comparisons']['C_minus_N']
        self.assertFalse(comp['complete_three_seed_pairing'])
        self.assertIsNone(comp['stats_pp']['mAP50_95']['sample_sd'])
        self.assertEqual(comp['evidence_status'], 'incomplete_descriptive_only')

    def test_missing_evaluation_receipt_is_not_complete(self):
        spec = fixture(self.root, 'C', 42, 55)
        (Path(spec['path']) / 'eval_evidence/run_receipt.json').unlink()
        self.assertEqual(self.analyze([spec])['records'][0]['status'], 'incomplete')

    def test_failed_receipt_invalidates_visible_metrics(self):
        spec = fixture(self.root, 'C', 42, 55)
        path = Path(spec['path']) / 'eval_evidence/run_receipt.json'
        receipt = ar._read(path)
        receipt['terminal_status'] = 'FAILED'
        write_json(path, receipt)
        self.assertEqual(self.analyze([spec])['records'][0]['status'], 'invalid')

    def test_metric_snapshot_must_equal_original(self):
        spec = fixture(self.root, 'C', 42, 55)
        path = Path(spec['path']) / 'evaluation_val.json'
        metric = ar._read(path)
        metric['mAP50_95'] = .9
        write_json(path, metric)
        self.assertEqual(self.analyze([spec])['records'][0]['status'], 'invalid')

    def test_mismatched_checkpoint_is_rejected(self):
        spec = fixture(self.root, 'C', 42, 55)
        path = Path(spec['path']) / 'evaluation_val.json'
        metric = ar._read(path)
        metric['checkpoint'] = '/remote/best.pt'
        write_json(path, metric)
        self.assertEqual(self.analyze([spec])['records'][0]['status'], 'invalid')

    def test_partial_training_is_rejected(self):
        spec = fixture(self.root, 'C', 42, 55)
        path = Path(spec['path']) / 'completion_receipt.json'
        complete = ar._read(path)
        complete['last_epoch'] = 12
        write_json(path, complete)
        self.assertEqual(self.analyze([spec])['records'][0]['status'], 'invalid')

    def test_no_cross_protocol_pairing(self):
        a, b = fixture(self.root, 'C', 42, 55), fixture(self.root, 'N', 42, 54)
        b['protocol_id'] = 'different_recipe'
        result = self.analyze([a, b])['comparisons']['C_minus_N']
        self.assertEqual(result['pairs'], [])
        self.assertTrue(result['issues'])

    def test_no_cross_population_pairing(self):
        a, b = fixture(self.root, 'C', 42, 55), fixture(self.root, 'N', 42, 54)
        path = Path(b['path'])
        for name in ['evaluation_val_roster.txt', 'eval_evidence/source_snapshot/roster.txt']:
            (path / name).write_text('/val/other.jpg\n')
        result = self.analyze([a, b])['comparisons']['C_minus_N']
        self.assertEqual(result['pairs'], [])
        self.assertTrue(result['issues'])

    def test_duplicate_attempt_cannot_select_best_outcome(self):
        spec = fixture(self.root, 'C', 42, 55)
        with self.assertRaises(ValueError):
            self.analyze([spec, dict(spec)])

    def test_no_mixed_protocol_arm_mean(self):
        a, b = fixture(self.root, 'C', 42, 55), fixture(self.root, 'C', 0, 54)
        b['protocol_id'] = 'different_recipe'
        result = self.analyze([a, b])['arms']['C']
        self.assertIsNone(result['metrics_percent'])
        self.assertTrue(result['issues'])

    def test_same_manifest_protocol_cannot_hide_batch_or_lr_change(self):
        for field, value in [('batch', 16), ('lr0', .02)]:
            with self.subTest(field=field):
                a, b = fixture(self.root, 'C', 42, 55), fixture(self.root, 'N', 42, 54)
                for name in ('protocol_config.yaml', 'run_evidence/source_snapshot/config.yaml'):
                    path = Path(b['path']) / name
                    config = yaml.safe_load(path.read_text())
                    config[field] = value
                    path.write_text(yaml.safe_dump(config))
                result = self.analyze([a, b])
                self.assertTrue(all(r['status'] == 'complete' for r in result['records']))
                comp = result['comparisons']['C_minus_N']
                self.assertEqual(comp['pairs'], [])
                self.assertIn(field, comp['issues'][0])

    def test_receipt_binding_and_intervention_fields(self):
        a, b = fixture(self.root, 'C', 0, 55), fixture(self.root, 'N', 0, 54)
        path = Path(b['path'])
        for name in ('protocol_config.yaml', 'run_evidence/source_snapshot/config.yaml'):
            target = path / name
            config = yaml.safe_load(target.read_text())
            config.update(teacher='/teacher/different.pt', kd_weight=0., method_id='other', seed=999)
            target.write_text(yaml.safe_dump(config))
        # Interventions may differ; YAML defaults do not override actual seed 0 receipts.
        result = self.analyze([a, b])['comparisons']['C_minus_N']
        self.assertEqual(len(result['pairs']), 1)
        target = path / 'protocol_config.yaml'
        config = yaml.safe_load(target.read_text())
        config['batch'] = 16
        target.write_text(yaml.safe_dump(config))
        record = self.analyze([b])['records'][0]
        self.assertEqual(record['status'], 'invalid')
        self.assertIn('Launch protocol differs', record['issues'][0])

    def test_pilot_exact_threshold_passes_with_valid_checks(self):
        specs = [fixture(self.root, 'C', 42, 54), fixture(self.root, 'CL', 42, 54.3), fixture(self.root, 'CGT', 42, 54.2)]
        pilot = self.analyze(specs, implementation_checks_passed=True)['pilot']
        self.assertTrue(pilot['auto_expansion_eligible'])
        self.assertFalse(pilot['stop_running_experiments'])

    def test_pilot_equal_gt_is_not_strictly_better(self):
        specs = [fixture(self.root, 'C', 42, 54), fixture(self.root, 'CL', 42, 54.3), fixture(self.root, 'CGT', 42, 54.3)]
        self.assertFalse(self.analyze(specs, implementation_checks_passed=True)['pilot']['auto_expansion_eligible'])

    def test_pilot_implementation_checks_default_to_hold(self):
        specs = [fixture(self.root, 'C', 42, 54), fixture(self.root, 'CL', 42, 55), fixture(self.root, 'CGT', 42, 54.2)]
        self.assertFalse(self.analyze(specs)['pilot']['auto_expansion_eligible'])

    def test_missing_pilot_endpoint_waits(self):
        out = self.analyze([fixture(self.root, 'C', 42, 54)])['pilot']
        self.assertEqual(out['decision'], 'await_complete_endpoints')
        self.assertFalse(out['stop_running_experiments'])

    def test_test_access_flag_rejected(self):
        spec = fixture(self.root, 'C', 42, 55)
        path = Path(spec['path']) / 'evaluation_val.json'
        metric = ar._read(path)
        metric['official_test_accessed'] = True
        write_json(path, metric)
        self.assertEqual(self.analyze([spec])['records'][0]['status'], 'invalid')


if __name__ == '__main__':
    unittest.main(verbosity=2)
