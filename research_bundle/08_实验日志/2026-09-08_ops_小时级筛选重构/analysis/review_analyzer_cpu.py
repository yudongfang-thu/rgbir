"""Independent bounded review; synthetic inputs only, never real AP records."""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).absolute().parent
source = HERE / 'analyze_hourly.py'
before = source.read_bytes()
spec = importlib.util.spec_from_file_location('hourly_author_synthetic_tests', HERE / 'test_analyze_hourly_cpu.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
a = module.a


class IndependentChecks(unittest.TestCase):
    def test_zero_difference_and_exact_timing_partition(self):
        ev, tr, q = module.fixture()
        for arm in a.ARMS:
            for metric in a.METRICS:
                ev[arm][metric] = ev['N'][metric]
            ev[arm]['per_class'] = copy.deepcopy(ev['N']['per_class'])
        q['seconds'] = 360.
        r = a.analyze(ev, tr, q)
        self.assertEqual(r['timing']['queue_seconds'], 360.)
        self.assertEqual(r['timing']['other_including_canaries_seconds'], 0.)
        self.assertEqual(r['timing']['queue_hours'], .1)
        for pair in r['differences'].values():
            self.assertTrue(all(value == 0. for value in pair['pp'].values()))

    def test_percent_and_pp_are_distinct_transforms(self):
        ev, tr, q = module.fixture()
        for arm, value in (('N', .31), ('C0', .305), ('C1', .32)):
            ev[arm]['precision'] = value
        r = a.analyze(ev, tr, q)
        self.assertAlmostEqual(r['values']['C0']['percent']['precision'], 30.5)
        self.assertAlmostEqual(r['differences']['C0-N']['pp']['precision'], -.5)
        self.assertAlmostEqual(r['differences']['C1-C0']['pp']['precision'], 1.5)

    def test_corrupted_identity_and_coefficients_rejected(self):
        for field, value in (('endpoint', 'SHORT_SCREEN_E8_LAST_EMA'),
                             ('classification_coefficient', True),
                             ('classification_coefficient', float('nan')),
                             ('localization_coefficient', True)):
            ev, tr, q = module.fixture()
            tr['C1'][field] = value
            with self.assertRaises(ValueError):
                a.analyze(ev, tr, q)

    def test_queue_seconds_not_added_again(self):
        ev, tr, q = module.fixture()
        q['seconds'] = 1200.
        r = a.analyze(ev, tr, q)
        t = r['timing']
        self.assertEqual(t['queue_seconds'], 1200.)
        self.assertEqual(t['training_seconds'] + t['evaluation_seconds'] +
                         t['other_including_canaries_seconds'], 1200.)
        self.assertEqual(t['other_including_canaries_seconds'], 840.)


suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(module.Tests),
    unittest.defaultTestLoader.loadTestsFromTestCase(IndependentChecks),
])
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
unchanged = before == source.read_bytes()
passed = result.wasSuccessful() and unchanged
receipt = dict(
    status='PASS_FOR_FIXED_HOURLY_DESCRIPTIVE_ANALYZER' if passed else 'FAIL',
    author_tests=14, independent_tests=4, tests_run=result.testsRun,
    failures=len(result.failures), errors=len(result.errors),
    source=a.source_stat(source), source_bytes_unchanged=unchanged,
    known_truth_only=True, new_AP_read=False, actual_analysis_executed=False,
    torch_imported='torch' in sys.modules, gpu_or_ssh=False, new_hash_computed=False,
    accepted_scope='Fixed N/C0/C1 seed42 mature RGB warm-start FT3 descriptive values, pp and serial queue timing only',
    scientific_claim_accepted=False, test_output=stream.getvalue(),
)
with (HERE / 'analyzer_independent_acceptance.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(stream.getvalue())
print(json.dumps({k: v for k, v in receipt.items() if k != 'test_output'}, ensure_ascii=False))
raise SystemExit(0 if passed else 1)
