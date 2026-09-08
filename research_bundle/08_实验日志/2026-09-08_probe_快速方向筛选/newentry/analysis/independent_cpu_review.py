"""Independent known-truth analyzer review; no actual direction AP is opened."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import analyze_direction as a
import test_analyze_direction_cpu as author
before=(HERE/'analyze_direction.py').read_bytes()


class ExtraTruths(unittest.TestCase):
    def test_nonuniform_perclass_delta_macro_and_percent(self):
        data=author.matrix();n=data[('drone','N')];r=data[('drone','C1')]
        changes=[.01,-.02,.03,-.01,.14]
        for i,delta in enumerate(changes):
            for metric in a.AP:r['per_class'][i][metric]=n['per_class'][i][metric]+delta
        for metric in a.AP:r[metric]=sum(x[metric] for x in r['per_class'])/5
        result=a.summarize(data);diff=result['datasets']['drone']['comparisons'][0]
        self.assertAlmostEqual(diff['delta_pp']['mAP50_95'],3.)
        self.assertAlmostEqual(sum(x['delta_pp']['AP50'] for x in diff['per_class'])/5,3.)
        self.assertAlmostEqual(result['datasets']['drone']['arms'][1]['display_percent']['mAP50_95'],33.2)

    def test_all_missing_remains_partial_and_null(self):
        result=a.summarize({});self.assertEqual(result['status'],'PARTIAL_DIRECTION_READOUT')
        self.assertEqual(sum(len(v['arms']) for v in result['datasets'].values()),7)
        self.assertEqual(sum(len(v['comparisons']) for v in result['datasets'].values()),8)
        for ds in result['datasets'].values():
            for r in ds['arms']:
                self.assertEqual(r['status'],'MISSING');self.assertIsNone(r['raw_fraction'])
            for r in ds['comparisons']:
                self.assertEqual(r['status'],'UNAVAILABLE');self.assertIsNone(r['delta_pp'])

    def test_exact_zero_contrast_and_no_cross_dataset_pooling(self):
        data=author.matrix()
        for ds,arms in a.ARMS.items():
            for arm in arms:
                for metric in a.METRICS:data[(ds,arm)][metric]=data[(ds,'N')][metric]
                data[(ds,arm)]['per_class']=copy.deepcopy(data[(ds,'N')]['per_class'])
        result=a.summarize(data)
        self.assertEqual(set(result['datasets']),{'drone','llvip'})
        for ds in result['datasets'].values():
            for row in ds['comparisons']:
                self.assertTrue(all(x==0. for x in row['delta_pp'].values()))
                self.assertEqual(set(row['direction'].values()),{'zero'})
        self.assertIsNone(result['standard_deviation'])

    def test_seed_and_scientific_promotion_flags_rejected(self):
        for field,value in [('seed',123),('formal_e200_complete',True),('accepted_endpoint_claim',True),
                            ('official_test_accessed',True),('formal_paper_gain_claim',True)]:
            row=author.receipt('drone','N');row[field]=value
            with self.assertRaises(ValueError):a.summarize({('drone','N'):row})


suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.Truths),
                         unittest.defaultTestLoader.loadTestsFromTestCase(ExtraTruths)])
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
p=HERE/'analyze_direction.py';s=p.stat();unchanged=before==p.read_bytes()
passed=result.wasSuccessful() and unchanged
receipt=dict(status='PASS_FOR_RECEIPT_ONLY_DIRECTION_ANALYZER' if passed else 'FAIL',
    tests_run=result.testsRun,independent_extra_tests=4,failures=len(result.failures),errors=len(result.errors),
    source=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns),source_bytes_unchanged=unchanged,
    actual_direction_AP_read=False,formal_analysis_executed=False,new_hash_computed=False,
    torch_imported='torch' in sys.modules,gpu_or_ssh=False,test_output=stream.getvalue(),
    accepted_scope='Fixed 7-arm seed42 FT3 receipt values and 8 descriptive pp contrasts; missing/failed/conflict remain unavailable')
with (HERE/'INDEPENDENT_ACCEPTANCE.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(stream.getvalue());print(receipt['status'],result.testsRun)
raise SystemExit(0 if passed else 1)
