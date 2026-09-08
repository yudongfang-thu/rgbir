"""Small independent synthetic review, no real confidence AP inputs."""
import io
import json
from pathlib import Path
import sys
import unittest
import analyze_confidence as a
import test_confidence_analysis_cpu as author

HERE=Path(__file__).absolute().parent
source=HERE/'analyze_confidence.py';before=source.read_bytes()


class Extra(unittest.TestCase):
    def test_exact_shared_zero_pp_and_bounded_scope(self):
        r=a.summarize({arm:author.receipt(arm) for arm in a.ARMS})
        self.assertEqual(set(r['comparison']['delta_pp'].values()),{0.})
        self.assertEqual(set(r['comparison']['direction'].values()),{'zero'})
        self.assertEqual(r['display_percent']['N']['mAP50_95'],30.)
        self.assertFalse(r['old_L2_N_reused']);self.assertFalse(r['automatically_extend_matrix'])
        self.assertFalse(r['formal_e200_complete']);self.assertIsNone(r['standard_deviation'])

    def test_no_duplicate_run_or_nearby_C0_dose(self):
        n,c=author.receipt('N'),author.receipt('C0')
        c['training_completion']=n['training_completion']
        with self.assertRaises(ValueError):a.summarize({'N':n,'C0':c})
        n,c=author.receipt('N'),author.receipt('C0');c['kd_coefficient']=.10000000000000002
        with self.assertRaises(ValueError):a.summarize({'N':n,'C0':c})


suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.Tests),
                         unittest.defaultTestLoader.loadTestsFromTestCase(Extra)])
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
unchanged=before==source.read_bytes();s=source.stat()
with (HERE/'INDEPENDENT_ACCEPTANCE.json').open('x',encoding='utf-8') as f:
    json.dump(dict(status='PASS_RECEIPT_ONLY_CONFIDENCE_ANALYZER' if result.wasSuccessful() and unchanged else 'FAIL',
        tests=result.testsRun,author_tests=4,independent_tests=2,failures=len(result.failures),errors=len(result.errors),
        actual_new_AP_read=False,GPU_or_SSH=False,new_hash_computed=False,torch_imported='torch' in sys.modules,
        source=dict(path=str(source),bytes=s.st_size,mtime_ns=s.st_mtime_ns),source_bytes_unchanged=unchanged,
        test_output=stream.getvalue()),f,ensure_ascii=False,indent=2)
print(stream.getvalue())
raise SystemExit(0 if result.wasSuccessful() and unchanged else 1)
