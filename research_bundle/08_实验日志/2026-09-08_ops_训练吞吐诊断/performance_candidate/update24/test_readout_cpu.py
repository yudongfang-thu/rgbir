"""Small deterministic truth/negative checks for JSON-only timing readout."""
import copy
import json
from pathlib import Path
import sys
import unittest

import summarize_24_updates as s


class ReadoutTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(batch=i,wall_seconds=3.,audit_seconds=1.,wall_minus_audit_seconds=2.,warmup=i<=6) for i in range(1,31)]
        return dict(result=dict(batches=30,updates=24),gpu_allocated_peak_mib=100.,gpu_reserved_peak_mib=120.,
            timing=dict(batches=rows,warmup_batches=6,post_warmup_batches=24,post_warmup_batch_seconds=[2.]*24,
                training_span_seconds=120.,training_audit_copy_save_check_seconds=40.,training_span_minus_audit_seconds=80.))

    def test_correct_overhead_and_warmup(self):
        r=s.timing_summary(self.fixture())
        self.assertEqual(r['post_warmup_batch_median_seconds'],2.)
        self.assertEqual(r['training_audit_fraction'],1/3)
        self.assertEqual(r['post_warmup_batches'],24)

    def test_wrong_subtraction_rejected(self):
        r=self.fixture();r['timing']['batches'][5]['wall_minus_audit_seconds']=2.1
        with self.assertRaises(ValueError):s.timing_summary(r)

    def test_missing_batch_rejected(self):
        r=self.fixture();r['timing']['batches'].pop()
        with self.assertRaises(ValueError):s.timing_summary(r)

    def test_nan_inf_negative_and_bool_rejected(self):
        for value in (float('nan'),float('inf'),-1,True):
            with self.assertRaises(ValueError):s.finite(value)

    def test_warmup_selection_cannot_change(self):
        r=self.fixture();r['timing']['batches'][6]['warmup']=True
        with self.assertRaises(ValueError):s.timing_summary(r)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReadoutTests))
    if len(sys.argv)>1:
        path=Path(sys.argv[1])
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
            scope='JSON-only timing closure; no torch/runtime/GPU imports',new_hash_computed=False),indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)
