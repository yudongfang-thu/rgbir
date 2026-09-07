"""CPU-only cadence/window truth checks; no torch/runtime import."""
import json
from pathlib import Path
import sys
import types
import unittest

import short_profile as p
from screen_common import write_new


class ProfileTests(unittest.TestCase):
    def test_exactly_first_composition_sanity(self):
        class Original:
            def __init__(self):self.calls=0;self.sanity=True;self.flags=[]
            def __call__(self,pred,batch):
                self.calls+=1;self.flags.append(self.sanity)
                return self.calls
        Private=p.first_composition_only_type(Original);c=Private()
        for i in range(5):self.assertEqual(c(None,None),i+1)
        self.assertEqual(c.flags,[True,False,False,False,False]);self.assertFalse(c.sanity)
        o=Original();o(None,None);o(None,None);self.assertEqual(o.flags,[True,True])

    def test_window_includes_between_batch_wait_and_exact_update_denominator(self):
        window=p.ContinuousWindow()
        for i in range(1,31):
            # Includes an extra0.5s of conceptual loader wait between each end.
            window.add(i,100.+i*2.5,max(0,i-6),i,min(i,6),{})
        r=window.summary()
        self.assertEqual(r['continuous_post_warmup_seconds'],60.)
        self.assertEqual(r['post_warmup_batches'],24);self.assertEqual(r['post_warmup_successful_updates'],24)
        self.assertEqual(r['seconds_per_batch'],2.5);self.assertFalse(r['periodic_100_batch_event_observed'])

    def test_nonstandard_skip_count_reported_not_assumed(self):
        window=p.ContinuousWindow()
        for i in range(1,28):window.add(i,float(i),max(0,i-3),i,min(i,3),{})
        r=window.summary();self.assertEqual(r['post_warmup_successful_updates'],21)
        self.assertEqual(r['post_warmup_batches'],21)

    def test_missing_final_or_out_of_order_rejected(self):
        window=p.ContinuousWindow()
        with self.assertRaises(ValueError):window.summary()
        with self.assertRaises(AssertionError):window.add(2,1.,0,1,1,{})

    def test_attempt_bound_rejected(self):
        window=p.ContinuousWindow()
        with self.assertRaises(RuntimeError):window.add(1,1.,0,97,97,{})

    def test_nan_and_counter_mismatch_rejected(self):
        with self.assertRaises(ValueError):p.ContinuousWindow().add(1,float('nan'),0,1,1,{})
        with self.assertRaises(AssertionError):p.ContinuousWindow().add(1,1.,1,1,1,{})

    def test_no_gpu_import(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('runtime',sys.modules)

    def test_candidate_fallback_cannot_be_reported_as_thin_speed(self):
        good=types.SimpleNamespace(thin_learning_batches=30,fallback_batches=0,full_diagnostics_batches=3)
        self.assertEqual(p.candidate_counters('C1',30,good)['thin_learning_batches'],30)
        bad=types.SimpleNamespace(thin_learning_batches=29,fallback_batches=1,full_diagnostics_batches=3)
        with self.assertRaises(AssertionError):p.candidate_counters('C1',30,bad)
        self.assertEqual(p.candidate_counters('N',30,object()),dict(applicable=False))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileTests))
    if len(sys.argv)>1:write_new(Path(sys.argv[1]),dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,torch_imported='torch' in sys.modules,runtime_imported='runtime' in sys.modules,
        scope='CPU cadence/continuous window only',new_hash_computed=False))
    if not result.wasSuccessful():raise SystemExit(1)
