"""Independent small arithmetic truths; does not run the dispatcher or inspect AP."""
import argparse,copy,importlib.util,json,math,shutil,unittest
from pathlib import Path
HERE=Path(__file__).absolute().parent
SOURCE=HERE.parent/'release/budget.py'
spec=importlib.util.spec_from_file_location('_independent_budget_under_review',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def canary(value=1.):return dict(normal_cadence_batch_seconds=[value]*24,seconds=24*value,flow_audit_seconds=0.)
def resource(gpu=4864.,rss=10240.):
    return dict(resources=dict(per_gpu_peak_vram_mib={'0':gpu},peak_rss_mib=rss),gpu_allocated_peak_mib=gpu-100,gpu_reserved_peak_mib=gpu-50)

class BudgetTruths(unittest.TestCase):
    def test_exact_2700_and_over(self):
        canaries={'N':canary(),'C0':canary()}
        exact=b.estimate(canaries,1351.2)
        self.assertEqual(exact['estimate_total_execution_seconds'],2700.)
        self.assertEqual(exact['status'],'PASS_WITHIN_EXECUTION_BUDGET')
        self.assertEqual(b.estimate(canaries,1351.20001)['status'],'BLOCKED_EXECUTION_BUDGET')
    def test_distinct_arms_no_common_fastest_assumption(self):
        r=b.estimate({'N':canary(1),'C0':canary(2)},0)
        self.assertAlmostEqual(r['arms']['N']['train_seconds_with_margin'],614.4)
        self.assertAlmostEqual(r['arms']['C0']['train_seconds_with_margin'],1228.8)
        self.assertAlmostEqual(r['estimate_total_execution_seconds'],1963.2)
        for arms in ({'N':canary()},{'N':canary(),'C1':canary()}):
            with self.assertRaises(ValueError):b.estimate(arms,0)
    def test_all_intervals_including_slow_batch(self):
        r=canary();r.update(normal_cadence_batch_seconds=[1.]*23+[25.],seconds=48.)
        self.assertEqual(b.training_estimate(r)['mean_normal_batch_seconds'],2.)
    def test_flow_IO_included_once_after_overhead_partition(self):
        r=canary();r.update(seconds=100.,flow_audit_seconds=5.)
        v=b.training_estimate(r)
        self.assertEqual(v['observed_non_batch_overhead_seconds'],71.)
        self.assertEqual(v['observed_flow_audit_seconds'],5.)
        self.assertAlmostEqual(v['train_seconds_with_margin'],1.2*(512+71+5))
    def test_missing_short_or_nonfinite_measurements_rejected(self):
        cases=[]
        r=canary();del r['flow_audit_seconds'];cases.append(r)
        r=canary();r['normal_cadence_batch_seconds']=[];cases.append(r)
        r=canary();r['normal_cadence_batch_seconds']=[1.]*23;cases.append(r)
        for key in ('seconds','flow_audit_seconds'):
            for value in (float('nan'),float('inf'),-1,True):
                r=canary();r[key]=value;cases.append(r)
        for value in (float('nan'),float('inf'),0.,-1.,True):
            r=canary();r['normal_cadence_batch_seconds'][0]=value;cases.append(r)
        for r in cases:
            with self.assertRaises((ValueError,KeyError)):b.training_estimate(r)
        with self.assertRaises(ValueError):b.estimate({'N':canary(),'C0':canary()},float('nan'))
    def test_reservation_rounds_up_exact_and_just_over(self):
        self.assertEqual(b.reservation(resource())['vram_mib'],5120)
        self.assertEqual(b.reservation(resource())['rss_mib'],12288)
        self.assertEqual(b.reservation(resource(4864.01,10240.01))['vram_mib'],5376)
        self.assertEqual(b.reservation(resource(4864.01,10240.01))['rss_mib'],13312)
        self.assertEqual(b.reservation(resource(20000,50000))['vram_mib'],21248)
        self.assertEqual(b.reservation(resource(20000,50000))['rss_mib'],53248)
    def test_reservation_uses_largest_actual_peak_and_finite(self):
        r=resource();r['gpu_reserved_peak_mib']=6000.
        self.assertEqual(b.reservation(r)['vram_mib'],6400)
        r=resource();r['resources']['per_gpu_peak_vram_mib']['1']=1000
        with self.assertRaises(ValueError):b.reservation(r)
        r=resource();r['gpu_allocated_peak_mib']=float('inf')
        with self.assertRaises(ValueError):b.reservation(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    before=SOURCE.read_bytes();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BudgetTruths))
    assert SOURCE.read_bytes()==before
    snapshot=args.output.with_name(args.output.stem+'_budget_source.py');snapshot.write_bytes(before)
    with args.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_ARITHMETIC_SCOPE' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),source=dict(path=str(SOURCE),bytes=len(before),mtime_ns=SOURCE.stat().st_mtime_ns),source_byte_unchanged=True,source_snapshot=str(snapshot),scope='Budget arithmetic only; RUNNING/queue wait clock and deadline enforcement not executed',actual_canary_measures_used=False,actual_AP_read=False,GPU_used=False,SSH_used=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not result.wasSuccessful())
