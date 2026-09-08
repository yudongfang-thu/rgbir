"""Synthetic truths only; no actual cache, torch, GPU or hashes."""
import argparse,json,math,unittest
from fractions import Fraction as F
from pathlib import Path
from probability_transport import affine_coefficients,scatter_complete,stable_softmax,transport_logits
from run_cached_transport import analyze,summarize

class Truths(unittest.TestCase):
    def test_identity_no_boundary_roundoff(self):
        gt=[103.5517578125,118.21175384521484,140.23672485351562,208.9588165283203]
        r=transport_logits([[0.]*16]*4,gt,gt,[120.,168.],[120.,168.],16,16)
        self.assertEqual(r['status'],'SUPPORTED');self.assertTrue(r['algebraic_identity'])
        self.assertEqual(r['source_probabilities'],r['target_probabilities'])
    def test_translation_and_scale_stride_cancel(self):
        a=affine_coefficients([0,0,10,10],[100,200,120,220],[5,5],[110,210],8,16)
        self.assertEqual(a,[(F(1),F(0))]*4)
    def test_fractional_scale_mass_mean_and_edge(self):
        r=transport_logits([[float(j)/10 for j in range(16)]]*4,[0,0,10,10],[0,0,5,5],[0,0],[0,0],8,8)
        self.assertEqual(r['status'],'SUPPORTED')
        for side in r['sides']:
            self.assertAlmostEqual(side['source_mass'],side['target_mass'],places=14)
            self.assertAlmostEqual(side['target_expectation_bin'],side['source_expectation_bin']/2,places=13)
            self.assertAlmostEqual(side['edge_closure_error'],0.,places=12)
    def test_integer_last_bin(self):
        p=[0.]*16;p[15]=1.
        self.assertEqual(scatter_complete(p,F(1),F(0))['target_probabilities'],p)
    def test_positive_tail_no_epsilon_waiver(self):
        p=[0.]*16;p[0]=1e-100;p[1]=1.
        r=scatter_complete(p,F(1),-F(1,10**30))
        self.assertEqual(r['status'],'OUTSIDE_STUDENT_SUPPORT');self.assertIsNone(r['target_probabilities'])
        self.assertEqual(r['rejected_source_bins'],[0])
    def test_shifted_anchor_rejects_whole_object(self):
        r=transport_logits([[0.]*16]*4,[0,0,10,10],[0,0,10,10],[5,5],[6,5],1,1)
        self.assertEqual(r['status'],'OUTSIDE_STUDENT_SUPPORT');self.assertIsNone(r['target_probabilities'])
    def test_zero_mass_outside_does_not_invent_mass(self):
        p=[0.]*16;p[1]=1.
        r=scatter_complete(p,F(1),F(-1));self.assertEqual(r['status'],'SUPPORTED')
        self.assertEqual(r['target_probabilities'][0],1.)
    def test_zero_extent_and_nonfinite_reject(self):
        for gt in ([0,0,0,10],[0,0,math.nan,10]):
            r=transport_logits([[0.]*16]*4,gt,[0,0,10,10],[5,5],[5,5],1,1)
            self.assertEqual(r['status'],'INVALID_INPUT')
    def test_softmax_positive_mass_underflow_reject(self):
        with self.assertRaises(ValueError):stable_softmax([0.]+[-10000.]*15)
    def test_missing_roles_stay_missing_no_substitution(self):
        o=dict(stable_rgb_gt_id='r',stable_ir_gt_id='t',image_index=0,frame_id='f',current_forward_id='new',bucket='fixed',
            roles={'S':{'historical_R_candidate':None},'T':{'same_historical_R_index':None,'own_native_iou50':None}})
        rows=analyze([o],{});self.assertEqual(len(rows),2)
        self.assertTrue(all(r['status']=='MISSING_ROLE' for r in rows))
        self.assertTrue(all(v['support_status']=='NO_SUPPORTED_TARGET' for v in summarize(rows).values()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    with a.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),actual_cache_read=False,GPU_used=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not r.wasSuccessful())
