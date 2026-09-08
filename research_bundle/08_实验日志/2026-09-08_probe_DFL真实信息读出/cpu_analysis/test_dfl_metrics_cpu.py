import argparse
import json
import math
from pathlib import Path
import unittest
import numpy as np
from dfl_metrics import distribution,own_anchor_readout


class Truths(unittest.TestCase):
    def test_same_mean_different_distribution(self):
        # Symmetric distributions about 7.5 have the same mean, unlike their shape/GT CE.
        a=np.full((4,16),-1000.);a[:,7:9]=0
        b=np.full((4,16),-1000.);b[:,[6,9]]=0
        da,db=distribution(a),distribution(b)
        np.testing.assert_allclose(da['expectation_bin'],db['expectation_bin'])
        self.assertNotEqual(da['variance_bin2'],db['variance_bin2'])
        self.assertNotEqual(own_anchor_readout(da,[7.5]*4)['four_edge_mean_DFL_CE_nat'],own_anchor_readout(db,[7.5]*4)['four_edge_mean_DFL_CE_nat'])
        c=np.zeros((4,16));dc=distribution(c)
        np.testing.assert_allclose(dc['expectation_bin'],da['expectation_bin'])
        self.assertNotEqual(dc['entropy_nat'],da['entropy_nat'])
    def test_adjacent_bin_interpolation(self):
        x=np.zeros((4,16));x[:,2]=math.log(3);d=distribution(x)
        r=own_anchor_readout(d,[2.25]*4)
        expected=.75*math.log(18/3)+.25*math.log(18)
        self.assertAlmostEqual(r['four_edge_mean_DFL_CE_nat'],expected)
        self.assertEqual(r['edges'][0]['left_weight'],.75)
    def test_invalid_distance_is_null(self):
        d=distribution(np.zeros((4,16)));r=own_anchor_readout(d,[-.01,15,float('nan'),14.999])
        self.assertEqual([e['valid'] for e in r['edges']],[False,False,False,True])
        self.assertIsNone(r['four_edge_mean_DFL_CE_nat']);self.assertFalse(r['target_clamped'])
        self.assertIsNone(r['edges'][0]['DFL_CE_nat'])
    def test_logit_shift_and_large_range(self):
        x=np.arange(64,dtype=float).reshape(4,16)*100;d=distribution(x);dd=distribution(x+10000)
        np.testing.assert_allclose(d['log_probabilities'],dd['log_probabilities'])
        self.assertTrue(math.isfinite(own_anchor_readout(d,[0,1,2,3])['four_edge_mean_DFL_CE_nat']))
        np.testing.assert_allclose(d['expectation_bin'],[15]*4)
    def test_uniform_moments_and_integer_distance(self):
        d=distribution(np.zeros((4,16)))
        np.testing.assert_allclose(d['entropy_nat'],[math.log(16)]*4)
        np.testing.assert_allclose(d['variance_bin2'],[21.25]*4)
        r=own_anchor_readout(d,[0,1,7,14])
        self.assertAlmostEqual(r['four_edge_mean_DFL_CE_nat'],math.log(16))
        self.assertTrue(all(e['right_weight']==0 for e in r['edges']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    a.output.write_text(json.dumps(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),new_GPU=False,new_forward=False,new_hash_computed=False),indent=2),encoding='utf-8')
    raise SystemExit(0 if r.wasSuccessful() else 1)
