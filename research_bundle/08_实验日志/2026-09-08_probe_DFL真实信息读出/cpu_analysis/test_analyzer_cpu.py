import argparse
import json
from pathlib import Path
import unittest
import numpy as np
from analyze_dfl import role_summary,check_distribution
from dfl_metrics import distribution,own_anchor_readout


class Truths(unittest.TestCase):
    def test_missing_and_invalid_keep_separate_denominators(self):
        d=distribution(np.zeros((4,16)))
        good=dict(distribution_id='d',distribution_metrics=d,GT_readout=own_anchor_readout(d,[1]*4))
        invalid=dict(distribution_id='d',distribution_metrics=d,GT_readout=own_anchor_readout(d,[-1,1,1,1]))
        empty=dict(distribution_id=None)
        x=role_summary([good,invalid,empty],3)
        self.assertEqual((x['anchor_present'],x['anchor_missing'],x['unique_distribution_ids']),(2,1,1))
        self.assertEqual((x['all_four_GT_edges_valid_objects'],x['valid_GT_edges'],x['invalid_GT_edges']),(1,7,1))
        self.assertEqual(x['mean4_DFL_CE_nat_valid_objects']['n'],1)
        z=role_summary([],0);self.assertIsNone(z['mean4_DFL_CE_nat_valid_objects']['mean'])
    def test_geometry_and_FP32_native_are_separate(self):
        row=dict(model='T',bins=16,side_order=['left','top','right','bottom'],fp32_decode_exact=True,native_decode_exact=True,
                 stride=8,level=0,raw_feature_shapes=[[1,4,2,2],[1,4,1,1],[1,4,1,1]],anchor_index=0,center_xy=[4.,4.],
                 raw_logits=np.zeros((4,16)).tolist(),probabilities_fp32=np.full((4,16),1/16).tolist(),
                 probabilities_native=np.full((4,16),1/16).tolist(),expectation_fp32_bins=[7.5]*4,
                 native_dfl_distances_bins=[7.5]*4,box_fp32_xyxy=[-56,-56,64,64],box_native_xyxy=[-56.001,-56,64,64])
        d,c=check_distribution(row)
        self.assertAlmostEqual(c['native_vs_FP32_box_max_abs_px'],.001)
        row['center_xy']=[5.,4.]
        with self.assertRaises(AssertionError):check_distribution(row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    a.output.write_text(json.dumps(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),new_GPU=False,new_forward=False,new_hash_computed=False),indent=2),encoding='utf-8')
    raise SystemExit(0 if r.wasSuccessful() else 1)
