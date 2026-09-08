import argparse
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from audit_targets import relative,quantities,smooth


def record(r,t,g=(0,0,10,10)):
    return dict(rgb_gt=list(g),reference_box=r,mapped_teacher_box=t)


class Truths(unittest.TestCase):
    def test_actual_source_coordinate_map(self):
        src=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        nodes=[n for n in src.body if isinstance(n,ast.FunctionDef) and n.name in ('_valid_boxes','object_relative','map_teacher_box')]
        env={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),env)
        box=torch.tensor([[1.,2,9,8]]);g=torch.tensor([[0.,0,10,10]]);rgb=torch.tensor([[10.,20,30,60]])
        np.testing.assert_allclose(relative(box[0].numpy(),g[0].numpy()),env['object_relative'](box,g)[0].numpy(),atol=1e-7)
        mapped=env['map_teacher_box'](box,g,rgb);torch.testing.assert_allclose(mapped,torch.tensor([[12.,28.,28.,52.]]))
    def test_denominator_and_actual_smooth_l1(self):
        r=record([2,2,12,12],[1,1,11,11]);a=quantities(r,5,32,1);b=quantities(r,10,32,1)
        self.assertAlmostEqual(a['proxy_teacher_loss_contribution'],2*b['proxy_teacher_loss_contribution'])
        e=torch.tensor(a['reference_minus_teacher'],dtype=torch.float64,requires_grad=True)
        loss=torch.nn.functional.smooth_l1_loss(e,torch.zeros_like(e),beta=.1,reduction='mean')/5;loss.backward()
        self.assertAlmostEqual(float(loss),a['proxy_teacher_loss_contribution'])
        np.testing.assert_allclose(e.grad.numpy(),a['derivative_teacher_normalized_loss'])
    def test_saturation_masks_target_difference(self):
        a=quantities(record([3,3,13,13],[1,1,11,11]),5,32,1)
        self.assertEqual(a['derivative_GT_unscaled'],a['derivative_teacher_unscaled'])
        self.assertNotEqual(a['proxy_GT_loss_contribution'],a['proxy_teacher_loss_contribution'])
        self.assertEqual(set(a['regimes']),{'both_saturated_same_sign'})
    def test_near_GT_opposite_correction(self):
        a=quantities(record([.2,.2,10.2,10.2],[.4,.4,10.4,10.4]),5,32,1)
        self.assertTrue(all(x*y<0 for x,y in zip(a['derivative_GT_unscaled'],a['derivative_teacher_unscaled'])))
        self.assertEqual(set(a['regimes']),{'both_quadratic'})
        self.assertFalse(a['reference_is_measured_student'])
    def test_pixel_chain_scale(self):
        a=quantities(record([1,4,11,24],[0,0,10,20],(0,0,10,20)),5,32,1)
        np.testing.assert_allclose(np.array(a['derivative_GT_normalized_loss'])/[10,20,10,20],a['derivative_GT_pixel_loss'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    SOURCE=a.source
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    a.output.write_text(json.dumps(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),CUDA_initialized=torch.cuda.is_initialized(),new_hash_computed=False),indent=2),encoding='utf-8')
    raise SystemExit(0 if r.wasSuccessful() else 1)
