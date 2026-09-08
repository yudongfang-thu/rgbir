"""Synthetic tensors only; no checkpoint, GPU, native inference or SSH."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import torch

HERE=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('swap_under_test',HERE/'evaluate_state_swap.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(2,2)
        self.bn=torch.nn.BatchNorm1d(2)
        self.register_buffer('extra',torch.tensor([1.]))
        self.register_buffer('temporary',torch.tensor([3.]),persistent=False)
    def forward(self,x):return self.bn(self.linear(x))


def fixture():
    initial=Toy();ft=copy.deepcopy(initial)
    with torch.no_grad():
        for p in ft.parameters():p.add_(.1)
        ft.bn.running_mean.add_(.2);ft.bn.running_var.add_(.3)
        ft.bn.num_batches_tracked.add_(4)
    return initial,ft


class Tests(unittest.TestCase):
    def test_parameter_only_exact_and_sources_unchanged(self):
        initial,ft=fixture();before=[{k:v.clone() for k,v in m.state_dict().items()} for m in (initial,ft)]
        hybrid,r=s.compose_model(initial,ft,'parameter_only')
        for name,p in hybrid.named_parameters():self.assertTrue(torch.equal(p,dict(ft.named_parameters())[name]))
        for name,b in hybrid.named_buffers():self.assertTrue(torch.equal(b,dict(initial.named_buffers())[name]))
        for m,old in zip((initial,ft),before):
            for key,v in m.state_dict().items():self.assertTrue(torch.equal(v,old[key]))
        self.assertTrue(r['all_changed_buffers_are_bn_statistics'])
        self.assertEqual(set(r['changed_buffer_keys']),{'bn.running_mean','bn.running_var','bn.num_batches_tracked'})

    def test_buffer_only_exact_and_bn_affine_is_parameter(self):
        initial,ft=fixture();hybrid,r=s.compose_model(initial,ft,'buffer_only')
        for name,p in hybrid.named_parameters():self.assertTrue(torch.equal(p,dict(initial.named_parameters())[name]))
        for name,b in hybrid.named_buffers():self.assertTrue(torch.equal(b,dict(ft.named_buffers())[name]))
        self.assertIn('bn.weight',r['changed_parameter_keys'])
        self.assertNotIn('bn.weight',r['changed_buffer_keys'])

    def test_all_buffers_include_nonpersistent_and_non_bn(self):
        initial,ft=fixture();ft.temporary.add_(2);ft.extra.add_(1)
        hybrid,r=s.compose_model(initial,ft,'buffer_only')
        self.assertTrue(torch.equal(hybrid.temporary,ft.temporary))
        self.assertFalse(r['all_changed_buffers_are_bn_statistics'])
        self.assertEqual(set(r['non_bn_changed_buffer_keys']),{'extra','temporary'})

    def test_raw_dtype_mismatch_rejected(self):
        initial,ft=fixture();ft.double()
        with self.assertRaises(ValueError):s.compose_model(initial,ft,'parameter_only')

    def test_shape_or_keys_mismatch_rejected(self):
        initial,ft=fixture();ft.linear=torch.nn.Linear(3,2)
        with self.assertRaises(ValueError):s.compose_model(initial,ft,'parameter_only')
        initial,ft=fixture();ft.register_buffer('new',torch.ones(1))
        with self.assertRaises(ValueError):s.compose_model(initial,ft,'buffer_only')

    def test_nan_rejected(self):
        initial,ft=fixture();ft.bn.running_var[0]=float('nan')
        with self.assertRaises(ValueError):s.compose_model(initial,ft,'buffer_only')

    def test_half_projection_no_grad_and_no_source_alias(self):
        initial,ft=fixture();initial.half();ft.half()
        hybrid,r=s.compose_model(initial,ft,'parameter_only')
        self.assertTrue(all(p.dtype==torch.float32 and not p.requires_grad and p.grad is None for p in hybrid.parameters()))
        self.assertFalse(hybrid.training)
        self.assertEqual(hybrid.bn.num_batches_tracked.dtype,torch.int64)
        with torch.no_grad():hybrid.linear.weight.add_(1)
        self.assertFalse(torch.equal(hybrid.linear.weight,ft.linear.weight.float()))
        self.assertTrue(r['fp32_projection_exact'])

    def test_known_forward_state_composition(self):
        initial,ft=fixture();initial.eval();ft.eval()
        hybrid,_=s.compose_model(initial,ft,'parameter_only')
        expected=copy.deepcopy(initial)
        with torch.no_grad():
            for name,p in expected.named_parameters():p.copy_(dict(ft.named_parameters())[name])
            x=torch.tensor([[.2,.8],[.3,.7]])
            self.assertTrue(torch.equal(hybrid(x),expected(x)))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    with (HERE/'STATE_SWAP_CPU_CHECKS.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),torch=str(torch.__version__),
            known_synthetic_only=True,checkpoint_read=False,gpu_or_ssh=False,new_hash_computed=False,
            source=s.stat(HERE/'evaluate_state_swap.py')),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
