"""CPU numerical/read-only contracts for fixed-epoch shared gradients."""
import json
import math
import random
import unittest
import numpy as np
import torch
from gradient_observation import FixedEpochGradientObserver, OBSERVATION_EPOCHS, shared_parameter_set


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = [0, 1]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.Linear(1,1,bias=False),
                                        torch.nn.Linear(1,1,bias=False), Head()])
        self.model[0].register_buffer('running_mean', torch.tensor([7.]))


class GradientObservationTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.p, self.q = (self.model.model[i].weight for i in (0,1))
        self.native = (3*self.p+4*self.q).sum()
        self.target = (2*self.p).sum()
        self.off = (-4*self.p).sum()
        self.full = self.target+.25*self.off

    def observe(self, observer=None, arm='C1', epoch=0, batch=2, coefficient=.5, **kwargs):
        return (observer or FixedEpochGradientObserver()).observe(self.model, self.native,
            self.target if arm=='C1_y' else self.full,
            arm=arm, epoch=epoch, batch_index=1, actual_batch=batch, coefficient=coefficient,
            target_unit=self.target, off_target_unit=self.off,
            off_target_weight=0. if arm=='C1_y' else .25, **kwargs)

    def test_known_gradient_norms_directions_and_component_cancellation(self):
        row = self.observe()
        self.assertAlmostEqual(row['native']['norm'],5.)
        self.assertAlmostEqual(row['weighted_kd']['norm'],1.)
        self.assertAlmostEqual(row['weighted_kd_native_ratio'],.2)
        self.assertAlmostEqual(row['weighted_kd_native_cosine'],.6)
        self.assertAlmostEqual(row['weighted_target']['norm'],2.)
        self.assertAlmostEqual(row['weighted_off_target']['norm'],1.)
        self.assertAlmostEqual(row['target_off_target_cosine'],-1.)
        y = self.observe(arm='C1_y')
        self.assertAlmostEqual(y['weighted_kd']['norm'],2.)
        self.assertAlmostEqual(y['weighted_off_target']['norm'],0.)
        self.assertIsNone(y['target_off_target_cosine'])
        self.assertGreater(y['weighted_kd']['norm'],row['weighted_kd']['norm'])

    def test_actual_partial_batch_scales_kd_once_native_already_scaled(self):
        row = self.observe(batch=3,coefficient=.2)
        self.assertAlmostEqual(row['weighted_kd']['norm'],.6,places=6)
        self.assertAlmostEqual(row['native']['norm'],5.)
        self.assertAlmostEqual(row['weighted_target']['norm'],1.2,places=6)

    def test_reads_do_not_modify_accumulated_gradients_state_or_rng_and_graph_survives(self):
        self.p.grad = torch.tensor([[11.]])
        self.q.grad = torch.tensor([[13.]])
        state = {name:value.clone() for name,value in self.model.state_dict().items()}
        before = (torch.get_rng_state().clone(),random.getstate(),np.random.get_state())
        self.observe()
        self.assertTrue(torch.equal(self.p.grad,torch.tensor([[11.]])))
        self.assertTrue(torch.equal(self.q.grad,torch.tensor([[13.]])))
        self.assertTrue(all(torch.equal(state[name],value) for name,value in self.model.state_dict().items()))
        self.assertTrue(torch.equal(before[0],torch.get_rng_state()))
        self.assertEqual(before[1],random.getstate())
        after=np.random.get_state()
        self.assertEqual(before[2][0],after[0])
        self.assertTrue(np.array_equal(before[2][1],after[1]))
        self.assertEqual(before[2][2:],after[2:])
        (self.native+self.full).backward()
        self.assertTrue(torch.equal(self.p.grad,torch.tensor([[15.]])))
        self.assertTrue(torch.equal(self.q.grad,torch.tensor([[17.]])))

    def test_fixed_epoch_schedule_observes_empty_signal_once_without_replacement(self):
        observer=FixedEpochGradientObserver()
        self.assertEqual(OBSERVATION_EPOCHS,(0,10,50,100,199))
        zero=self.p.sum()*0
        row=observer.observe(self.model,self.native,zero,arm='L1',epoch=0,batch_index=1,
                             actual_batch=32,coefficient=.1)
        self.assertEqual(row['status'],'ZERO_KD_SIGNAL')
        self.assertEqual(row['weighted_kd_native_ratio'],0.)
        self.assertIsNone(row['weighted_kd_native_cosine'])
        self.assertFalse(observer.should_observe(0))
        self.assertIsNone(self.observe(observer=observer,epoch=0))
        for epoch in (1,2,9,11,49,51,198):
            self.assertFalse(observer.should_observe(epoch))
        for epoch in (10,50,100,199):
            self.assertTrue(observer.should_observe(epoch))
        self.assertEqual(observer.__dict__,{'observed_epochs':{0}})

    def test_localization_observation_and_unused_parameters(self):
        kd=(-self.p+3*self.q).sum()
        row=FixedEpochGradientObserver().observe(self.model,self.native,kd,arm='L_GT',epoch=10,
            batch_index=42,actual_batch=2,coefficient=.5)
        self.assertAlmostEqual(row['weighted_kd']['norm'],math.sqrt(10))
        self.assertAlmostEqual(row['weighted_kd_native_cosine'],9/(5*math.sqrt(10)))
        self.assertNotIn('weighted_target',row)
        row=self.observe()
        self.assertEqual(row['weighted_kd']['absent_parameter_tensors'],1)

    def test_nonfinite_gradient_is_json_safe_observation_and_does_not_mutate_grad(self):
        zero=self.p-self.p.detach()
        kd=zero.sqrt().sum()  # finite loss with an infinite derivative
        self.assertTrue(bool(torch.isfinite(kd)))
        row=FixedEpochGradientObserver().observe(self.model,self.native,kd,arm='L1',epoch=0,
            batch_index=1,actual_batch=2,coefficient=.5)
        self.assertEqual(row['status'],'NONFINITE_OBSERVATION')
        self.assertIsNone(row['weighted_kd']['norm'])
        self.assertIsNone(row['weighted_kd_native_ratio'])
        self.assertEqual(row['weighted_kd']['nonfinite_parameter_tensors'],1)
        self.assertIsNone(self.p.grad)
        json.dumps(row,allow_nan=False)

    def test_zero_native_has_no_epsilon_ratio(self):
        row=FixedEpochGradientObserver().observe(self.model,self.native*0,self.full,arm='L1',epoch=0,
            batch_index=1,actual_batch=2,coefficient=.5)
        self.assertEqual(row['native']['norm'],0.)
        self.assertIsNone(row['weighted_kd_native_ratio'])

    def test_parameter_names_and_no_arbitrary_model_fallback(self):
        indices,named=shared_parameter_set(self.model)
        self.assertEqual(indices,[0,1])
        self.assertEqual([name for name,_ in named],['model.0.weight','model.1.weight'])
        row=self.observe()
        self.assertEqual(row['parameter_names'],[name for name,_ in named])
        self.model.model[-1].f=[0,0]
        with self.assertRaises(ValueError):
            self.observe()

    def test_json_only_record_no_graph_or_tensor_retention(self):
        row=self.observe()
        self.assertEqual(json.loads(json.dumps(row,allow_nan=False)),row)
        self.assertEqual(row['gradient_stage'],'unscaled_unclipped_per_batch_autograd_grad')
        self.assertFalse(row['parameter_grad_accumulation_modified'])

    def test_historical_and_wrong_eta_rejected(self):
        for arm in ('N','C0','CL'):
            with self.assertRaises(ValueError):
                self.observe(arm=arm)
        with self.assertRaises(ValueError):
            FixedEpochGradientObserver().observe(self.model,self.native,self.full,arm='C1_y',epoch=0,
                batch_index=1,actual_batch=2,coefficient=.5,target_unit=self.target,
                off_target_unit=self.off,off_target_weight=.25)


if __name__=='__main__':
    unittest.main(verbosity=2)
