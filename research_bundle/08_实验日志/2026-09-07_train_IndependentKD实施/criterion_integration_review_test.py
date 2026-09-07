"""Independent CPU integration fixture, never a real model/data/GPU receipt."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
sys.path.insert(0, str(MODULE))
from test_classification_logit import evidence_raw, labels
from selection_adapter import EvidenceConfig
from test_gradient_observation import Model


class FrozenRaw(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw

    def forward(self, _):
        return self.raw


class Base:
    def __init__(self, native, teacher, reference, cfg, arm, trainer, sanity):
        self.native, self.teacher, self.reference = native, teacher, reference
        self.cfg, self.arm, self.trainer, self.sanity = cfg, arm, trainer, sanity
        self.evidence_cfg = EvidenceConfig(**cfg['evidence'])
        self.calls, self.selected_total, self.gradient_checks = 0, 0, []

    def __call__(self, prediction, batch):
        return ('historical', self.arm)


class IntegratedCriterionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = []
        legacy = types.SimpleNamespace(
            raw_prediction=lambda p: p,
            combine_loss=lambda native,kd,b,w: native.sum()+b*w*kd,
            append_json=lambda p,row: cls.events.append((str(p),json.loads(json.dumps(row,allow_nan=False)))))
        fake = types.ModuleType('runtime')
        fake.legacy, fake.ORIGINAL_CRITERION = legacy, Base
        spec = importlib.util.spec_from_file_location('_independent_review_actual_criterion', MODULE/'independent_criterion.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'runtime': fake}):
            spec.loader.exec_module(module)
        cls.criterion_cls = module.IndependentCriterion

    def fixture(self, arm='C1', observe=True):
        model = Model()
        model.stride = torch.tensor([8,16,32])
        with torch.no_grad():
            model.model[0].weight.fill_(.3)
            model.model[1].weight.fill_(.7)
        student, teacher, reference = evidence_raw(0), evidence_raw(4), evidence_raw(1)
        pattern = torch.arange(student['scores'].shape[-1],dtype=torch.float32)
        student['scores'] = student['scores'].detach()+model.model[0].weight.reshape(1,1,1)*(pattern%3)[None,None]+model.model[1].weight.reshape(1,1,1)*(pattern%5)[None,None]
        # The actual criterion still sees the normal raw class/DFL dictionary.
        native = lambda raw,b: ((raw['scores'].square().mean()+raw['boxes'].square().mean()).reshape(1),torch.zeros(3))
        batch = labels()
        batch['teacher_batch'] = labels()
        batch['img'] = torch.zeros(1,3,64,64)
        batch['strong_img'] = torch.zeros(1,3,64,64)
        batch['im_file'] = ['synthetic-fixture-only']
        trainer = types.SimpleNamespace(model=model, epoch=0, real_updates=0, update_attempts=0,
                skipped_amp_updates=0, ema=types.SimpleNamespace(updates=0), amp=False,
                scaler=types.SimpleNamespace(get_scale=lambda:1.), save_dir=Path('/synthetic'),
                wall_started=0.)
        cfg = dict(arm=arm,source='paired',seed=42,classification_coefficient=.2,
                   localization_coefficient=0.,log_every_batches=50,
                   evidence=dict(levels=(0,1),rho=.5,input_size=64))
        criterion = self.criterion_cls(native,FrozenRaw(teacher),FrozenRaw(reference),cfg,arm,trainer,False)
        if not observe and arm in ('C1','C1_y'):
            criterion.shared_gradient_observer.observed_epochs.add(0)
        return criterion, model, student, batch, trainer

    def test_real_criterion_observation_preserves_total_backward_and_rng(self):
        for arm in ('C1','C1_y'):
            observed, model, prediction, batch, trainer = self.fixture(arm,True)
            plain, plain_model, plain_prediction, plain_batch, _ = self.fixture(arm,False)
            state = {k:v.clone() for k,v in model.state_dict().items()}
            rng = (torch.get_rng_state().clone(),random.getstate(),np.random.get_state())
            before = len(self.events)
            total, _ = observed(prediction,batch)
            self.assertTrue(torch.equal(rng[0],torch.get_rng_state()))
            self.assertEqual(rng[1],random.getstate())
            self.assertTrue(np.array_equal(rng[2][1],np.random.get_state()[1]))
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            total_plain,_ = plain(plain_prediction,plain_batch)
            self.assertTrue(torch.equal(total,total_plain))
            total.backward()
            total_plain.backward()
            for a,b in zip(model.parameters(),plain_model.parameters()):
                self.assertTrue(torch.equal(a.grad,b.grad))
            observations=[r for p,r in self.events[before:] if p.endswith('shared_gradient_observations.jsonl')]
            self.assertEqual(len(observations),1)
            self.assertEqual(observations[0]['off_target_weight'],.25 if arm=='C1' else 0.)
            self.assertEqual(observations[0]['actual_B'],1)
            self.assertEqual(trainer.ema.updates,0)
            self.assertEqual(trainer.real_updates,0)

    def test_historical_dispatch_does_not_enter_observer_or_new_loss(self):
        for arm,legacy_arm in (('N','weight0'),('C0','paired')):
            criterion,_,prediction,batch,_=self.fixture(arm)
            self.assertFalse(hasattr(criterion,'shared_gradient_observer'))
            self.assertEqual(criterion(prediction,batch),('historical',legacy_arm))


if __name__=='__main__':
    unittest.main(verbosity=2)
