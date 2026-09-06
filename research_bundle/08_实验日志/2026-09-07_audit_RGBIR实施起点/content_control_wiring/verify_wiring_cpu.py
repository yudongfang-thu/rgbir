"""Actual TaskCriterion with synthetic CPU interfaces; no real model/GPU claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Preserve previous review result')
sys.path[:0] = [str(args.module_dir), str(args.module_dir/'legacy_oev1')]
import torch
from test_content_controls import evidence_raw
from test_localization_loss import labels
from object_evidence_loss import EvidenceConfig, object_evidence_loss
from content_controls import object_evidence_content_loss, same_modal_evidence_loss


class BaseCriterion:
    def __init__(self, native, teacher, reference, cfg, arm, trainer, sanity):
        self.native, self.teacher, self.reference = native, teacher, reference
        self.cfg, self.arm, self.trainer, self.sanity = cfg, arm, trainer, sanity
        self.calls, self.selected_total, self.gradient_checks = 0, 0, []
        self.evidence_cfg = EvidenceConfig(**cfg['evidence'])


def combine(native, kd, batch_size, weight):
    return native.sum() + float(batch_size)*float(weight)*kd


bridge = types.ModuleType('legacy_bridge')
bridge.legacy = types.SimpleNamespace(EvidenceCriterion=BaseCriterion,
    raw_prediction=lambda x: x, combine_loss=combine, append_json=lambda *args: None)
bridge.object_evidence_loss, bridge.EvidenceConfig = object_evidence_loss, EvidenceConfig
sys.modules['legacy_bridge'] = bridge
from task_criterion import TaskCriterion


class Frozen(torch.nn.Module):
    def __init__(self, foreground):
        super().__init__()
        self.foreground, self.calls = foreground, 0
    def forward(self, image):
        self.calls += 1
        foreground = self.foreground if float(image.mean()) > .5 else 1.
        return evidence_raw(foreground, requires_grad=False)


class Native:
    assigner = object()
    def __call__(self, prediction, batch):
        values = torch.stack([prediction['scores'].square().mean(),
                              prediction['boxes'].square().mean(), prediction['boxes'].sum()*0])
        return values, values.detach()


cfg = dict(evidence={'input_size': 64}, localization={'input_size': 64}, geometry_contract=None,
           seed=42, kd_weight=.1, localization_coefficient=None, log_every_batches=100)
trainer = types.SimpleNamespace(model=types.SimpleNamespace(stride=torch.tensor([8,16,32])),
    epoch=0, real_updates=0, save_dir=Path('/no-write'))
rows = []
for arm in ('c_shuffled', 'c_same_modal'):
    student, teacher, reference, native = evidence_raw(0), Frozen(4), Frozen(1), Native()
    batch = labels()
    batch['teacher_batch'] = labels()
    batch.update(img=torch.ones(1,3,64,64), strong_img=torch.ones(1,3,64,64),
                 content_img=torch.zeros(1,3,64,64), im_file=['fixture'])
    criterion = TaskCriterion(native, teacher, reference, cfg, arm, trainer, True)
    actual, _ = criterion(student, batch)
    t, r = evidence_raw(4, requires_grad=False), evidence_raw(1, requires_grad=False)
    if arm == 'c_shuffled':
        loss, stats = object_evidence_content_loss(student, t, r, batch,
            content_teacher=evidence_raw(1, requires_grad=False), config=EvidenceConfig(input_size=64))
    else:
        loss, stats = same_modal_evidence_loss(student, t, r, batch, config=EvidenceConfig(input_size=64))
    assert torch.equal(actual, combine(native(student, batch)[0], loss, 1, .1))
    assert criterion.last_stats['lambda_L'] == 0 and criterion.last_stats['c'] == stats
    assert criterion.gradient_checks[0]['kd_score_gradient_l2'] > 0
    assert teacher.calls == (2 if arm == 'c_shuffled' else 1)
    rows.append(dict(arm=arm, exact_composed_loss=True, exact_content_stats=True,
                     nonzero_C_gradient=True, lambda_L=0.0, teacher_forward_calls=teacher.calls))
result = dict(status='PASSED', scope='actual TaskCriterion with synthetic CPU native/frozen interfaces',
              torch_version=str(torch.__version__), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
              real_model_canary=False, cases=rows)
args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
