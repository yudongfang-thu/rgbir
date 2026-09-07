"""Bounded CPU truth checks; no Ultralytics model, data loader, GPU or receipts."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import torch
import yaml

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
MODULE = WORKSPACE / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
sys.path.insert(0, str(MODULE))
from classification_logit import classification_relative_kd
from localization_adapter import LocalizationAdapter
from protocol import validate_config


def extract_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    compile_tree = ast.Module(body=[node], type_ignores=[])
    exec(compile(compile_tree, str(path), 'exec'), namespace)
    return namespace[name]


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    checks = {}
    s = torch.tensor([[[-1.2], [.3]], [[2.1], [-.8]]], requires_grad=True)
    t = torch.tensor([[[.8], [-.4]], [[-.1], [1.7]]])
    valid = torch.tensor([[True, True], [True, False]])
    selected = torch.tensor([True, True])
    labels = torch.zeros(2, dtype=torch.long)
    c1 = classification_relative_kd(s, t, valid, selected, labels, off_target_weight=.25)
    c1y = classification_relative_kd(s, t, valid, selected, labels, off_target_weight=0.)
    g1 = torch.autograd.grad(c1, s, retain_graph=True)[0]
    gy = torch.autograd.grad(c1y, s, retain_graph=True)[0]
    assert torch.equal(c1, c1y) and torch.equal(g1, gy) and c1 > 0 and g1.norm() > 0
    checks['single_class_C1_equals_C1_y_loss_and_gradient_not_zero'] = {'loss': float(c1), 'gradient_l2': float(g1.norm()), 'exact': True}
    combine = extract_function(MODULE / 'task_conditional_reference/legacy_oev1/train_object_evidence.py', 'combine_loss', {})
    x = torch.tensor([.2, -.4], requires_grad=True)
    native = x.square().sum()
    kd = (x - 3).square().sum()
    total = combine(native, kd, 32, 0.)
    assert torch.equal(total, native)
    assert torch.equal(torch.autograd.grad(total, x, retain_graph=True)[0], torch.autograd.grad(native, x)[0])
    checks['actual_zero_weight_combiner_native_loss_gradient_exact'] = 'passed'
    class LegacyStub:
        def build_trainer(self, *args):
            return args
    legacy = LegacyStub()
    symbols = dict(legacy=legacy, ORIGINAL_CRITERION=object(), ORIGINAL_DATASET=object(),
                   IndependentCriterion=object(), TrackedDualLabelRGBIRDataset=object())
    build = extract_function(MODULE / 'train_independent.py', 'build_trainer', symbols)
    result = build({'arm': 'N'}, 'config', 'output')
    assert result[3] == 'weight0' and result[0]['kd_weight'] == .1
    assert legacy.EvidenceCriterion is symbols['IndependentCriterion']
    assert legacy.DualLabelRGBIRDataset is symbols['TrackedDualLabelRGBIRDataset']
    checks['actual_build_trainer_N_maps_to_weight0_and_tracked_loader'] = 'passed'
    try:
        LocalizationAdapter(None)
    except ValueError as e:
        assert 'accepted' in str(e)
        checks['empty_geometry_blocks_L1_L_GT_constructor'] = str(e)
    else:
        raise AssertionError('Missing geometry unexpectedly allowed')
    profile_spec = importlib.util.spec_from_file_location('_llvip_eval_profile', MODULE / 'evaluator_profile.py')
    profile = importlib.util.module_from_spec(profile_spec); profile_spec.loader.exec_module(profile)
    cfg = yaml.safe_load((HERE / 'prepared_v1/llvip_N_seed42_NOT_ADMITTED.yaml').read_text(encoding='utf-8'))
    try:
        profile.configuration_identity(cfg)
    except ValueError as e:
        assert 'Drone' in str(e)
        checks['current_evaluator_profile_rejects_LLVIP_confirmed_blocker'] = str(e)
    else:
        raise AssertionError('Expected known Drone-only evaluation profile restriction changed')
    for path in sorted((HERE / 'prepared_v1').glob('*NOT_ADMITTED.yaml')):
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert validate_config(cfg, formal=False)['valid']
        assert cfg['protocol_status'] == 'NOT_ADMITTED' and cfg['formal_training_authorized'] is False
        assert all(cfg[k] is None for k in ('readiness_receipt', 'geometry_contract', 'calibration_receipt', 'canary_acceptance', 'd2_receipt'))
        if cfg['arm'] != 'N':
            assert cfg['localization_coefficient'] is None and not validate_config(cfg, formal=True)['valid']
    checks['nine_drafts_valid_unadmitted_null_evidence_and_uncalibrated_L'] = 'passed'
    assert not torch.cuda.is_initialized()
    receipt = {'status': 'CPU_CHECKS_PASSED_NOT_ADMITTED', 'torch_version': torch.__version__,
               'pinned_training_environment': False, 'cuda_initialized': False, 'tests': checks,
               'formal_training_authorized': False, 'scope': 'Math and config/dispatch logic only; not real-data compatibility or calibration'}
    (HERE / 'cpu_checks.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
