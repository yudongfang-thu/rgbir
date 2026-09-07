"""Frozen single-task identities and configuration checks; no Torch or GPU I/O.

This validates configuration logic. The trainer still must verify the actual
receipt files, model lifecycle, data flow and bound lease before admitting a run.
An empty error list is NOT a formal-training authorization.
"""
from __future__ import annotations

import math

ARMS = ('N', 'C0', 'C1', 'C1_y', 'L1', 'L_GT')
SOURCES = ('paired', 'shuffled', 'same_modal')
SEEDS = (0, 42, 123)
ENDPOINT = 'fixed_budget_last_ema'
MIN_GAIN_PP = 0.10
HARM_AP50_DROP_PP = 0.20
CLASS_ARMS = ('C0', 'C1', 'C1_y')
LOC_ARMS = ('L1', 'L_GT')


def identity_errors(arm, source='paired', seed=None):
    errors = []
    if arm not in ARMS:
        errors.append('Unsupported arm: joint/fusion/router and legacy arm aliases are forbidden')
    if source not in SOURCES:
        errors.append('source must be paired, shuffled, or same_modal')
    if arm == 'N' and source != 'paired':
        errors.append('N is one zero-KD baseline, not a separate content-control arm')
    if seed is not None and (type(seed) is not int or seed not in SEEDS):
        errors.append('seed must be exactly one of 0, 42, 123')
    return errors


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def required_receipts(arm, source='paired'):
    names = ['baseline_identity', 'recipe', 'compatibility', 'canary', 'resource_lease']
    if arm in ('C1', 'C1_y', 'L1', 'L_GT'):
        names.append('calibration')
    if arm in LOC_ARMS:
        names.extend(['geometry_contract', 'verified_d2'])
    if source != 'paired':
        names.extend(['source_control_identity', 'source_control_canary'])
    return names


def validate_config(cfg, arm=None, formal=False, source=None):
    """Return errors instead of mutating cfg or silently applying defaults.

    Native project keys are supported: classification_coefficient,
    localization_coefficient, teacher/reference, paths, and optional cfg['arm'].
    Missing new coefficients may remain None in a draft, never in formal mode.
    Receipt schema is deliberately owned by the trainer, not guessed here.
    """
    if not isinstance(cfg, dict):
        return {'valid': False, 'errors': ['config must be a mapping']}
    arm = cfg.get('arm') if arm is None else arm
    source = cfg.get('source', 'paired') if source is None else source
    errors = identity_errors(arm, source, cfg.get('seed'))
    if cfg.get('arm', arm) != arm:
        errors.append('CLI arm differs from cfg arm')
    if cfg.get('source', source) != source:
        errors.append('CLI source differs from cfg source')
    if formal and cfg.get('seed') is None:
        errors.append('Formal configuration requires its actual seed')
    if cfg.get('allow_joint_kd') or cfg.get('allow_new_routing'):
        errors.append('Joint KD and learned routing are disabled in this phase')
    for key in ('allow_joint_kd', 'allow_new_routing'):
        if cfg.get('run_policy', {}).get(key):
            errors.append('run_policy.' + key + ' must be false')
    c = cfg.get('classification_coefficient')
    loc = cfg.get('localization_coefficient')
    for key, value in (('classification_coefficient', c), ('localization_coefficient', loc)):
        if value is not None and (not _number(value) or value < 0):
            errors.append(key + ' must be finite and nonnegative')
    if _number(c) and _number(loc) and c > 0 and loc > 0:
        errors.append('Two nonzero KD coefficients are forbidden')
    if arm == 'N' and (c != 0 or loc != 0):
        errors.append('N requires both coefficients to be explicitly zero')
    if arm in CLASS_ARMS and loc != 0:
        errors.append('Classification-only requires localization_coefficient=0')
    if arm in LOC_ARMS and c != 0:
        errors.append('Localization-only requires classification_coefficient=0')
    if arm == 'C0' and c != 0.1:
        errors.append('C0 coefficient must remain exactly 0.1')
    active = c if arm in CLASS_ARMS else loc
    if arm in ('C1', 'C1_y', 'L1', 'L_GT'):
        if active is None:
            if formal:
                errors.append('Formal KD coefficient is not calibrated')
        elif _number(active) and not 0 < active <= 1:
            errors.append('Calibrated coefficient must be in (0,1]')
    classification = cfg.get('classification', {})
    if arm in ('C1', 'C1_y'):
        expected_eta = 0.25 if arm == 'C1' else 0.0
        if 'off_target_weight' in classification and classification['off_target_weight'] != expected_eta:
            errors.append('off_target_weight differs from frozen arm definition')
    if source == 'same_modal':
        if cfg.get('teacher') and cfg.get('teacher') == cfg.get('reference'):
            errors.append('same_modal teacher must differ from reference')
        for key in ('teacher_labels_used_by_kd', 'privileged_ir_labels_used_by_kd'):
            if cfg.get(key) is True:
                errors.append('same_modal cannot inherit IR labels/masks: ' + key)
    if formal:
        for key in ('method_id', 'model', 'teacher', 'reference'):
            if not cfg.get(key):
                errors.append('Missing model/method identity: ' + key)
        if not cfg.get('paths', {}).get('student_data_yaml'):
            errors.append('Missing student data identity')
        for key, expected in (('epochs', 200), ('imgsz', 640), ('batch', 32), ('nbs', 64), ('workers', 4)):
            if cfg.get(key) != expected:
                errors.append('Frozen recipe mismatch: ' + key)
    return {'valid': not errors, 'errors': errors, 'arm': arm, 'source': source,
            'required_receipts': required_receipts(arm, source),
            'formal_receipts_verified': False,
            'scope': 'configuration logic only; actual receipt and lease verification remains mandatory'}


def require_valid_config(cfg, arm=None, formal=False, source=None):
    result = validate_config(cfg, arm, formal, source)
    if not result['valid']:
        raise ValueError('; '.join(result['errors']))
    return result


def core_budget(localization_dataset='drone', classification_ablation=True, localization_enabled=True):
    """Frozen upper-bound accounting, never a queue or authorization."""
    if localization_dataset not in ('drone', 'llvip'):
        raise ValueError('Only the authorized Drone/LLVIP localization branch exists')
    baseline = 3 if localization_enabled and localization_dataset == 'llvip' else 0
    core = 3 + (3 if classification_ablation else 0) + (6 if localization_enabled else 0) + baseline
    return {'initial_classification': 3,
            'first_two_line_stage': 3 + (4 if localization_enabled else 0) + baseline,
            'max_core': core,
            'both_new_winners_with_four_arms': core + (12 if localization_enabled else 6),
            'new_localization_baselines': baseline,
            'localization_dataset': localization_dataset,
            'is_launch_authorization': False}
