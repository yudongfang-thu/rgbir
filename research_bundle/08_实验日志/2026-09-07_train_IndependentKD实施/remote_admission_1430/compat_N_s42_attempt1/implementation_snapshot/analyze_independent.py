"""Independent-KD descriptive analyzer with outcome-blind, receipt-bound inputs.

CLI manifest: {runs:[{arm, source, seed, path, source_arm, protocol_id,
evaluator_contract}], implementation_checks_passed:false}. The file loader uses
the existing v2 receipt verifier; it never accepts inline AP as a file endpoint.
analyze_records is a pure kernel for already-verified records and small truth
fixtures. It cannot prove that caller-created dictionaries came from real runs.

The implementation is NOT accepted merely because its tests pass. Automatic
expansion remains false unless a separate accepted review binds these sources.
No hash computation, training, Torch import or GPU access occurs here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import copy

try:
    from .protocol import ARMS, SOURCES, SEEDS, ENDPOINT, MIN_GAIN_PP, HARM_AP50_DROP_PP, identity_errors
except ImportError:
    from protocol import ARMS, SOURCES, SEEDS, ENDPOINT, MIN_GAIN_PP, HARM_AP50_DROP_PP, identity_errors

METRICS = ('mAP50_95', 'AP50', 'AP75', 'precision', 'recall')
PAIR_FIELDS = ('dataset', 'protocol_id', 'endpoint', 'roster', 'frozen_recipe', 'evaluator_contract')
PRIMARY_PAIRS = (('C0', 'N'), ('C1', 'N'), ('C1', 'C0'), ('C1', 'C1_y'),
                 ('C1_y', 'N'), ('C1_y', 'C0'), ('L1', 'N'), ('L_GT', 'N'), ('L1', 'L_GT'))
ERROR_CONTRACT = {'confidence': 0.25, 'match_iou': 0.50, 'coarse_iou': 0.10,
                  'background_definition': 'all_gt_iou_below_0.10',
                  'background_unit': 'false_positives_per_image'}


def to_percent(value, units):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Metric must be a finite number, not bool/null')
    if units == 'fraction_0_to_1' and 0 <= value <= 1:
        return float(value) * 100.0
    if units == 'percent_0_to_100' and 0 <= value <= 100:
        return float(value)
    raise ValueError('Explicit units or metric range is invalid')


def summarize(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) >= 2 else None, 'ddof': 1}


def validate_endpoint(record):
    """Validate a normalized, externally file-verified record, without mutation."""
    errors = identity_errors(record.get('arm'), record.get('source', 'paired'), record.get('seed'))
    if record.get('status') != 'complete':
        errors.append('Endpoint is incomplete or invalid')
    if record.get('receipt_validation') != 'passed':
        errors.append('Actual training/evaluation receipt verification is required')
    if record.get('endpoint') != ENDPOINT:
        errors.append('Endpoint must be fixed-budget last/EMA')
    if record.get('official_test_accessed') is not False:
        errors.append('Test exposure flag must be explicitly false')
    for field in ('dataset', 'protocol_id', 'frozen_recipe', 'evaluator_contract'):
        if not record.get(field):
            errors.append('Missing comparison identity: ' + field)
    if record.get('evaluation_contract_validation') != 'passed':
        errors.append('Evaluator contract/full population must be bound to actual evaluation evidence')
    if record.get('arm') != 'N' and (record.get('intervention_validation') != 'passed'
                                   or not isinstance(record.get('intervention_identity'), dict)):
        errors.append('KD intervention identity must be receipt-bound')
    recipe = record.get('frozen_recipe') or {}
    if recipe.get('epochs') != 200:
        errors.append('Full E200 budget is required')
    roster = record.get('roster')
    if not isinstance(roster, list) or not roster or any(not isinstance(x, str) for x in roster):
        errors.append('Complete development roster is missing')
    elif len(roster) != len(set(roster)):
        errors.append('Development roster contains duplicates')
    if record.get('expected_val_images') != (len(roster) if isinstance(roster, list) else None):
        errors.append('Development roster does not match the frozen full population')
    for name in METRICS:
        try:
            to_percent((record.get('metrics_percent') or {}).get(name), 'percent_0_to_100')
        except ValueError as error:
            errors.append(name + ': ' + str(error))
    return {'valid': not errors, 'errors': errors}


def _rows(records, arm, source, seed):
    return [r for r in records if r.get('arm') == arm and r.get('source', 'paired') == source and r.get('seed') == seed]


def paired_comparison(records, treatment, control, treatment_source='paired', control_source='paired'):
    pairs, issues, members = [], [], []
    for seed in SEEDS:
        left, right = _rows(records, treatment, treatment_source, seed), _rows(records, control, control_source, seed)
        if len(left) != 1 or len(right) != 1:
            issues.append('seed %s: require exactly one treatment and one control endpoint' % seed)
            continue
        a, b = left[0], right[0]
        validations = [validate_endpoint(a), validate_endpoint(b)]
        if not all(x['valid'] for x in validations):
            issues.append('seed %s: %s' % (seed, '; '.join(e for x in validations for e in x['errors'])))
            continue
        different = [f for f in PAIR_FIELDS if a.get(f) != b.get(f)]
        if different:
            issues.append('seed %s: mismatched %s' % (seed, ', '.join(different)))
            continue
        intervention_errors = _intervention_errors(a, b, treatment, control, treatment_source, control_source)
        if intervention_errors:
            issues.append('seed %s: intervention identity mismatch: %s' % (seed, ', '.join(intervention_errors)))
            continue
        members.extend([a, b])
        pairs.append({'seed': seed, 'deltas_pp': {m: a['metrics_percent'][m] - b['metrics_percent'][m] for m in METRICS}})
    intervention_mixed = False
    for arm, source in {(r['arm'], r.get('source', 'paired')) for r in members if r['arm'] != 'N'}:
        values = [r.get('intervention_identity') for r in members if r['arm'] == arm and r.get('source', 'paired') == source]
        if values and any(value != values[0] for value in values[1:]):
            intervention_mixed = True
            issues.append('Cross-seed intervention differs within %s/%s; aggregate withheld' % (arm, source))
    if intervention_mixed or (members and any(any(r.get(f) != members[0].get(f) for f in PAIR_FIELDS) for r in members[1:])):
        issues.append('Cross-seed recipe/population/evaluator mismatch; aggregate withheld')
        stats = None
    else:
        stats = {m: dict(summarize([p['deltas_pp'][m] for p in pairs]),
                         positive_seeds=sum(p['deltas_pp'][m] > 0 for p in pairs),
                         negative_seeds=sum(p['deltas_pp'][m] < 0 for p in pairs)) for m in METRICS}
    complete = len(pairs) == 3 and not issues
    return {'treatment': treatment, 'control': control, 'treatment_source': treatment_source,
            'control_source': control_source, 'pairs': pairs, 'stats_pp': stats,
            'complete_three_seed_pairing': complete, 'issues': issues,
            'stop_running_experiments': False}


def _intervention_errors(a, b, treatment, control, treatment_source, control_source):
    """Allow only the intervention differences specified by each contrast."""
    if control == 'N':
        return []  # The common native recipe, data and endpoint still must match.
    ia, ib = a.get('intervention_identity'), b.get('intervention_identity')
    if not isinstance(ia, dict) or not isinstance(ib, dict):
        return ['missing receipt-bound intervention identity']
    if a.get('intervention_validation') != 'passed' or b.get('intervention_validation') != 'passed':
        return ['intervention evidence was not verified']
    fields = ['reference', 'selection']
    if control_source != 'same_modal' and treatment_source != 'same_modal':
        fields += ['teacher', 'privileged_data']
    if (treatment, control) in (('C1', 'C1_y'), ('L1', 'L_GT')) or treatment == control:
        fields += ['coefficient', 'payload_common']
    if (treatment, control) == ('L1', 'L_GT') or (treatment == control == 'L1' and control_source != 'same_modal'):
        fields += ['geometry_content']
        if ia.get('geometry_content') is None or ib.get('geometry_content') is None:
            return ['localization geometry was not bound']
    # A fully RGB-only control constructs a different eligible set; threshold
    # rules still match, while teacher/IR labels and geometry may differ.
    if control_source == 'same_modal' or treatment_source == 'same_modal':
        fields += ['selection_algorithm']
    return [field for field in fields if field not in ia or field not in ib or ia[field] != ib[field]]


def passes_gain(comparison, threshold=MIN_GAIN_PP):
    if not comparison or not comparison['complete_three_seed_pairing'] or comparison['stats_pp'] is None:
        return False
    stat = comparison['stats_pp']['mAP50_95']
    return stat['positive_seeds'] == 3 and stat['mean'] > 0 and stat['mean'] >= threshold - 1e-10


def harm_review(records, arm, source='paired', control='N'):
    comparison = paired_comparison(records, arm, control, source, 'paired')
    triggers, missing = [], []
    if not comparison['complete_three_seed_pairing']:
        return {'status': 'INCOMPLETE', 'triggers': [], 'missing': comparison['issues'], 'stop_training': False}
    if comparison['stats_pp']['AP50']['mean'] < -HARM_AP50_DROP_PP - 1e-10:
        triggers.append('Mean AP50 drop exceeds 0.20 pp')
    class_deltas, background_deltas, class_identity = {}, [], None
    for seed in SEEDS:
        a, b = _rows(records, arm, source, seed)[0], _rows(records, control, 'paired', seed)[0]
        ca, cb = a.get('per_class_percent'), b.get('per_class_percent')
        expected_a, expected_b = a.get('expected_class_ids'), b.get('expected_class_ids')
        if (not isinstance(ca, dict) or not ca or not isinstance(cb, dict)
                or not expected_a or expected_a != expected_b or set(ca) != set(expected_a) or set(cb) != set(expected_b)):
            missing.append('seed %s: all-class metrics missing or mismatched' % seed)
        else:
            if class_identity is None:
                class_identity = set(ca)
            if set(ca) != class_identity:
                missing.append('Cross-seed class inventory differs')
            for cls in ca:
                try:
                    delta = to_percent(ca[cls], 'percent_0_to_100') - to_percent(cb[cls], 'percent_0_to_100')
                    class_deltas.setdefault(cls, []).append(delta)
                except ValueError:
                    missing.append('Invalid per-class AP: ' + str(cls))
        ea, eb = a.get('error_analysis', {}), b.get('error_analysis', {})
        if ea.get('contract') != ERROR_CONTRACT or eb.get('contract') != ERROR_CONTRACT:
            missing.append('seed %s: fixed operating-point error contract missing' % seed)
        else:
            va, vb = ea.get('background_fp_per_image'), eb.get('background_fp_per_image')
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (va, vb)):
                missing.append('seed %s: invalid background FP/image' % seed)
            else:
                background_deltas.append(va - vb)
    for cls, values in class_deltas.items():
        if len(values) == 3 and all(v < 0 for v in values):
            triggers.append('Class %s AP falls on all three seeds' % cls)
    if len(background_deltas) == 3 and all(v > 0 for v in background_deltas):
        triggers.append('Background FP/image rises on all three seeds')
    return {'status': 'REVIEW_REQUIRED' if triggers else ('INCOMPLETE' if missing else 'CLEAR'),
            'triggers': triggers, 'missing': missing, 'per_class_deltas_pp': class_deltas,
            'background_deltas_per_image': background_deltas,
            'stop_training': False, 'interpretation': 'review trigger, not proof of material negative transfer'}


def analyze_records(records, implementation_checks_passed=False, analyzer_accepted=False):
    """Pure logic on normalized records; raw-file verification belongs to loader."""
    identities = [(r.get('dataset'), r.get('arm'), r.get('source', 'paired'), r.get('seed')) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate dataset/arm/source/seed; attempts must be chosen independently of outcome')
    problems = [e for r in records for e in identity_errors(r.get('arm'), r.get('source', 'paired'), r.get('seed'))]
    if problems:
        raise ValueError('; '.join(problems))
    datasets = sorted({r.get('dataset', 'UNKNOWN') for r in records})
    groups = {}
    for dataset in datasets:
        rows = [r for r in records if r.get('dataset', 'UNKNOWN') == dataset]
        comparisons = {a + '_minus_' + b: paired_comparison(rows, a, b) for a, b in PRIMARY_PAIRS}
        gains = lambda a, b, t=MIN_GAIN_PP: passes_gain(comparisons[a + '_minus_' + b], t)
        complete = lambda a, b: comparisons[a + '_minus_' + b]['complete_three_seed_pairing']
        class_status = 'AWAIT_C1_ENDPOINTS'
        winner = 'C0'
        if complete('C1', 'N') and complete('C1', 'C0'):
            class_status = 'PROMOTE_C1' if gains('C1', 'N') and gains('C1', 'C0') else 'KEEP_C0'
            winner = 'C1' if class_status == 'PROMOTE_C1' else 'C0'
        y_fallback = (gains('C1_y', 'N') and gains('C1_y', 'C0') and complete('C1', 'C1_y')
                      and not gains('C1', 'C1_y', 0.0))
        if y_fallback:
            class_status, winner = 'PROMOTE_C1_Y', 'C1_y'
        class_harm = harm_review(rows, winner) if class_status.startswith('PROMOTE') else None
        if class_status.startswith('PROMOTE') and class_harm['status'] != 'CLEAR':
            expansion = 'REVIEW_REQUIRED' if class_harm['status'] == 'REVIEW_REQUIRED' else 'AWAIT_HARM_DIAGNOSTICS'
        else:
            expansion = class_status
        loc_status = 'AWAIT_L1_ENDPOINTS'
        if complete('L1', 'N'):
            loc_status = 'L1_EFFECTIVE' if gains('L1', 'N') else 'L1_NO_PRACTICAL_GAIN'
        loc_harm = harm_review(rows, 'L1') if loc_status == 'L1_EFFECTIVE' else None
        loc_consistency = all(_rows(rows, 'L1', 'paired', s) and
                              _rows(rows, 'L1', 'paired', s)[0].get('localization_diagnostics_consistent') is True
                              for s in SEEDS)
        loc_expansion = loc_status
        if loc_status == 'L1_EFFECTIVE':
            if loc_harm['status'] != 'CLEAR':
                loc_expansion = 'REVIEW_REQUIRED' if loc_harm['status'] == 'REVIEW_REQUIRED' else 'AWAIT_HARM_DIAGNOSTICS'
            elif not loc_consistency:
                loc_expansion = 'AWAIT_LOCALIZATION_DIAGNOSTICS'
        controls = {}
        for arm in ('C0', 'C1', 'C1_y', 'L1'):
            controls[arm] = {source: paired_comparison(rows, arm, arm, 'paired', source)
                             for source in ('shuffled', 'same_modal')}
            controls[arm]['complete_four_arms'] = (paired_comparison(rows, arm, 'N')['complete_three_seed_pairing']
                                                   and all(controls[arm][s]['complete_three_seed_pairing']
                                                           for s in ('shuffled', 'same_modal')))
        valid_rows = [r for r in rows if validate_endpoint(r)['valid']]
        summaries = {}
        for arm, source in sorted({(r['arm'], r.get('source', 'paired')) for r in rows}):
            group = [r for r in valid_rows if r['arm'] == arm and r.get('source', 'paired') == source]
            homogeneous = not group or all(all(r.get(f) == group[0].get(f) for f in PAIR_FIELDS)
                                          and (arm == 'N' or r.get('intervention_identity') == group[0].get('intervention_identity'))
                                          for r in group)
            summaries[arm + '/' + source] = {'seeds': [r['seed'] for r in group],
                'metrics_percent': {m: summarize([r['metrics_percent'][m] for r in group]) for m in METRICS} if homogeneous else None,
                'complete_three_seeds': len(group) == 3 and homogeneous}
        verified = implementation_checks_passed is True and analyzer_accepted is True
        groups[dataset] = {'arms': summaries, 'comparisons': comparisons, 'four_arm_attribution': controls,
            'classification': {'proposed_decision': class_status, 'candidate': winner, 'expansion_status': expansion,
                'harm': class_harm, 'non_target_content_supported': gains('C1', 'C1_y', 0.0),
                'auto_expansion_eligible': verified and expansion.startswith('PROMOTE')},
            'localization': {'proposed_decision': loc_status, 'expansion_status': loc_expansion,
                'harm': loc_harm, 'teacher_content_supported': gains('L1', 'L_GT', 0.0),
                'auto_expansion_eligible': verified and loc_expansion == 'L1_EFFECTIVE'}}
    return {'schema': 'rgbir-independent-analysis-v1', 'seed_order': list(SEEDS),
            'report_units': 'percentage_and_percentage_points', 'datasets': groups,
            'analyzer_acceptance': 'ACCEPTED' if analyzer_accepted is True else 'NOT_ACCEPTED',
            'implementation_checks_passed': implementation_checks_passed is True,
            'stop_running_experiments': False,
            'records': records,
            'claim_policy': 'Descriptive development evidence; thresholds are not significance. Four arms, actual review, geometry and test-exposure audit remain separate requirements.'}


def _legacy_path():
    here = Path(__file__).resolve().parent
    frozen = here / 'task_conditional_reference' / 'analyze_results.py'
    return frozen if frozen.is_file() else here.parent / 'rgbir_task_conditional_v1' / 'analyze_results.py'


def _legacy_loader():
    path = _legacy_path()
    spec = importlib.util.spec_from_file_location('_independent_receipt_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound_configs(path, receipt, folder):
    import yaml  # Only the actual file adapter uses the existing YAML runtime.
    configs, json_sources = [], []
    for relative in receipt.get('source_snapshots', {}).get('config', []):
        source = path / folder / relative
        if source.suffix.lower() in ('.yaml', '.yml'):
            value = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
            if isinstance(value, dict) and 'model' in value and 'paths' in value:
                configs.append(value)
        elif source.suffix.lower() == '.json':
            json_sources.append((relative, json.loads(source.read_text(encoding='utf-8-sig'))))
    if not configs or any(value != configs[0] for value in configs[1:]):
        raise ValueError('Missing or conflicting receipt-bound effective configurations')
    return configs[0], json_sources


def _validate_evaluation_contract(value, roster, metric):
    if not isinstance(value, dict) or value.get('schema') != 'rgbir-evaluation-contract-v1':
        raise ValueError('Missing bound evaluator-contract schema')
    if value.get('expected_val_images') != len(roster):
        raise ValueError('Bound full development population count differs from actual roster')
    if value.get('roster') != roster:
        raise ValueError('Bound evaluator contract does not identify the actual full roster')
    if value.get('endpoint') != ENDPOINT or value.get('official_test_accessed') is not False:
        raise ValueError('Bound evaluator endpoint/test contract differs')
    effective = value.get('effective_kwargs', {})
    required = ('imgsz', 'batch', 'workers', 'half', 'conf', 'iou', 'max_det', 'agnostic_nms')
    if any(key not in effective for key in required):
        raise ValueError('Bound evaluator lacks resolved precision/NMS/size/batch settings')
    if not value.get('evaluator_identity'):
        raise ValueError('Bound evaluator implementation identity is missing')
    # Per-run checkpoint/seed are checked separately and must not make otherwise
    # matched model endpoints appear to be different evaluation protocols.
    return {key: copy.deepcopy(value[key]) for key in
            ('schema', 'evaluator_identity', 'effective_kwargs', 'expected_val_images', 'endpoint', 'official_test_accessed')}


def _evaluation_evidence(path, spec, base_dir, train_cfg, metric, roster):
    receipt = json.loads((path / 'eval_evidence/run_receipt.json').read_text(encoding='utf-8-sig'))
    eval_cfg, sources = _bound_configs(path, receipt, 'eval_evidence')
    actual_sources = receipt.get('source_snapshots', {}).get('trainer', [])
    if not actual_sources:
        raise ValueError('Evaluator source snapshots are missing')
    evaluator_source_contents = [(path / 'eval_evidence' / relative).read_text(encoding='utf-8-sig')
                                 for relative in sorted(actual_sources)]
    contracts = [value for _, value in sources if value.get('schema') == 'rgbir-evaluation-contract-v1']
    if contracts:
        if any(value != contracts[0] for value in contracts[1:]):
            raise ValueError('Conflicting bound evaluator contracts')
        value = contracts[0]
    elif spec.get('evaluation_compatibility_receipt'):
        bridge_path = Path(spec['evaluation_compatibility_receipt'])
        bridge_path = bridge_path if bridge_path.is_absolute() else Path(base_dir) / bridge_path
        bridge = json.loads(bridge_path.read_text(encoding='utf-8-sig'))
        if bridge.get('status') != 'ACCEPTED' or not bridge.get('reviewer'):
            raise ValueError('Legacy evaluator compatibility is not independently accepted')
        entries = [row for row in bridge.get('entries', [])
                   if row.get('checkpoint') == metric.get('checkpoint') and row.get('seed') == spec['seed']]
        if (len(entries) != 1 or entries[0].get('bound_training_config') != train_cfg
                or entries[0].get('bound_evaluation_config') != eval_cfg):
            raise ValueError('Legacy evaluator bridge differs from actual run/config')
        entry = entries[0]
        copies = entry.get('evaluation_source_copies', {})
        if not actual_sources or set(copies) != set(actual_sources):
            raise ValueError('Legacy evaluator bridge does not cover actual executed source set')
        for relative in actual_sources:
            source = path / 'eval_evidence' / relative
            accepted = Path(copies[relative])
            accepted = accepted if accepted.is_absolute() else bridge_path.parent / accepted
            if accepted.resolve() == source.resolve() or accepted.read_bytes() != source.read_bytes():
                raise ValueError('Legacy evaluator source differs from reviewed independent copy')
        value = entry.get('evaluation_contract')
        canonical_copies = bridge.get('canonical_evaluator_source_copies', [])
        if not canonical_copies:
            raise ValueError('Legacy bridge must bind the reviewed canonical evaluation implementation')
        evaluator_source_contents = []
        for name in canonical_copies:
            accepted = Path(name)
            accepted = accepted if accepted.is_absolute() else bridge_path.parent / accepted
            evaluator_source_contents.append(accepted.read_text(encoding='utf-8-sig'))
    else:
        raise ValueError('No receipt-bound evaluation contract or accepted legacy bridge')
    canonical = _validate_evaluation_contract(value, roster, metric)
    canonical['executed_or_accepted_equivalent_source_contents'] = evaluator_source_contents
    for key in ('imgsz', 'batch', 'workers', 'half', 'conf', 'iou', 'max_det', 'agnostic_nms'):
        if key in eval_cfg and canonical['effective_kwargs'][key] != eval_cfg[key]:
            raise ValueError('Bound evaluator effective setting differs from actual evaluation config: ' + key)
    frozen_counts = {'drone': 1469, 'dronevehicle': 1469, 'llvip': 2406}
    expected = frozen_counts.get(str(receipt.get('dataset', '')).lower())
    if expected is not None and canonical['expected_val_images'] != expected:
        raise ValueError('Development count differs from the authorized dataset protocol')
    return canonical


def _intervention_identity(cfg, arm, json_sources):
    is_class = arm in ('C0', 'C1', 'C1_y')
    coefficient = cfg.get('classification_coefficient') if is_class else cfg.get('localization_coefficient')
    if arm == 'C0' and coefficient is None:
        coefficient = cfg.get('kd_weight')
    if arm == 'N':
        coefficient = 0.0
    elif (not cfg.get('teacher') or not cfg.get('reference') or type(coefficient) not in (int, float)
          or not math.isfinite(coefficient) or coefficient <= 0):
        raise ValueError('KD intervention lacks bound teacher/reference/positive coefficient')
    payload = copy.deepcopy(cfg.get('classification', {})) if is_class else copy.deepcopy(cfg.get('localization', {}))
    for allowed in ('off_target_weight', 'enabled', 'mode', 'target_kind'):
        payload.pop(allowed, None)
    if arm in ('C1', 'C1_y'):
        # These are fixed in the implementation, including when omitted from
        # the legacy-shaped effective config. Keep explicit changed values.
        payload.setdefault('temperature', 2.0)
        payload.setdefault('raw_teacher_clip', 16.0)
    selection = copy.deepcopy(cfg.get('evidence', {})) if is_class else copy.deepcopy(cfg.get('localization', {}))
    for allowed in ('mode', 'target_kind'):
        selection.pop(allowed, None)
    geometry = [value for _, value in json_sources if value.get('mode') == 'verified_identity_grid']
    if len(geometry) > 1 and any(value != geometry[0] for value in geometry[1:]):
        raise ValueError('Conflicting bound geometry contracts')
    return {'teacher': cfg.get('teacher'), 'reference': cfg.get('reference'),
            'privileged_data': {key: cfg.get('paths', {}).get(key) for key in ('privileged_data_yaml', 'paired_train_mapping')},
            'selection': selection,
            'selection_algorithm': 'exact_oev1' if is_class else 'strict07_single_anchor',
            'coefficient': coefficient, 'payload_common': payload,
            'geometry_content': geometry[0] if geometry else None}


def load_endpoint(spec, base_dir):
    """Read real original files and their bound source/metric snapshots via v2."""
    errors = identity_errors(spec.get('arm'), spec.get('source', 'paired'), spec.get('seed'))
    if errors:
        raise ValueError('; '.join(errors))
    record = _legacy_loader().load_endpoint(spec, base_dir)
    record['source'] = spec.get('source', 'paired')
    record['receipt_validation'] = 'passed' if record['status'] == 'complete' else 'failed'
    record['official_test_accessed'] = False if record['status'] == 'complete' else None
    record['evaluator_contract'] = None
    record['expected_val_images'] = None
    record['evaluation_contract_validation'] = 'failed'
    record['intervention_validation'] = 'failed'
    path = Path(record['path'])
    if record['status'] == 'complete':
        raw = json.loads((path / 'evaluation_val.json').read_text(encoding='utf-8-sig'))
        aliases = {'N': {'weight0', 'N'}, 'C0': {'paired', 'c', 'C0', 'c_shuffled', 'c_same_modal'},
                   'C1': {'C1'}, 'C1_y': {'C1_y'}, 'L1': {'l', 'L1'}, 'L_GT': {'L_GT'}}
        inferred_source = {'c_shuffled': 'shuffled', 'c_same_modal': 'same_modal'}.get(raw.get('arm'), 'paired')
        actual_source = raw.get('source', inferred_source)
        if raw.get('arm') not in aliases[spec['arm']] or actual_source != record['source']:
            record['status'], record['receipt_validation'] = 'invalid', 'failed'
            record['issues'].append('Manifest attempted to relabel the actual method arm/source')
            return record
        try:
            train_receipt = json.loads((path / 'run_evidence/run_receipt.json').read_text(encoding='utf-8-sig'))
            cfg, bound_json = _bound_configs(path, train_receipt, 'run_evidence')
            if type(cfg.get('expected_nc')) is int and cfg['expected_nc'] > 0:
                record['expected_class_ids'] = [str(i) for i in range(cfg['expected_nc'])]
            record['intervention_identity'] = _intervention_identity(cfg, spec['arm'], bound_json)
            record['intervention_validation'] = 'passed'
            record['evaluator_contract'] = _evaluation_evidence(path, spec, base_dir, cfg, raw, record['roster'])
            record['expected_val_images'] = record['evaluator_contract']['expected_val_images']
            record['evaluation_contract_validation'] = 'passed'
        except (ValueError, KeyError, OSError, TypeError) as error:
            record['issues'].append(str(error))
        record['per_class_percent'] = {str(r['class_id']): to_percent(r['mAP50_95'], raw['metric_units'])
                                      for r in raw.get('per_class', [])}
        # Fixed-threshold diagnostics are separate from YOLO's reported best-F1
        # precision/recall. Never infer this artifact from those metric columns.
        if spec.get('error_analysis_file'):
            target = Path(spec['error_analysis_file'])
            target = target if target.is_absolute() else Path(base_dir) / target
            data = json.loads(target.read_text(encoding='utf-8-sig'))
            if data.get('checkpoint') != raw.get('checkpoint') or data.get('seed') != spec['seed']:
                record['issues'].append('Error-analysis artifact checkpoint/seed mismatch')
            elif data.get('roster') != record['roster']:
                record['issues'].append('Error-analysis population mismatch')
            else:
                record['error_analysis'] = data
                record['localization_diagnostics_consistent'] = data.get('localization_diagnostics_consistent')
    return record


def verify_analyzer_acceptance(path):
    """Accept only an external review receipt with directly equal source copies."""
    if path is None:
        return False
    path = Path(path)
    receipt = json.loads(path.read_text(encoding='utf-8-sig'))
    if receipt.get('status') != 'ACCEPTED' or not receipt.get('reviewer'):
        return False
    dependencies = {'analyze_independent.py': Path(__file__),
                    'protocol.py': Path(__file__).parent / 'protocol.py',
                    'legacy_analyze_results.py': _legacy_path()}
    for name, actual in dependencies.items():
        copy_name = receipt.get('source_snapshots', {}).get(name)
        if not copy_name:
            return False
        copy_path = Path(copy_name)
        copy_path = copy_path if copy_path.is_absolute() else path.parent / copy_path
        if not copy_path.is_file() or copy_path.read_bytes() != actual.read_bytes():
            return False
    return True


def analyze_manifest(manifest, base_dir, accepted_receipt=None):
    records = [load_endpoint(spec, base_dir) for spec in manifest['runs']]
    return analyze_records(records, manifest.get('implementation_checks_passed', False),
                           verify_analyzer_acceptance(accepted_receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accepted-receipt', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    result = analyze_manifest(manifest, args.manifest.parent, args.accepted_receipt)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
