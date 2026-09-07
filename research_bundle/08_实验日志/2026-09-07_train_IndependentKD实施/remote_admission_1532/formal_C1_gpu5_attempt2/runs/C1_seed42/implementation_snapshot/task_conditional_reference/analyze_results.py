"""Receipt-backed development endpoints and frozen, paired seed comparisons.

Input JSON: {"runs": [{"arm": "C", "seed": 42, "path": "run_dir",
"protocol_id": "frozen_recipe_id"}], "implementation_checks_passed": false}.
Paths are relative to the manifest. Metrics remain fractions in original files;
all reported metrics and differences are percentages / percentage points.
The analyzer never accepts CSV training metrics as independent endpoints.
"""
import argparse
import json
import math
import statistics
from pathlib import Path
import yaml

SEEDS = (0, 42, 123)
METRICS = ('mAP50_95', 'AP50', 'AP75', 'precision', 'recall')
COMPARISONS = (('C', 'N'), ('R', 'C'), ('CL', 'C'), ('CL', 'L'),
               ('CL', 'CGT'), ('CL', 'CL_random'), ('CL', 'CL_shuffled'),
               ('CL', 'CL_same_modal'), ('C', 'C_shuffled'), ('C', 'C_same_modal'))
RECIPE_FIELDS = ('model', 'imgsz', 'epochs', 'batch', 'nbs', 'workers', 'optimizer',
                 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
                 'warmup_momentum', 'warmup_bias_lr', 'cos_lr', 'close_mosaic',
                 'patience', 'amp', 'deterministic', 'torch_version', 'ultralytics_version')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def to_percent(value, units):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Metric must be a finite number')
    if units == 'fraction_0_to_1':
        if not 0 <= value <= 1:
            raise ValueError('Fraction metric is outside [0, 1]')
        return 100.0 * value
    if units == 'percent_0_to_100':
        if not 0 <= value <= 100:
            raise ValueError('Percent metric is outside [0, 100]')
        return float(value)
    raise ValueError(f'Explicit metric units required, received {units!r}')


def summarize(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) >= 2 else None,
            'ddof': 1}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _recipe(config):
    """Compare the common training recipe, not the intervention or config seed.

    Teacher, reference, KD coefficients, method ID and the geometry contract are
    intentionally excluded: those define the experimental intervention. Seeds
    come from actual training/evaluation receipts, not a copied YAML default.
    """
    missing = [name for name in RECIPE_FIELDS if name not in config]
    _require(not missing, 'Bound protocol is missing recipe fields: ' + ', '.join(missing))
    _require(isinstance(config.get('augmentation'), dict), 'Bound augmentation recipe is missing')
    _require(bool(config.get('paths', {}).get('student_data_yaml')), 'Bound student data identity is missing')
    recipe = {name: config[name] for name in RECIPE_FIELDS}
    recipe['augmentation'] = dict(config['augmentation'])
    recipe['student_data_yaml'] = config['paths']['student_data_yaml']
    return recipe


def bound_recipe(path, train_receipt):
    """Read the receipt-bound protocol and cross-check the launch protocol copy."""
    candidates = []
    for relative in train_receipt.get('source_snapshots', {}).get('config', []):
        source = path / 'run_evidence' / relative
        _require(source.is_file(), 'Receipt-bound training config is missing: ' + relative)
        if source.suffix.lower() not in ('.yaml', '.yml'):
            continue
        config = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
        if isinstance(config, dict) and 'model' in config and 'paths' in config:
            candidates.append((relative, _recipe(config)))
    _require(bool(candidates), 'Receipt contains no bound experiment protocol')
    recipe = candidates[0][1]
    _require(all(value == recipe for _, value in candidates), 'Conflicting bound training protocols')
    protocol_copy = path / 'protocol_config.yaml'
    if protocol_copy.is_file():
        local_recipe = _recipe(yaml.safe_load(protocol_copy.read_text(encoding='utf-8-sig')))
        different = [name for name in recipe if local_recipe[name] != recipe[name]]
        _require(not different, 'Launch protocol differs from bound recipe: ' + ', '.join(different))
        source_status = 'launch_protocol_cross_checked_against_train_receipt'
    else:
        # Actual legacy OEv1 endpoints predate the standalone protocol copy;
        # their source is the config saved by the completed training receipt.
        source_status = 'legacy_receipt_bound_protocol_only'
    return recipe, {'status': source_status, 'bound_paths': [relative for relative, _ in candidates]}


def load_endpoint(spec, base_dir):
    path = Path(spec['path'])
    path = path if path.is_absolute() else Path(base_dir) / path
    result = {'arm': spec['arm'], 'seed': spec['seed'], 'path': str(path),
              'protocol_id': spec['protocol_id'], 'status': 'incomplete', 'issues': [],
              'metrics_percent': None}
    required = ('completion_receipt.json', 'evaluation_val.json',
                'run_evidence/run_receipt.json', 'eval_evidence/run_receipt.json',
                'evaluation_val_roster.txt')
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        result['issues'] = ['Missing ' + name for name in missing]
        return result
    try:
        complete = _read(path / required[0])
        metric = _read(path / required[1])
        train = _read(path / required[2])
        evaluation = _read(path / required[3])
        _require(complete.get('status') == 'training_completed', 'Full training did not complete')
        _require(complete.get('last_epoch') == complete.get('epochs_configured') and complete.get('last_epoch', 0) > 0,
                 'Frozen training epoch budget not completed')
        _require(complete.get('official_test_accessed') is False and metric.get('official_test_accessed') is False,
                 'Missing or non-false official-test-access flag')
        _require(metric.get('split') == 'val' and metric.get('endpoint') == 'fixed_budget_last_ema',
                 'Expected independent development-val fixed last/EMA endpoint')
        _require(metric.get('checkpoint') == complete.get('checkpoint') and str(metric.get('checkpoint', '')).endswith('/weights/last.pt'),
                 'Metric and training checkpoint identity differ')
        for receipt, kind, role, expected_metric, folder in (
                (train, 'train', 'development_train', complete, 'run_evidence'),
                (evaluation, 'eval', 'development_val', metric, 'eval_evidence')):
            _require(receipt.get('terminal_status') == 'COMPLETED' and receipt.get('run_kind') == kind,
                     f'{kind} receipt is not a completed {kind} receipt')
            _require(receipt.get('data_role') == role, f'{kind} data role differs')
            _require(receipt.get('seed') == spec['seed'], f'{kind} receipt seed differs')
            _require(receipt.get('inputs', {}).get('arm') == metric.get('arm'), f'{kind} receipt arm differs')
            _require(receipt.get('inputs', {}).get('method_id') == metric.get('method_id'), f'{kind} method identity differs')
            _require(bool(receipt.get('resources', {}).get('lease_id')), f'{kind} bound lease missing')
            copies = receipt.get('metric_snapshots', [])
            _require(bool(copies), f'{kind} metric snapshot is missing')
            _require(any((path / folder / name).is_file() and _read(path / folder / name) == expected_metric for name in copies),
                     f'{kind} bound metric snapshot does not match original')
        _require(metric.get('seed') == complete.get('seed') == spec['seed'], 'Seed mismatch')
        _require(metric.get('arm') == complete.get('arm'), 'Completion and metric arms differ')
        if 'source_arm' in spec:
            _require(metric.get('arm') == spec['source_arm'], 'Manifest source arm differs from receipt')
        _require(train.get('dataset') == evaluation.get('dataset'), 'Training and evaluation dataset differ')
        roster = (path / 'evaluation_val_roster.txt').read_text(encoding='utf-8-sig').splitlines()
        _require(bool(roster) and len(roster) == len(set(roster)), 'Empty or duplicate evaluation roster')
        bound_rosters = evaluation.get('source_snapshots', {}).get('split_roster', [])
        _require(bool(bound_rosters) and any((path / 'eval_evidence' / name).is_file() and
                 (path / 'eval_evidence' / name).read_text(encoding='utf-8-sig').splitlines() == roster for name in bound_rosters),
                 'Bound evaluation roster differs or is missing')
        for names in evaluation.get('source_snapshots', {}).values():
            _require(all((path / 'eval_evidence' / name).is_file() for name in names), 'Evaluation source/config snapshot is missing')
        recipe, recipe_evidence = bound_recipe(path, train)
        result.update(status='complete', metrics_percent={name: to_percent(metric[name], metric.get('metric_units')) for name in METRICS},
                      endpoint=metric['endpoint'], dataset=evaluation['dataset'], roster=roster,
                      original_units=metric['metric_units'], source_arm=metric['arm'], method_id=metric['method_id'],
                      frozen_recipe=recipe, recipe_evidence=recipe_evidence)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(status='invalid', issues=[str(error)])
    return result


def paired_comparison(records, treatment, control):
    pairs = []
    issues = []
    for seed in SEEDS:
        arm_rows = [[r for r in records if r['arm'] == arm and r['seed'] == seed and r['status'] == 'complete']
                    for arm in (treatment, control)]
        if not all(arm_rows):
            continue
        a, b = arm_rows[0][0], arm_rows[1][0]
        if any(a.get(field) != b.get(field) for field in ('protocol_id', 'endpoint', 'dataset', 'roster', 'frozen_recipe')):
            differences = [name for name in a.get('frozen_recipe', {})
                           if a['frozen_recipe'][name] != b.get('frozen_recipe', {}).get(name)]
            issues.append(f'seed {seed}: protocol, endpoint, dataset, roster or bound recipe differs; recipe fields={differences}')
            continue
        pairs.append({'seed': seed, 'deltas_pp': {key: a['metrics_percent'][key] - b['metrics_percent'][key] for key in METRICS}})
    complete = len(pairs) == len(SEEDS) and not issues
    stats = {key: dict(summarize([p['deltas_pp'][key] for p in pairs]),
                       positive_seeds=sum(p['deltas_pp'][key] > 0 for p in pairs),
                       negative_seeds=sum(p['deltas_pp'][key] < 0 for p in pairs)) for key in METRICS}
    return {'treatment': treatment, 'control': control, 'pairs': pairs, 'stats_pp': stats,
            'complete_three_seed_pairing': complete, 'issues': issues,
            'evidence_status': 'three_seed_development_descriptive' if complete else 'incomplete_descriptive_only',
            'claim_supported': 'not_automatically_assessed'}


def pilot_decision(records, implementation_checks_passed=False):
    comparisons = [paired_comparison(records, 'CL', arm) for arm in ('C', 'CGT')]
    deltas = [next((p['deltas_pp']['mAP50_95'] for p in item['pairs'] if p['seed'] == 42), None) for item in comparisons]
    ready = all(delta is not None for delta in deltas)
    eligible = ready and implementation_checks_passed is True and deltas[0] >= 0.3 - 1e-10 and deltas[1] > 0.0
    return {'seed': 42, 'metric': 'mAP50_95', 'threshold_cl_minus_c_pp': 0.3,
            'threshold_cl_minus_cgt_pp_strict': 0.0, 'cl_minus_c_pp': deltas[0], 'cl_minus_cgt_pp': deltas[1],
            'endpoints_ready': ready, 'implementation_checks_passed': implementation_checks_passed is True,
            'auto_expansion_eligible': eligible,
            'decision': 'expand' if eligible else ('hold_new_L_expansion' if ready else 'await_complete_endpoints'),
            'stop_running_experiments': False, 'scientific_claim': 'single_seed_budget_decision_only'}


def analyze_manifest(manifest, base_dir):
    specs = manifest['runs']
    identities = [(r['arm'], r['seed']) for r in specs]
    _require(len(identities) == len(set(identities)), 'Duplicate arm/seed; select attempt independently of outcome before analysis')
    _require(all(r['seed'] in SEEDS for r in specs), 'Only frozen seeds 0, 42, 123 are allowed')
    records = [load_endpoint(spec, base_dir) for spec in specs]
    summaries = {}
    for arm in sorted({r['arm'] for r in records}):
        valid = [r for r in records if r['arm'] == arm and r['status'] == 'complete']
        homogeneous = not valid or all(all(r.get(field) == valid[0].get(field) for field in
                                      ('protocol_id', 'endpoint', 'dataset', 'roster', 'frozen_recipe')) for r in valid)
        summaries[arm] = {'completed_seeds': [s for s in SEEDS if any(r['seed'] == s for r in valid)],
                          'metrics_percent': {key: summarize([r['metrics_percent'][key] for r in valid]) for key in METRICS} if homogeneous else None,
                          'issues': [] if homogeneous else ['Mixed protocol or evaluation population; arm summary withheld'],
                          'complete_three_seeds': len(valid) == 3 and homogeneous}
    comparisons = {f'{a}_minus_{b}': paired_comparison(records, a, b) for a, b in COMPARISONS
                   if any(r['arm'] == a for r in records) and any(r['arm'] == b for r in records)}
    pilot = pilot_decision(records, manifest.get('implementation_checks_passed', False))
    for record in records:
        if 'roster' in record:
            record['roster_count'] = len(record.pop('roster'))
    return {'schema': 'rgbir-task-conditional-results-v2', 'seed_order': list(SEEDS),
            'report_units': 'percentage_and_percentage_points', 'records': records, 'arms': summaries,
            'comparisons': comparisons, 'pilot': pilot,
            'claim_policy': 'Development evidence only; three seeds and valid endpoints do not replace independent analyzer review, four-arm attribution, or test exposure audit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze_manifest(_read(args.manifest), args.manifest.parent)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError('Choose a new output; do not overwrite original analysis evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    else:
        print(serialized, end='')


if __name__ == '__main__':
    main()
