"""Explicit CPU-only post-hoc class adapter; frozen training/analyzer stay intact.

The accepted aggregate bridge is an input, not acceptance of this adapter.
Outputs remain non-authorizing until separately reviewed. No old JSON is edited.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def class_values(metric, expected_ids, to_percent):
    """All classes exactly once; fractions and macro AP checked independently."""
    rows = metric.get('per_class')
    require(isinstance(rows, list) and bool(rows), 'Missing post-hoc class metrics')
    require(all(type(r.get('class_id')) is int for r in rows), 'Invalid class ID')
    ids = [str(r['class_id']) for r in rows]
    require(len(ids) == len(set(ids)) and set(ids) == set(expected_ids), 'Class inventory differs')
    result = {}
    for key in ('mAP50_95', 'AP50', 'AP75'):
        result[key] = {str(r['class_id']): to_percent(r[key], metric['metric_units']) for r in rows}
        average = statistics.mean(result[key].values())
        require(math.isclose(average, to_percent(metric[key], metric['metric_units']),
                             rel_tol=0, abs_tol=1e-9), 'Class macro average differs: '+key)
    return result


def validate_identity(record, old_metric, entry, metric, receipt, complete, contract, roster):
    expected = {'seed': record['seed'], 'arm': entry['actual_source_arm'],
                'checkpoint': old_metric['checkpoint'], 'endpoint': old_metric['endpoint'],
                'source': record['source'], 'method_id': old_metric['method_id']}
    require(metric.get('status') == 'completed' and complete.get('status') == 'completed',
            'Post-hoc evaluation incomplete')
    require(metric.get('normalized_method_arm') == record['arm'] and
            complete.get('normalized_method_arm') == record['arm'], 'Method relabeling')
    require(metric.get('evaluation_kind') == 'posthoc_legacy_checkpoint_diagnostics', 'Not a post-hoc evaluation')
    require(metric.get('split') == 'val' and metric.get('dataset') == record['dataset'], 'Dataset/split differs')
    require(metric.get('official_test_accessed') is False and complete.get('official_test_accessed') is False,
            'Test exposure must be false')
    for key, value in expected.items():
        # The established receipt schema stores seed at top level, not inputs.
        observed = receipt.get(key) if key == 'seed' else receipt.get('inputs', {}).get(key)
        require(metric.get(key) == value and observed == value,
                'Observed identity differs: '+key)
    for key in ('checkpoint', 'seed', 'arm'):
        require(complete.get(key) == expected[key], 'Completion differs: '+key)
    require(receipt.get('terminal_status') == 'COMPLETED' and receipt.get('run_kind') == 'eval'
            and receipt.get('data_role') == 'development_val'
            and receipt.get('seed') == record['seed'] and receipt.get('dataset') == record['dataset'],
            'Invalid execution receipt')
    require(complete.get('new_training_receipt_created') is False and complete.get('old_results_modified') is False,
            'Old evidence was replaced')
    require(contract == entry['evaluation_contract'] and roster == record['roster']
            and contract.get('roster') == roster and contract.get('expected_val_images') == len(roster)
            and contract.get('observed_images') == len(roster)
            and len(contract.get('actual_loader_roster', [])) == len(roster)
            and set(contract['actual_loader_roster']) == set(roster), 'Observed population differs')


def load_with_posthoc(spec, base_dir, analyzer):
    record = analyzer.load_endpoint(spec, base_dir)
    if spec.get('posthoc_class_metrics') is not True:
        return record
    require(analyzer.validate_endpoint(record)['valid'], 'Original endpoint did not pass the frozen analyzer')
    require(not record.get('per_class_percent'), 'Original already has classes; no silent override')
    old = Path(record['path'])
    old_metric = read(old/'evaluation_val.json')
    bridge_path = Path(spec['evaluation_compatibility_receipt'])
    if not bridge_path.is_absolute():
        bridge_path = Path(base_dir)/bridge_path
    bridge = read(bridge_path)
    require(bridge.get('schema') == 'rgbir-observed-legacy-evaluation-bridge-v1'
            and bridge.get('status') == 'ACCEPTED' and bridge.get('reviewer'), 'Observed bridge not accepted')
    entries = [e for e in bridge['entries'] if e['checkpoint'] == old_metric['checkpoint']
               and e['seed'] == record['seed'] and e['actual_source_arm'] == old_metric['arm']]
    require(len(entries) == 1, 'Require one matching observed entry')
    entry = entries[0]
    obs = entry['observed_reevaluation']

    def bound(relative):
        path = bridge_path.parent/relative
        mappings = [m for m in bridge['source_mappings'] if m['independent_copy'] == relative]
        require(bool(mappings), 'Missing accepted evidence mapping: '+relative)
        payload = path.read_bytes()
        for mapping in mappings:
            require(Path(mapping['local_source']).read_bytes() == payload,
                    'Observed artifact changed since independent copy: '+relative)
        return path

    metric_path = bound(obs['metric_copy'])
    metric = read(metric_path)
    receipt = read(bound(obs['receipt_copy']))
    complete = read(bound(obs['completion_copy']))
    contract = read(bound(obs['contract_copy']))
    roster_path = metric_path.parent/'evaluation_val_roster.txt'
    bound(roster_path.relative_to(bridge_path.parent).as_posix())
    roster = roster_path.read_text(encoding='utf-8-sig').splitlines()
    validate_identity(record, old_metric, entry, metric, receipt, complete, contract, roster)
    aliases = obs['metric_snapshot_independent_copies']
    require(any(r in receipt.get('metric_snapshots', []) and bound(target).read_bytes() == metric_path.read_bytes()
                for r, target in aliases.items()), 'Class metrics not bound by actual evaluation receipt')
    actual_sources = receipt['source_snapshots']['trainer']
    require(set(actual_sources) == set(obs['source_copies']), 'Observed evaluator source inventory differs')
    for relative in actual_sources:
        bound(obs['source_copies'][relative])
    canonical = bridge['canonical_evaluator_source_copies']
    common_order = [r['expected_formal_receipt_relative'] for r in bridge['canonical_source_order']]
    require(actual_sources[:len(common_order)] == common_order, 'Observed source order differs')
    for original, accepted in zip(common_order, canonical):
        require(bound(obs['source_copies'][original]).read_bytes() == bound(accepted).read_bytes(),
                'Observed canonical evaluator differs')
    for key in analyzer.METRICS:
        require(analyzer.to_percent(metric[key], metric['metric_units']) == record['metrics_percent'][key],
                'Post-hoc aggregate differs from original: '+key)
    values = class_values(metric, record['expected_class_ids'], analyzer.to_percent)
    # Enrichment is explicit and only in memory; metric/checkpoint/seed stay original.
    enriched = copy.deepcopy(record)
    enriched['per_class_percent'] = values['mAP50_95']
    enriched['posthoc_class_metrics_percent'] = values
    enriched['per_class_provenance'] = {
        'kind': 'separately_executed_posthoc_legacy_checkpoint_diagnostics',
        'metric': str(metric_path), 'receipt': str(bridge_path.parent/obs['receipt_copy']),
        'observed_bridge': str(bridge_path), 'checkpoint': metric['checkpoint'], 'seed': metric['seed'],
        'old_per_class_recorded': False, 'old_files_modified': False,
        'adapter_review_required': True}
    return enriched


def module(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('frozen_independent_analyzer', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--analyzer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    analyzer = module(args.analyzer)
    records = [load_with_posthoc(spec, args.manifest.parent, analyzer) for spec in manifest['runs']]
    output = analyzer.analyze_records(records, implementation_checks_passed=False, analyzer_accepted=False)
    output['posthoc_adapter_status'] = 'DRAFT_AWAITING_INDEPENDENT_REVIEW'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'records': len(records), 'with_posthoc_classes': sum(bool(r.get('per_class_provenance')) for r in records),
                      'status': output['posthoc_adapter_status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
