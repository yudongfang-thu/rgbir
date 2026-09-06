"""CPU-only raw endpoint/context check; historical differences are not the estimand."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = HERE / 'oev1_snapshot/raw_runs/full_paired_s42_attempt1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

evaluation = read(RUN / 'evaluation_val.json')
completion = read(RUN / 'completion_receipt.json')
summary = read(HERE / 'oev1_snapshot/oev1_endpoints/summary.json')
eval_receipt = read(RUN / 'eval_evidence/run_receipt.json')
train_receipt = read(RUN / 'run_evidence/run_receipt.json')
checks = {
    'evaluation_identity': evaluation['seed'] == 42 and evaluation['arm'] == 'paired',
    'fixed_endpoint': evaluation['endpoint'] == 'fixed_budget_last_ema' and evaluation['split'] == 'val',
    'completion_e200': completion['last_epoch'] == completion['epochs_configured'] == 200,
    'same_checkpoint': completion['checkpoint'] == evaluation['checkpoint'],
    'eval_completed': eval_receipt['terminal_status'] == 'COMPLETED',
    'train_completed': train_receipt['terminal_status'] == 'COMPLETED',
}
for label, receipt, raw in [('eval', eval_receipt, evaluation), ('train', train_receipt, completion)]:
    receipt_root = RUN / ('eval_evidence' if label == 'eval' else 'run_evidence')
    checks[label + '_metric_copy_equal'] = any(read(receipt_root / p) == raw for p in receipt['metric_snapshots'])
    checks[label + '_referenced_source_files_exist'] = all(
        (receipt_root / p).is_file() for entries in receipt['source_snapshots'].values() for p in entries)
roster = RUN / 'eval_evidence' / eval_receipt['source_snapshots']['split_roster'][0]
ids = roster.read_text(encoding='utf-8').splitlines()
checks['val_roster_1469_unique'] = len(ids) == len(set(ids)) == 1469
cell = next(c for c in summary['cells'] if c['seed'] == 42 and c['arm'] == 'paired')
checks['collector_endpoint_matches'] = cell['status'] == 'completed' and all(
    cell['metrics'][m] == evaluation[m] for m in cell['metrics'])
assert all(checks.values()), checks

comparisons = {}
for label, name in [('historical_native', 'N_llvip_seed42_metrics_record.json'),
                    ('historical_cmdistill', 'L_dronevehicle_seed42_metrics_record.json')]:
    source = ROOT / '09_外部审计_rgbir/02_raw_results_dronevehicle' / name
    record = read(source)
    assert record['dataset'] == 'dronevehicle' and record['seed'] == 42 and record['split'] == 'val'
    comparisons[label] = {
        'source': str(source),
        'record_identity': {k: record[k] for k in ('dataset', 'seed', 'arm', 'checkpoint', 'data_yaml', 'split')},
        'historical_metrics_pct': {m: 100*v for m, v in record['metrics'].items()},
        'oev1_minus_historical_pp': {m: 100*(evaluation[m]-v) for m, v in record['metrics'].items()},
        'interpretation': 'Descriptive historical context only; not same-code paired-minus-weight0 estimand.',
    }
with (RUN / 'results.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.reader(f))
report = {
    'schema': 'oev1-first-endpoint-context-review-v1',
    'captured_at_utc': summary['captured_at_utc'],
    'raw_checks': checks,
    'complete_endpoints': summary['complete_endpoints'],
    'complete_seed_pairs': summary['complete_seed_pairs'],
    'three_seed_summary': summary['three_seed_summary'],
    'new_endpoint_percent': {m: 100*evaluation[m] for m in cell['metrics']},
    'comparisons': comparisons,
    'csv_format': {'header_columns': len(rows[0]), 'penultimate_columns': len(rows[-2]),
                   'last_columns': len(rows[-1]), 'last_epoch': rows[-1][0],
                   'interpretation': 'Final-epoch validate() returns empty metrics; dynamic writer omits seven placeholders. Independent endpoint JSON is unaffected.'},
    'optimizer_updates': {k: completion[k] for k in ('optimizer_updates', 'optimizer_update_attempts', 'amp_skipped_updates', 'ema_updates')},
    'limitations': ['Single seed development endpoint.', 'Same-code weight0 seed42 endpoint pending.',
                    'No four-arm attribution or accepted analyzer claim.'],
}
(HERE / 'oev1_first_endpoint_context_check.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'all_checks_pass': all(checks.values()), 'complete_endpoints': report['complete_endpoints'],
    'mAP50_95_percent': report['new_endpoint_percent']['mAP50_95'],
    'historical_native_difference_pp': comparisons['historical_native']['oev1_minus_historical_pp']['mAP50_95'],
    'historical_cmdistill_difference_pp': comparisons['historical_cmdistill']['oev1_minus_historical_pp']['mAP50_95'],
    'csv_format': report['csv_format']}, ensure_ascii=False))
