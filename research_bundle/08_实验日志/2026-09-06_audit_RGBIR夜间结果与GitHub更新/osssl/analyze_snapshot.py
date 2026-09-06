"""Recalculate frozen-epoch CSV contrasts; no inference, checkpoint loading, or hashes."""
import csv
import json
import math
import pathlib
from decimal import Decimal
import yaml

OUT = pathlib.Path(__file__).resolve().parent
RAW = OUT / 'raw/runs'
SNAPSHOT = json.loads((OUT / 'snapshot.json').read_text(encoding='utf-8'))
METRICS = ('metrics/mAP50(B)', 'metrics/mAP50-95(B)')


def last_csv(path):
    rows = list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))
    assert rows, path
    assert all(None not in row for row in rows), path
    return rows, rows[-1]


def difference(left, right):
    return {key: {
        'left_pp': float(Decimal(left[key]) * 100),
        'right_pp': float(Decimal(right[key]) * 100),
        'delta_pp': float((Decimal(left[key]) - Decimal(right[key])) * 100),
    } for key in METRICS}


def args_diff(left_dir, right_dir):
    left = yaml.safe_load((left_dir / 'args.yaml').read_text(encoding='utf-8'))
    right = yaml.safe_load((right_dir / 'args.yaml').read_text(encoding='utf-8'))
    return {key: {'left': left.get(key), 'right': right.get(key)}
            for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)}


report = {
    'timestamp': SNAPSHOT['timestamp'], 'finished_at': SNAPSHOT['finished_at'],
    'metric_identity': 'epoch 200 training CSV validation; not independent eval_rgbt_detector last/EMA endpoint',
    'test_used': False, 'hashes_generated': False,
    'initialization_evidence_reused_from': '2026-09-06T17:31:43.773744+08:00',
    'initialization_evidence_file': 'initialization_inspection_173143_reused.json',
    'progress': [], 'contrasts_vs_historical_native': [], 'internal_ssl_contrasts': [],
    'collection_integrity': {
        'source_files': len(SNAPSHOT['inventory']),
        'changed_during_read': sum(row['changed_during_read'] for row in SNAPSHOT['inventory']),
        'partial_tail_files': sum(row['partial_tail'] for row in SNAPSHOT['inventory']),
        'snapshot_atomic': False,
    },
}
completed = {}
for arm in ('paired', 'sar_only', 'shuffled'):
    for seed in (0, 42, 123):
        run = RAW / 'osssl_ir_20260906' / f'{arm}_rgb_s{seed}_e200'
        progress = {'arm': arm, 'seed': seed, 'status': 'queued', 'epoch_completed': 0,
                    'has_independent_metrics_record': (run / 'metrics_record.json').exists()}
        if (run / 'results.csv').exists():
            rows, last = last_csv(run / 'results.csv')
            progress.update(status='running', epoch_completed=int(last['epoch']), csv_rows=len(rows))
            assert all(math.isfinite(float(last[key])) for key in METRICS)
            if (run / 'completion_receipt.json').exists():
                receipt = json.loads((run / 'completion_receipt.json').read_text(encoding='utf-8'))
                expected_run = f'/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/osssl_ir_20260906/{run.name}'
                checks = {
                    'exact_200_rows': len(rows) == 200,
                    'contiguous_1_to_200': [int(row['epoch']) for row in rows] == list(range(1, 201)),
                    'last_epoch_200': int(last['epoch']) == 200,
                    'receipt_seed_matches': receipt.get('seed') == seed,
                    'receipt_output_matches': receipt.get('output') == expected_run,
                    'receipt_config_matches': receipt.get('config', '').endswith(f'/protocol_clean_{arm}.yaml'),
                }
                assert all(checks.values()), checks
                progress.update(status='completed', completion_checks=checks,
                                receipt_legacy_identity={'arm': receipt.get('arm'), 'method_identity': receipt.get('method_identity')})
                completed[(arm, seed)] = (run, last)
                native = RAW / 'cgkd_w1' / f'native_rgb_s{seed}_e200'
                native_rows, native_last = last_csv(native / 'results.csv')
                assert len(native_rows) == 200 and int(native_last['epoch']) == 200
                report['contrasts_vs_historical_native'].append({
                    'arm': arm, 'seed': seed, 'epoch': 200,
                    'metrics': difference(last, native_last), 'args_differences': args_diff(run, native),
                    'causal_limit': 'SSL backbone and detector initialization are confounded; not a clean SSL benefit estimate',
                })
        report['progress'].append(progress)

for seed in (0, 42, 123):
    for control in ('shuffled', 'sar_only'):
        if ('paired', seed) in completed and (control, seed) in completed:
            left_dir, left = completed[('paired', seed)]
            right_dir, right = completed[(control, seed)]
            contrast = {
                'left_arm': 'paired', 'right_arm': control, 'seed': seed,
                'metric_identity': report['metric_identity'],
                'metrics': difference(left, right),
                'args_differences': args_diff(left_dir, right_dir),
                'shared_nonbackbone_tensor_count': 259,
                'independent_last_endpoint': False, 'three_seed_gate_evaluable': False,
            }
            report['internal_ssl_contrasts'].append(contrast)

report.update(completed_finetunes=len(completed),
              running_finetunes=sum(row['status'] == 'running' for row in report['progress']),
              queued_finetunes=sum(row['status'] == 'queued' for row in report['progress']),
              independent_osssl_metrics_records=sum(row['has_independent_metrics_record'] for row in report['progress']))
report['conclusions'] = {
    'positive_ssl_claim_supported': False, 'paired_attribution_supported': False,
    'new_observation': 'First complete same-seed paired-versus-shuffled CSV comparison is positive for seed123, without satisfying final evaluation or three-seed requirements',
    'ssl_vs_native_initialization_confounded': True,
    'ssl_internal_nonbackbone_template_shared': True, 'target_rgb_only_ssl_control_missing': True,
    'primary_gate_ap50_pp': 1.0, 'attribution_gate_ap50_pp': 0.5,
    'pretraining_seed_repetitions': 1,
    'threshold_warning': 'paired123 minus shuffled123 AP50 is 0.495 pp in saved CSV, not rounded to a passing 0.5; no final gate decision from this single CSV contrast',
    'action_this_audit': 'Read-only collection and local analysis; no GPU, inference, queue edits or threshold changes',
}
(OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: report[key] for key in ('timestamp', 'completed_finetunes', 'running_finetunes', 'queued_finetunes', 'independent_osssl_metrics_records', 'internal_ssl_contrasts', 'collection_integrity')}, ensure_ascii=False, indent=2))
