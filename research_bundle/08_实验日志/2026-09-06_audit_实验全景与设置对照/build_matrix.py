"""Build a source-labelled current experiment matrix from saved evidence only."""
import csv
import json
from pathlib import Path

base = Path(__file__).resolve().parent
progress = json.loads((base / 'progress_snapshot.json').read_text(encoding='utf-8'))
ssl = json.loads((base / 'osssl_new_eval/summary.json').read_text(encoding='utf-8'))
rows = []
for cell in progress['oev1']['cells']:
    metric = cell.get('metrics') or {}
    rows.append(dict(route='OEv1', arm=cell['arm'], seed=cell['seed'],
                     completed_epochs=cell['completed_csv_epochs'],
                     endpoint_status=cell['status'],
                     metric_source='independent_last_ema' if metric else 'pending',
                     mAP50_95_percent=100*metric['mAP50_95'] if metric else '',
                     AP50_percent=100*metric['AP50'] if metric else '',
                     AP75_percent=100*metric['AP75'] if metric else '',
                     run=cell['run']))
for cell in progress['osssl']:
    key = str(cell['arm']) + str(cell['seed'])
    record = ssl['runs'].get(key)
    metric = record['metrics_percent'] if record else {}
    # Only terminal CSV values are retained, and never promoted to independent eval.
    terminal_csv = cell['last_csv_metrics'] if cell['completed_epochs'] == 200 else None
    status = 'completed' if cell['has_completion_receipt'] else ('running' if cell['completed_epochs'] else 'queued')
    rows.append(dict(route='OS-SSL-IR', arm='ir_only' if cell['arm']=='sar_only' else cell['arm'],
                     seed=cell['seed'], completed_epochs=cell['completed_epochs'],
                     endpoint_status=status,
                     metric_source='independent_last_limited_receipt' if metric else ('terminal_csv_only' if terminal_csv else 'pending'),
                     mAP50_95_percent=metric.get('mAP50_95', 100*float(terminal_csv['metrics/mAP50-95(B)']) if terminal_csv else ''),
                     AP50_percent=metric.get('AP50', 100*float(terminal_csv['metrics/mAP50(B)']) if terminal_csv else ''),
                     AP75_percent=metric.get('AP75',''), run=cell['run']))
assert len(rows) == 15
assert progress['oev1']['complete_seed_pairs'] == 0
with (base / 'experiment_matrix.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'rows':len(rows),'oev1_complete_pairs':0,'osssl_independent_last':2,'source_snapshot':'2026-09-06 22:26/22:28 Asia/Shanghai'}))
