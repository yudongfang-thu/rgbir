"""Collect exact small source files over SSH and compute endpoint contrasts."""
from pathlib import Path
import base64
import csv
import io
import json
import subprocess

here = Path(__file__).resolve().parent
proc = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94',
 '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],
 input=(here/'collect_remote.py').read_bytes(), capture_output=True)
(here/'collection_stderr.txt').write_bytes(proc.stderr)
if proc.returncode:
    raise RuntimeError(proc.stderr.decode(errors='replace'))
snap = json.loads(proc.stdout)
for row in snap['files']:
    p = here / row['target']
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(base64.b64decode(row.pop('content_base64')))
(here/'source_inventory.json').write_text(json.dumps(snap,indent=2),encoding='utf-8')

def record(rel):
    p = here/'raw/runs'/rel
    r = json.loads((p/'metrics_record.json').read_text())
    receipt = json.loads((p/'completion_receipt.json').read_text())
    rows = list(csv.DictReader(io.StringIO((p/'results.csv').read_text())))
    clean = [{k.strip():v.strip() for k,v in row.items()} for row in rows]
    last = clean[-1]
    return {'run':rel,'metric_record':r, 'metrics_percent':{k:v*100 for k,v in r['metrics'].items()},
            'completed_csv_epoch':int(last['epoch']),
            'completion_receipt_arm':receipt.get('arm'),
            'completion_receipt_config':receipt.get('config'),
            'csv_E200_percent':{'AP50':float(last['metrics/mAP50(B)'])*100,
                               'mAP50_95':float(last['metrics/mAP50-95(B)'])*100}}

runs = {
    'paired123':record('osssl_ir_20260906/paired_rgb_s123_e200'),
    'shuffled123':record('osssl_ir_20260906/shuffled_rgb_s123_e200'),
    'historical_native123':record('cgkd_w1/native_rgb_s123_e200')}
contrasts = {}
for a,b in [('paired123','shuffled123'),('paired123','historical_native123'),('shuffled123','historical_native123')]:
    contrasts[a+' minus '+b] = {k:runs[a]['metrics_percent'][k]-runs[b]['metrics_percent'][k]
                              for k in runs[a]['metrics_percent']}
roster = snap['current_roster']
(here/'current_val_roster_generated.txt').write_text('\n'.join(roster)+'\n',encoding='utf-8')
assert all(not f['changed_during_read'] for f in snap['files'])
assert all(r['completed_csv_epoch']==200 for r in runs.values())
assert len({r['metric_record']['data_yaml'] for r in runs.values()})==1
assert all(r['metric_record']['checkpoint'].endswith('/weights/last.pt') for r in runs.values())
summary = {'captured_at_utc':snap['captured_at_utc'],'runs':runs,'contrasts_pp':contrasts,
 'current_val_roster_count':len(roster),'current_val_roster_unique':len(set(roster)),
 'sar_only42_has_independent_metric':(here/'raw/runs/osssl_ir_20260906/sar_only_rgb_s42_e200/metrics_record.json').is_file(),
 'limitations':['One pretraining realization per arm and one completed same-seed finetuning comparison; no three-seed claim.',
 'Old native has a known head-initialization mismatch versus clean SSL templates.',
 'Metric records do not store full eval CLI, image size, batch, workers, or source snapshot; current script is contextual evidence.',
 'Current validation roster is collected but no execution-time roster receipt exists for these OS-SSL endpoints.',
 'Remote README records D3: both short GPU2 evaluations bypassed the resource guard.']}
(here/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
