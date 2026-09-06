"""Read-only CPU metadata audit. Run via: ssh 94 python3 - < this_file.

No checkpoint loads, writes, training imports, or GPU calls.
"""
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path

ROOT = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')

def read_entry(p):
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
            'mtime_utc': datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat(),
            'text': b.decode('utf-8', errors='replace')}

out = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'root': str(ROOT), 'runs': [], 'metadata': [], 'errors': []}
for p in sorted((ROOT/'runs').rglob('results.csv')):
    try:
        e = read_entry(p)
        raw = e.pop('text')
        e['raw_csv'] = raw
        parsed = list(csv.reader(io.StringIO(raw)))
        header = [k.strip() for k in parsed[0]]
        e['row_width_anomalies'] = [{'line': i+2, 'expected': len(header), 'actual': len(r)} for i, r in enumerate(parsed[1:]) if len(r) != len(header)]
        rows = [{k: v.strip() for k, v in zip(header, r)} for r in parsed[1:]]
        valid = [r for r in rows if r.get('epoch')]
        e['n_rows'] = len(valid)
        e['last'] = valid[-1] if valid else None
        metric = 'metrics/mAP50-95(B)'
        normal = [r for r in valid if len(r) == len(header) and r.get(metric)]
        e['max_map_row'] = max(normal, key=lambda r: float(r[metric])) if normal else None
        e['last10'] = valid[-10:]
        for name in ('args.yaml', 'completion_receipt.json'):
            a = p.parent/name
            if a.is_file(): e[name] = read_entry(a)
        out['runs'].append(e)
    except Exception as ex: out['errors'].append({'path': str(p), 'error': repr(ex)})

selected = [
    ROOT/'governance/dataset_audit_20260831.md',
    ROOT/'governance/dataset_audit_clean_20260831.md',
    ROOT/'artifacts/p2_feature_probe_20260905/summary.json',
    ROOT/'artifacts/p3_daynight_20260905/summary.json',
]
for pattern in ('*cmdistill*', '*hnewa*', '*p3*', '*cgkd*'):
    selected.extend((ROOT/'configs').glob(pattern))
for p in sorted(set(selected)):
    if p.is_file() and p.stat().st_size < 200000:
        out['metadata'].append(read_entry(p))
print(json.dumps(out, ensure_ascii=False, indent=2))
