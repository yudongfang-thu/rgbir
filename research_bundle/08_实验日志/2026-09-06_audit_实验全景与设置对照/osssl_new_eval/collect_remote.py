"""Read-only collection of two newly created OS-SSL last-checkpoint metrics.

Run through SSH stdin. Does not import torch or write any remote file.
"""
from pathlib import Path
import base64
import datetime
import json
import yaml

base = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
code = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
names = ['paired_rgb_s123_e200', 'shuffled_rgb_s123_e200', 'sar_only_rgb_s42_e200']
files = []
missing = []
weights = []

def take(p, target):
    if not p.is_file():
        missing.append(str(p))
        return
    a = p.stat()
    b = p.read_bytes()
    z = p.stat()
    if len(b) > 2_000_000:
        raise ValueError(f'Unexpected large file: {p}')
    files.append({'source': str(p), 'target': target, 'bytes': len(b),
                  'mtime_utc': datetime.datetime.fromtimestamp(a.st_mtime, datetime.timezone.utc).isoformat(),
                  'changed_during_read': (a.st_size,a.st_mtime_ns) != (z.st_size,z.st_mtime_ns),
                  'content_base64': base64.b64encode(b).decode()})

for name in names:
    run = base / 'runs/osssl_ir_20260906' / name
    for fn in ['metrics_record.json', 'completion_receipt.json', 'args.yaml', 'results.csv']:
        take(run / fn, f'raw/runs/osssl_ir_20260906/{name}/{fn}')
    for fn in ['last.pt', 'best.pt']:
        p = run / 'weights' / fn
        if p.is_file():
            st = p.stat()
            weights.append({'source': str(p), 'bytes': st.st_size,
                            'mtime_utc': datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),
                            'downloaded': False})
run = base / 'runs/cgkd_w1/native_rgb_s123_e200'
for fn in ['metrics_record.json','completion_receipt.json','args.yaml','results.csv']:
    take(run/fn, f'raw/runs/cgkd_w1/native_rgb_s123_e200/{fn}')
for fn in ['protocol_clean_paired.yaml','protocol_clean_shuffled.yaml','protocol_clean_sar_only.yaml','README.md']:
    take(base/'artifacts/osssl_ir_20260906'/fn, f'raw/artifacts/osssl_ir_20260906/{fn}')
for fn in ['eval_rgbt_detector.py','eval_yolo_detector.py']:
    take(code/'tools'/fn, f'raw/code/tools/{fn}')
data = base/'artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/rgb.data.yaml'
take(data, 'raw/data/rgb.data.yaml')
cfg = yaml.safe_load(data.read_text())
val = Path(cfg['val'])
if not val.is_absolute():
    val = Path(cfg.get('path', data.parent)) / val
if val.is_dir():
    roster = sorted(str(p) for p in val.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
    roster_kind = 'generated from current directory image filenames; not an execution-time evaluation receipt'
else:
    take(val, 'raw/data/val_roster.txt')
    roster = val.read_text().splitlines()
    roster_kind = 'existing text roster'
print(json.dumps({'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'read_only': True, 'files': files, 'missing': missing, 'checkpoint_metadata_only': weights,
                  'val_roster_source': str(val), 'current_roster': roster, 'roster_kind': roster_kind}))
