"""Read-only CCLKD endpoint inventory; no model loading or GPU access."""
import base64
import csv
import datetime
import io
import json
import pickletools
from pathlib import Path
import zipfile

ROOT = Path('/mnt/dataset/yudongfang/projects')
REPO = ROOT / 'SpaceNet6_OTD_official_reproduction'
CAM = ROOT / 'RGBT_campaign'
RUNS = CAM / 'runs/rgbt_cclkd_adapted_v1'
files = {}
def take(p, label=None):
    p = Path(p)
    if p.is_file():
        data = p.read_bytes()
        files[label or str(p.relative_to(ROOT))] = {
            'source': str(p), 'size_bytes': len(data),
            'mtime': datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            'base64': base64.b64encode(data).decode('ascii'),
        }

for p in [
    REPO/'tools/train_rgbt_cclkd.py', REPO/'yolo_osssl/rgbt_cclkd_kd.py',
    REPO/'yolo_osssl/rgbt_hnewa_pairing.py', REPO/'tools/eval_yolo_detector.py',
    REPO/'tools/project_resource_guard.py', REPO/'tools/write_jstars_run_receipt.py',
    REPO/'configs/research/rgbt_cclkd_protocol_drone.yaml', REPO/'AGENTS.md',
    CAM/'artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/rgb.data.yaml',
    CAM/'artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/infrared.data.yaml',
    CAM/'artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/prepare_receipt.json',
]: take(p)

inventory = []
for seed in (0,42,123):
    run = RUNS/f'cclkd_drone_seed{seed}_b32_e200'
    for name in ('args.yaml','completion_receipt.json','results.csv'):
        take(run/name)
    checkpoint = run/'weights/last.pt'
    info = {'seed': seed, 'run': str(run), 'checkpoint': str(checkpoint),
            'checkpoint_exists': checkpoint.exists()}
    if checkpoint.exists():
        info['checkpoint_size_bytes'] = checkpoint.stat().st_size
        info['checkpoint_mtime'] = datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat()
        with zipfile.ZipFile(checkpoint) as z:
            member = next(n for n in z.namelist() if n.endswith('data.pkl'))
            payload = z.read(member)
        info['pickle_globals'] = [arg for op,arg,pos in pickletools.genops(payload) if op.name=='GLOBAL']
        info['pickle_custom_names'] = [v for v in ('CCLKD','criterion','teacher','__main__','train_rgbt') if v.encode() in payload]
    rows = list(csv.reader(io.StringIO((run/'results.csv').read_text())))
    info['csv_header'] = rows[0]
    info['csv_last_row'] = rows[-1]
    info['csv_data_rows'] = len(rows)-1
    info['eval_files_in_run'] = [str(p) for p in run.rglob('*') if p.is_file() and ('eval' in p.name or 'metric' in p.name)]
    log = RUNS/f'logs/cclkd_dr_s{seed}_v2.log'
    if log.exists():
        with log.open('rb') as f:
            head = f.read(11500)
            f.seek(max(0,log.stat().st_size-12000))
            tail = f.read()
        for label,data in [('head',head),('tail',tail)]:
            files[f'log_excerpts/cclkd_dr_s{seed}_v2_{label}.txt'] = {
                'source': str(log), 'source_byte_section': label,
                'base64': base64.b64encode(data).decode('ascii'), 'size_bytes':len(data),
            }
    inventory.append(info)

# Search names, without opening checkpoints or image data.
existing = []
for top in (CAM/'artifacts', REPO/'artifacts'):
    for p in top.rglob('*'):
        if p.is_file() and p.suffix == '.json' and ('cclkd' in str(p).lower()):
            if any(t in p.name.lower() for t in ('eval','metric','summary')):
                existing.append(str(p))
                if p.stat().st_size < 200000: take(p)
print(json.dumps({'captured_at_server':datetime.datetime.now().isoformat(),
                  'inventory':inventory,'existing_named_eval_files':existing,'files':files}))
