"""Read selected small evaluation records and implementation metadata on 94."""
import datetime
import hashlib
import json
from pathlib import Path
R = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
S = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
paths = list((R/'runs/rgbt_cmdistill_adapted_v2').glob('*/metrics_record.json'))
paths += list((R/'runs/rgbt_hnewa_cmkd_mse_inspired_v1/eval_records').glob('*.json'))
paths += list((R/'artifacts/cgkd_20260905').glob('*/completion_receipt.json'))
paths += list((S/'configs/research').glob('rgbt_cmdistill_protocol*.yaml'))
paths += [S/'yolo_osssl/rgbt_cmdistill_kd.py', S/'tools/train_rgbt_cmdistill.py',
          S/'yolo_osssl/rgbt_hnewa_pairing.py', S/'tools/train_rgbt_hnewa_mse.py',
          R/'artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/prepare_receipt.json',
          R/'artifacts/rgbt_p3_causal_v1/prepared/llvip/prepare_receipt.json']
out = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': [], 'errors': []}
for p in sorted(set(paths)):
    try:
        b = p.read_bytes()
        if len(b) > 250000: raise ValueError('small-file size limit')
        out['files'].append({'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(),
                             'bytes': len(b), 'text': b.decode('utf-8')})
    except Exception as ex: out['errors'].append({'path': str(p), 'error': repr(ex)})
print(json.dumps(out, ensure_ascii=False, indent=2))
