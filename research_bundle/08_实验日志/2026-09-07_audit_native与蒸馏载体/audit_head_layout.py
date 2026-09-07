"""Read known frozen model heads on CPU only; no prediction/training/CUDA."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PYTHON = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE_CODE = '''import json, torch
torch.set_num_threads(1)
base = '/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/'
rows = []
for role, suffix in [('teacher', 'infrared_seed42_native_b32a2'), ('reference', 'rgb_seed42_native_b32a2')]:
    path = base + suffix + '/weights/last.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = ckpt.get('ema') or ckpt.get('model')
    head = model.model[-1]
    rows.append(dict(role=role, path=path, nc=head.nc, reg_max=head.reg_max, no=head.no,
                     strides=head.stride.tolist(), names=model.names,
                     device=str(next(model.parameters()).device)))
print(json.dumps(dict(status='read_only_cpu', heads=rows)))
'''

proc = subprocess.run(['ssh', '94', PYTHON + ' -'], input=REMOTE_CODE, text=True, capture_output=True, check=True)
data = json.loads(proc.stdout)
assert all(x['device'] == 'cpu' and x['nc'] == 5 and x['reg_max'] == 16 for x in data['heads'])
(HERE/'head_layout_receipt.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(data, ensure_ascii=False, indent=2))
