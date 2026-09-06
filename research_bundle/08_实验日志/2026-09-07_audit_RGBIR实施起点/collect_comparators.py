"""Read-only collection of already-trained comparator evidence; no evaluations."""
import base64
import json
import subprocess
from pathlib import Path
from collect_readonly import REMOTE_PY

HERE=Path(__file__).resolve().parent
CODE=r'''
import base64,datetime,json
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
files={};manifest=[]
for seed in [0,42,123]:
 folders={'cmdistill':root/'runs/rgbt_cmdistill_adapted_v2'/f'dronevehicle_seed{seed}_b32_e200',
          'historical_native':root/'runs/cgkd_w1'/f'native_rgb_s{seed}_e200',
          'cclkd_train':root/'runs/rgbt_cclkd_adapted_v1'/f'cclkd_drone_seed{seed}_b32_e200',
          'cclkd_eval':root/'runs/oev1_comparators_20260906'/f'cclkd_partial_s{seed}_full_attempt1'}
 for method,folder in folders.items():
  names=['metrics_record.json','completion_receipt.json','args.yaml','evaluation_val.json','protocol.json','evaluation_val_roster.txt','eval_evidence/run_receipt.json']
  if (folder/'eval_evidence').is_dir():
   names.extend(str(p.relative_to(folder)) for p in (folder/'eval_evidence').rglob('*') if p.is_file() and p.stat().st_size<2_000_000)
  for name in dict.fromkeys(names):
   path=folder/name
   if path.is_file():
    data=path.read_bytes();key=f'{method}/s{seed}/{name}'
    files[key]=base64.b64encode(data).decode()
    manifest.append({'remote':str(path),'local':key,'bytes':len(data),'mtime':path.stat().st_mtime})
print(json.dumps({'captured_at':datetime.datetime.now().astimezone().isoformat(),'files':files,'manifest':manifest}))
'''

def main():
 result=subprocess.run(['ssh','-o','BatchMode=yes','94',REMOTE_PY,'-'],input=CODE.encode(),capture_output=True)
 if result.returncode:raise RuntimeError(result.stderr.decode())
 payload=json.loads(result.stdout)
 out=HERE/'comparator_snapshot'
 out.mkdir(exist_ok=False)
 for relative,encoded in payload.pop('files').items():
  path=out/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
 (out/'source_manifest.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'captured_at':payload['captured_at'],'files':len(payload['manifest'])}))

if __name__=='__main__':main()
