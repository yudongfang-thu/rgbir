"""Collect completed D1/D2 evidence only; large-file approval is explicit."""
import argparse
import base64
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
BASE = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907'
RUNS = {key: f'd2_{key}_attempt1' for key in ('drone_train','drone_val','llvip_train','llvip_val')}
RUNS['llvip_verified_train'] = 'd2_llvip_verified_attempt1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--allow-large-drone-train',action='store_true')
    args=parser.parse_args()
    local_capture=HERE/'collection_receipts'/args.capture
    local_capture.mkdir(parents=True,exist_ok=False)
    already=[p.relative_to(HERE).as_posix() for key in RUNS if (HERE/key).is_dir() for p in (HERE/key).rglob('*') if p.is_file()]
    code='import json,base64,datetime\nfrom pathlib import Path\n'
    code+=f'base=Path({BASE!r})\nruns={RUNS!r}\nalready=set({already!r})\nallow_large={args.allow_large_drone_train!r}\n'
    code+='''files={};manifest=[];inventory=[]
for key,name in runs.items():
 path=base/name
 summary=json.loads((path/'summary.json').read_text()) if (path/'summary.json').is_file() else None
 receipt=json.loads((path/'run_evidence/run_receipt.json').read_text()) if (path/'run_evidence/run_receipt.json').is_file() else None
 completed=summary is not None and summary.get('status')=='completed' and receipt is not None and receipt.get('terminal_status')=='COMPLETED'
 sizes={p.name:p.stat().st_size for p in path.iterdir() if p.is_file()} if path.is_dir() else {}
 inventory.append({'key':key,'remote':str(path),'completed_with_receipt':completed,'files':sizes,
   'summary_totals':summary.get('totals') if summary else None,
   'progress':json.loads((path/'progress.json').read_text()) if (path/'progress.json').is_file() else None})
 if not completed:continue
 names=['summary.json','d1_objects.jsonl','d2_anchors.jsonl','images.jsonl','frozen_roster.json','protocol_config.yaml']
 names.extend(str(p.relative_to(path)) for p in (path/'run_evidence').rglob('*') if p.is_file())
 for name in names:
  source=path/name;relative=key+'/'+name
  if relative in already:continue
  size=source.stat().st_size
  if size>25_000_000 and not (allow_large and key=='drone_train' and name in ['d1_objects.jsonl','d2_anchors.jsonl']):
   manifest.append({'remote':str(source),'local':relative,'bytes':size,'status':'AWAITING_LARGE_FILE_APPROVAL'});continue
  data=source.read_bytes();files[relative]=base64.b64encode(data).decode()
  manifest.append({'remote':str(source),'local':relative,'bytes':len(data),'mtime':source.stat().st_mtime,'status':'COLLECTED_ORIGINAL_BYTES'})
print(json.dumps({'captured_at':datetime.datetime.now().astimezone().isoformat(),'inventory':inventory,'manifest':manifest,'files':files}))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),capture_output=True)
    (local_capture/'stderr.txt').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    payload=json.loads(result.stdout)
    files=payload.pop('files')
    for relative,encoded in files.items():
        path=HERE/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        path.write_bytes(base64.b64decode(encoded))
    (local_capture/'manifest.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'captured_at':payload['captured_at'],'files':len(files),
          'runs':[{'key':r['key'],'complete':r['completed_with_receipt'],'totals':r['summary_totals']} for r in payload['inventory']],
          'oversize':[r for r in payload['manifest'] if r['status']!='COLLECTED_ORIGINAL_BYTES']},indent=2))


if __name__=='__main__':main()
