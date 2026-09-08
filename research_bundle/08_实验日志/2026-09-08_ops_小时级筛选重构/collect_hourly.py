"""Collect small executed screen evidence; never images, checkpoints or hashes."""
import argparse
import base64
import datetime
import json
from pathlib import Path
import subprocess
from prepare_hourly_release import HERE,PY,CAMPAIGN

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    dest=HERE/a.name
    if dest.exists():raise FileExistsError(dest)
    code="""import base64,json
from pathlib import Path
root=Path(ROOT);files={};summary={}
names={'hourly_training_receipt.json','hourly_evaluation_receipt.json','canary.json','initialization_check.json',
 'runtime_ready.json','hourly_failure.json','progress.json','hourly_config.yaml','results.csv','sample_stream.jsonl',
 'gradient_checks.jsonl','shared_gradient_observations.jsonl','evaluation_identity_projection.json','development_roster.txt',
 'completion.json','failure.json','manifest.json','canary_checks.json','training_evaluation_jobs.json'}
for folder in ('queue','canaries/N','canaries/C0','canaries/C1','runs/N','runs/C0','runs/C1','evaluations/N','evaluations/C0','evaluations/C1'):
 parent=root/folder
 if not parent.is_dir():continue
 for p in parent.iterdir():
  if p.is_file() and (p.name in names or p.name.endswith('_status.json') or p.name.endswith('_admission.json') or p.name.endswith('_completed.json')):
   if p.stat().st_size<=4000000:files[str(p.relative_to(root))]=base64.b64encode(p.read_bytes()).decode()
for arm in ('N','C0','C1'):
 r={}
 for stage,parent,receipt in [('canary',root/'canaries'/arm,'canary.json'),('train',root/'runs'/arm,'hourly_training_receipt.json'),('eval',root/'evaluations'/arm,'hourly_evaluation_receipt.json')]:
  if (parent/receipt).is_file():
   x=json.loads((parent/receipt).read_text());r[stage]={k:x[k] for k in ('status','seconds','batches','successful_updates','optimizer_updates','amp_skips','metrics') if k in x}
  elif (parent/'hourly_failure.json').is_file():r[stage]=json.loads((parent/'hourly_failure.json').read_text())
  elif (parent/'progress.json').is_file():r[stage]=json.loads((parent/'progress.json').read_text())
 summary[arm]=r
print(json.dumps(dict(files=files,summary=summary,queue_failure=json.loads((root/'queue/failure.json').read_text()) if (root/'queue/failure.json').is_file() else None)))
""".replace('ROOT',repr(CAMPAIGN+'/ft_screen_attempt1'))
    r=subprocess.run(['ssh','94',PY,'-'],input=code.encode(),capture_output=True,timeout=60);r.check_returncode()
    payload=json.loads(r.stdout.decode());dest.mkdir()
    for name,encoded in payload.pop('files').items():
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(encoded))
    payload['recorded_at']=datetime.datetime.now().astimezone().isoformat()
    (dest/'collection_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False))

if __name__=='__main__':main()
