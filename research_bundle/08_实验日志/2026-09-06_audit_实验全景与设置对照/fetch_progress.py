from pathlib import Path
import json
import subprocess

here=Path(__file__).resolve().parent
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=(here/'refresh_progress.py').read_bytes(),capture_output=True)
(here/'progress_snapshot.json').write_bytes(r.stdout)
(here/'progress_stderr.txt').write_bytes(r.stderr)
if r.returncode:raise RuntimeError(r.stderr.decode('utf-8',errors='replace'))
data=json.loads(r.stdout)
print(json.dumps({'captured_at_utc':data['captured_at_utc'],'oev1':[{k:c[k] for k in ('seed','arm','status','completed_csv_epochs','metrics')} for c in data['oev1']['cells']],
                 'osssl':[{k:c[k] for k in ('arm','seed','completed_epochs','has_completion_receipt','independent_metric_files')} for c in data['osssl']]}))
