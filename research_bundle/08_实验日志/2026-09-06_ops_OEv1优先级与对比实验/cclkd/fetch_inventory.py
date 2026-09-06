"""Fetch read-only remote evidence without transmitting checkpoint tensors."""
import base64
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94',
         '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-']
result=subprocess.run(command,input=(HERE/'inspect_remote.py').read_bytes(),capture_output=True,check=True)
blob=json.loads(result.stdout)
files=blob.pop('files')
manifest=[]
for key,item in files.items():
    target=HERE/'raw'/key
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(base64.b64decode(item.pop('base64')))
    manifest.append({'local_relative':str(target.relative_to(HERE)),**item})
(HERE/'inventory.json').write_text(json.dumps(blob,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(HERE/'fetch_stderr.txt').write_bytes(result.stderr)
print(json.dumps({'captured_at_server':blob['captured_at_server'],'inventory':blob['inventory'],
                  'existing_named_eval_files':blob['existing_named_eval_files'],'copied_files':len(files)},indent=2))
