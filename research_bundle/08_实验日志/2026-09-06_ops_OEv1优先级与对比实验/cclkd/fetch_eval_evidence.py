import base64,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94',
     '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-']
r=subprocess.run(cmd,input=(HERE/'collect_eval_remote.py').read_bytes(),capture_output=True,check=True)
b=json.loads(r.stdout);manifest=[]
for key,value in b['files'].items():
    target=HERE/'eval_raw'/key;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(base64.b64decode(value.pop('base64')))
    manifest.append({'local_relative':str(target.relative_to(HERE)),**value})
(HERE/'eval_source_manifest.json').write_text(json.dumps({'captured_at_server':b['captured_at_server'],'files':manifest},indent=2)+'\n',encoding='utf-8')
(HERE/'eval_fetch_stderr.txt').write_bytes(r.stderr)
print(json.dumps({'captured_at_server':b['captured_at_server'],'files':len(manifest),'total_bytes':sum(v['size_bytes'] for v in manifest)}))
