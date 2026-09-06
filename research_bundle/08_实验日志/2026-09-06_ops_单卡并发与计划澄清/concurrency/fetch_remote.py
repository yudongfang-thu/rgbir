from pathlib import Path
import subprocess,json
here=Path(__file__).resolve().parent
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=(here/'read_remote.py').read_bytes(),capture_output=True)
(here/'remote_snapshot_v2.json').write_bytes(r.stdout)
(here/'remote_stderr_v2.txt').write_bytes(r.stderr)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
j=json.loads(r.stdout)
for path,value in j['files'].items():
    p=here/'raw'/Path(path).relative_to('/mnt/dataset/yudongfang/projects')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value['text'].encode('utf-8'))
print(json.dumps({k:v for k,v in j.items() if k not in ['files','runs','relevant_processes']},ensure_ascii=False,indent=2))
print('RELEVANT PROCESSES')
for p in j['relevant_processes']:
    if 'worker' in p['cmd'] or p['pid'] in [x['job_pid'] for x in json.loads(next(v['text'] for k,v in j['files'].items() if k.endswith('.project_resource_leases.json')))['leases'].values()]:print(p)
