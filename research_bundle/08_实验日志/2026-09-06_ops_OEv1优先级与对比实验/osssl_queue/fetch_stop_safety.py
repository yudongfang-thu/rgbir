from pathlib import Path
import subprocess,json
here=Path(__file__).resolve().parent
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=(here/'read_stop_safety.py').read_bytes(),capture_output=True)
(here/'stop_safety_snapshot.json').write_bytes(r.stdout)
(here/'stop_safety_stderr.txt').write_bytes(r.stderr)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
d=json.loads(r.stdout)
print(json.dumps({'captured_at':d['captured_at'],'checkpoints':d['checkpoints'],'processes':[p for p in d['processes'] if '/worker' in p['cmd'] or 'project_resource_guard.py' in p['cmd'] or p['ppid'] in (878944,)]},ensure_ascii=False))
