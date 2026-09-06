from pathlib import Path
import subprocess,json
here=Path(__file__).resolve().parent
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=(here/'read_remote_queue.py').read_bytes(),capture_output=True)
(here/'queue_snapshot.json').write_bytes(r.stdout)
(here/'fetch_stderr.txt').write_bytes(r.stderr)
if r.returncode: raise RuntimeError(r.stderr.decode(errors='replace'))
d=json.loads(r.stdout)
sources=here/'remote_source'
sources.mkdir(exist_ok=True)
for path,data in d['files'].items():
    (sources/Path(path).name).write_bytes(data['text'].encode('utf-8'))
print(json.dumps({'captured_at':d['captured_at'],'files':list(d['files']),'progress':[{k:v for k,v in x.items() if k in ('arm','seed','epochs')} for x in d['progress']],'gpu':d['gpu']},ensure_ascii=False))
