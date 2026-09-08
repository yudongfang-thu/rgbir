"""Observe existing coordinator-owned jobs; never start or stop them."""
from pathlib import Path
import json,subprocess
root=Path(__file__).parent
source=(root.parent/'2026-09-08_probe_快速方向筛选/inspect_remaining_jobs.py').read_bytes()
with (root/'old_progress_source.py').open('xb') as f:f.write(source)
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=source,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
r=json.loads(p.stdout);assert r['scope']=='READ_ONLY_OLD_QUEUE_CSV_PROGRESS_NOT_ENDPOINT'
with (root/'OLD_QUEUE_PROGRESS.json').open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(utc=r['utc'],progress=[dict(run=Path(x['run']).name,epoch=x.get('last_logged_epoch'),complete=x['completion_receipt_exists']) for x in r['runs']])))
