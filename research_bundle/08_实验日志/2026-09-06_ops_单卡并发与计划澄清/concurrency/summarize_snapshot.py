import json
from pathlib import Path
H=Path(__file__).resolve().parent
j=json.loads((H/'remote_snapshot_v2.json').read_text(encoding='utf-8'))
leases=json.loads(next(v['text'] for k,v in j['files'].items() if k.endswith('.project_resource_leases.json')))['leases']
for l in leases.values():
    print('LEASE',json.dumps(l,ensure_ascii=False))
for r in j['runs']:
    if '/full_' not in r['run']:continue
    p=r.get('progress.json',{})
    rows=r.get('last_rows',[])
    print('RUN',r['run'], 'progress', p)
    print('ROWS',r.get('csv_header'), rows)
print('CANARY')
for k,v in j['files'].items():
    if 'canary_comparison' in k:
        q=json.loads(v['text']); print(k,q)
