"""Read-only SSH snapshot of OS-SSL-IR small artifacts; no GPU, inference or hash."""
import base64
import json
import pathlib
import subprocess

OUT = pathlib.Path(__file__).resolve().parent
REMOTE = r'''
import base64, csv, datetime, io, json, pathlib, subprocess
root = pathlib.Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
artifact = root/'artifacts/osssl_ir_20260906'
stamp = datetime.datetime.now().astimezone().isoformat()
files, inventory, summaries = [], [], []
selected = []
for p in artifact.iterdir():
    if p.is_file() and p.suffix in ('.json', '.yaml', '.py', '.sh', '.txt', '.md', '.log'):
        selected.append(p)
for p in (artifact/'ssl').glob('*/config.json'):
    selected.append(p)
for subroot in (root/'runs/osssl_ir_20260906', root/'runs/cgkd_w1'):
    for run in sorted(subroot.iterdir()):
        if not run.is_dir(): continue
        item = {'run': str(run), 'files': []}
        for p in sorted(run.iterdir()):
            if p.is_file():
                item['files'].append({'name':p.name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
                if p.suffix in ('.json','.csv','.yaml'):
                    selected.append(p)
        item['weights'] = [{'name':p.name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for p in sorted((run/'weights').glob('*.pt'))]
        if (run/'results.csv').exists():
            rows = list(csv.DictReader((run/'results.csv').read_text().splitlines()))
            item['csv_rows'] = len(rows)
            item['csv_last'] = rows[-1] if rows else None
        summaries.append(item)
for name in ('train_native_rgbt.py','eval_natives.sh','eval_natives.log'):
    selected.append(root/'artifacts/cgkd_20260905'/name)
for p in sorted(set(selected)):
    before = p.stat()
    partial = p.suffix == '.log' and before.st_size > 65536
    with p.open('rb') as f:
        if partial: f.seek(max(0, before.st_size-65536))
        body = f.read()
    after = p.stat()
    rel = str(p.relative_to(root)) + ('.partial_tail.txt' if partial else '')
    files.append({'relative':rel,'body':base64.b64encode(body).decode()})
    inventory.append({'source':str(p),'relative':rel,'source_bytes':before.st_size,'copied_bytes':len(body),'partial_tail':partial,'offset':max(0,before.st_size-65536) if partial else 0,'mtime_before':before.st_mtime,'mtime_after':after.st_mtime,'changed_during_read':(before.st_size,before.st_mtime)!=(after.st_size,after.st_mtime)})
ps = subprocess.run(['ps','-u','yudongfang','-o','pid,ppid,etime,args'],capture_output=True,text=True,check=True).stdout
processes = '\n'.join(line for line in ps.splitlines() if 'osssl' in line and ('worker' in line or 'project_resource_guard.py' in line))
print(json.dumps({'timestamp':stamp,'finished_at':datetime.datetime.now().astimezone().isoformat(),'root':str(root),'files':files,'inventory':inventory,'runs':summaries,'queue_processes':processes},ensure_ascii=False))
'''

result = subprocess.run(['ssh','-o','BatchMode=yes','94','python3 -'],input=REMOTE.encode(),capture_output=True,check=True)
data = json.loads(result.stdout)
for row in data.pop('files'):
    path = OUT/'raw'/row['relative']
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(row['body']))
(OUT/'snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'timestamp':data['timestamp'],'finished_at':data['finished_at'],'files':len(data['inventory']),'runs':[{'run':p['run'],'rows':p.get('csv_rows'),'last':p.get('csv_last')} for p in data['runs']]},ensure_ascii=False,indent=2))
