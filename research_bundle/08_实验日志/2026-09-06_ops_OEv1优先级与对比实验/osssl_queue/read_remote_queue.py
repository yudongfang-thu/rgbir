"""Read-only remote queue audit. Never writes on the server or signals processes."""
from pathlib import Path
import subprocess, json, datetime, hashlib, csv

base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
art=base/'artifacts/osssl_ir_20260906'
guard=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/tools/project_resource_guard.py')
files={}
for p in sorted(art.iterdir()):
    if p.is_file() and (p.suffix in ('.sh','.txt') or any(s in p.name.lower() for s in ('stop','pause','freeze','hold','queue'))):
        if p.stat().st_size<500000:
            data=p.read_bytes()
            files[str(p)]={'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode('utf-8',errors='replace')}
files[str(guard)]={'text':guard.read_text(),'sha256':hashlib.sha256(guard.read_bytes()).hexdigest()}
procs=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        status=(p/'status').read_text()
        ppid=int(next(l.split()[1] for l in status.splitlines() if l.startswith('PPid:')))
        if any(s in c for s in ('osssl_ir_20260906','rgbir_object_evidence','project_resource_guard')):
            procs.append({'pid':int(p.name),'ppid':ppid,'cmd':c})
    except (FileNotFoundError,PermissionError,StopIteration):pass
logs={str(p):p.read_text(errors='replace')[-10000:] for p in art.glob('worker*.log') if p.is_file()}
progress=[]
for arm in ('paired','shuffled','sar_only'):
    for seed in (0,42,123):
        d=base/'runs/osssl_ir_20260906'/f'{arm}_rgb_s{seed}_e200'
        p=d/'results.csv'
        rows=list(csv.DictReader(p.open())) if p.exists() else []
        progress.append({'arm':arm,'seed':seed,'epochs':len(rows),'last_csv':rows[-1] if rows else None,'files':[x.name for x in d.iterdir()] if d.exists() else []})
def cmd(args):
    r=subprocess.run(args,capture_output=True,text=True)
    return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
out={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'processes':procs,'worker_log_tails':logs,'progress':progress,'screen':cmd(['screen','-ls']),'gpu':cmd(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader']), 'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'])}
print(json.dumps(out,ensure_ascii=False))
