"""Read-only small running-state snapshot; does not recollect completed evidence."""
import datetime
import json
from pathlib import Path
import subprocess

LOG=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE=r'''
import csv,datetime,io,json,statistics,subprocess,time
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
repo=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
def read(path):return json.loads(path.read_text()) if path.exists() else None
campaign=root/'artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2'
runs={'C1_s'+str(s):campaign/'runs'/('C1_seed'+str(s)) for s in (42,0,123)}
runs.update({a:root/'runs/rgbir_task_conditional_c_attribution_20260907'/('full_'+a+'_s42_attempt1')
             for a in ('c_shuffled','c_same_modal')})
rows={}
for name,path in runs.items():
    progress=path/'progress.json'
    row=dict(run=str(path),progress=read(progress),progress_age_seconds=time.time()-progress.stat().st_mtime,
        completion=read(path/'completion_receipt.json'),failure=read(path/'failure_receipt.json'),
        evaluation_exists=(path/'evaluation_val.json').exists(),
        evaluation_receipt_exists=(path/'eval_evidence/run_receipt.json').exists())
    metrics=path/'results.csv'
    if metrics.exists():
        values=[]
        for raw in csv.DictReader(io.StringIO(metrics.read_text())):
            try:
                clean={k.strip():v.strip() for k,v in raw.items() if k is not None and isinstance(v,str)}
                values.append((int(float(clean['epoch'])),float(clean['time'])))
            except (KeyError,ValueError):continue
        elapsed=[b[1]-a[1] for a,b in zip(values[-6:],values[-5:])] if len(values)>=6 else [b[1]-a[1] for a,b in zip(values,values[1:])]
        if elapsed:
            seconds=statistics.median(elapsed)
            row['throughput']=dict(completed_epochs=values[-1][0],median_recent_epoch_seconds=seconds,
                remaining_training_hours=(200-values[-1][0])*seconds/3600,
                scope='Recent observed training throughput only; excludes evaluation, queue and future load changes')
    rows[name]=row
leases=read(repo/'runs/.project_resource_leases.json')
procs={};roots=set()
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        status=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
        cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        pid=int(p.name)
        procs[pid]=dict(ppid=int(status['PPid']),rss_kib=int(status.get('VmRSS','0 kB').strip().split()[0]))
        if any(s in cmd for s in ('RGBT_campaign/','SpaceNet6_OTD_official_reproduction/')):roots.add(pid)
    except (OSError,ValueError,KeyError):continue
selected=set(roots)
while True:
    more={p for p,v in procs.items() if v['ppid'] in selected}-selected
    if not more:break
    selected.update(more)
def query(args):return list(csv.reader(io.StringIO(subprocess.check_output(args,text=True))))
gpus={}
for r in query(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits']):
    i,uuid,total,used,free=[x.strip() for x in r]
    gpus[uuid]=dict(index=int(i),total_mib=int(total),used_mib=int(used),free_mib=int(free))
cuda=[]
for r in query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']):
    pid,uuid,mem=[x.strip() for x in r]
    if int(pid) in selected:
        cuda.append(dict(pid=int(pid),gpu=gpus[uuid]['index'],vram_mib=int(mem)))
summary=dict(read_at=datetime.datetime.now().astimezone().isoformat(),runs=rows,
    gpus=list(gpus.values()),project_cuda=cuda,project_processes={str(p):procs[p] for p in sorted(selected)},
    project_rss_gib=sum(procs[p]['rss_kib'] for p in selected)/2**20,leases=leases,
    raw_monitor_only=True,new_gpu_tasks=0,remote_mutations=False)
print(json.dumps(summary))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode('utf-8'),capture_output=True,check=True)
data=json.loads(r.stdout)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output=LOG/('running_state_'+stamp+'.json')
with output.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
print(json.dumps(dict(output=str(output),rss_gib=data['project_rss_gib'],cuda=data['project_cuda'],
    runs={name:dict(progress=row['progress'],completion=bool(row['completion']),failure=bool(row['failure']),
        evaluation=row['evaluation_exists'],age=row['progress_age_seconds'],throughput=row.get('throughput'))
        for name,row in data['runs'].items()})))
