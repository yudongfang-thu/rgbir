"""Strictly read-only 94 process, lease and queue audit; prints small raw evidence."""
import csv,datetime,hashlib,json,os,runpy,subprocess
from pathlib import Path
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
R=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
files={}
def capture(p):
    if p.is_file() and p.stat().st_size < 1000000:
        raw=p.read_bytes()
        files[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'mtime':p.stat().st_mtime,'text':raw.decode(errors='replace')}
for name in ['rgbir_oev1_random_20260906','rgbir_object_evidence_expand_20260906','rgbir_object_evidence_v1_20260906']:
    art=B/'artifacts'/name
    if art.exists():
        for p in list(art.iterdir())+[p for release in art.glob('release_*') for p in release.iterdir()]:
            if any(x in p.name.lower() for x in ['worker','queue','status','canary','launch','receipt']) and p.suffix in ['.py','.sh','.json']:capture(p)
capture(R/'tools/project_resource_guard.py')
capture(R/'runs/.project_resource_leases.json')
procs={}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        status=dict((line.split(':',1)[0],line.split(':',1)[1].strip()) for line in (p/'status').read_text().splitlines() if ':' in line)
        c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        procs[int(p.name)]={'pid':int(p.name),'ppid':int(status['PPid']),'rss_kib':int(status.get('VmRSS','0 kB').split()[0]),'cmd':c,'uid':status['Uid'].split()[0]}
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
selected={pid for pid,p in procs.items() if any(s in p['cmd'] for s in ('rgbir_oev1_random','rgbir_object_evidence','project_resource_guard'))}
while True:
    expanded=selected|{pid for pid,p in procs.items() if p['ppid'] in selected}
    if expanded==selected:break
    selected=expanded
runs=[]
for campaign in ['rgbir_oev1_random_20260906','rgbir_object_evidence_expand_20260906','rgbir_object_evidence_v1_20260906']:
    folder=B/'runs'/campaign
    if not folder.exists():continue
    for run in folder.iterdir():
        if not run.is_dir():continue
        rec={'run':str(run)}
        p=run/'results.csv'
        if p.exists():
            rows=list(csv.reader(p.read_text().splitlines()))
            rec['csv_header']=rows[0] if rows else []; rec['last_rows']=rows[-6:];rec['csv_mtime']=p.stat().st_mtime
        for name in ['progress.json','runtime_ready.json','completion_receipt.json']:
            p=run/name
            if p.exists():capture(p);rec[name]=json.loads(p.read_text())
        runs.append(rec)
def cmd(args):
    x=subprocess.run(args,capture_output=True,text=True);return {'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
g=runpy.run_path(str(R/'tools/project_resource_guard.py'))
state=json.loads((R/'runs/.project_resource_leases.json').read_text())
guard=g['ProjectResourceGuard'](R/'runs/.project_resource_leases.json')
usage=guard._usage(state,g['process_snapshot'](),g['gpu_process_snapshot']())
out={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'runs':runs,
     'relevant_processes':[procs[p] for p in sorted(selected)],'project_related_tree_rss_mib':sum(procs[p]['rss_kib'] for p in selected)/1024,
     'guard_usage_no_refresh':usage,'screen':cmd(['screen','-ls']),
     'gpus':cmd(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
     'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']),
     'free':cmd(['free','-m'])}
print(json.dumps(out,ensure_ascii=False,indent=2))
