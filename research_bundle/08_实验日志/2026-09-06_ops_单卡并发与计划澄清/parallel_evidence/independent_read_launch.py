"""Independent CPU-only live launch audit; no remote writes or guard.inspect()."""
import datetime,json,runpy,subprocess
from pathlib import Path
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
R=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
A=B/'artifacts/oev1_concurrency_20260906'
files={}
for seed in [0,123]:
    for name in [f'seed{seed}_status.json',f'seed{seed}_ownership.json',f'concurrent_canary_s{seed}.json',f'parallel_canary_comparison_s{seed}.json']:
        p=A/name
        if p.exists():files[name]=json.loads(p.read_text())
    cp=B/f'runs/rgbir_oev1_random_20260906/canary_paired_random_s{seed}_attempt1/completion_receipt.json'
    if cp.exists():files[f'canary_completion_s{seed}.json']=json.loads(cp.read_text())
lease=json.loads((R/'runs/.project_resource_leases.json').read_text())
m=runpy.run_path(str(R/'tools/project_resource_guard.py'))
ps=m['process_snapshot']();gp=m['gpu_process_snapshot']();gpus=m['gpu_snapshot']()
usage=m['ProjectResourceGuard'](R/'runs/.project_resource_leases.json')._usage(lease,ps,gp)
selected={int(l['job_pid']) for l in lease['leases'].values() if l.get('job_pid')}
selected|={975898,975933,975949,3743513,3743565,1564095,1564206,3966414,3966512}
for info in files.values():
    if isinstance(info,dict) and 'pid' in info:selected.add(int(info['pid']))
while True:
    expanded=selected|{pid for pid,p in ps.items() if p['ppid'] in selected}
    if expanded==selected:break
    selected=expanded
processes={}
for pid in sorted(selected & set(ps)):
    try:
        c=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        processes[pid]={**ps[pid],'cmd':c}
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
runs={}
relative={'R0':'rgbir_oev1_random_20260906/full_paired_random_s0_attempt1',
          'R42':'rgbir_oev1_random_20260906/full_paired_random_s42_attempt1',
          'R123':'rgbir_oev1_random_20260906/full_paired_random_s123_attempt1',
          'N42':'rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1',
          'P0':'rgbir_object_evidence_expand_20260906/full_paired_s0_attempt1',
          'N123':'rgbir_object_evidence_expand_20260906/full_weight0_s123_attempt1'}
for name,part in relative.items():
    p=B/'runs'/part
    runs[name]={'path':str(p),'exists':p.exists(),'progress':json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None,
                'runtime_ready':json.loads((p/'runtime_ready.json').read_text()) if (p/'runtime_ready.json').exists() else None,
                'completion_exists':(p/'completion_receipt.json').exists(),'evaluation_exists':(p/'eval_evidence/run_receipt.json').exists()}
print(json.dumps({'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'state':lease,'usage':usage,
                  'gpus':gpus,'cuda_processes':gp,'processes':processes,'runs':runs},ensure_ascii=False,indent=2))
