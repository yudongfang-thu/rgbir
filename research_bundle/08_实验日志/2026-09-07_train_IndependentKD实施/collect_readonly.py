"""Read-only SSH snapshot; never invokes the mutating guard inspect operation."""
import base64
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE_CODE = r'''
import base64,csv,datetime,io,json,statistics,subprocess
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
repo=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
runs={}
for seed in [0,42,123]:
    campaign='rgbir_object_evidence_v1_20260906' if seed==42 else 'rgbir_object_evidence_expand_20260906'
    for arm,label in [('weight0','N'),('paired','C')]:
        runs[f'{label}{seed}']=root/'runs'/campaign/f'full_{arm}_s{seed}_attempt1'
    runs[f'R{seed}']=root/'runs/rgbir_oev1_random_20260906'/f'full_paired_random_s{seed}_attempt1'

for arm, name in [('c_shuffled','S42'),('c_same_modal','M42')]:
    runs[name]=root/'runs/rgbir_task_conditional_c_attribution_20260907'/f'full_{arm}_s42_attempt1'
files={};manifest=[];summary={}
def capture(path,key):
    if path.is_file():
        data=path.read_bytes()
        files[key]=base64.b64encode(data).decode()
        manifest.append({'remote':str(path),'local':key,'bytes':len(data),'mtime':path.stat().st_mtime})
def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None
for label,path in runs.items():
    names=['progress.json','completion_receipt.json','failure_receipt.json','evaluation_val.json','evaluation_val_roster.txt','results.csv','training_metrics.csv','args.yaml','config.yaml','initialization_receipt.json']
    if path.is_dir():
        for dirname in ['eval_evidence','run_evidence']:
            folder=path/dirname
            if folder.is_dir():
                names.extend(str(p.relative_to(path)) for p in folder.rglob('*') if p.is_file() and p.stat().st_size<2_000_000)
    for name in dict.fromkeys(names):capture(path/name,f'raw/{label}/{name}')
    progress=read_json(path/'progress.json')
    comp=read_json(path/'completion_receipt.json')
    eval_result=read_json(path/'evaluation_val.json')
    metrics_path=path/'results.csv'
    eta=None
    if metrics_path.is_file():
        rows=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(io.StringIO(metrics_path.read_text())) if all(k is not None and isinstance(v,str) for k,v in row.items())]
        if len(rows)>2 and 'time' in rows[-1]:
            times=[float(x['time']) for x in rows[-11:]]
            sec=statistics.median([b-a for a,b in zip(times,times[1:])])
            epoch=int(float(rows[-1]['epoch']))
            eta={'completed_epoch':epoch,'recent_median_epoch_seconds':sec,'remaining_training_seconds_estimate':max(0,200-epoch)*sec,'assumption':'recent throughput unchanged; excludes independent evaluation and resource queue delay'}
    summary[label]={'run':str(path),'exists':path.is_dir(),'progress':progress,'completion':comp,'evaluation':eval_result,'eval_receipt_exists':(path/'eval_evidence/run_receipt.json').is_file(),'eta':eta}
lease_path=repo/'runs/.project_resource_leases.json'
capture(lease_path,'resource_leases_raw.json')
for name in ['seed0_status.json','seed123_status.json','seed0_ownership.json','seed123_ownership.json','parallel_dispatch.json']:
    capture(root/'artifacts/oev1_concurrency_20260906'/name,f'queue/{name}')
capture(root/'artifacts/rgbir_oev1_random_20260906/queue_status.json','queue/old_random_queue_status.json')
procs={}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        status=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
        cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        procs[int(p.name)]={'ppid':int(status['PPid']),'rss_kib':int(status.get('VmRSS','0 kB').strip().split()[0]),'cmdline':cmd}
    except (OSError,ValueError,KeyError):pass
roots={pid for pid,x in procs.items() if any(s in x['cmdline'] for s in ['RGBT_campaign/','SpaceNet6_OTD_official_reproduction/']) and not x['cmdline'].endswith('bin/python - ')}
selected=set(roots)
while True:
    additional={pid for pid,x in procs.items() if x['ppid'] in selected}-selected
    if not additional:break
    selected.update(additional)
snapshot={'captured_at':datetime.datetime.now().astimezone().isoformat(),'runs':summary,
 'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 'cuda_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv,noheader,nounits'],text=True),
 'project_processes':{str(pid):procs[pid] for pid in sorted(selected)},
 'project_rss_kib_readonly':sum(procs[pid]['rss_kib'] for pid in selected),
 'rss_scope':'project path cmdline roots plus all descendants; shared pages double-counted as RSS; raw lease snapshot retained',
 'remote_mutations':False}
print(json.dumps({'files':files,'manifest':manifest,'snapshot':snapshot}))
'''


def main():
    result = subprocess.run(['ssh','-o','BatchMode=yes','94',REMOTE_PY,'-'], input=REMOTE_CODE.encode(), capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    payload=json.loads(result.stdout)
    stamp=payload['snapshot']['captured_at'].replace(':','').replace('+','_')
    out=HERE/'snapshots'/stamp
    out.mkdir(parents=True,exist_ok=False)
    for relative,encoded in payload['files'].items():
        target=out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))
    for name,obj in [('source_manifest.json',payload['manifest']),('snapshot.json',payload['snapshot'])]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'snapshot_dir':str(out),'captured_at':payload['snapshot']['captured_at'],'files':len(payload['files']),'runs':{k:{'epoch':(v['progress'] or {}).get('epoch'),'complete':bool(v['completion']),'evaluation':v['evaluation'],'eta':v['eta']} for k,v in payload['snapshot']['runs'].items()},'gpus':payload['snapshot']['gpus'],'project_rss_gib':payload['snapshot']['project_rss_kib_readonly']/2**20},indent=2))

if __name__=='__main__':main()
