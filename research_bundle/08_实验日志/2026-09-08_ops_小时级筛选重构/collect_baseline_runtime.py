"""Read baseline CSV time and args, without model loading or GPU work."""
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).absolute().parent
PYTHON='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE=r'''
import csv,datetime,io,json,statistics
from pathlib import Path
import yaml
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
runs={
 'original_rgb42':root/'runs/rgbt_p3_causal_v1/formal_native/dronevehicle/rgb_seed42_native_b32a2',
 'original_ir42':root/'runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2',
 'current_C1_42':root/'artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2/runs/C1_seed42',
}
out={}
for name,path in runs.items():
    record=dict(path=str(path),exists=path.exists())
    for leaf in ['args.yaml','results.csv']:
        q=path/leaf
        if not q.exists():continue
        s=q.stat();assert s.st_size<3000000
        record[leaf]=dict(path=str(q),bytes=s.st_size,mtime_ns=s.st_mtime_ns,text=q.read_text())
    if 'results.csv' in record:
        rows=list(csv.DictReader(io.StringIO(record['results.csv']['text'])))
        times=[]
        for row in rows:
            row={k.strip():v.strip() for k,v in row.items() if k is not None and isinstance(v,str)}
            times.append(dict(epoch=int(float(row['epoch'])),seconds=float(row['time'])))
        durations=[b['seconds']-a['seconds'] for a,b in zip(times,times[1:])]
        record['timing']=dict(completed_epochs=times[-1]['epoch'],total_hours=times[-1]['seconds']/3600,
              last10_median_seconds=statistics.median(durations[-10:]),
              all_excluding_first_median_seconds=statistics.median(durations),last10=durations[-10:])
    if 'args.yaml' in record:
        cfg=yaml.safe_load(record['args.yaml']['text'])
        keys=['model','data','epochs','imgsz','batch','nbs','workers','optimizer','amp','cache','rect','deterministic','seed','device','lr0','warmup_epochs']
        record['selected_args']={k:cfg.get(k) for k in keys}
    out[name]=record
print(json.dumps(dict(captured_at=datetime.datetime.now().astimezone().isoformat(),runs=out,
                     new_hash_computed=False,new_cuda_workloads=0,checkpoint_read=False)))
'''
out=HERE/'baseline_runtime';out.mkdir(exist_ok=False)
r=subprocess.run(['ssh','94',PYTHON,'-'],input=REMOTE.encode('utf-8'),capture_output=True,check=True,timeout=30)
result=json.loads(r.stdout)
(out/'raw.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
for name,row in result['runs'].items():
    target=out/name;target.mkdir()
    for leaf in ['args.yaml','results.csv']:
        if leaf in row:(target/leaf).write_bytes(row[leaf]['text'].encode('utf-8'))
print(json.dumps({k:{field:v.get(field) for field in ['exists','timing','selected_args']} for k,v in result['runs'].items()},ensure_ascii=False))
