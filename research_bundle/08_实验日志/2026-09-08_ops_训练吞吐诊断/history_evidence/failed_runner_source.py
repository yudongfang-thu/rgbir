"""Read existing training times only; no AP analysis, GPU, SSH, hashes or run mutation."""
import csv,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;LOG=HERE.parent
IKD=LOG/'2026-09-07_train_IndependentKD实施'
SNAP=IKD/'snapshots/2026-09-07T165357.033246_0800/raw'
OUT=HERE/'history_evidence';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stat(p):
    v=p.stat();return dict(path=str(p),bytes=v.st_size,mtime_ns=v.st_mtime_ns)
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
all_epochs=[];full=[];input_files=[]
for name in ('N0','N42','N123','C0','C42','C123','R0','R42','R123','S42','M42'):
    folder=SNAP/name;path=folder/'results.csv'
    if not path.exists():continue
    with path.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    raw=[{k.strip():v.strip() for k,v in row.items()} for row in raw]
    times=[float(row['time']) for row in raw];epochs=[int(row['epoch']) for row in raw]
    assert all(math.isfinite(t) for t in times) and all(b>a for a,b in zip(times,times[1:]))
    durations=[times[0]]+[b-a for a,b in zip(times,times[1:])]
    for ep,t,d in zip(epochs,times,durations):all_epochs.append(dict(endpoint=name,epoch=ep,cumulative_seconds=t,epoch_seconds=d))
    completion_path=folder/'completion_receipt.json';completion=read(completion_path) if completion_path.exists() else {}
    full.append(dict(endpoint=name,arm='N' if name.startswith('N') else 'C0' if name.startswith('C') else 'random' if name.startswith('R') else 'shuffled' if name.startswith('S') else 'same_modal',
        seed=int(''.join(c for c in name if c.isdigit())),rows=len(epochs),last_completed_epoch=epochs[-1],first_epoch_seconds=durations[0],
        median_epoch_2plus_seconds=statistics.median(durations[1:]),mean_epoch_2plus_seconds=statistics.mean(durations[1:]),
        median_first20_excluding_start_seconds=statistics.median(durations[1:20]),median_last20_seconds=statistics.median(durations[-20:]),
        csv_total_seconds=times[-1],csv_total_hours=times[-1]/3600,completion_status=completion.get('status'),completion_seconds=completion.get('seconds'),
        gpu_ids=completion.get('resources',{}).get('gpu_ids'),batches=completion.get('batches'),optimizer_updates=completion.get('optimizer_updates'),
        checkpoint=completion.get('checkpoint'),source=str(path),completion_source=str(completion_path) if completion else None))
    input_files.append(stat(path))
    if completion:input_files.append(stat(completion_path))
canaries=[]
roots=[LOG/'2026-09-06_ops_单卡并发与计划澄清/concurrency/raw/RGBT_campaign/runs',IKD/'remote_admission_1532']
paths=[]
for root in roots:
    for path in root.rglob('completion_receipt.json'):
        if any(k in path.parent.name.lower() for k in ('canary','profile2')):paths.append(path)
for path in paths:
    value=read(path);sec=value.get('seconds');batches=value.get('batches');updates=value.get('optimizer_updates')
    canaries.append(dict(name=path.parent.name,status=value.get('status'),seed=value.get('seed'),arm=value.get('arm'),seconds=sec,
        batches=batches,optimizer_updates=updates,amp_skips=value.get('amp_skipped_updates'),
        seconds_per_batch=sec/batches if sec and batches else None,seconds_per_successful_update=sec/updates if sec and updates else None,
        gpu_ids=value.get('resources',{}).get('gpu_ids'),source=str(path)))
    input_files.append(stat(path))
states=[]
for path in sorted(IKD.glob('running_state_*.json')):
    value=read(path)
    if 'read_at' not in value or 'runs' not in value:continue
    for name,run in value['runs'].items():
        prog=run.get('progress') or {};through=run.get('throughput') or {}
        states.append(dict(read_at=value['read_at'],run=name,epoch_entered=prog.get('epoch'),completed_epochs=through.get('completed_epochs'),
            seconds=prog.get('seconds'),batches=prog.get('batches'),updates=prog.get('optimizer_updates'),
            recent_epoch_seconds=through.get('median_recent_epoch_seconds'),remaining_training_hours=through.get('remaining_training_hours'),source=str(path)))
    input_files.append(stat(path))
latest=read(IKD/'running_state_20260908_011232.json')
leases=[dict(job_id=v['job_id'],gpu=v['gpus'],pids=v['observed_cuda_pids'],vram=v['per_gpu_peak_vram_mib'],rss=v['peak_rss_mib']) for v in latest['leases']['leases'].values()]
result=dict(status='COMPLETED_HISTORY_ONLY',date='2026-09-08',snapshot='2026-09-07T165357.033246_0800',
    full=full,canaries=canaries,latest_read_at=latest['read_at'],latest_leases=leases,
    metric_scope='Training cumulative time/epoch differences; no AP values loaded into outputs. Snapshots can be stale; not current ETA.',
    new_hash_computed=False,gpu_or_ssh_used=False,inputs=input_files)
write(OUT/'summary.json',result);csvwrite(OUT/'completed_and_partial.csv',full);csvwrite(OUT/'epoch_times.csv',all_epochs)
csvwrite(OUT/'canary_times.csv',canaries);csvwrite(OUT/'running_throughput_snapshots.csv',states)
(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(full=full,canaries=canaries,latest_leases=leases),ensure_ascii=False,indent=2))
