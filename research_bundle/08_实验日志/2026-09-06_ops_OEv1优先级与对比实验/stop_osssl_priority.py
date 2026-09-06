"""Stop only the verified OS-SSL schedulers and newly begun IR-only seed0 run."""
import csv,json,os,signal,time,zipfile
from datetime import datetime,timezone
from pathlib import Path

root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
artifact=root/'artifacts/oev1_priority_comparators_20260906'
artifact.mkdir(parents=True,exist_ok=True)
target=artifact/'osssl_priority_stop_receipt.json'
assert not target.exists(), 'This action must not be blindly repeated'
worker=root/'artifacts/osssl_ir_20260906/worker3.sh'
run=root/'runs/osssl_ir_20260906/sar_only_rgb_s0_e200'
trainer_script=str(root/'artifacts/cgkd_20260905/train_native_rgbt.py')

def state(pid):
    p=Path('/proc')/str(pid)
    if not p.exists(): return None
    try:
        st=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,cmd=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x],
                    state=st[0],ppid=int(st[1]),start_ticks=int(st[19]))
    except FileNotFoundError:return None

def stat_file(p):
    s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

workers={3894177:0,3894186:2}
before={}
for pid,gpu in workers.items():
    before[pid]=state(pid)
    expected=['bash',str(worker),str(gpu),str(root/f'artifacts/osssl_ir_20260906/jobs2_gpu{gpu}.txt')]
    assert before[pid] and before[pid]['cmd']==expected,(pid,before[pid])
    assert before[pid]['start_ticks']==183404002
train=state(878988);guard=state(878944)
assert train and train['ppid']==878944 and train['cmd'][1]==trainer_script
assert train['cmd'][train['cmd'].index('--output')+1]==str(run)
assert guard and guard['ppid']==3894186 and 'osssl-ir-sar_only-s0' in guard['cmd']
last=run/'weights/last.pt'
assert last.is_file() and last.stat().st_size>1_000_000
assert not (run/'completion_receipt.json').exists()
csv_rows=list(csv.DictReader((run/'results.csv').open()))
last_epoch=int(float(csv_rows[-1]['epoch']))
assert last_epoch<20, 'Only the newly begun early run is authorized for termination here'
all_states=[state(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
children={878988}
while True:
    expanded=children|{s['pid'] for s in all_states if s and s['ppid'] in children}
    if expanded==children:break
    children=expanded
desc={s['pid']:s for s in all_states if s and s['pid'] in children}
for s in desc.values():
    assert s['cmd'][1]==trainer_script and str(run) in s['cmd'],s

receipt=dict(reason='User-directed prioritization of OEv1 and comparator evidence; not an efficacy-based success/failure decision',
             utc_start=datetime.now(timezone.utc).isoformat(),workers_before=before,trainer_before=train,
             guard_before=guard,training_descendants=desc,completed_epoch_before=last_epoch,
             checkpoint_before=stat_file(last),actions=[])
intent=artifact/'osssl_priority_stop_intent.json'
with intent.open('x') as f:json.dump(receipt,f,indent=2)

# Freeze only parent schedulers first; no process-group signals touch unrelated work.
for pid in workers:
    assert state(pid)['cmd']==before[pid]['cmd']
    os.kill(pid,signal.SIGSTOP);receipt['actions'].append([pid,'SIGSTOP_scheduler'])
time.sleep(.2)
assert all(state(pid)['state'] in ('T','t') for pid in workers)
os.kill(878988,signal.SIGTERM);receipt['actions'].append([878988,'SIGTERM_training'])
time.sleep(2)
for pid,s in desc.items():
    now=state(pid)
    if now and now['state']!='Z' and now['start_ticks']==s['start_ticks'] and now['cmd']==s['cmd']:
        os.kill(pid,signal.SIGTERM);receipt['actions'].append([pid,'SIGTERM_verified_training_descendant'])
# A pending SIGTERM kills the stopped bash when continued; no script rewrite/retry.
for pid in workers:
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    receipt['actions'].append([pid,'SIGTERM_then_SIGCONT_scheduler'])
for _ in range(15):
    live=[pid for pid in [*workers,*children,878944] if (s:=state(pid)) and s['state']!='Z']
    if not live:break
    time.sleep(1)
receipt['remaining_target_pids']=live
receipt['checkpoint_after']=stat_file(last)
with zipfile.ZipFile(last) as z:receipt['checkpoint_zip_crc_error']=z.testzip()
receipt['official_training_completion_absent']=not (run/'completion_receipt.json').exists()
receipt['utc_finished']=datetime.now(timezone.utc).isoformat()
with target.open('x') as f:json.dump(receipt,f,indent=2)
with (run/'priority_stop_20260906.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
assert not live,live
assert receipt['checkpoint_zip_crc_error'] is None
