"""Explicit, identity-checked resource reservation revision; does not edit guard code.

Default is read-only validation. --apply is required to modify the two named
live leases under the existing guard lock. Never releases or rebinds a lease.
"""
import argparse,copy,datetime,json,runpy
from pathlib import Path

REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
BASE=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
ART=BASE/'artifacts/oev1_concurrency_20260906'
PROFILE_RUN=BASE/'runs/oev1_concurrency_20260906/profile_paired_random_s0_attempt1'
LEASE_FILE=REPO/'runs/.project_resource_leases.json'
VRAM=8300
RSS=32768
TARGETS=(
    dict(lease_id='oev1_random_full_s42-66b55165489e',job_id='oev1_random_full_s42',guard_pid=975933,cuda_pid=975949,gpu=2,
         run=str(BASE/'runs/rgbir_oev1_random_20260906/full_paired_random_s42_attempt1'),arm='paired_random',seed=42),
    dict(lease_id='rgbir_oev1_expand_full_paired_s0_attempt1-7f1297f1b60b',job_id='rgbir_oev1_expand_full_paired_s0_attempt1',guard_pid=3743513,cuda_pid=3743565,gpu=5,
         run=str(BASE/'runs/rgbir_object_evidence_expand_20260906/full_paired_s0_attempt1'),arm='paired',seed=0),
)

def require(value,message):
    if not value:raise ValueError(message)

def is_descendant(pid,root,processes):
    seen=set()
    while pid in processes and pid not in seen:
        if pid==root:return True
        seen.add(pid);pid=int(processes[pid]['ppid'])
    return False

def validate_revision(state,processes,gpu_processes,gpus,profile,summary,receipt,progress):
    """Pure validation and planning. Does not mutate inputs or access filesystem."""
    require(state.get('schema')=='jstars-project-resource-leases-v1','unexpected lease schema')
    require(profile.get('returncode')==0 and summary.get('returncode')==0 and summary.get('status')=='passed','concurrent profile not completed/passed')
    require(summary.get('run')==str(PROFILE_RUN),'unexpected profile run')
    require(summary.get('existing_run')==TARGETS[0]['run'],'unexpected concurrent existing run')
    require(summary.get('minimum_observed_free_mib',0)>=2048,'profile free VRAM below 2 GiB')
    require(summary.get('cuda_peak_process_count')==2 and summary.get('samples_with_two_cuda_pids',0)>0,'no proven two-CUDA-process profile')
    require(summary.get('canary_check',{}).get('status')=='passed','canary validation not passed')
    require(receipt.get('optimizer_updates',0)>=24,'profile has fewer than 24 real optimizer updates')
    require(receipt.get('arm')=='paired_random','profile arm mismatch')
    require(receipt.get('resources',{}).get('gpu_ids')==[2],'profile physical GPU mismatch')
    require(receipt.get('official_test_accessed') is False,'unexpected official test access')
    seen_profile_pids={int(p) for s in profile.get('samples',[]) for p in s.get('cuda_pids',[])}-{TARGETS[0]['cuda_pid']}
    require(seen_profile_pids,'profile CUDA PID evidence missing')
    for pid in seen_profile_pids:
        require(pid not in processes and all(int(x['pid'])!=pid for x in gpu_processes),f'profile CUDA PID still alive: {pid}')
    require(all('oev1_concurrent_profile_s0' not in p.get('cmd','') and str(PROFILE_RUN) not in p.get('cmd','') for p in processes.values()),'profile launcher or loader still alive')
    require(all(l.get('job_id')!='oev1_concurrent_profile_s0' for l in state['leases'].values()),'profile lease has not been released')
    changed=copy.deepcopy(state)
    records=[]
    for t in TARGETS:
        l=state['leases'].get(t['lease_id'])
        require(l is not None,f'missing exact lease: {t["lease_id"]}')
        require(l.get('job_id')==t['job_id'] and l.get('job_pid')==t['guard_pid'] and l.get('owner_pid')==t['guard_pid'],'lease PID/job identity mismatch')
        require(l.get('gpus')==[t['gpu']] and l.get('kind')=='train' and l.get('formal_train') is True,'lease device/kind mismatch')
        require(l.get('expected_vram_mib')==10000 and l.get('expected_rss_mib')==49152,'reservation already revised or unexpected initial values')
        require(t['guard_pid'] in processes and t['cuda_pid'] in processes,'full train/guard PID disappeared')
        gc=processes[t['guard_pid']].get('cmd','');tc=processes[t['cuda_pid']].get('cmd','')
        require('project_resource_guard.py run' in gc and '--job-id '+t['job_id']+' ' in gc,'guard command mismatch')
        require(t['run'] in gc and 'train_object_evidence.py' in tc and '--output '+t['run']+' ' in tc,'full training command mismatch')
        require('--arm '+t['arm']+' ' in tc and '--seed '+str(t['seed'])+' ' in tc,'student arm/seed mismatch')
        require(is_descendant(t['cuda_pid'],t['guard_pid'],processes),'CUDA PID not descendant of guard')
        matching=[x for x in gpu_processes if int(x['gpu'])==t['gpu']]
        require(len(matching)==1 and int(matching[0]['pid'])==t['cuda_pid'],'target GPU no longer single expected CUDA PID')
        require(t['cuda_pid'] in l.get('observed_cuda_pids',[]),'lease has not observed expected CUDA PID')
        require(l.get('peak_cuda_pid_counts',{}).get(str(t['gpu']))==1,'unexpected historical CUDA PID multiplicity')
        require(gpus[t['gpu']]['memory_free_mib']>=2048,'current physical free VRAM below 2 GiB')
        p=progress[t['run']]
        require(p.get('status')=='running' and p.get('arm')==t['arm'] and 2<=p.get('epoch',0)<200,'full training progress is not active beyond startup')
        require(p.get('optimizer_updates',0)>24,'not a full-training profile')
        live_vram=int(matching[0]['used_mib'])
        peak_vram=max(int(l.get('per_gpu_peak_vram_mib',{}).get(str(t['gpu']),0)),live_vram)
        live_rss=sum(int(row['rss_mib']) for pid,row in processes.items() if is_descendant(pid,t['guard_pid'],processes))
        peak_rss=max(int(l.get('peak_rss_mib',0)),live_rss)
        require(peak_vram>0 and VRAM-peak_vram>=512,f'VRAM headroom <512 MiB for {t["job_id"]}: {peak_vram}')
        require(peak_rss>0 and RSS-peak_rss>=1024,f'RSS headroom <1024 MiB for {t["job_id"]}: {peak_rss}')
        revised=changed['leases'][t['lease_id']]
        revised['expected_vram_mib']=VRAM;revised['expected_rss_mib']=RSS
        records.append({'identity':t,'before':{'expected_vram_mib':10000,'expected_rss_mib':49152},
                        'after':{'expected_vram_mib':VRAM,'expected_rss_mib':RSS},'observed_peak_vram_mib':peak_vram,
                        'observed_peak_rss_mib':peak_rss,'current_tree_rss_mib':live_rss,'full_progress':p,
                        'vram_margin_mib':VRAM-peak_vram,'rss_margin_mib':RSS-peak_rss})
    # Explicit proof that only four reservation scalar values change.
    restored=copy.deepcopy(changed)
    for t in TARGETS:
        restored['leases'][t['lease_id']]['expected_vram_mib']=10000
        restored['leases'][t['lease_id']]['expected_rss_mib']=49152
    require(restored==state,'unexpected non-reservation mutation')
    return changed,records

def load(path):return json.loads(path.read_text())

def evidence_inputs(module):
    processes=module['process_snapshot']()
    for pid,row in processes.items():
        try:row['cmd']=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):row['cmd']=''
    return dict(processes=processes,gpu_processes=module['gpu_process_snapshot'](),gpus=module['gpu_snapshot'](),
                profile=load(ART/'concurrency_profile.json'),summary=load(ART/'concurrency_summary.json'),
                receipt=load(PROFILE_RUN/'completion_receipt.json'),progress={t['run']:load(Path(t['run'])/'progress.json') for t in TARGETS})

def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--revision-id',default='profiled_reservation_revision_20260906')
    args=parser.parse_args()
    require(args.revision_id and all(x.isalnum() or x in '_-' for x in args.revision_id),'unsafe revision-id')
    module=runpy.run_path(str(REPO/'tools/project_resource_guard.py'))
    guard=module['ProjectResourceGuard'](LEASE_FILE)
    if not args.apply:
        before=load(LEASE_FILE);after,records=validate_revision(before,**evidence_inputs(module))
        print(json.dumps({'status':'DRY_RUN_PASSED','changes':records,'note':'No file or lease changed; revalidate under lock on --apply.'},indent=2));return
    paths={suffix:ART/(args.revision_id+'_'+suffix+'.json') for suffix in ['before','intent','after','receipt']}
    require(all(not p.exists() for p in paths.values()),'revision evidence already exists; do not repeat blindly')
    # The exact shared guard lock prevents concurrent admission/lease replacement.
    with guard._locked_state() as state:
        inputs=evidence_inputs(module)
        before=copy.deepcopy(state);after,records=validate_revision(before,**inputs)
        created=datetime.datetime.now(datetime.timezone.utc).isoformat()
        intent={'status':'VALIDATED_BEFORE_COMMIT','captured_at_utc':created,'changes':records,
                'reason':'Measured full-training peaks plus a passed completed concurrent GPU2 canary; only reservations revised. No model, optimizer, checkpoint, PID, binding, guard policy or historical admission changed.',
                'concurrency_summary':inputs['summary'],'profile_source':str(ART/'concurrency_profile.json'),
                'lease_file':str(LEASE_FILE),'source_script':str(Path(__file__).resolve())}
        write_new(paths['before'],before);write_new(paths['intent'],intent)
        for t in TARGETS:
            state['leases'][t['lease_id']]['expected_vram_mib']=VRAM
            state['leases'][t['lease_id']]['expected_rss_mib']=RSS
    # The existing context manager committed its atomic replacement on exit.
    write_new(paths['after'],after)
    receipt={'status':'COMMITTED','captured_at_utc':created,'changes':records,
             'only_changed_fields':['expected_vram_mib','expected_rss_mib'],'lease_count_unchanged':len(before['leases'])==len(after['leases']),
             'evidence':{k:str(v) for k,v in paths.items()}}
    write_new(paths['receipt'],receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
