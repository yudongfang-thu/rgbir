"""Two-arm subset check using only the existing global lease and bounded execution."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

from budget import ARMS,LIMIT_SECONDS,estimate,reservation
from screen_common import ENDPOINT,load_config,read,stat,write_new

SCOPE='DRONE_SUBSET2048_PRETRAIN_E8_CHECK'
BASE=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python')
REFERENCE=BASE/'rgbir_independent_kd_v2_20260907/release_gpu5'
DISPATCH=BASE/'rgbir_task_conditional_v1_20260907/release_v8'
NATIVE_BINDING=BASE/'rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evaluation_profile_binding/binding.json'

def require(condition,message):
    if not condition:raise ValueError(message)

def check_identity(r,arm,status):
    fields=dict(status=status,scope=SCOPE,endpoint=ENDPOINT,arm=arm,seed=42,single_seed=True,
        formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False)
    for k,v in fields.items():require(r.get(k)==v,'Stage identity differs: '+k)

def check_canary(path,arm,config,reference):
    r=read(path);check_identity(r,arm,'SUBSET_SCREEN_CANARY_COMPLETED')
    require(r['successful_updates']==r['optimizer_updates']>=24,'Canary lacks24 successful updates')
    require(24<=r['attempts']<=48 and r['batches']>=30,'Canary attempt/prefix budget differs')
    require(r['attempts']==r['optimizer_updates']+r['amp_skips'],'Attempt accounting differs')
    require(Path(r['configuration']).read_bytes()==config.read_bytes(),'Canary config differs')
    flow=read(r['flow_prefix_receipt']);initial=read(r['initialization_receipt'])
    require(flow['status']=='PASS' and flow['prefix_batches']==30 and flow['exact_pixels_and_labels'] is True,
        'Full first30 actual pixel/label comparison required')
    require(Path(flow['reference_dir']).resolve()==reference.resolve(),'Wrong within-attempt flow reference')
    require(initial['status']=='PASS','Actual initialization verification failed')
    if arm=='N':require(initial.get('state_reference_written_and_verified') is True,'N reference not readback verified')
    else:require(initial.get('cross_arm_initial_state_exact') is True,'Cross-arm initial tensors differ')
    if arm=='C0':require(r.get('nonzero_kd_gradient') is True and r['selected_objects']>0,'C0 signal not demonstrated')
    budget=reservation(r)
    require(budget['vram_mib']<=8192 and budget['rss_mib']<=32768,'Measured canary plus margin exceeds initial ceiling')
    return r,dict(receipt=stat(path),flow=stat(r['flow_prefix_receipt']),initialization=stat(r['initialization_receipt']),reservation=budget)

def verify_eval_resource_basis():
    path=NATIVE_BINDING.parent.parent/'evidence_metrics.json';r=read(path)
    require(r.get('status')=='completed' and r.get('observed_images')==1469,'Existing full-dev evaluator evidence missing')
    require(r.get('official_test_accessed') is False,'Wrong evaluation evidence scope')
    v=list(r['resources']['per_gpu_peak_vram_mib'].values());rss=r['resources']['peak_rss_mib']
    require(v and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in v+[rss]),'Missing finite measured peaks')
    require(max(v)<=2048 and rss<=8192,'Evaluator reservation below existing measured peaks')
    return dict(source=stat(path),vram_mib=2048,rss_mib=8192,measured_nvml_peak_mib=max(v),measured_rss_peak_mib=rss)

class ExecutionClock:
    def __init__(self):
        self.started=time.monotonic();self.excluded_admission_seconds=0.;self.stages=[]
    def spent(self):return max(0.,time.monotonic()-self.started-self.excluded_admission_seconds)
    def remaining(self):return max(0.,LIMIT_SECONDS-self.spent())

def execute(dispatch,job,queue,clock):
    """Observe original status writes; only this job has an execution deadline."""
    require(clock.remaining()>0,'Execution budget exhausted before stage')
    t0=time.monotonic();limit=clock.remaining();state={};done=threading.Event()
    original_dump=dispatch.dump
    status_path=queue/(job['id']+'_status.json')
    def observe(path,value):
        original_dump(path,value)
        if Path(path)==status_path and value.get('status')=='RUNNING':
            require('running_at' not in state,'Duplicate launch for one stage')
            state.update(running_at=time.monotonic(),running_wall=value['time'],launch=dict(value['launch']))
            clock.excluded_admission_seconds+=state['running_at']-t0
    def watchdog():
        while not done.wait(.25):
            if 'running_at' in state and time.monotonic()-state['running_at']>=limit:
                state['timed_out']=True
                write_new(queue/(job['id']+'_budget_timeout.json'),dict(status='STOPPED_ON_EXECUTION_BUDGET',
                    job_id=job['id'],stage_execution_limit_seconds=limit,owned_pid=state['launch']['pid'],
                    scientific_negative_result=False,new_hash_computed=False))
                dispatch.stop_owned_tree(state['launch']['pid']);return
    dispatch.dump=observe
    thread=threading.Thread(target=watchdog,daemon=True);thread.start()
    try:
        report=dispatch.run_job(job,queue)
        require(not state.get('timed_out'),'Stage exceeded remaining execution budget')
        require(report.get('status')=='COMPLETED' and report.get('exit_code')==0 and not report.get('monitor_errors'),
            'Dispatcher did not report a clean stage')
        require('running_at' in state,'Actual RUNNING observation missing')
        return report
    finally:
        done.set();thread.join(timeout=6);dispatch.dump=original_dump
        elapsed=time.monotonic()-t0
        active=max(0.,time.monotonic()-state['running_at']) if 'running_at' in state else 0.
        if 'running_at' not in state:clock.excluded_admission_seconds+=elapsed
        row=dict(job_id=job['id'],stage=job['stage'],arm=job['arm'],wall_seconds=elapsed,
            observed_execution_seconds=active,admission_wait_seconds=elapsed-active,
            execution_limit_seconds=limit,budget_timed_out=state.get('timed_out',False),
            actual_launch_observed='running_at' in state,new_hash_computed=False)
        clock.stages.append(row);write_new(queue/(job['id']+'_timing.json'),row)

def make_job(release,output,arm,stage,remaining,resources=None):
    config=release/'configs'/('drone_'+arm+'_s42_E8.yaml')
    canary=output/'canaries'/arm;run=output/'runs'/arm;flow=output/'canaries/N/flow_reference'
    job=dict(id='subset_e8_'+output.name+'_'+arm+'_'+stage,arm=arm,stage=stage,
        kind='eval' if stage=='eval' else 'train',formal=False)
    if stage in ('canary','train'):
        dest=canary if stage=='canary' else run
        command=[str(PY),str(release/'train_short_screen.py'),'--reference-dir',str(REFERENCE),
            '--config',str(config),'--output',str(dest),'--flow-reference-dir',str(flow),
            '--amp-prior',str(release/'validated_amp_prior.json'),'--wall-seconds',str(remaining)]
        if stage=='canary':
            command+=['--canary']
            if arm=='N':command+=['--write-flow-reference']
            receipt=dest/'canary.json';vram,rss=8192,32768
        else:
            command+=['--canary-receipt',str(canary/'canary.json')]
            receipt=dest/'short_training_receipt.json';vram,rss=resources['vram_mib'],resources['rss_mib']
    else:
        require(stage=='eval','Unknown stage');dest=output/'evaluations'/arm
        command=[str(PY),str(release/'evaluate_short_screen.py'),'--reference-dir',str(REFERENCE),
            '--run',str(run),'--output',str(dest),'--native-profile-binding',str(NATIVE_BINDING),
            '--wall-seconds',str(remaining)]
        receipt=dest/'short_evaluation_receipt.json';vram,rss=2048,8192
    job.update(command=command,expected_receipt=str(receipt),vram_mib=vram,rss_mib=rss)
    return job

def check_terminal(path,arm,stage,config):
    r=read(path);check_identity(r,arm,'SUBSET_SCREEN_TRAINING_COMPLETED' if stage=='train' else 'SUBSET_SCREEN_EVALUATION_COMPLETED')
    if stage=='train':
        require(r['batches']==512 and r['epochs_configured']==8 and r['last_epoch']==8,'Incomplete fixed E8 exposure')
        require(r['attempts']==r['optimizer_updates']+r['amp_skips'] and r['optimizer_updates']>=24,'Training update accounting differs')
        require(Path(r['configuration']).read_bytes()==config.read_bytes(),'Training config changed')
        flow=read(r['flow_prefix_receipt']);initial=read(r['initialization_receipt'])
        require(flow['status']=='PASS' and flow['prefix_batches']==30 and flow['exact_pixels_and_labels'] is True,'Training stream differs')
        require(initial['status']=='PASS' and initial['cross_arm_initial_state_exact'] is True,'Training fresh state differs')
    else:
        require(r['epochs']==8 and r['full_dev_images']==1469 and r['full_dev_gt_objects']==22462,'Independent full-dev endpoint differs')
    return r

def run(args):
    release=args.release_dir.resolve();output=args.output.resolve()
    require(Path('/mnt/dataset/yudongfang') in output.parents and not output.exists(),'Fresh data-disk attempt required')
    require(shutil.disk_usage(output.parent).free>=6*2**30,'Need at least6GiB for private first30 inputs and outputs')
    configs={a:release/'configs'/('drone_'+a+'_s42_E8.yaml') for a in ARMS}
    cfg={a:load_config(p) for a,p in configs.items()};frozen={a:p.read_bytes() for a,p in configs.items()}
    require(all(cfg[a]['arm']==a for a in ARMS),'Config arm names differ')
    eval_basis=verify_eval_resource_basis()
    sys.path.insert(0,str(DISPATCH));import resource_dispatch as dispatch
    require(Path(dispatch.__file__).resolve()==DISPATCH/'resource_dispatch.py','Wrong original dispatcher')
    output.mkdir(parents=True,exist_ok=False);queue=output/'queue';queue.mkdir()
    shutil.copyfile(Path(__file__),queue/'executed_driver.py')
    require(Path(__file__).read_bytes()==(queue/'executed_driver.py').read_bytes(),'Driver copy changed')
    write_new(queue/'manifest.json',dict(scope=SCOPE,endpoint=ENDPOINT,arms=list(ARMS),
        sequence='both canaries -> measured budget -> N train/eval -> C0 train/eval',
        limit_execution_seconds=LIMIT_SECONDS,queue_admission_wait_separate=True,
        dispatcher=stat(DISPATCH/'resource_dispatch.py'),python=str(PY),eval_basis=eval_basis,
        private_pixel_reference='server_only_first30_no_hash_no_upload',new_scheduler_created=False,
        new_hash_computed=False,ap_adaptive=False))
    clock=ExecutionClock();completed=[];status='RUNNING';error=None
    try:
        canaries={};checks={}
        for arm in ARMS:
            job=make_job(release,output,arm,'canary',clock.remaining());write_new(queue/(job['id']+'_job.json'),job)
            execute(dispatch,job,queue,clock)
            canaries[arm],checks[arm]=check_canary(Path(job['expected_receipt']),arm,configs[arm],output/'canaries/N/flow_reference')
            completed.append(dict(arm=arm,stage='canary',receipt=job['expected_receipt'],validated=True))
        write_new(queue/'canary_checks.json',dict(status='BOTH_CANARIES_VALID',arms=checks,new_hash_computed=False))
        cost=estimate(canaries,clock.spent());write_new(queue/'budget_decision.json',cost)
        if cost['status']!='PASS_WITHIN_EXECUTION_BUDGET':status='SUBSET_SCREEN_BLOCKED_BUDGET'
        else:
            for arm in ARMS:
                for stage in ('train','eval'):
                    require(configs[arm].read_bytes()==frozen[arm],'Frozen source config changed')
                    job=make_job(release,output,arm,stage,clock.remaining(),checks[arm]['reservation'])
                    write_new(queue/(job['id']+'_job.json'),job);execute(dispatch,job,queue,clock)
                    check_terminal(Path(job['expected_receipt']),arm,stage,configs[arm])
                    completed.append(dict(arm=arm,stage=stage,receipt=job['expected_receipt'],validated=True))
            require(clock.spent()<=LIMIT_SECONDS,'Final matrix exceeded execution budget')
            status='SUBSET_SCREEN_MATRIX_COMPLETED'
    except BaseException as exc:
        status='SUBSET_SCREEN_STOPPED_INCOMPLETE';error=dict(error=repr(exc),traceback=traceback.format_exc())
    result=dict(status=status,scope=SCOPE,completed=completed,error=error,
        execution_seconds=clock.spent(),admission_wait_seconds=clock.excluded_admission_seconds,
        total_wall_seconds=time.monotonic()-clock.started,stage_timings=clock.stages,
        single_seed=True,seed=42,epochs=8,formal_e200_complete=False,formal_paper_gain_claim=False,
        automatic_retry=False,ap_adaptive=False,new_hash_computed=False)
    write_new(queue/'completion.json',result)
    return 0 if status in ('SUBSET_SCREEN_MATRIX_COMPLETED','SUBSET_SCREEN_BLOCKED_BUDGET') else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
