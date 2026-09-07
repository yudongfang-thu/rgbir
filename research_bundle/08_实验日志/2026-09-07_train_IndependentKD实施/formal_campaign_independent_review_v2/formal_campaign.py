"""Persistent C1 three-seed coordinator, outside the frozen training release.

Preparation does not launch. Workers call the real release run_job separately
for train then evaluation. Only a restrictive campaign admission gate is added
in process memory; GPU selection and resource leases remain the real guard's.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import yaml

ORDER = (42,0,123)
POLL_SECONDS = 30
VRAM_MARGIN_MIB = 256
RSS_MARGIN_MIB = 4096
RSS_MINIMUM_MIB = 8192
SCHEMA = 'rgbir-persistent-c1-campaign-v1'


class AdmissionPaused(RuntimeError):pass


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def event(campaign,value):
    with (Path(campaign)/'campaign_events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time=time.time(),**value),ensure_ascii=False,allow_nan=False)+'\n')


def save_state(campaign,state):
    # Only this derived scheduler state is replaced. Raw results/logs are never replaced.
    path=Path(campaign)/'state.json';temporary=path.with_name('state.tmp.'+str(os.getpid()))
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(state,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


@contextmanager
def lock(path,nonblocking=False):
    if os.name!='posix':raise RuntimeError('Persistent workers require the Linux server; CPU tests do not launch them')
    import fcntl
    with Path(path).open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def module_from_release(release,name):
    path=(Path(release)/(name+'.py')).resolve()
    if name in sys.modules:
        existing=sys.modules[name]
        if Path(existing.__file__).resolve()!=path:raise ValueError('A different release module is loaded: '+name)
        return existing
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def require_data_path(path):
    path=Path(path).resolve()
    if os.name!='posix' or not str(path).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Campaign output must be on /mnt/dataset/yudongfang')
    return path


def discover_configs(directory):
    configs={}
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.yaml','.yml','.json'):continue
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(cfg,dict) or cfg.get('arm')!='C1':continue
        seed=cfg.get('seed')
        if seed not in ORDER:continue
        if seed in configs:raise ValueError('Duplicate C1 seed config: '+str(seed))
        if (cfg.get('source')!='paired' or cfg.get('epochs')!=200 or cfg.get('protocol_status')!='FROZEN'
                or cfg.get('formal_training_authorized') is not True):
            raise ValueError('Only real frozen, authorized C1 paired E200 configurations are eligible')
        configs[seed]=(path.resolve(),cfg)
    if set(configs)!=set(ORDER):raise ValueError('Require exactly one real C1 config for seeds42/0/123')
    return configs


def profile_resources(profile_path,expected_stage):
    profile=read(profile_path)
    if (profile.get('schema')!='rgbir-independent-resource-profile-v1' or profile.get('status')!='COMPLETED'
            or profile.get('measurement_valid') is not True or profile.get('stage')!=expected_stage):
        raise ValueError('Require an actual successful '+expected_stage+' resource profile')
    reservation=profile.get('reservation',{})
    resources=profile.get('resources',{})
    peaks=dict(vram_mib=max(resources.get('per_gpu_peak_vram_mib',{}).values(),default=0),
               rss_mib=resources.get('peak_rss_mib',0))
    for key,measured in peaks.items():
        if (type(reservation.get(key)) is not int or type(measured) not in (int,float)
                or not math.isfinite(measured) or not 0<measured<=reservation[key]):
            raise ValueError('Profile lacks its actual sufficient reservation: '+key)
    # A bootstrap cap is not a measurement. Reserve from actual measured peaks
    # with these frozen margins; the real dispatcher still checks profile binding.
    return profile,dict(vram_mib=math.ceil(peaks['vram_mib'])+VRAM_MARGIN_MIB,
        rss_mib=max(RSS_MINIMUM_MIB,math.ceil(peaks['rss_mib'])+RSS_MARGIN_MIB))


def build_jobs(release,campaign,python,seed,config,train_profile,eval_profile,train_resources,eval_resources,train_key,eval_key):
    release,campaign=Path(release),Path(campaign)
    run=campaign/'runs'/('C1_seed'+str(seed));prefix='ikdv2_C1_'+str(seed)
    common=dict(queue_owner='independent_v2',config=str(config))
    train=dict(common,id=prefix+'_train_a1',kind='train',stage='train',formal=True,
        profile_key=train_key,requires_profile=str(train_profile),result_receipt=str(run/'completion_receipt.json'),
        command=[str(python),str(release/'train_independent.py'),'--config',str(config),'--output',str(run),
                 '--arm','C1','--source','paired','--seed',str(seed)],**train_resources)
    evaluation=dict(common,id=prefix+'_eval_a1',kind='eval',stage='evaluation',formal=False,
        profile_key=eval_key,requires_profile=str(eval_profile),result_receipt=str(run/'evaluation_val.json'),
        command=[str(python),str(release/'evaluate_independent.py'),'--config',str(config),'--run',str(run),'--attempt','1'],
        **eval_resources)
    return dict(seed=seed,screen='ikdv2_C1_'+str(seed),config=str(config),run=str(run),train=train,evaluation=evaluation)


def initialize(args):
    campaign=require_data_path(args.campaign)
    release=Path(args.release).resolve();configs=discover_configs(args.formal_config_dir)
    dispatch=module_from_release(release,'resource_dispatch')
    protocol=module_from_release(release,'protocol');admission=module_from_release(release,'admission')
    train,tr=profile_resources(args.canary_profile,'canary')
    evaluation,er=profile_resources(args.evaluation_profile,'evaluation_profile')
    seeds=[]
    for seed in ORDER:
        path,cfg=configs[seed]
        protocol.require_valid_config(cfg,formal=True);admission.check_readiness(cfg,release)
        row=build_jobs(release,campaign,Path(args.python).resolve(),seed,path,
                       Path(args.canary_profile).resolve(),Path(args.evaluation_profile).resolve(),tr,er,
                       train['profile_key'],evaluation['profile_key'])
        dispatch.measured_reservation(row['train']);dispatch.measured_reservation(row['evaluation'])
        seeds.append(row)
    if campaign.exists():raise FileExistsError('Preserve existing campaign; coordinate it rather than reinitializing')
    campaign.mkdir(parents=True,exist_ok=False);(campaign/'dispatch').mkdir();(campaign/'runs').mkdir()
    snapshots=[]
    originals=[Path(__file__).resolve(),*sorted({p for p,_ in configs.values()}),Path(args.canary_profile).resolve(),Path(args.evaluation_profile).resolve()]
    folder=campaign/'frozen_inputs';folder.mkdir()
    for index,source in enumerate(originals):
        destination=folder/f'{index:03d}_{source.name}';destination.write_bytes(source.read_bytes())
        snapshots.append(dict(original=str(source),copy=str(destination)))
    manifest=dict(schema=SCHEMA,release=str(release),repo=str(Path(args.repo).resolve()),
        python=str(Path(args.python).resolve()),worker_script=str(Path(__file__).resolve()),seed_order=list(ORDER),
        seeds=seeds,input_snapshots=snapshots,status='PREPARED_NOT_LAUNCHED',no_ap_early_stop=True,
        measured_reservation_policy=dict(vram_margin_mib=VRAM_MARGIN_MIB,rss_margin_mib=RSS_MARGIN_MIB,
            rss_minimum_mib=RSS_MINIMUM_MIB,rounding='ceil each actual measured peak in MiB'),
        evaluation_priority='at each atomic resource admission; running evaluation does not block later training')
    write_new(campaign/'manifest.json',manifest)
    write_new(campaign/'state.json',dict(paused=False,pause_reason=None,seeds={str(s):dict(status='NOT_STARTED') for s in ORDER}))
    event(campaign,dict(status='PREPARED',seeds=list(ORDER),training_launched=False))
    return manifest


def validated_manifest(campaign):
    manifest=read(Path(campaign)/'manifest.json')
    if manifest.get('schema')!=SCHEMA or manifest.get('seed_order')!=list(ORDER):raise ValueError('Unsupported campaign identity')
    for row in manifest['input_snapshots']:
        if Path(row['original']).read_bytes()!=Path(row['copy']).read_bytes():raise ValueError('Frozen campaign input changed: '+row['original'])
    return manifest


def launched(campaign,job):
    path=Path(campaign)/'dispatch'/(job['id']+'_events.jsonl')
    if not path.is_file():return False
    for line in path.read_text(encoding='utf-8').splitlines():
        try:record=json.loads(line)
        except json.JSONDecodeError:continue  # writer may currently be appending its last line
        if record.get('status')=='LAUNCHED':return True
    return False


def complete_training(row):
    path=Path(row['train']['result_receipt'])
    if not path.exists():return False
    value=read(path)
    return (value.get('status')=='training_completed' and value.get('last_epoch')==200
            and value.get('epochs_configured')==200 and value.get('seed')==row['seed']
            and value.get('arm')=='C1' and value.get('source')=='paired')


def complete_evaluation(campaign,row):
    path=Path(row['evaluation']['result_receipt'])
    profile=Path(campaign)/'dispatch'/(row['evaluation']['id']+'_resource_profile.json')
    if not path.is_file() or not profile.is_file():return False
    try:value,measurement=read(path),read(profile)
    except json.JSONDecodeError:return False  # either writer may be publishing its final receipt
    return (value.get('status')=='completed' and value.get('seed')==row['seed'] and value.get('arm')=='C1'
        and value.get('endpoint')=='fixed_budget_last_ema' and value.get('official_test_accessed') is False
        and measurement.get('status')=='COMPLETED' and measurement.get('measurement_valid') is True)


def require_training_measurement(campaign,row):
    """A workload completion alone cannot clear an interrupted/failed guard."""
    path=Path(campaign)/'dispatch'/(row['train']['id']+'_resource_profile.json')
    if not path.is_file():raise RuntimeError('Completed training lacks its successful resource receipt; technical review required')
    measurement=read(path)
    if (measurement.get('status')!='COMPLETED' or measurement.get('measurement_valid') is not True
            or measurement.get('stage')!='train' or measurement.get('result_receipt')!=row['train']['result_receipt']):
        raise RuntimeError('Completed training has an invalid resource receipt; technical review required')


def pending_evaluations(campaign,manifest):
    pending=[]
    for row in manifest['seeds']:
        try:done=complete_training(row)
        except json.JSONDecodeError:done=True  # completion is being published: defer new training safely
        if done and not complete_evaluation(campaign,row) and not launched(campaign,row['evaluation']):pending.append(row['seed'])
    return pending


def training_gate_reason(campaign,manifest,state,seed):
    if state.get('paused'):return 'PAUSED: '+str(state.get('pause_reason'))
    pending=pending_evaluations(campaign,manifest)
    if pending:return 'Pending evaluation has priority: '+str(pending)
    previous=ORDER[:ORDER.index(seed)]
    rows={r['seed']:r for r in manifest['seeds']}
    if any(not launched(campaign,rows[s]['train']) for s in previous):return 'Earlier seed has no actual training LAUNCHED event'
    return None


def install_campaign_gate(dispatch,campaign,manifest,seed):
    original=dispatch.atomic_acquire
    def gated(guard,guard_module,job,reservation,*args,**kwargs):
        with lock(Path(campaign)/'admission.lock'):
            state=read(Path(campaign)/'state.json')
            if job['stage']=='train':
                reason=training_gate_reason(campaign,manifest,state,seed)
                if reason and reason.startswith('PAUSED:'):raise AdmissionPaused(reason)
                if reason:raise guard_module.ResourceUnavailable([reason])
            # The real dispatcher acquires the actual shared lease under its own
            # lock, with a fresh GPU snapshot. This wrapper only delays admission.
            return original(guard,guard_module,job,reservation,*args,**kwargs)
    dispatch.atomic_acquire=gated
    return original


def update(campaign,seed,status,**extra):
    with lock(Path(campaign)/'admission.lock'):
        state=read(Path(campaign)/'state.json');state['seeds'][str(seed)].update(status=status,**extra)
        save_state(campaign,state);event(campaign,dict(seed=seed,status=status,**extra))


def pause(campaign,seed,error):
    with lock(Path(campaign)/'admission.lock'):
        state=read(Path(campaign)/'state.json');state.update(paused=True,pause_reason=f'seed{seed}: {error}')
        state['seeds'][str(seed)].update(status='TECHNICAL_FAILURE',error=str(error))
        save_state(campaign,state);event(campaign,dict(seed=seed,status='TECHNICAL_FAILURE',error=str(error)))


def worker(campaign,seed):
    campaign=Path(campaign).resolve();manifest=validated_manifest(campaign)
    row=next(r for r in manifest['seeds'] if r['seed']==seed)
    with lock(campaign/('worker_'+str(seed)+'.lock'),nonblocking=True):
        dispatch=module_from_release(manifest['release'],'resource_dispatch')
        if dispatch.POLL_SECONDS!=30:raise ValueError('Expected actual dispatcher resource queue interval of30 seconds')
        guard=dispatch.load_guard(Path(manifest['repo']))
        original=install_campaign_gate(dispatch,campaign,manifest,seed)
        try:
            if complete_evaluation(campaign,row):update(campaign,seed,'DONE');return
            if not complete_training(row):
                # No automatic restart/overwrite of an incomplete E200 attempt.
                if Path(row['run']).exists():raise RuntimeError('Incomplete prior training attempt requires explicit technical review')
                update(campaign,seed,'TRAIN_WAITING',worker_pid=os.getpid())
                dispatch.run_job(row['train'],campaign/'dispatch',Path(manifest['repo']),guard)
                if not complete_training(row):raise RuntimeError('Training returned without the exact E200 completion')
            require_training_measurement(campaign,row)
            update(campaign,seed,'EVAL_PENDING',worker_pid=os.getpid())
            # Sequential call is deliberate: ordered_jobs would incorrectly sort
            # this dependent evaluation before its own training.
            if Path(row['evaluation']['result_receipt']).exists():
                raise RuntimeError('Published evaluation without completed resource receipt requires reconciliation; do not re-infer')
            dispatch.run_job(row['evaluation'],campaign/'dispatch',Path(manifest['repo']),guard)
            if not complete_evaluation(campaign,row):raise RuntimeError('Evaluation obligation not completed')
            update(campaign,seed,'DONE')
        except AdmissionPaused as error:
            update(campaign,seed,'DEFERRED_PAUSED',reason=str(error))
        except BaseException as error:
            pause(campaign,seed,error)
            path=campaign/f'worker_{seed}_failure_{time.time_ns()}.json'
            write_new(path,dict(status='FAILED',seed=seed,error=repr(error),traceback=traceback.format_exc(),
                               evaluation_obligation_pending=not complete_evaluation(campaign,row)))
            raise
        finally:dispatch.atomic_acquire=original


def screen_names():
    result=subprocess.run(['screen','-ls'],capture_output=True,text=True,check=False)
    if result.returncode not in (0,1):raise RuntimeError('Cannot inspect screen sessions: '+result.stderr)
    return set(re.findall(r'\d+\.([^\s]+)',result.stdout))


def next_seed_to_launch(campaign,manifest,state):
    if state.get('paused') or pending_evaluations(campaign,manifest):return None
    for seed in ORDER:
        if state['seeds'][str(seed)]['status']=='NOT_STARTED':
            return seed if training_gate_reason(campaign,manifest,state,seed) is None else None
    return None


def coordinate(campaign):
    campaign=Path(campaign).resolve();manifest=validated_manifest(campaign)
    with lock(campaign/'coordinator.lock',nonblocking=True):
        while True:
            with lock(campaign/'admission.lock'):
                state=read(campaign/'state.json')
                if all(complete_evaluation(campaign,row) for row in manifest['seeds']):
                    event(campaign,dict(status='CAMPAIGN_COMPLETED'));return
                active=screen_names()
                for row in manifest['seeds']:
                    status=state['seeds'][str(row['seed'])]['status']
                    if status in ('SCREEN_REQUESTED','TRAIN_WAITING','EVAL_PENDING') and row['screen'] not in active:
                        state.update(paused=True,pause_reason=f"Worker screen disappeared: {row['screen']}; preserve attempt and review")
                        event(campaign,dict(status='MISSING_WORKER',seed=row['seed'],evaluation_obligation_pending=not complete_evaluation(campaign,row)))
                seed=next_seed_to_launch(campaign,manifest,state)
                if seed is not None:
                    row=next(r for r in manifest['seeds'] if r['seed']==seed)
                    if row['screen'] in active:raise RuntimeError('Reserved worker screen already exists; do not adopt an unrelated campaign')
                    state['seeds'][str(seed)].update(status='SCREEN_REQUESTED')
                    save_state(campaign,state)
                    command=['screen','-dmS',row['screen'],manifest['python'],manifest['worker_script'],
                             'worker','--campaign',str(campaign),'--seed',str(seed)]
                    # Keep campaign admission lock through screen creation. The
                    # new worker waits for it, so it cannot race launch bookkeeping.
                    try:subprocess.run(command,check=True)
                    except BaseException as error:
                        state.update(paused=True,pause_reason='screen launch failed: '+repr(error));save_state(campaign,state);raise
                    event(campaign,dict(status='SCREEN_STARTED',seed=seed,screen=row['screen']))
                else:save_state(campaign,state)
            time.sleep(POLL_SECONDS)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    init=sub.add_parser('prepare')
    for name in ('campaign','release','repo','formal-config-dir','canary-profile','evaluation-profile','python'):
        init.add_argument('--'+name,type=Path,required=True)
    for action in ('coordinate','worker'):
        p=sub.add_parser(action);p.add_argument('--campaign',type=Path,required=True)
        if action=='worker':p.add_argument('--seed',type=int,choices=ORDER,required=True)
    args=parser.parse_args()
    if args.action=='prepare':initialize(args);print('PREPARED_NOT_LAUNCHED')
    elif args.action=='coordinate':coordinate(args.campaign)
    else:worker(args.campaign,args.seed)


if __name__=='__main__':main()
