"""Run frozen paired_random controls on one GPU, checking each canary before full training."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
BASE=ROOT/'artifacts/rgbir_oev1_random_20260906'
SOURCE=BASE/'release_v1'
RUNS=ROOT/'runs/rgbir_oev1_random_20260906'
PY=str(REPO/'environments/sn6-int8-kd/bin/python')
GUARD=str(REPO/'tools/project_resource_guard.py')
GPU=2
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')

def write_status(**values):
    (BASE/'queue_status.json').write_text(json.dumps({'pid':os.getpid(),'gpu':GPU,'time':time.time(),**values},indent=2))

def admission_policy():
    current=json.loads(subprocess.check_output([PY,GUARD,'inspect'],text=True))
    active=set(current['usage']['active_gpus'])|{GPU}
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    empty={int(x.split(',')[0]) for x in raw.splitlines() if int(x.split(',')[1])<100}
    return len(active)<=3 or (len(active)<=4 and len(empty-active)>=2)

def guarded(job,kind,command,nonformal=False):
    cmd=[PY,GUARD,'run','--job-id',job,'--kind',kind,'--candidate-gpu',str(GPU),
         '--expected-vram-mib','10000','--expected-rss-mib','49152','--free-safety-mib','2048']
    if nonformal:cmd+=['--non-formal-train']
    elif kind=='train':cmd+=['--formal-train']
    cmd+=['--',*command]
    with (BASE/(job+'.log')).open('x') as log:
        while True:
            if not admission_policy():
                log.write('Waiting for user AGENTS 2.1 empty-card condition\n');log.flush();time.sleep(30);continue
            launched=queued=False
            proc=subprocess.Popen(cmd,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                log.write(line);log.flush()
                try:event=json.loads(line)
                except ValueError:continue
                if isinstance(event,dict):
                    launched|=event.get('status')=='LAUNCHED';queued|=event.get('status')=='QUEUED'
            rc=proc.wait()
            if rc==2 and queued and not launched:time.sleep(30);continue
            if rc:raise RuntimeError(f'{job} failed with exit {rc}; original attempt preserved')
            return

def canary(seed):
    if seed==42:
        check=BASE/'canary_comparison_s42_attempt2.json'
    else:
        out=RUNS/f'canary_paired_random_s{seed}_attempt1'
        assert not out.exists()
        cmd=[PY,str(SOURCE/'train_object_evidence.py'),'--config',str(SOURCE/'config_drone.yaml'),
             '--output',str(out),'--arm','paired_random','--seed',str(seed),'--max-steps','24']
        guarded(f'oev1_random_canary_s{seed}','train',cmd,True)
        check=BASE/f'canary_comparison_s{seed}_attempt1.json'
        old=ROOT/f'runs/rgbir_object_evidence_expand_20260906/canary_paired_s{seed}_attempt1'
        cmd=[PY,str(SOURCE/'validate_random_canary_reviewed.py'),'--paired-run',str(old),'--random-run',str(out),
             '--gpu',str(GPU),'--output',str(check)]
        subprocess.run(cmd,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert json.loads(check.read_text())['status']=='passed'

try:
    for seed in (42,0,123):
        write_status(status='canary_or_validation',seed=seed)
        canary(seed)
        out=RUNS/f'full_paired_random_s{seed}_attempt1'
        assert not out.exists()
        write_status(status='training',seed=seed,run=str(out),remaining_seeds=[x for x in (42,0,123) if (42,0,123).index(x)>(42,0,123).index(seed)])
        cmd=[PY,str(SOURCE/'train_object_evidence.py'),'--config',str(SOURCE/'config_drone.yaml'),
             '--output',str(out),'--arm','paired_random','--seed',str(seed)]
        guarded(f'oev1_random_full_s{seed}','train',cmd)
        write_status(status='evaluating',seed=seed,run=str(out))
        guarded(f'oev1_random_eval_s{seed}','eval',[PY,str(SOURCE/'evaluate_object_evidence.py'),
            '--config',str(SOURCE/'config_drone.yaml'),'--run',str(out)])
    write_status(status='completed',seeds=[42,0,123])
except BaseException as error:
    write_status(status='failed',error=repr(error))
    raise
