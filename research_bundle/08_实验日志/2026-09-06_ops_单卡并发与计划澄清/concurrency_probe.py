"""Observe one existing full train plus a bounded 24-update train on the same GPU."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PY=str(REPO/'environments/sn6-int8-kd/bin/python')
SOURCE=ROOT/'artifacts/rgbir_oev1_random_20260906/release_v1'
BASE=ROOT/'artifacts/oev1_concurrency_20260906'
RUN=ROOT/'runs/oev1_concurrency_20260906/profile_paired_random_s0_attempt1'
EXISTING=ROOT/'runs/rgbir_oev1_random_20260906/full_paired_random_s42_attempt1'

def read_progress():
    try:return json.loads((EXISTING/'progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def gpu_sample():
    output=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    used,free,util=[int(x) for x in output.strip().split(',')]
    processes=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(time=time.time(),used_mib=used,free_mib=free,util_percent=util,
                cuda_pids=[int(x.split(',')[0]) for x in processes.splitlines() if x.strip()],existing=read_progress())

os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
assert not RUN.exists()
before=gpu_sample()
assert len(before['cuda_pids'])==1 and before['free_mib']>6500+2048
assert not (EXISTING/'completion_receipt.json').exists()
cmd=[PY,str(REPO/'tools/project_resource_guard.py'),'run','--job-id','oev1_concurrent_profile_s0',
     '--kind','train','--candidate-gpu','2','--non-formal-train','--expected-vram-mib','6500',
     '--expected-rss-mib','32768','--free-safety-mib','2048','--',PY,str(SOURCE/'train_object_evidence.py'),
     '--config',str(SOURCE/'config_drone.yaml'),'--output',str(RUN),'--arm','paired_random','--seed','0','--max-steps','24']
samples=[before]
with (BASE/'profile.log').open('x') as log:
    proc=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    while proc.poll() is None:
        samples.append(gpu_sample());time.sleep(1)
    rc=proc.returncode
samples.append(gpu_sample())
record={'returncode':rc,'command':cmd,'samples':samples,'run':str(RUN),'existing_run':str(EXISTING),
        'cuda_peak_process_count':max(len(x['cuda_pids']) for x in samples),
        'observed_peak_total_used_mib':max(x['used_mib'] for x in samples),
        'minimum_observed_free_mib':min(x['free_mib'] for x in samples),
        'samples_with_two_cuda_pids':sum(len(x['cuda_pids'])==2 for x in samples)}
if (RUN/'completion_receipt.json').exists():record['completion']=json.loads((RUN/'completion_receipt.json').read_text())
with (BASE/'concurrency_profile.json').open('x') as f:json.dump(record,f,indent=2)
assert rc==0,rc
check=BASE/'profile_canary_comparison_s0.json'
subprocess.run([PY,str(SOURCE/'validate_random_canary_reviewed.py'),'--paired-run',str(ROOT/'runs/rgbir_object_evidence_expand_20260906/canary_paired_s0_attempt1'),
                '--random-run',str(RUN),'--gpu','2','--output',str(check)],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
assert record['samples_with_two_cuda_pids']>=3
assert record['minimum_observed_free_mib']>=2048
summary={k:v for k,v in record.items() if k not in ('samples','completion','command')}
summary['canary_check']=json.loads(check.read_text())
summary['status']='passed'
with (BASE/'concurrency_summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary),flush=True)
