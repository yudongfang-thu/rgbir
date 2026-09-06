"""Serial two-arm queue locked to one physical GPU; all files on dataset disk."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
GUARD=REPO/'tools/project_resource_guard.py'

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def guarded(a,job_id,kind,command,nonformal=False):
    cmd=[sys.executable,str(GUARD),'run','--job-id',job_id,'--kind',kind,
        '--candidate-gpu',str(a.gpu),'--gpu-count','1','--cuda-processes-per-gpu','1',
        '--expected-vram-mib',str(a.vram_mib),'--expected-rss-mib',str(a.rss_mib),
        '--free-safety-mib','2048']
    if nonformal: cmd+=['--non-formal-train']
    cmd+=['--',*command]
    # Admission failures wait on this same GPU. Training failures are not retried.
    while True:
        queued=launched=False
        process=subprocess.Popen(cmd,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            try: record=json.loads(line)
            except (ValueError,TypeError): continue
            if isinstance(record,dict):
                queued=queued or record.get('status')=='QUEUED'
                launched=launched or record.get('status')=='LAUNCHED'
        status=process.wait()
        if status!=2 or not queued or launched: return status
        print('Resource admission queued; retrying same physical GPU in 30 seconds',flush=True)
        time.sleep(30)

def compare_canaries(paths,out):
    import torch
    records=[json.loads((p/'completion_receipt.json').read_text()) for p in paths]
    result={'status':'passed','cuda_tensor_comparison':False,'arms':['paired','weight0'],
        'checks':{},'receipts':[str(p/'completion_receipt.json') for p in paths]}
    for filename in ('initial_student.pt','first_batch.pt'):
        values=[torch.load(p/filename,map_location='cpu',weights_only=True) for p in paths]
        equal=set(values[0])==set(values[1]) and all(torch.equal(v,values[1][k]) for k,v in values[0].items())
        result['checks'][filename]=equal
        if not equal: raise AssertionError(f'P/N direct tensor equality failed: {filename}')
    for r in records:
        if r['status']!='canary_completed' or r['optimizer_updates']<24: raise AssertionError('Incomplete canary')
        if not r['selected_objects'] or not any(c['kd_score_gradient_l2']>0 for c in r['gradient_checks']):
            raise AssertionError('No active KD signal')
    result['gpu_reserved_peak_mib']=max(r['gpu_reserved_peak_mib'] for r in records)
    result['gpu_allocated_peak_mib']=max(r['gpu_allocated_peak_mib'] for r in records)
    result['optimizer_updates']=[r['optimizer_updates'] for r in records]
    result['batches']=[r['batches'] for r in records]
    result['amp_skipped_updates']=[r['amp_skipped_updates'] for r in records]
    write(out,result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['canary','full'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--vram-mib',type=int,required=True)
    p.add_argument('--rss-mib',type=int,default=16384)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--attempt',default='attempt1')
    a=p.parse_args()
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['MKL_NUM_THREADS']='4'
    # Fixed allocation applies to canaries, training and evaluation alike.
    allocation=json.loads((a.root/'allocation.json').read_text())
    if a.gpu!=allocation['physical_gpu']: raise RuntimeError('Campaign GPU changed')
    with (a.root/'one_gpu_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.stage=='full':
            verified=json.loads((a.root/'canary_comparison.json').read_text())
            if verified['status']!='passed': raise RuntimeError('Canaries not accepted')
        outputs=[]
        for arm in ('paired','weight0'):
            run=a.run_root/f'{a.stage}_{arm}_s42_{a.attempt}'
            outputs.append(run)
            if run.exists(): raise FileExistsError(f'Preserve existing attempt: {run}')
            write(a.root/f'{a.stage}_queue_status.json',{'stage':a.stage,'status':'running',
                'active_arm':arm,'queued_arms':['weight0'] if arm=='paired' else [],
                'physical_gpu':a.gpu,'pid':os.getpid(),'run':str(run),
                'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            command=[sys.executable,str(HERE/'train_object_evidence.py'),'--config',str(HERE/'config_drone.yaml'),
                '--output',str(run),'--arm',arm]
            if a.stage=='canary': command+=['--max-steps','24']
            code=guarded(a,f'rgbir_oev1_{a.stage}_{arm}_s42_{a.attempt}','train',command,a.stage=='canary')
            if code: raise RuntimeError(f'{arm} training failed with exit {code}; attempt preserved')
            if a.stage=='full':
                code=guarded(a,f'rgbir_oev1_eval_{arm}_s42_{a.attempt}','eval',
                    [sys.executable,str(HERE/'evaluate_object_evidence.py'),'--config',str(HERE/'config_drone.yaml'),'--run',str(run)])
                if code: raise RuntimeError(f'{arm} evaluation failed with exit {code}')
        if a.stage=='canary': compare_canaries(outputs,a.root/'canary_comparison.json')
        write(a.root/f'{a.stage}_queue_status.json',{'stage':a.stage,'status':'completed','physical_gpu':a.gpu,
            'outputs':[str(p) for p in outputs],'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__': main()
