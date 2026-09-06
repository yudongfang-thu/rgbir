"""Take ownership of one pending random-control seed without interrupting seed42."""
import argparse,json,os,subprocess,threading,time
from pathlib import Path
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PY=str(REPO/'environments/sn6-int8-kd/bin/python')
BASE=ROOT/'artifacts/oev1_concurrency_20260906'
OLD=ROOT/'artifacts/rgbir_oev1_random_20260906'
SOURCE=OLD/'release_v1'
RUNS=ROOT/'runs/rgbir_oev1_random_20260906'

def dump(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2)

def gpu_sample(gpu):
    values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    used,free=map(int,values.strip().split(','))
    procs=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {'time':time.time(),'used_mib':used,'free_mib':free,'cuda_pids':[int(x) for x in procs.splitlines() if x.strip()]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[0,123],required=True);parser.add_argument('--gpu',type=int,choices=[2,4],required=True)
    a=parser.parse_args();os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    assert (a.seed,a.gpu) in ((0,2),(123,4))
    assert json.loads((BASE/'concurrency_summary.json').read_text())['status']=='passed'
    canary=RUNS/f'canary_paired_random_s{a.seed}_attempt1';full=RUNS/f'full_paired_random_s{a.seed}_attempt1'
    assert not canary.exists() and not full.exists()
    claim=BASE/f'seed{a.seed}_ownership.json'
    dump(claim,{'seed':a.seed,'gpu':a.gpu,'pid':os.getpid(),'old_worker_retains_seed42':True,
                'canary':str(canary),'full':str(full),'source':str(SOURCE),'method_changed':False,
                'expected_old_queue_exit':'After seed42 training AND evaluation, pre-existing canonical seed0 canary prevents old serial queue from duplicating transferred seeds.'})
    status_path=BASE/f'seed{a.seed}_status.json'
    def status(**values):status_path.write_text(json.dumps({'seed':a.seed,'gpu':a.gpu,'pid':os.getpid(),'time':time.time(),**values},indent=2))

    def guarded(kind,stage,command,profile=False):
        job=f'oev1_random_parallel_{stage}_s{a.seed}'
        cmd=[PY,str(REPO/'tools/project_resource_guard.py'),'run','--job-id',job,'--kind',kind,
             '--candidate-gpu',str(a.gpu),'--expected-vram-mib',('6500' if stage=='canary' else '8300'),'--expected-rss-mib','32768',
             '--free-safety-mib','2048']
        if kind=='train':cmd+=['--non-formal-train'] if stage=='canary' else ['--formal-train','--profiled-second-train']
        cmd+=['--',*command]
        log_path=BASE/(job+'.log')
        with log_path.open('x') as log:
            while True:
                current=json.loads(subprocess.check_output([PY,str(REPO/'tools/project_resource_guard.py'),'inspect'],cwd=REPO,text=True))
                active=set(current['usage']['active_gpus'])
                cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                empty={int(row.split(',')[0]) for row in cards.splitlines() if int(row.split(',')[1])<100}
                admitted=active|{a.gpu}
                if not (len(admitted)<=3 or (len(admitted)<=4 and len(empty-admitted)>=2)):
                    log.write(json.dumps({'status':'WAIT_USER_GPU_POLICY','active':sorted(active),'empty':sorted(empty)})+'\n');log.flush();time.sleep(30);continue
                samples=[];stop=threading.Event()
                def sample_loop():
                    while not stop.is_set():
                        samples.append(gpu_sample(a.gpu));stop.wait(1)
                monitor=threading.Thread(target=sample_loop,daemon=True) if profile else None
                if monitor:monitor.start()
                queued=launched=False
                proc=subprocess.Popen(cmd,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in proc.stdout:
                    log.write(line);log.flush()
                    try:event=json.loads(line)
                    except ValueError:continue
                    if isinstance(event,dict):queued|=event.get('status')=='QUEUED';launched|=event.get('status')=='LAUNCHED'
                rc=proc.wait();stop.set()
                if monitor:monitor.join()
                if rc==2 and queued and not launched:time.sleep(30);continue
                if rc:raise RuntimeError(f'{job} exit {rc}; no retry after launch')
                if profile:
                    report={'samples':samples,'minimum_free_mib':min(x['free_mib'] for x in samples),
                            'maximum_used_mib':max(x['used_mib'] for x in samples),
                            'dual_samples':sum(len(x['cuda_pids'])==2 for x in samples),
                            'max_cuda_pids':max(len(x['cuda_pids']) for x in samples)}
                    dump(BASE/f'concurrent_canary_s{a.seed}.json',report)
                    assert report['minimum_free_mib']>=2048 and report['max_cuda_pids']<=2
                    assert report['dual_samples']>=3
                return
    try:
        status(status='canary')
        command=[PY,str(SOURCE/'train_object_evidence.py'),'--config',str(SOURCE/'config_drone.yaml'),
                 '--output',str(canary),'--arm','paired_random','--seed',str(a.seed),'--max-steps','24']
        guarded('train','canary',command,True)
        comparison=BASE/f'parallel_canary_comparison_s{a.seed}.json'
        subprocess.run([PY,str(SOURCE/'validate_random_canary_reviewed.py'),'--paired-run',
             str(ROOT/f'runs/rgbir_object_evidence_expand_20260906/canary_paired_s{a.seed}_attempt1'),
             '--random-run',str(canary),'--gpu',str(a.gpu),'--output',str(comparison)],check=True,
             env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        assert json.loads(comparison.read_text())['status']=='passed'
        if a.seed==123:
            primary=ROOT/'runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1'
            status(status='waiting_primary_N42_evaluation',primary=str(primary))
            while not ((primary/'evaluation_val.json').is_file() and (primary/'eval_evidence/run_receipt.json').is_file()):
                time.sleep(30)
        status(status='training',run=str(full))
        command=[PY,str(SOURCE/'train_object_evidence.py'),'--config',str(SOURCE/'config_drone.yaml'),
                 '--output',str(full),'--arm','paired_random','--seed',str(a.seed)]
        guarded('train','full',command)
        status(status='evaluating',run=str(full))
        guarded('eval','eval',[PY,str(SOURCE/'evaluate_object_evidence.py'),'--config',str(SOURCE/'config_drone.yaml'),'--run',str(full)])
        status(status='completed',run=str(full))
        # Record the legacy scheduler's expected retirement without rewriting its raw log/status.
        old42=RUNS/'full_paired_random_s42_attempt1'
        if a.seed==0 and (old42/'completion_receipt.json').is_file() and (old42/'eval_evidence/run_receipt.json').is_file():
            dump(BASE/'legacy_queue_seed42_completed_handoff.json',{'seed42_training_and_eval_completed':True,
                'transferred_seeds':[0,123],'legacy_status':json.loads((OLD/'queue_status.json').read_text()),
                'note':'Seed42 endpoint receipt exists. A generic failed queue status is not retirement evidence; separately verify that the traceback points to the transferred canonical seed0 canary existence assertion.'})
    except BaseException as error:
        status(status='failed',error=repr(error));raise
if __name__=='__main__':main()
