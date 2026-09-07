"""A sequential stage runner using the existing project-wide lease, never a second resource pool."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil

REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PY=str(REPO/'environments/sn6-int8-kd/bin/python')
sys.path.insert(0,str(REPO))
from tools.project_resource_guard import ProjectResourceGuard, gpu_snapshot, DEFAULT_LEASE_FILE


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def stop_owned_tree(pid):
    """Terminate only descendants of this launched workload; never other leases."""
    try:
        parent=psutil.Process(pid)
        owned=parent.children(recursive=True)+[parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(owned):
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=5)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass


def run_job(job, output):
    guard=ProjectResourceGuard(DEFAULT_LEASE_FILE)
    log_path=output/(job['id']+'.log')
    if log_path.exists():raise FileExistsError('A stage was already attempted: '+job['id'])
    samples=[]
    with log_path.open('x') as log:
        while True:
            state=guard.inspect()
            rows=gpu_snapshot()
            active=set(state['usage']['active_gpus'])
            empty={g for g,r in rows.items() if r['memory_used_mib']<100}
            candidates=[]
            for gpu in sorted(rows,key=lambda g:(g not in active,rows[g]['memory_used_mib'],g)):
                occupied=active|{gpu}
                if len(occupied)<=3 or (len(occupied)<=4 and len(empty-occupied)>=2):
                    candidates.append(gpu)
            dump(output/(job['id']+'_status.json'),{'status':'ADMISSION','time':time.time(),
                'active':sorted(active),'empty':sorted(empty),'candidate_gpus':candidates,
                'four_gpu_exception':len(active)>=4,'job':job})
            if not candidates:
                time.sleep(30);continue
            command=[PY,str(REPO/'tools/project_resource_guard.py'),'run','--job-id',job['id'],
                '--kind',job['kind'],'--expected-vram-mib',str(job['vram_mib']),
                '--expected-rss-mib',str(job['rss_mib']),'--free-safety-mib','2048']
            for gpu in candidates:command+=['--candidate-gpu',str(gpu)]
            if job['kind']=='train':
                command+=['--formal-train','--profiled-second-train'] if job.get('formal') else ['--non-formal-train']
            command+=['--',*job['command']]
            launched={};stop=threading.Event();monitor_errors=[]
            def watch():
                while not stop.wait(1):
                    if launched:
                        try:
                            now=gpu_snapshot();usage=guard.inspect()['usage']
                            for gpu in launched['gpu_ids']:
                                row={'time':time.time(),'gpu':gpu,**now[gpu],
                                     'project_rss_mib':usage['project_rss_mib']};samples.append(row)
                                if row['memory_free_mib']<2048:
                                    raise RuntimeError('Full GPU free memory below 2048 MiB')
                                if usage['actual_vram_mib'].get(str(gpu),usage['actual_vram_mib'].get(gpu,0))>=.7*row['memory_total_mib']:
                                    raise RuntimeError('Project GPU memory reached 70 percent')
                                if usage['actual_cuda_pids'].get(str(gpu),usage['actual_cuda_pids'].get(gpu,0))>2:
                                    raise RuntimeError('Project CUDA task count exceeded two')
                            if usage['project_rss_mib']*2**20>300_000_000_000:
                                raise RuntimeError('Project host memory exceeded 300 GB')
                        except Exception as error:
                            monitor_errors.append(repr(error))
                            log.write('RESOURCE_FAILURE '+repr(error)+'\n');log.flush()
                            stop_owned_tree(launched['pid']);stop.set()
            watcher=threading.Thread(target=watch,daemon=True);watcher.start()
            proc=subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                  env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
            queued=False
            for line in proc.stdout:
                log.write(line);log.flush()
                try:event=json.loads(line)
                except ValueError:continue
                if isinstance(event,dict) and event.get('status')=='QUEUED':queued=True
                if isinstance(event,dict) and event.get('status')=='LAUNCHED':
                    launched.update(event)
                    after=active|set(event['gpu_ids'])
                    admission={'active_gpus_after':sorted(after),'empty_gpus_after':sorted(empty-after),
                               'four_gpu_exception':len(after)==4,'minimum_empty_required':2}
                    dump(output/(job['id']+'_admission.json'),admission)
                    dump(output/(job['id']+'_status.json'),{'status':'RUNNING','time':time.time(),'launch':event,'job':job,'admission':admission})
            rc=proc.wait();stop.set();watcher.join(timeout=3)
            if rc==2 and queued and not launched:
                time.sleep(30);continue
            report={'status':'COMPLETED' if rc==0 and not monitor_errors else 'FAILED','exit_code':rc,'time':time.time(),
                'monitor_errors':monitor_errors,
                'launch':launched,'samples':samples,'minimum_free_mib':min((r['memory_free_mib'] for r in samples),default=None)}
            dump(output/(job['id']+'_resource_profile.json'),report)
            dump(output/(job['id']+'_status.json'),{k:v for k,v in report.items() if k!='samples'})
            if rc or monitor_errors:raise RuntimeError('Launched stage failed; preserve attempt and inspect: '+job['id'])
            if report['minimum_free_mib'] is not None and report['minimum_free_mib']<2048:
                raise RuntimeError('Stage exceeded required free-memory margin')
            return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text())
    dump(a.output/'manifest.json',manifest)
    for job in manifest['jobs']:
        if job.get('requires_profile'):
            probe=json.loads(Path(job['requires_profile']).read_text())
            resources=probe.get('resources',{})
            measured_vram=max(resources.get('per_gpu_peak_vram_mib',{}).values(),default=0)
            measured_rss=resources.get('peak_rss_mib',0)
            if probe.get('status') not in ('completed','canary_completed','COMPLETED'):
                raise ValueError('Prerequisite profile did not complete')
            if measured_vram<=0 or measured_rss<=0:
                raise ValueError('Prerequisite has no measured NVML/process-tree profile')
            if max(measured_vram,probe.get('gpu_reserved_peak_mib',0),probe.get('gpu_allocated_peak_mib',0))>job['vram_mib']:
                raise ValueError('Observed canary memory exceeds requested reservation')
            if measured_rss>job['rss_mib']:
                raise ValueError('Observed canary process-tree RSS exceeds reservation')
        run_job(job,a.output)
    dump(a.output/'completion.json',{'status':'COMPLETED','time':time.time(),'jobs':[j['id'] for j in manifest['jobs']]})


if __name__=='__main__':main()
