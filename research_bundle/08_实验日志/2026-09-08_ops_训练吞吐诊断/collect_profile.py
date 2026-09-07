"""Read-only 20-second host/process/GPU sample, no CUDA workload or hashes."""
import datetime
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE = r'''
import datetime,json,os,shutil,subprocess,time
from pathlib import Path
ROOT = Path('/mnt/dataset/yudongfang/projects')
lease_file=ROOT/'SpaceNet6_OTD_official_reproduction/runs/.project_resource_leases.json'
leases=json.loads(lease_file.read_text())
cuda_pids=set(p for v in leases['leases'].values() for p in v.get('observed_cuda_pids', []))
def processes():
    rows={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            stat=(path/'stat').read_text().split(') ',1)[1].split()
            rows[int(path.name)]={'ppid':int(stat[1]),'state':stat[0],
                'utime':int(stat[11]),'stime':int(stat[12]),'threads':int(stat[17])}
        except (OSError,ValueError,IndexError):pass
    selected=set(cuda_pids)
    while True:
        more={pid for pid,v in rows.items() if v['ppid'] in selected}-selected
        if not more:break
        selected.update(more)
    result={}
    for pid in sorted(selected):
        if pid not in rows:continue
        path=Path('/proc')/str(pid)
        row=rows[pid]
        for key in ('wchan','io','status'):
            try:row[key]=(path/key).read_text()
            except OSError:row[key]=None
        try:
            row['affinity_cpus']=len(os.sched_getaffinity(pid))
            row['command']=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:pass
        result[str(pid)]=row
    return result
def run(args):
    r=subprocess.run(args,capture_output=True,text=True)
    return {'args':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
start=time.time()
before=processes()
pmon=subprocess.Popen(['nvidia-smi','pmon','-c','20','-d','1','-s','um'],
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
samples=[]
for i in range(20):
    samples.append({'time':time.time(),'loadavg':os.getloadavg(),
        'gpus':run(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits'])})
    if i<19:time.sleep(1)
after=processes()
pout,perr=pmon.communicate(timeout=10)
tools={name:shutil.which(name) for name in ('py-spy','nsys','perf','strace','iostat','pidstat')}
filtered=[]
for line in pout.splitlines():
    parts=line.split()
    if line.startswith('#') or (len(parts)>1 and parts[1].isdigit() and int(parts[1]) in cuda_pids):filtered.append(line)
result={'captured_at':datetime.datetime.now().astimezone().isoformat(),
        'wall_seconds':time.time()-start,'cpu_count':os.cpu_count(),'clock_ticks':os.sysconf('SC_CLK_TCK'),
        'tools':tools,'cuda_pids':sorted(cuda_pids),'leases':leases,
        'before':before,'after':after,'gpu_samples':samples,
        'project_pmon':'\n'.join(filtered),'pmon_stderr':perr,'pmon_exit':pmon.returncode,
        'memory_info':Path('/proc/meminfo').read_text(),
        'storage':run(['df','-h','/mnt/dataset/yudongfang']),
        'new_cuda_workloads':0,'remote_mutations':False}
print(json.dumps(result))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=REMOTE.encode('utf-8'),capture_output=True,check=True)
data=json.loads(r.stdout)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
path=OUT/('host_profile_'+stamp+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
cpu=[]
for pid,row in data['after'].items():
    old=data['before'].get(pid)
    if old:
        cpu.append({'pid':pid,'cpu_pct_one_core':100*(row['utime']+row['stime']-old['utime']-old['stime'])/data['clock_ticks']/data['wall_seconds'],
                    'state':row['state'],'wchan':row.get('wchan'),'threads':row['threads']})
print(json.dumps({'output':str(path),'tools':data['tools'],'cpu':cpu,
                  'gpu_first':data['gpu_samples'][0]['gpus']['stdout'],
                  'gpu_last':data['gpu_samples'][-1]['gpus']['stdout'],
                  'pmon':data['project_pmon'][:4200]}))
