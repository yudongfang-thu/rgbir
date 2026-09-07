import os,sys,json,subprocess,time
from pathlib import Path
r=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction'); b=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907');sys.path.insert(0,str(r))
from tools.project_resource_guard import ProjectResourceGuard,ResourceRequest,ResourceUnavailable,DEFAULT_LEASE_FILE,LEASE_ID_ENV,LEASE_FILE_ENV,LEASE_GPUS_ENV
g=ProjectResourceGuard(DEFAULT_LEASE_FILE)
request=ResourceRequest(job_id='ikdv2_cpu_a3',kind='cpu',candidate_gpus=(),expected_vram_mib=0,expected_rss_mib=16384,gpu_count=0,cuda_processes_per_gpu=0)
while True:
 try: lease=g.acquire(request,owner_pid=os.getpid());break
 except ResourceUnavailable as e: print('QUEUED '+str(e),flush=True);time.sleep(30)
g.bind(lease['lease_id'],os.getpid())
os.environ.update({LEASE_ID_ENV:lease['lease_id'],LEASE_FILE_ENV:str(DEFAULT_LEASE_FILE),LEASE_GPUS_ENV:'', 'CUDA_VISIBLE_DEVICES':''})
try:
 p=subprocess.Popen(['/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python',str(b/'cpu_sequence1.py')])
 while p.poll() is None:
  usage=g.inspect()['usage']
  if usage['project_rss_mib']*2**20>300_000_000_000:
   p.terminate();raise RuntimeError('Project RSS limit')
  time.sleep(2)
 sys.exit(p.returncode)
finally:g.release(lease['lease_id'])
