"""Two read-only LLVIP diagnostics through the existing global resource pool."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PY=str(REPO/'environments/sn6-int8-kd/bin/python')
sys.path.insert(0,'/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v8')
from resource_dispatch import run_job

def main():
    queue=ROOT/'queue_attempt1';queue.mkdir(exist_ok=False)
    for model in ('N42','T42'):
        cmd=[PY,str(ROOT/'export_full_dev.py'),'--spec',str(ROOT/'spec.json'),'--model',model]
        canary=ROOT/(model+'_canary_attempt1')
        run_job(dict(id='llvip_full_'+model+'_canary',kind='eval',vram_mib=4096,rss_mib=12288,
            command=cmd+['--output',str(canary),'--canary']),queue)
        s=json.loads((canary/'summary.json').read_text())
        if s['status']!='completed' or not s['native_capture_exact'] or s.get('gt_count',0)<=0:raise ValueError('Canary failed or has no GT')
        measured=max(s['resources']['per_gpu_peak_vram_mib'].values())
        rss=s['resources']['peak_rss_mib']
        if measured<=0 or rss<=0:raise ValueError('Missing measured resources')
        reserve=int(max(measured,s['gpu_reserved_peak_mib'],s['gpu_allocated_peak_mib']))+512
        rss_reserve=max(12288,int(rss)+4096)
        admission=dict(canary=str(canary),vram_measured_mib=measured,reservation_vram_mib=reserve,
            rss_measured_mib=rss,reservation_rss_mib=rss_reserve,source='this_path_actual_canary',time=time.time())
        with (queue/(model+'_canary_acceptance.json')).open('x') as f:json.dump(admission,f,indent=2)
        run_job(dict(id='llvip_full_'+model+'_full',kind='eval',vram_mib=reserve,rss_mib=rss_reserve,
            command=cmd+['--output',str(ROOT/(model+'_full_attempt1'))]),queue)
    with (queue/'completion.json').open('x') as f:json.dump(dict(status='COMPLETED',time=time.time()),f)

if __name__=='__main__':main()
