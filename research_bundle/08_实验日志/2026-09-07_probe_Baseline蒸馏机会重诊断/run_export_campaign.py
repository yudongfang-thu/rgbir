"""One sequential diagnostic queue using the existing global lease."""
from pathlib import Path
import json, sys, shutil, time
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v8')
from resource_dispatch import run_job
PY=str(REPO/'environments/sn6-int8-kd/bin/python')

def main():
    queue=ROOT/'queue_attempt1';queue.mkdir(exist_ok=False)
    for dataset in ('dronevehicle','llvip'):
        canary=ROOT/(dataset+'_canary_attempt1')
        command=[PY,str(ROOT/'export_baseline_information.py'),'--spec',str(ROOT/'probe_spec.json'),'--dataset',dataset]
        run_job({'id':'baseline_info_'+dataset+'_canary_a1','kind':'feature','vram_mib':2048,'rss_mib':8192,
                 'command':command+['--output',str(canary),'--canary']},queue)
        summary=json.loads((canary/'summary.json').read_text())
        measured=max(summary['resources']['per_gpu_peak_vram_mib'].values())
        assert summary['status']=='completed' and measured>0
        # Identical batch=1/canvas/models. Add margin to measured framework/NVML.
        reserve=max(2048,int(max(measured,summary['gpu_reserved_peak_mib'])*1.25)+256)
        full=ROOT/(dataset+'_full_attempt1')
        admission={'canary':str(canary),'measured_vram_mib':measured,'reservation_vram_mib':reserve,
                   'rss_reservation_mib':8192,'rss_reason':'resident model plus bounded 1224-image ROI arrays; no DataLoader workers',
                   'timestamp':time.time(),'gpu_selection':'dynamic_shared_lease','training_modified':False}
        (queue/(dataset+'_canary_acceptance.json')).write_text(json.dumps(admission,indent=2)+'\n')
        run_job({'id':'baseline_info_'+dataset+'_full_a1','kind':'feature','vram_mib':reserve,'rss_mib':8192,
                 'command':command+['--output',str(full)]},queue)
    (queue/'completion.json').write_text(json.dumps({'status':'COMPLETED','time':time.time()})+'\n')

if __name__=='__main__':main()
