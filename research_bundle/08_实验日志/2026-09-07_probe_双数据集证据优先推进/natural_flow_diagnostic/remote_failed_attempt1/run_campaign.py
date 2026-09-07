"""Actual natural-flow inference diagnostics using the existing global lease."""
from pathlib import Path
import json,sys,time,math
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
RUNTIME=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
sys.path.insert(0,'/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v8')
from resource_dispatch import run_job

def validate_summary(s,dataset,batches):
    expected_dataset={'llvip':'llvip','drone':'dronevehicle'}[dataset]
    checks=[s['status']=='COMPLETED',s['dataset']==expected_dataset,s['batches']==batches,
      s['seed']==20260907,s['canary_only'] is (batches==2),s['trace_exact_all_recorded_fields'] is True,
      s['diagnostic_status']=='UNVERIFIED_GEOMETRY_DIAGNOSTIC',
      s['classification']['C0_C1_selection_identical'] is True,
      s['localization']['geometry_verified'] is False,s['localization']['formal_L1_admitted'] is False]
    checks += [s[k] is False for k in ['backward_executed','training_executed','calibration_executed','validation_or_test_accessed','new_hash_computed']]
    if not all(checks):raise ValueError('Diagnostic summary contract is invalid')
    resources=s['resources']
    peaks=list(resources['per_gpu_peak_vram_mib'].values())+[resources['peak_rss_mib'],s['gpu_reserved_peak_mib'],s['gpu_allocated_peak_mib']]
    if not resources['per_gpu_peak_vram_mib'] or not all(math.isfinite(x) and x>0 for x in peaks):
        raise ValueError('No empirical resource measurement')

def main():
    queue=ROOT/'queue_attempt1';queue.mkdir(exist_ok=False)
    for dataset in ('llvip','drone'):
        cmd=[PY,str(ROOT/'diagnose_natural_flow.py'),'--release',str(RUNTIME/'release_gpu5'),
          '--config',str(RUNTIME/'configs_draft_v1'/(dataset+'_C1.yaml')),
          '--coverage-dir',str(RUNTIME/('coverage_'+dataset+'_attempt1'))]
        canary=ROOT/(dataset+'_canary_attempt1')
        # Initial bounded measurement ceiling, not an asserted empirical peak.
        run_job(dict(id='natural_'+dataset+'_canary',kind='eval',vram_mib=4096,rss_mib=24576,
          command=cmd+['--output',str(canary),'--batches','2']),queue)
        s=json.loads((canary/'summary.json').read_text())
        validate_summary(s,dataset,2)
        resources=s['resources'];measured=max(resources['per_gpu_peak_vram_mib'].values());rss=resources['peak_rss_mib']
        if measured<=0 or rss<=0:raise ValueError('No empirical resource measurement')
        vram=int(max(measured,s['gpu_reserved_peak_mib'],s['gpu_allocated_peak_mib']))+512
        rss_reserve=max(24576,int(rss)+4096)
        with (queue/(dataset+'_canary_acceptance.json')).open('x') as f:
            json.dump(dict(status='ACCEPTED_FOR_FIXED_FLOW_DIAGNOSTIC_ONLY',canary=str(canary),
              vram_measured_mib=measured,reserved_mib=vram,rss_measured_mib=rss,rss_reserved_mib=rss_reserve,
              new_training_admitted=False,geometry_verified=False,time=time.time()),f,indent=2)
        run_job(dict(id='natural_'+dataset+'_full',kind='eval',vram_mib=vram,rss_mib=rss_reserve,
          command=cmd+['--output',str(ROOT/(dataset+'_full_attempt1')),'--batches','64']),queue)
        validate_summary(json.loads((ROOT/(dataset+'_full_attempt1')/'summary.json').read_text()),dataset,64)
    with (queue/'completion.json').open('x') as f:json.dump(dict(status='COMPLETED',time=time.time()),f)

if __name__=='__main__':main()
