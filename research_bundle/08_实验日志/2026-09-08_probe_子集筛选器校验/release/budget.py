"""Frozen wall-cost and measured-resource arithmetic; no GPU, scheduler or AP."""
import math

LIMIT_SECONDS=2700.
ARMS=('N','C0')

def finite(value,positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError('Expected finite nonnegative measured number')
    return float(value)

def training_estimate(receipt):
    samples=receipt['normal_cadence_batch_seconds']
    if not isinstance(samples,list) or len(samples)<24:raise ValueError('At least24 measured batch intervals required')
    samples=[finite(x,True) for x in samples]
    seconds=finite(receipt['seconds'],True)
    io=finite(receipt['flow_audit_seconds'])
    if io>seconds or sum(samples)+io>seconds+1.:raise ValueError('Timing decomposition cannot exceed stage time')
    overhead=max(0.,seconds-sum(samples)-io)
    mean=sum(samples)/len(samples)
    value=1.2*(512*mean+overhead+io)
    return dict(mean_normal_batch_seconds=mean,measured_batches=len(samples),
        observed_non_batch_overhead_seconds=overhead,observed_flow_audit_seconds=io,
        train_seconds_with_margin=value,margin_factor=1.2)

def estimate(canaries,spent_seconds):
    if set(canaries)!=set(ARMS):raise ValueError('Exactly both N/C0 canaries required')
    spent=finite(spent_seconds)
    arms={a:training_estimate(canaries[a]) for a in ARMS}
    total=spent+sum(v['train_seconds_with_margin'] for v in arms.values())+120.
    return dict(status='PASS_WITHIN_EXECUTION_BUDGET' if total<=LIMIT_SECONDS else 'BLOCKED_EXECUTION_BUDGET',
        spent_execution_seconds=spent,arms=arms,eval_reserve_seconds_each=60.,
        estimate_total_execution_seconds=total,limit_seconds=LIMIT_SECONDS,
        queue_admission_wait_excluded=True,ap_used=False)

def reservation(receipt):
    resources=receipt['resources']
    v=list(resources['per_gpu_peak_vram_mib'].values())
    if len(v)!=1:raise ValueError('Exactly one measured GPU required')
    measured=finite(v[0],True)
    raw=max(measured,finite(receipt['gpu_allocated_peak_mib'],True),finite(receipt['gpu_reserved_peak_mib'],True))
    rss=finite(resources['peak_rss_mib'],True)
    return dict(vram_mib=math.ceil((raw+max(256.,.05*raw))/256.)*256,
        rss_mib=math.ceil((rss+max(2048.,.05*rss))/1024.)*1024,
        measured_nvml_peak_mib=measured,measured_rss_peak_mib=rss,
        measured_combined_peak_mib=raw,source='this_arm_new_actual_canary')
