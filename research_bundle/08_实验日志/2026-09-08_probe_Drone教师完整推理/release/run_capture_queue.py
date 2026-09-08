"""One Drone IR canary and full-dev capture through the original global resource lease."""
from pathlib import Path
import argparse,json,shutil,sys,time,traceback

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
DISPATCH=B/'artifacts/rgbir_task_conditional_v1_20260907/release_v8'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'

def write_new(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)

def check_summary(p,canary):
    s=json.loads((p/'summary.json').read_text())
    expected=dict(status='completed',scope='DRONE_IR42_FULL_DEV_CAPTURE',dataset='dronevehicle',model='T42',images=32 if canary else 1469,
        full_population=1469,canary=canary)
    for k,v in expected.items():
        if s.get(k)!=v:raise ValueError('Producer contract differs: '+k)
    if s.get('gt_count',0)<=0:raise ValueError('Missing IR GT')
    if canary and s['gt_count']!=514:raise ValueError('Fixed first32 IR GT differs')
    if canary and s.get('native_capture_exact') is not True:raise ValueError('Canary native capture differs')
    if not canary and s['gt_count']!=24490:raise ValueError('Full IR GT count differs')
    if not (p/'capture/objects.jsonl.gz').is_file():raise ValueError('Missing cached output')
    return s

def run(a):
    release=a.release_dir.resolve();output=a.output.resolve()
    if Path('/mnt/dataset/yudongfang') not in output.parents:raise ValueError('Only data-disk output')
    output.mkdir(parents=True,exist_ok=False);q=output/'queue';q.mkdir()
    shutil.copyfile(__file__,q/'executed_queue.py')
    for p in [Path(PY),DISPATCH/'resource_dispatch.py',release/'export_drone_teacher.py',release/'drone_teacher_spec.json']:
        if not p.is_file():raise FileNotFoundError(p)
    sys.path.insert(0,str(DISPATCH));import resource_dispatch
    if Path(resource_dispatch.__file__).resolve()!=DISPATCH/'resource_dispatch.py':raise ValueError('Wrong dispatcher')
    cmd=[PY,str(release/'export_drone_teacher.py'),'--spec',str(release/'drone_teacher_spec.json'),'--model','T42']
    started=time.time()
    try:
        canary=output/'T42_canary_attempt1'
        canary_job=dict(id='drone_teacher_'+output.name+'_canary',kind='eval',formal=False,vram_mib=4096,rss_mib=12288,
            expected_receipt=str(canary/'summary.json'),command=cmd+['--output',str(canary),'--canary'])
        write_new(q/'canary_job.json',canary_job)
        r=resource_dispatch.run_job(canary_job,q);s=check_summary(canary,True)
        measured=max(s['resources']['per_gpu_peak_vram_mib'].values());rss=s['resources']['peak_rss_mib']
        if measured<=0 or rss<=0:raise ValueError('Missing measured resources')
        reserve=int(max(measured,s['gpu_reserved_peak_mib'],s['gpu_allocated_peak_mib']))+512
        rss_reserve=max(12288,int(rss)+4096)
        write_new(q/'canary_acceptance.json',dict(status='CANARY_ACCEPTED',canary=str(canary),
            vram_measured_mib=measured,reservation_vram_mib=reserve,rss_measured_mib=rss,
            reservation_rss_mib=rss_reserve,source='this_path_actual_canary',resource_status=r['status'],time=time.time()))
        if a.canary_only:
            write_new(q/'completion.json',dict(status='DRONE_IR42_CANARY_COMPLETED',seconds=time.time()-started,
                full_capture_started=False,new_resource_pool=False,new_hash_computed=False));return
        full=output/'T42_full_attempt1'
        job=dict(id='drone_teacher_'+output.name+'_full',kind='eval',formal=False,vram_mib=reserve,rss_mib=rss_reserve,
            expected_receipt=str(full/'summary.json'),command=cmd+['--output',str(full)])
        write_new(q/'full_job.json',job)
        result=resource_dispatch.run_job(job,q);s=check_summary(full,False)
        write_new(q/'completion.json',dict(status='DRONE_IR42_FULL_CAPTURE_COMPLETED',images=s['images'],gt_count=s['gt_count'],
            receipt=str(full/'summary.json'),seconds=time.time()-started,resource_status=result['status'],
            new_resource_pool=False,new_hash_computed=False))
    except BaseException as e:
        write_new(q/'failure.json',dict(status='DRONE_IR42_CAPTURE_QUEUE_FAILED',error=repr(e),
            traceback=traceback.format_exc(),seconds=time.time()-started,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--canary-only',action='store_true');run(p.parse_args())
