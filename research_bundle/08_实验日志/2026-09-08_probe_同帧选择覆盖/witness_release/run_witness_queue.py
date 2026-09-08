"""One bounded feature/inference job through the original global resource lease."""
import argparse,json,shutil,sys,time,traceback
from pathlib import Path

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
DISPATCH=B/'rgbir_task_conditional_v1_20260907/release_v8'
SCREEN=B/'rgbir_direction_screen_20260908/release_v1'
REFERENCE=B/'rgbir_independent_kd_v2_20260907/release_gpu5'
MAIN=B/'rgbir_direction_screen_20260908/screen_attempt1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'

def write_new(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)

def run(a):
    output=a.output.resolve();release=a.release_dir.resolve()
    if Path('/mnt/dataset/yudongfang') not in output.parents:raise ValueError('Only data-disk output')
    output.mkdir(parents=True,exist_ok=False);q=output/'queue';q.mkdir()
    cfg=MAIN/'effective_configs/llvip_N_s42_FT3.yaml'
    if (release/'llvip_N_s42_FT3.yaml').read_bytes()!=cfg.read_bytes():raise ValueError('Frozen inherited config differs')
    cp=release/'llvip_N_s42_FT3.yaml'
    expected=MAIN/'runs/llvip/N/sample_stream.jsonl'
    for p in (Path(PY),cfg,expected,DISPATCH/'resource_dispatch.py',release/'run_witness.py'):
        if not p.is_file():raise FileNotFoundError(p)
    shutil.copyfile(__file__,q/'executed_driver.py')
    job=dict(id='detector_witness_'+output.name,kind='feature',formal=False,vram_mib=8192,rss_mib=32768,
        expected_receipt=str(output/'probe/completion_receipt.json'),
        command=[PY,str(release/'run_witness.py'),'--screen-release',str(SCREEN),'--reference-dir',str(REFERENCE),
            '--config',str(cp),'--expected-stream',str(expected),'--previous-objects',str(B/'rgbir_selection_coverage_20260908/attempt1/probe/objects.jsonl'),'--output',str(output/'probe')])
    write_new(q/'job.json',job);started=time.time()
    try:
        sys.path.insert(0,str(DISPATCH));import resource_dispatch
        if Path(resource_dispatch.__file__).resolve()!=DISPATCH/'resource_dispatch.py':raise ValueError('Wrong global dispatcher')
        result=resource_dispatch.run_job(job,q)
        receipt=json.loads(Path(job['expected_receipt']).read_text())
        for k,v in dict(status='SAME_FORWARD_DETECTOR_WITNESS_COMPLETED',batches=1,frames=32,
            optimizer_updates=0,backward=0,training=0,first_batch_stream_exact=True,previous_objects_exact=True).items():
            if receipt.get(k)!=v:raise ValueError('Producer contract differs: '+k)
        write_new(q/'completion.json',dict(status='SAME_FORWARD_DETECTOR_WITNESS_QUEUE_COMPLETED',receipt=job['expected_receipt'],
            seconds=time.time()-started,resource_status=result['status'],new_resource_pool=False,new_hash_computed=False))
    except Exception as e:
        write_new(q/'failure.json',dict(status='SAME_FORWARD_DETECTOR_WITNESS_QUEUE_FAILED',error=repr(e),
            traceback=traceback.format_exc(),seconds=time.time()-started,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
