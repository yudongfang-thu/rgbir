"""One measured zero-update batch, through the project's sole existing GPU lease."""
import argparse,json,shutil,sys,time,traceback
from pathlib import Path
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
DISPATCH=B/'rgbir_task_conditional_v1_20260907/release_v8'
SCREEN=B/'rgbir_direction_screen_20260908/release_v1'
REFERENCE=B/'rgbir_independent_kd_v2_20260907/release_gpu5'
PREVIOUS=B/'rgbir_selection_coverage_20260908/witness_attempt1/probe'
WITNESS=B/'rgbir_selection_coverage_20260908/witness_release_v1'
ANCHORS=B/'rgbir_localization_learning_target_20260908/review_v1/anchor_join/output_attempt3/objects.jsonl'
CFG=B/'rgbir_direction_screen_20260908/screen_attempt1/effective_configs/llvip_N_s42_FT3.yaml'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def run(a):
    out=a.output.resolve();release=a.release_dir.resolve()
    if Path('/mnt/dataset/yudongfang') not in out.parents:raise ValueError('Data disk only')
    for p in (Path(PY),CFG,ANCHORS,PREVIOUS/'first_batch_stream.json',DISPATCH/'resource_dispatch.py',release/'run_dfl_probe.py'):
        if not p.is_file():raise FileNotFoundError(p)
    if (release/'llvip_N_s42_FT3.yaml').read_bytes()!=CFG.read_bytes():raise ValueError('Inherited config differs')
    out.mkdir(parents=True,exist_ok=False);q=out/'queue';q.mkdir();shutil.copyfile(__file__,q/'executed_driver.py')
    job=dict(id='raw_dfl_'+out.name,kind='feature',formal=False,vram_mib=8192,rss_mib=32768,
        expected_receipt=str(out/'probe/completion_receipt.json'),
        command=[PY,str(release/'run_dfl_probe.py'),'--screen-release',str(SCREEN),'--reference-dir',str(REFERENCE),
          '--config',str(release/'llvip_N_s42_FT3.yaml'),'--witness-release',str(WITNESS),'--previous-probe',str(PREVIOUS),
          '--anchor-objects',str(ANCHORS),'--output',str(out/'probe')])
    write(q/'job.json',job);start=time.time()
    write(q/'measurement_scope.json',dict(single_measured_batch_is_complete_probe=True,additional_full_stage=False,
        reservation_basis='Unchanged three-forward B32 witness path measured NVML1646MiB/RSS26771MiB; conservative existing8192/32768 reservation for new readout.',
        new_path_peak_to_be_measured=True,old_training_unchanged=True,new_hash_computed=False))
    try:
        sys.path.insert(0,str(DISPATCH));import resource_dispatch
        if Path(resource_dispatch.__file__).resolve()!=DISPATCH/'resource_dispatch.py':raise ValueError('Wrong resource dispatcher')
        result=resource_dispatch.run_job(job,q)
        receipt=json.loads(Path(job['expected_receipt']).read_text())
        required=dict(status='RAW_DFL_SINGLE_BATCH_COMPLETED',batches=1,frames=32,objects_n=80,optimizer_updates=0,backward=0,
            first_batch_stream_exact=True,identity_exact=True,auxiliary_full_state_unchanged=True)
        for k,v in required.items():
            if receipt.get(k)!=v:raise ValueError('Producer contract differs: '+k)
        if receipt.get('raw_forward_counts')!=dict(student=1,teacher=1,reference=1):raise ValueError('Measured batch forward count differs')
        write(q/'completion.json',dict(status='RAW_DFL_SINGLE_BATCH_QUEUE_COMPLETED',receipt=job['expected_receipt'],
            resource_status=result['status'],seconds=time.time()-start,additional_full_stage=False,new_resource_pool=False,new_hash_computed=False))
    except BaseException as e:
        write(q/'failure.json',dict(status='RAW_DFL_SINGLE_BATCH_QUEUE_FAILED',error=repr(e),traceback=traceback.format_exc(),
            seconds=time.time()-start,automatic_retry=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
