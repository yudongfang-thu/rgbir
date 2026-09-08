"""Fixed LLVIP N/C0 mini queue after the original direction queue terminates.

Uses only the existing resource_dispatch/globallease. No work starts on import.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import yaml

ARMS=('N','C0')
SCOPE='LLVIP_CONFIDENCE_FT3'
ENDPOINT='LLVIP_CONFIDENCE_FT3_LAST_EMA'
BASE=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python')
REFERENCE=BASE/'rgbir_independent_kd_v2_20260907/release_gpu5'
DISPATCH=BASE/'rgbir_task_conditional_v1_20260907/release_v8'
WAIT_FOR=BASE/'rgbir_direction_screen_20260908/screen_attempt1/queue/completion.json'
VRAM_LIMIT,RSS_LIMIT=8192,32768


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def file_stat(path):
    path=Path(path);s=path.stat()
    return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def finite_positive(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('Missing finite positive measured resource: '+name)
    return value


def training_reservation(canary):
    resources=canary['resources'];peaks=list(resources['per_gpu_peak_vram_mib'].values())
    if not peaks:raise ValueError('Canary has no measured NVML peak')
    nvml=max(finite_positive(x,'NVML') for x in peaks)
    rss=finite_positive(resources['peak_rss_mib'],'process-tree RSS')
    allocated=finite_positive(canary['gpu_allocated_peak_mib'],'allocated')
    reserved=finite_positive(canary['gpu_reserved_peak_mib'],'reserved')
    peak=max(nvml,allocated,reserved)
    vram_budget=math.ceil((peak+max(256,.05*peak))/256)*256
    rss_budget=math.ceil((rss+max(2048,.05*rss))/1024)*1024
    if vram_budget>VRAM_LIMIT or rss_budget>RSS_LIMIT:
        raise ValueError('Measured canary plus margin exceeds fixed 8192/32768 ceiling')
    return dict(vram_mib=vram_budget,rss_mib=rss_budget,measured_nvml_peak_mib=nvml,
        measured_allocated_peak_mib=allocated,measured_reserved_peak_mib=reserved,measured_rss_peak_mib=rss,
        margin='GPU max(256 MiB,5%), ceil256; RSS max(2048 MiB,5%), ceil1024')


def first_thirty(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('Sample stream record must be dictionary')
            row.pop('arm',None)
            for key in ('batch','im_file','cls','bboxes','teacher_cls','teacher_bboxes'):
                if key not in row:raise ValueError('Sample stream field missing: '+key)
            if type(row['batch']) is not int or not isinstance(row['im_file'],list) or len(row['im_file'])!=32:
                raise ValueError('Sample stream requires full B32')
            json.dumps(row,allow_nan=False);rows.append(row)
            if len(rows)==30:break
    if len(rows)!=30:raise ValueError('Need thirty actual batches')
    first=rows[0]['batch']
    if first not in (0,1) or [r['batch'] for r in rows]!=list(range(first,first+30)):
        raise ValueError('Sample stream is not the first contiguous thirty batches')
    return rows


def validate_configs(configs):
    if set(configs)!=set(ARMS):raise ValueError('Exactly new N/C0 configurations required')
    common=dict(dataset='llvip',scope=SCOPE,seed=42,epochs=3,imgsz=640,batch=32,nbs=64,workers=4,
        amp=True,optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,expected_train_images=2048,
        expected_val_images=2406,expected_nc=1,freeze_bn_running_statistics=True,localization_coefficient=0.)
    for arm,cfg in configs.items():
        if cfg.get('arm')!=arm:raise ValueError('Config arm differs')
        for key,value in common.items():
            if cfg.get(key)!=value:raise ValueError('Frozen confidence config differs: '+key)
        dose=0. if arm=='N' else .1
        if cfg.get('kd_coefficient')!=dose or cfg.get('classification_coefficient')!=dose:
            raise ValueError('Fixed original C0 dose differs; calibration is forbidden')
    for key in ('model','teacher','reference','paths','auxiliary_data_identity','augmentation','evidence',
                'torch_version','ultralytics_version'):
        if configs['N'][key]!=configs['C0'][key]:raise ValueError('Common confidence protocol differs: '+key)


def main_queue_terminal(value):
    if value.get('status')=='DIRECTION_MATRIX_PARTIAL_FAILURE':
        raise ValueError('Main queue partial failure; stop pending shared implementation review')
    if value.get('status') not in ('DIRECTION_MATRIX_COMPLETED','DIRECTION_MATRIX_COMPLETED_WITH_BLOCKED_ARMS'):
        raise ValueError('Original direction queue is not terminal')
    if value.get('seed')!=42 or value.get('epochs')!=3 or value.get('new_hash_computed') is not False:
        raise ValueError('Wrong original queue completion identity')
    return value['status']


def wait_for_main(path):
    # This background driver holds no GPU lease while waiting. A partially
    # written completion JSON is retried; a valid nonterminal identity fails.
    print('Waiting for original direction queue completion: '+str(path),flush=True)
    while True:
        if path.is_file():
            try:value=read(path)
            except json.JSONDecodeError:
                time.sleep(30);continue
            status=main_queue_terminal(value)
            return dict(completion=file_stat(path),status=status)
        time.sleep(30)


def check_terminal(stage,arm,result):
    statuses=dict(canary='CONFIDENCE_CANARY_COMPLETED',train='CONFIDENCE_TRAINING_COMPLETED',
                  eval='DIRECTION_EVALUATION_COMPLETED')
    fixed=dict(status=statuses[stage],scope=SCOPE,endpoint=ENDPOINT,dataset='llvip',arm=arm,seed=42,
        single_seed=True,formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False)
    for key,value in fixed.items():
        if result.get(key)!=value or (type(value) is bool and result.get(key) is not value):
            raise ValueError('Wrong confidence completion: '+stage+'/'+arm+'/'+key)
    if stage in ('canary','train'):
        if result.get('bn_running_buffers_unchanged') is not True or result.get('bn_affine_trainable') is not True or result.get('bn_buffer_count',0)<=0:
            raise ValueError('Actual BN freeze evidence missing')
        if type(result.get('successful_updates')) is not int or result['successful_updates']<24:
            raise ValueError('At least 24 successful canary/train updates required')
        if stage=='train' and (result.get('epochs_configured'),result.get('last_epoch'),result.get('batches'))!=(3,3,192):
            raise ValueError('Fixed full FT3 duration differs')
    elif (result.get('epochs'),result.get('full_dev_images'),result.get('full_dev_gt_objects'))!=(3,2406,7879):
        raise ValueError('Full LLVIP development evaluation required')


def check_canary(path,cfg,config_path):
    result=read(path);check_terminal('canary',cfg['arm'],result)
    for key in ('model','teacher','reference','kd_coefficient','classification_coefficient','localization_coefficient'):
        if result.get(key)!=cfg[key]:raise ValueError('Canary config identity differs: '+key)
    if Path(result['config_copy']).read_bytes()!=config_path.read_bytes():raise ValueError('Canary config bytes differ')
    initial=read(path.parent/'initialization_check.json')
    if (initial.get('status')!='PASS_FULL_STATE_WARM_START' or
        initial.get('initial_checkpoint',{}).get('path')!=str(Path(cfg['model']).resolve()) or
        any(initial.get(k) is not True for k in ('fresh_optimizer','fresh_ema','teacher_reference_isolated','head_included'))):
        raise ValueError('Actual fresh full-state initialization failed')
    return dict(initialization=initial,receipt=file_stat(path),resources=training_reservation(result))


def make_job(release,output,arm,stage,budget=None,reference=REFERENCE):
    cfg=output/'frozen_configs'/('llvip_'+arm+'_s42_FT3.yaml')
    canary=output/'canaries'/arm;run_dir=output/'runs'/arm
    name=re.sub(r'[^A-Za-z0-9_-]','_',output.name)
    if stage in ('canary','train'):
        destination=canary if stage=='canary' else run_dir
        command=[str(PY),str(release/'train_confidence.py'),'--reference-dir',str(reference),
                 '--config',str(cfg),'--output',str(destination)]
        command+=['--canary'] if stage=='canary' else ['--canary-receipt',str(canary/'canary.json')]
        receipt=destination/('canary.json' if stage=='canary' else 'completion_receipt.json')
        vram,rss=(VRAM_LIMIT,RSS_LIMIT) if stage=='canary' else (budget['vram_mib'],budget['rss_mib'])
    elif stage=='eval':
        destination=output/'evaluations'/arm
        command=[str(PY),str(release/'evaluate_confidence.py'),'--reference-dir',str(reference),
            '--config',str(cfg),'--checkpoint',str(run_dir/'weights/last.pt'),'--output',str(destination),
            '--native-config',str(release/'configs/llvip_native_evaluation.yaml')]
        receipt=destination/'direction_evaluation_receipt.json';vram,rss=2048,8192
    else:raise ValueError('Only fixed canary/train/eval stages; no calibration')
    return dict(id='confidence_'+name+'_'+arm+'_'+stage,dataset='llvip',arm=arm,stage=stage,formal=False,
        kind='eval' if stage=='eval' else 'train',command=command,expected_receipt=str(receipt),vram_mib=vram,rss_mib=rss)


def process(release,output,configs,executor,reference=REFERENCE):
    completed=[];checks={};frozen=output/'frozen_configs'
    config_paths={a:frozen/('llvip_'+a+'_s42_FT3.yaml') for a in ARMS}
    saved={a:p.read_bytes() for a,p in config_paths.items()}
    def execute(arm,stage,budget=None):
        if config_paths[arm].read_bytes()!=saved[arm]:raise ValueError('Frozen config changed')
        job=make_job(release,output,arm,stage,budget,reference)
        write_new(output/'queue'/(job['id']+'_job.json'),job);executor(job,output/'queue')
        path=Path(job['expected_receipt']);result=read(path);check_terminal(stage,arm,result)
        completed.append(dict(id=job['id'],arm=arm,stage=stage,receipt=str(path),receipt_validated=True))
        return path,result
    for arm in ARMS:
        path,_=execute(arm,'canary');checks[arm]=check_canary(path,configs[arm],config_paths[arm])
    streams={a:first_thirty(output/'canaries'/a/'sample_stream.jsonl') for a in ARMS}
    if streams['N']!=streams['C0']:raise ValueError('New N/C0 first30 RGB/IR streams differ')
    if checks['N']['initialization']['initial_checkpoint']!=checks['C0']['initialization']['initial_checkpoint']:
        raise ValueError('New N/C0 initialization checkpoint stat differs')
    write_new(output/'queue/canary_checks.json',dict(status='CONFIDENCE_CANARIES_MATCHED',scope=SCOPE,
        checks=checks,first30_stream_exact=True,new_hash_computed=False))
    for arm in ARMS:
        for stage in ('train','eval'):
            path,result=execute(arm,stage,checks[arm]['resources'])
            if stage=='train':
                for key in ('model','teacher','reference','kd_coefficient','classification_coefficient','localization_coefficient'):
                    if result.get(key)!=configs[arm][key]:raise ValueError('Training identity differs: '+key)
                if Path(result['config_copy']).read_bytes()!=saved[arm]:raise ValueError('Training config differs from canary')
                if first_thirty(path.parent/'sample_stream.jsonl')!=streams[arm]:raise ValueError('Full train first30 differs from canary')
                write_new(output/'queue'/(arm+'_training_stream_check.json'),dict(first30_exact=True,new_hash_computed=False))
    return completed


def run(args):
    release,output,reference=args.release_dir.resolve(),args.output.resolve(),args.reference_dir.resolve()
    if Path('/mnt/dataset/yudongfang') not in output.parents or output.exists():raise ValueError('New project data-disk output required')
    for p in (PY,reference/'runtime.py',DISPATCH/'resource_dispatch.py',release/'train_confidence.py',
              release/'evaluate_confidence.py',release/'configs/llvip_native_evaluation.yaml'):
        if not p.is_file():raise FileNotFoundError(p)
    paths={a:release/'configs'/('llvip_'+a+'_s42_FT3.yaml') for a in ARMS}
    configs={a:yaml.safe_load(p.read_text(encoding='utf-8')) for a,p in paths.items()};validate_configs(configs)
    output.mkdir(parents=True,exist_ok=False);(output/'queue').mkdir();(output/'frozen_configs').mkdir()
    for arm,p in paths.items():
        q=output/'frozen_configs'/p.name;shutil.copyfile(p,q)
        if p.read_bytes()!=q.read_bytes():raise ValueError('Frozen config copy differs')
    shutil.copyfile(__file__,output/'queue/executed_driver.py')
    if Path(__file__).read_bytes()!=(output/'queue/executed_driver.py').read_bytes():raise ValueError('Driver copy differs')
    write_new(output/'queue/manifest.json',dict(scope=SCOPE,endpoint=ENDPOINT,arms=list(ARMS),coefficients={'N':0.,'C0':.1},
        wait_for=str(args.wait_for),sequence='wait main terminal -> canary N/C0 -> first30 exact -> N train/eval -> C0 train/eval',
        python=str(PY),dispatcher=file_stat(DISPATCH/'resource_dispatch.py'),source=file_stat(Path(__file__)),
        calibration=False,new_resource_pool=False,ap_adaptive=False,new_hash_computed=False))
    started=time.time()
    try:
        preceding=wait_for_main(args.wait_for)
        write_new(output/'queue/main_queue_gate.json',preceding)
        sys.path.insert(0,str(DISPATCH));import resource_dispatch
        if Path(resource_dispatch.__file__).resolve()!=(DISPATCH/'resource_dispatch.py').resolve():raise ValueError('Wrong existing dispatcher')
        completed=process(release,output,configs,resource_dispatch.run_job,reference)
        write_new(output/'queue/completion.json',dict(status='CONFIDENCE_MATRIX_COMPLETED',scope=SCOPE,endpoint=ENDPOINT,
            dataset='llvip',seed=42,single_seed=True,epochs=3,train_images=2048,batches_per_arm=192,
            completed=completed,seconds=time.time()-started,preceding_main_queue=preceding,
            formal_e200_complete=False,formal_paper_gain_claim=False,ap_adaptive=False,new_hash_computed=False))
        return 0
    except Exception as error:
        write_new(output/'queue/failure.json',dict(status='CONFIDENCE_MATRIX_FAILED',scope=SCOPE,error=repr(error),
            traceback=traceback.format_exc(),seconds=time.time()-started,automatic_retry=False,new_hash_computed=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,default=REFERENCE);p.add_argument('--wait-for',type=Path,default=WAIT_FOR)
    raise SystemExit(run(p.parse_args()))
