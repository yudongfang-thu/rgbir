"""One fixed F-rel-GM canary/train/eval, using the existing global lease.

Waits for confidence success without a GPU lease. The completed main N/C1
controls are reused only after explicit configuration, initialization and
actual first-thirty-batch checks. No metrics choose or modify this queue.
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

ARM='F-rel-GM'
SCOPE='FEATURE_RELATION_GM_FT3'
ENDPOINT='FEATURE_RELATION_GM_FT3_LAST_EMA'
CONTROL_SCOPE='DIRECTION_FT3_BNFROZEN'
CONTROL_ENDPOINT='DIRECTION_FT3_BNFROZEN_LAST_EMA'
COEFFICIENT=14.438521129817886
BASE=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python')
REFERENCE=BASE/'rgbir_independent_kd_v2_20260907/release_gpu5'
DISPATCH=BASE/'rgbir_task_conditional_v1_20260907/release_v8'
NATIVE_BINDING=BASE/'rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evaluation_profile_binding/binding.json'
MAIN=BASE/'rgbir_direction_screen_20260908/screen_attempt1'
WAIT_FOR=BASE/'rgbir_direction_screen_20260908/confidence_attempt1/queue/completion.json'
VRAM_LIMIT,RSS_LIMIT=8192,32768
COMMON_FIELDS=('dataset','student_modality','privileged_modality','model','teacher','reference','paths',
    'auxiliary_data_identity','native_contract_config','augmentation','evidence','source',
    'seed','epochs','imgsz','batch','nbs','workers','amp','optimizer','lr0','lrf','momentum',
    'weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','cos_lr','close_mosaic',
    'patience','deterministic','expected_train_images','expected_val_images','expected_nc',
    'freeze_bn_running_statistics','torch_version','ultralytics_version','teacher_cache_images')
SCREEN_FIELDS=('subset_seed','training_images','batches_per_epoch','initialization',
    'fresh_optimizer','fresh_ema','not_from_scratch')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def file_stat(path):
    p=Path(path);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def config(path):return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def finite_positive(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('Missing finite positive measured resource: '+name)
    return value


def training_reservation(canary):
    r=canary['resources'];peaks=list(r['per_gpu_peak_vram_mib'].values())
    if not peaks:raise ValueError('Canary has no measured NVML peak')
    nvml=max(finite_positive(x,'NVML') for x in peaks)
    rss=finite_positive(r['peak_rss_mib'],'process-tree RSS')
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
            if not isinstance(row,dict):raise ValueError('Stream record must be dictionary')
            row.pop('arm',None)
            for key in ('batch','im_file','cls','bboxes','batch_idx','teacher_cls','teacher_bboxes','teacher_batch_idx'):
                if key not in row:raise ValueError('Stream field missing: '+key)
            if type(row['batch']) is not int or not isinstance(row['im_file'],list) or len(row['im_file'])!=32:
                raise ValueError('Sample stream requires full B32')
            json.dumps(row,allow_nan=False);rows.append(row)
            if len(rows)==30:break
    if len(rows)!=30:raise ValueError('Need thirty actual batches')
    first=rows[0]['batch']
    if first not in (0,1) or [r['batch'] for r in rows]!=list(range(first,first+30)):
        raise ValueError('Stream must contain first contiguous thirty batches')
    return rows


def validate_candidate(cfg):
    fixed=dict(dataset='drone',scope=SCOPE,arm=ARM,seed=42,epochs=3,imgsz=640,batch=32,nbs=64,workers=4,
        amp=True,optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,expected_train_images=2048,
        expected_val_images=1469,expected_nc=5,freeze_bn_running_statistics=True,localization_coefficient=0.,
        kd_coefficient=COEFFICIENT,classification_coefficient=COEFFICIENT)
    for key,value in fixed.items():
        if cfg.get(key)!=value:raise ValueError('Frozen F-rel-GM configuration differs: '+key)
    if cfg['direction_screen']['scope']!=SCOPE or cfg['direction_screen']['endpoint']!=ENDPOINT:
        raise ValueError('Wrong nested candidate identity')


def compare_common(candidate,control,arm):
    if control.get('scope')!=CONTROL_SCOPE or control.get('arm')!=arm:
        raise ValueError('Wrong main control scope/arm')
    for key in COMMON_FIELDS:
        if key not in candidate or key not in control or candidate[key]!=control[key]:
            raise ValueError('Cross-scope common configuration differs: '+arm+'/'+key)
    for key in SCREEN_FIELDS:
        if candidate['direction_screen'][key]!=control['direction_screen'][key]:
            raise ValueError('Cross-scope screen configuration differs: '+arm+'/'+key)
    if control.get('localization_coefficient')!=0.:
        raise ValueError('Main control localization must be zero')
    if arm=='N' and (control.get('kd_coefficient'),control.get('classification_coefficient'))!=(0.,0.):
        raise ValueError('Main N must have zero KD')
    if arm=='C1' and not (finite_positive(control.get('kd_coefficient'),'main C1 dose')==control.get('classification_coefficient')):
        raise ValueError('Main C1 dose fields differ')


def preceding_success(value):
    fixed=dict(status='CONFIDENCE_MATRIX_COMPLETED',scope='LLVIP_CONFIDENCE_FT3',seed=42,epochs=3,
        formal_e200_complete=False,new_hash_computed=False)
    for key,expected in fixed.items():
        if value.get(key)!=expected:raise ValueError('Confidence queue has not completed successfully: '+key)
    return value['status']


def wait_for_confidence(path):
    print('Waiting without GPU lease for confidence success: '+str(path),flush=True)
    while True:
        if path.is_file():
            try:value=read(path)
            except json.JSONDecodeError:time.sleep(30);continue
            return dict(completion=file_stat(path),status=preceding_success(value))
        if (path.parent/'failure.json').is_file():
            raise ValueError('Confidence queue failed; stop pending implementation review')
        time.sleep(30)


def check_terminal(stage,arm,result,control=False):
    fixed=dict(status={'canary':'DIRECTION_CANARY_COMPLETED','train':'DIRECTION_TRAINING_COMPLETED',
        'eval':'DIRECTION_EVALUATION_COMPLETED'}[stage],scope=CONTROL_SCOPE if control else SCOPE,
        endpoint=CONTROL_ENDPOINT if control else ENDPOINT,dataset='drone',arm=arm,seed=42,single_seed=True,
        formal_e200_complete=False,new_hash_computed=False,official_test_accessed=False)
    for key,value in fixed.items():
        if result.get(key)!=value or (type(value) is bool and result.get(key) is not value):
            raise ValueError('Wrong completion identity: '+stage+'/'+arm+'/'+key)
    if stage in ('canary','train'):
        if result.get('bn_running_buffers_unchanged') is not True or result.get('bn_affine_trainable') is not True or result.get('bn_buffer_count',0)<=0:
            raise ValueError('Actual BN freeze evidence missing')
        if type(result.get('successful_updates')) is not int or result['successful_updates']<24:
            raise ValueError('At least 24 successful updates required')
        if stage=='train' and (result.get('epochs_configured'),result.get('last_epoch'),result.get('batches'))!=(3,3,192):
            raise ValueError('Fixed FT3 training duration differs')
    elif (result.get('epochs'),result.get('full_dev_images'),result.get('full_dev_gt_objects'))!=(3,1469,22462):
        raise ValueError('Full Drone development evaluation required')


def training_identity(result,cfg,config_path):
    for key in ('model','teacher','reference','kd_coefficient','classification_coefficient','localization_coefficient'):
        if result.get(key)!=cfg[key]:raise ValueError('Training/config identity differs: '+key)
    if Path(result['config_copy']).read_bytes()!=config_path.read_bytes():raise ValueError('Executed config bytes differ')


def initialization(path,cfg):
    value=read(path)
    if (value.get('status')!='PASS_FULL_STATE_WARM_START' or
        value.get('initial_checkpoint',{}).get('path')!=str(Path(cfg['model']).resolve()) or
        any(value.get(k) is not True for k in ('fresh_optimizer','fresh_ema','teacher_reference_isolated','head_included'))):
        raise ValueError('Actual fresh full-state initialization failed')
    if value['initial_checkpoint']!=file_stat(Path(cfg['model']).resolve()):raise ValueError('Initial checkpoint stat changed')
    return value['initial_checkpoint']


def matched_controls(main,output,cfg,config_path):
    canary_path=output/'canaries'/ARM/'canary.json';canary=read(canary_path)
    training_identity(canary,cfg,config_path)
    initial=initialization(canary_path.parent/'initialization_check.json',cfg)
    stream=first_thirty(canary_path.parent/'sample_stream.jsonl');controls={}
    for arm in ('N','C1'):
        cp=main/'effective_configs'/('drone_'+arm+'_s42_FT3.yaml');ccfg=config(cp)
        compare_common(cfg,ccfg,arm)
        cap=main/'canaries/drone'/arm/'canary.json';trp=main/'runs/drone'/arm/'completion_receipt.json'
        evp=main/'evaluations/drone'/arm/'direction_evaluation_receipt.json'
        ca,tr,ev=read(cap),read(trp),read(evp)
        for stage,value in (('canary',ca),('train',tr),('eval',ev)):check_terminal(stage,arm,value,True)
        for result in (ca,tr):training_identity(result,ccfg,cp)
        for directory in (cap.parent,trp.parent):
            if initialization(directory/'initialization_check.json',ccfg)!=initial:
                raise ValueError('Initial checkpoint stat differs: '+arm)
            if first_thirty(directory/'sample_stream.jsonl')!=stream:
                raise ValueError('Actual first30 differs: '+arm+'/'+directory.name)
        if ev['checkpoint']!=tr['checkpoint'] or tr['checkpoint']!=file_stat(trp.parent/'weights/last.pt'):
            raise ValueError('Main train/evaluation checkpoint stat differs: '+arm)
        if ev.get('training_model')!=ccfg['model'] or ev.get('kd_coefficient')!=ccfg['kd_coefficient']:
            raise ValueError('Main evaluation method identity differs: '+arm)
        if ev.get('training_configuration')!=str(cp.resolve()) or ev.get('training_completion')!=str(trp.resolve()):
            raise ValueError('Main evaluation provenance path differs: '+arm)
        controls[arm]=dict(common_config_fields_exact=True,first30_canary_exact=True,
            initial_checkpoint_stat_exact=True,control_full_train_first30_exact=True,
            control_evaluation_receipt=file_stat(evp),control_training_receipt=file_stat(trp),control_config=file_stat(cp),
            control_canary_receipt=file_stat(cap),common_config_fields=list(COMMON_FIELDS),screen_fields=list(SCREEN_FIELDS))
    result=dict(status='FEATURE_GM_MATCHED_CONTROL_VERIFIED',scope=SCOPE,candidate_scope=SCOPE,candidate_arm=ARM,
        control_scope=CONTROL_SCOPE,dataset='drone',seed=42,controls=controls,initial_checkpoint=initial,
        candidate_configuration=file_stat(config_path),candidate_canary_receipt=file_stat(canary_path),
        fixed_candidate_coefficient=COEFFICIENT,batch_records=30,batch_size=32,
        comparison_scope='single-seed mature warm-start FT3, explicit cross-scope control projection',
        checkpoint_tensor_equality_source='existing executed full-state initialization receipts; no new weight loads',
        new_hash_computed=False,formal_e200_complete=False,formal_paper_gain_claim=False)
    write_new(output/'queue/matched_control_projection.json',result)
    return result,stream


def make_job(release,output,stage,budget=None,reference=REFERENCE,native_config=None):
    cfg=output/'frozen_configs/drone_F-rel-GM_s42_FT3.yaml';ca=output/'canaries'/ARM;run=output/'runs'/ARM
    name=re.sub(r'[^A-Za-z0-9_-]','_',output.name)
    if stage in ('canary','train'):
        destination=ca if stage=='canary' else run
        command=[str(PY),str(release/'train_feature_gm.py'),'--reference-dir',str(reference),
            '--config',str(cfg),'--output',str(destination)]
        command+=['--canary'] if stage=='canary' else ['--canary-receipt',str(ca/'canary.json')]
        receipt=destination/('canary.json' if stage=='canary' else 'completion_receipt.json')
        vram,rss=(VRAM_LIMIT,RSS_LIMIT) if stage=='canary' else (budget['vram_mib'],budget['rss_mib'])
    elif stage=='eval':
        if native_config is None:raise ValueError('Explicit original native configuration required')
        destination=output/'evaluations'/ARM
        command=[str(PY),str(release/'evaluate_feature_gm.py'),'--reference-dir',str(reference),
            '--config',str(cfg),'--checkpoint',str(run/'weights/last.pt'),'--output',str(destination),
            '--native-config',str(native_config),'--native-profile-binding',str(NATIVE_BINDING)]
        receipt=destination/'direction_evaluation_receipt.json';vram,rss=2048,8192
    else:raise ValueError('Only canary/train/eval; no calibration or retries')
    return dict(id='feature_gm_'+name+'_'+stage,dataset='drone',arm=ARM,stage=stage,formal=False,
        kind='eval' if stage=='eval' else 'train',command=command,expected_receipt=str(receipt),vram_mib=vram,rss_mib=rss)


def process(release,output,cfg,executor,main=MAIN,reference=REFERENCE):
    cp=output/'frozen_configs/drone_F-rel-GM_s42_FT3.yaml';saved=cp.read_bytes();completed=[]
    def execute(stage,budget=None):
        if cp.read_bytes()!=saved:raise ValueError('Frozen candidate configuration changed')
        job=make_job(release,output,stage,budget,reference,cfg['native_contract_config'])
        write_new(output/'queue'/(job['id']+'_job.json'),job);executor(job,output/'queue')
        path=Path(job['expected_receipt']);value=read(path);check_terminal(stage,ARM,value)
        completed.append(dict(id=job['id'],stage=stage,receipt=str(path),receipt_validated=True))
        return path,value
    _,canary=execute('canary');budget=training_reservation(canary)
    projection,stream=matched_controls(main,output,cfg,cp)
    write_new(output/'queue/canary_budget.json',budget)
    trp,tr=execute('train',budget);training_identity(tr,cfg,cp)
    if initialization(trp.parent/'initialization_check.json',cfg)!=projection['initial_checkpoint']:
        raise ValueError('Full training initial checkpoint differs')
    if first_thirty(trp.parent/'sample_stream.jsonl')!=stream:raise ValueError('Full candidate train first30 differs')
    write_new(output/'queue/candidate_training_stream_check.json',dict(status='PASS',first30_exact=True,new_hash_computed=False))
    _,ev=execute('eval')
    if ev['checkpoint']!=tr['checkpoint'] or tr['checkpoint']!=file_stat(trp.parent/'weights/last.pt'):
        raise ValueError('Candidate train/evaluation checkpoint differs')
    if ev.get('kd_coefficient')!=COEFFICIENT or ev.get('training_model')!=cfg['model']:
        raise ValueError('Candidate evaluation dose/init differs')
    if ev.get('training_configuration')!=str(cp.resolve()) or ev.get('training_completion')!=str(trp.resolve()):
        raise ValueError('Candidate evaluation provenance path differs')
    return completed


def run(args):
    release,output,reference=args.release_dir.resolve(),args.output.resolve(),args.reference_dir.resolve()
    if Path('/mnt/dataset/yudongfang') not in output.parents or output.exists():raise ValueError('New project data-disk output required')
    cp=release/'configs/drone_F-rel-GM_s42_FT3.yaml';cfg=config(cp);validate_candidate(cfg)
    for p in (PY,reference/'runtime.py',DISPATCH/'resource_dispatch.py',release/'train_feature_gm.py',
              release/'evaluate_feature_gm.py',Path(cfg['native_contract_config']),NATIVE_BINDING):
        if not p.is_file():raise FileNotFoundError(p)
    output.mkdir(parents=True,exist_ok=False);(output/'queue').mkdir();(output/'frozen_configs').mkdir()
    shutil.copyfile(cp,output/'frozen_configs'/cp.name)
    if cp.read_bytes()!=(output/'frozen_configs'/cp.name).read_bytes():raise ValueError('Frozen configuration copy differs')
    shutil.copyfile(__file__,output/'queue/executed_driver.py')
    if Path(__file__).read_bytes()!=(output/'queue/executed_driver.py').read_bytes():raise ValueError('Driver copy differs')
    write_new(output/'queue/manifest.json',dict(scope=SCOPE,endpoint=ENDPOINT,arm=ARM,coefficient=COEFFICIENT,
        wait_for=str(args.wait_for),main_screen=str(args.main_screen),controls=['N','C1'],
        sequence='wait confidence success -> F canary24 -> matched controls -> F train192 -> full native eval',
        python=str(PY),source=file_stat(Path(__file__)),dispatcher=file_stat(DISPATCH/'resource_dispatch.py'),
        calibration=False,new_resource_pool=False,ap_adaptive=False,new_hash_computed=False))
    started=time.time()
    try:
        preceding=wait_for_confidence(args.wait_for);write_new(output/'queue/confidence_queue_gate.json',preceding)
        sys.path.insert(0,str(DISPATCH));import resource_dispatch
        if Path(resource_dispatch.__file__).resolve()!=(DISPATCH/'resource_dispatch.py').resolve():raise ValueError('Wrong existing dispatcher')
        completed=process(release,output,cfg,resource_dispatch.run_job,args.main_screen,reference)
        write_new(output/'queue/completion.json',dict(status='FEATURE_GM_QUEUE_COMPLETED',scope=SCOPE,endpoint=ENDPOINT,
            dataset='drone',arm=ARM,seed=42,single_seed=True,epochs=3,train_images=2048,batches=192,
            completed=completed,seconds=time.time()-started,preceding_confidence_queue=preceding,
            matched_control_projection=file_stat(output/'queue/matched_control_projection.json'),
            formal_e200_complete=False,formal_paper_gain_claim=False,ap_adaptive=False,new_hash_computed=False))
        return 0
    except Exception as error:
        write_new(output/'queue/failure.json',dict(status='FEATURE_GM_QUEUE_FAILED',scope=SCOPE,error=repr(error),
            traceback=traceback.format_exc(),seconds=time.time()-started,automatic_retry=False,new_hash_computed=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,default=REFERENCE);p.add_argument('--main-screen',type=Path,default=MAIN)
    p.add_argument('--wait-for',type=Path,default=WAIT_FOR)
    raise SystemExit(run(p.parse_args()))
