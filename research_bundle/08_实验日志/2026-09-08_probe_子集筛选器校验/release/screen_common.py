"""Independent subset E8 contracts. No GPU, publication, or training on import."""
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import yaml

SCOPE='DRONE_SUBSET2048_PRETRAIN_E8_CHECK'
ENDPOINT='SUBSET2048_PRETRAIN_E8_LAST_EMA'
COEFFICIENTS={'N':0.,'C0':.1}
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign'
SUBSET=BASE+'/artifacts/rgbir_hourly_screen_20260908/subset_v1'
MODEL='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt'
FULL=BASE+'/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle'
FIXED=dict(dataset='dronevehicle',seed=42,epochs=8,expected_nc=5,imgsz=640,batch=32,nbs=64,
    workers=4,expected_train_images=2048,expected_val_images=1469,amp=True,source='paired',
    optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,warmup_epochs=3.,
    warmup_momentum=.8,warmup_bias_lr=.1,cos_lr=False,close_mosaic=0,patience=0,
    deterministic=True,localization_coefficient=0.,formal_training_authorized=False,
    torch_version='2.10.0+cu128',ultralytics_version='8.4.115',model=MODEL,
    teacher=BASE+'/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt',
    reference=BASE+'/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/rgb_seed42_native_b32a2/weights/last.pt',
    teacher_cache_images=64,log_every_batches=100,bn_running_statistics='normal_training',scope=SCOPE)
AUGMENTATION=dict(mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,shear=0.,perspective=0.,
    translate=.1,scale=.5,fliplr=.5,flipud=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,erasing=0.,bgr=0.)
EVIDENCE=dict(input_size=640,levels=[0,1],temperature=2.,background_scale=2.,minimum_foreground=1,
    minimum_background=4,match_iou=.5,reference_conf=.05,reference_iou=.1,teacher_conf=.25,
    teacher_iou=.5,rho=.5,target_clip=8.,smooth_l1_beta=1.)
PATHS=dict(student_data_yaml=SUBSET+'/data_rgb.yaml',privileged_data_yaml=SUBSET+'/data_infrared.yaml',
    paired_train_mapping=SUBSET+'/rgb_to_infrared_train.json')
AUXILIARY=dict(student_data_yaml=FULL+'/rgb.data.yaml',privileged_data_yaml=FULL+'/infrared.data.yaml')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def append_json(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def stat(path):
    p=Path(path);s=p.stat();return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def explicit(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def data_output(path):
    p=Path(path).resolve()
    if os.name!='posix' or not str(p).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('New outputs must be on the project data disk')
    return p
def copy_sources(output,paths):
    folder=Path(output)/'source_copies';folder.mkdir(exist_ok=False);rows=[]
    for i,p in enumerate(sorted({Path(p).resolve() for p in paths},key=str)):
        target=folder/('%04d_'%i+p.name);shutil.copyfile(p,target)
        if p.read_bytes()!=target.read_bytes():raise AssertionError('Source bytes differ')
        rows.append(dict(stat(p),copy=str(target),byte_identity=True))
    write_new(Path(output)/'source_manifest.json',dict(new_hash_computed=False,files=rows))

def validate_config(cfg):
    arm=cfg.get('arm')
    if arm not in COEFFICIENTS:raise ValueError('Only subset N/C0 allowed')
    for key,value in dict(FIXED,classification_coefficient=COEFFICIENTS[arm]).items():
        if cfg.get(key)!=value:raise ValueError('Subset fixed config differs: '+key)
        if type(value) in (int,bool) and type(cfg.get(key)) is not type(value):raise ValueError('Type differs: '+key)
    for key,value in [('augmentation',AUGMENTATION),('evidence',EVIDENCE),('paths',PATHS),('auxiliary_data_identity',AUXILIARY)]:
        if cfg.get(key)!=value:raise ValueError('Frozen nested config differs: '+key)
    s=cfg.get('short_screen',{})
    if (s.get('scope')!=SCOPE or s.get('endpoint')!=ENDPOINT or s.get('comparison_arms')!=['N','C0']
        or s.get('scheduler_horizon_epochs')!=8 or s.get('full_dev_gt_objects')!=22462
        or s.get('single_seed') is not True):raise ValueError('Explicit subset E8 identity required')
    return cfg
def load_config(path):return validate_config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
def common_identity(cfg):
    return dict({k:cfg[k] for k in FIXED},paths=cfg['paths'],auxiliary_data_identity=cfg['auxiliary_data_identity'],
        augmentation=cfg['augmentation'],evidence=cfg['evidence'],short_screen=cfg['short_screen'])

def canary_decision(successful,batches,attempts):
    if min(successful,batches,attempts)<0 or successful>attempts or attempts>batches:raise ValueError('Invalid counters')
    if attempts>48:raise RuntimeError('Canary exceeded48 optimizer attempts')
    if successful>=24 and batches>=30:return 'completed'
    if attempts>=48:raise RuntimeError('Canary48 attempts without24 successes/30 batches')
    return 'continue'

class ExecutionDeadline(TimeoutError):pass
class Deadline:
    def __init__(self,seconds):
        if not math.isfinite(seconds) or not 0<seconds<=2700:raise ValueError('Stage wall must be in (0,2700]')
        self.seconds=float(seconds);self.started=time.perf_counter()
    def check(self):
        if time.perf_counter()-self.started>=self.seconds:raise ExecutionDeadline('Stage wall budget exhausted; incomplete')
    @contextlib.contextmanager
    def armed(self):
        # The root watchdog independently enforces hard process-tree time limits.
        if hasattr(signal,'setitimer'):
            previous=signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(ExecutionDeadline('Stage wall expired')))
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.seconds-(time.perf_counter()-self.started)))
            try:yield
            finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        else:yield

def validate_amp_prior(cfg,path):
    binding=read(path)
    if binding.get('schema')!='executed_e8_amp_prior_v1':raise ValueError('Explicit executed AMP provenance required')
    paths={key:Path(binding[key]) for key in ('configuration','runtime_ready','training_receipt')}
    c=yaml.safe_load(paths['configuration'].read_text(encoding='utf-8'))
    ready=read(paths['runtime_ready']);receipt=read(paths['training_receipt'])
    if (ready.get('status')!='ready' or ready.get('amp') is not True or ready.get('train_images')!=17990
        or ready.get('class_names')!={str(i):n for i,n in enumerate(('car','freight car','truck','bus','van'))}
        or receipt.get('status')!='SHORT_SCREEN_TRAINING_COMPLETED' or receipt.get('scope')!='SHORT_SCREEN'
        or receipt.get('arm')!='N' or receipt.get('last_epoch')!=8):raise ValueError('Missing actual completed same-model E8 AMP')
    for key in ('model','teacher','reference','torch_version','ultralytics_version','amp','expected_nc','imgsz','batch','workers'):
        if c.get(key)!=cfg[key]:raise ValueError('AMP prior configuration differs: '+key)
    for key in ('model','teacher','reference'):
        if receipt.get(key)!=cfg[key]:raise ValueError('AMP prior receipt model differs: '+key)
    if paths['configuration'].resolve()!=Path(receipt['configuration']).resolve():raise ValueError('AMP prior config provenance differs')
    if len({p.resolve().parent for p in paths.values()})!=1:raise ValueError('AMP prior must be one original executed run')
    return dict(actual_amp=True,prior_sources={k:stat(p) for k,p in paths.items()},same_version_model_paths=True,
        historical_checkpoint_content_equality_claim=False)

def setup_with_validated_amp(trainer,setup,trainer_module):
    original=trainer_module.check_amp;calls=[]
    def checked(model):
        if model is not trainer.model:raise ValueError('Wrong setup AMP model')
        calls.append(1);return True
    trainer_module.check_amp=checked
    try:setup()
    finally:trainer_module.check_amp=original
    if calls!=[1] or bool(trainer.amp) is not True:raise ValueError('Validated AMP setup changed')
    return dict(new_scope_setup_check_amp_bypass=True,actual_autocast=True,unrelated_model_forward=False,
        original_check_amp_restored=True,setup_bitwise_equivalence_claim=False)

TENSOR_KEYS=('img','strong_img','cls','bboxes','batch_idx','strong_cls','strong_bboxes','strong_batch_idx')
def flow_payload(batch,torch):
    # Exact augmented loader batch, BEFORE preprocessing/float normalization.
    tensors={k:batch[k].detach().cpu().contiguous() for k in TENSOR_KEYS}
    for key in ('img','strong_img'):
        t=tensors[key]
        if t.dtype!=torch.uint8 or tuple(t.shape)!=(32,3,640,640):raise ValueError('Loader image contract differs: '+key)
        if str(batch[key].device)!='cpu':raise ValueError('Expected pre-device loader images')
    return dict(tensors=tensors,metadata=dict(im_file=list(batch['im_file']),pair_info=batch['pair_info']))
def tensor_equal(actual,expected,torch):
    return (actual.dtype==expected.dtype and actual.shape==expected.shape and
        str(actual.device)==str(expected.device) and torch.equal(actual,expected))
def assert_flow_equal(actual,expected,torch):
    if actual['metadata']!=expected['metadata'] or set(actual['tensors'])!=set(expected['tensors']):raise AssertionError('Source/augmentation/GT keys differ')
    for key in TENSOR_KEYS:
        if not tensor_equal(actual['tensors'][key],expected['tensors'][key],torch):raise AssertionError('Full tensor differs: '+key)
def load_tensors(path,torch):
    # Files are only our private local project artifacts, never arbitrary inputs.
    try:return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(path,map_location='cpu')

class FlowAudit:
    def __init__(self,folder,writer,cfg,output,torch):
        self.folder=Path(folder);self.writer=writer;self.torch=torch;self.output=Path(output)
        self.seconds=0.;self.initial_seconds=0.;self.batch_seconds={};self.rows=[]
        if writer:
            self.folder.mkdir(parents=True,exist_ok=False)
            write_new(self.folder/'identity.json',dict(scope=SCOPE,owner='N_canary',common=common_identity(cfg),
                augmented_loader_uint8_not_original_image_files=True,remote_only=True,new_hash_computed=False))
        elif read(self.folder/'identity.json')['common']!=common_identity(cfg):raise ValueError('Flow common config differs')
    def initial(self,model,cfg):
        start=time.perf_counter();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        path=self.folder/'initial_student.pt'
        if self.writer:self.torch.save(state,path)
        expected=load_tensors(path,self.torch)
        if state.keys()!=expected.keys() or any(not tensor_equal(state[k],expected[k],self.torch) for k in state):
            raise AssertionError('Full initialized5-class student state differs')
        del expected
        result=dict(status='PASS',reference_state=stat(path),tensors=len(state),head_included=True,
            cross_arm_initial_state_exact=None if self.writer else True,
            state_reference_written_and_verified=self.writer,scope='initialized_student_all_parameters_and_buffers',
            equality_to_original80class_pretrained_not_claimed=True,remote_only_tensor=True)
        del state;self.initial_seconds=time.perf_counter()-start;return result
    def batch(self,batch,index):
        if index>30:return
        start=time.perf_counter();payload=flow_payload(batch,self.torch);path=self.folder/('batch_%02d.pt'%index)
        if self.writer:
            if path.exists():raise FileExistsError(path)
            self.torch.save(payload,path)
        expected=load_tensors(path,self.torch);assert_flow_equal(payload,expected,self.torch);del expected
        row=dict(batch=index,reference=stat(path),mode='reference_created' if self.writer else 'torch_equal',
            full_rgb_ir_pixels_and_dual_labels_exact=True,scope='first30_augmented_loader_batch_uint8_before_preprocess',
            tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),device=str(v.device)) for k,v in payload['tensors'].items()},
            im_file=payload['metadata']['im_file'],pair_info=payload['metadata']['pair_info'],
            labels={k:v.tolist() for k,v in payload['tensors'].items() if k not in ('img','strong_img')})
        del payload;append_json(self.output/'sample_stream.jsonl',row)
        elapsed=time.perf_counter()-start;self.seconds+=elapsed;self.batch_seconds[index]=elapsed
        self.rows.append(dict(batch=index,reference=row['reference'],exact=True,seconds=elapsed))
    def finish(self):
        if [r['batch'] for r in self.rows]!=list(range(1,31)):raise AssertionError('Exactly30 actual flow batches required')
        result=dict(status='PASS',reference_dir=str(self.folder),reference_written=self.writer,
            prefix_batches=30,exact_pixels_and_labels=True,records=30,first30_pixels_labels_exact=True,
            equality_method='shape_dtype_device_and_torch_equal_all_elements',remote_only_tensors=True,
            files=self.rows,new_hash_computed=False)
        write_new(self.output/'flow_check.json',result);return result
