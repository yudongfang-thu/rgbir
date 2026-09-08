"""Fixed subset fine-tuning screen. Separate from formal E200 evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import yaml
import math

ENDPOINT='OBJECT_DFL_FT3_BNFROZEN_LAST_EMA'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def stat(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def load_config(path):
    c=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if c.get('dataset') != 'llvip':raise ValueError('Dataset not admitted')
    arms=('N','L3-DFL','L3-GT')
    if c.get('arm') not in arms:raise ValueError('Arm not admitted')
    fixed=dict(seed=42,epochs=3,imgsz=640,batch=32,nbs=64,workers=4,amp=True,
        source='paired',optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,
        expected_train_images=2048,freeze_bn_running_statistics=True,scope='OBJECT_DFL_FT3_BNFROZEN')
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError('Direction config differs: '+k)
    if not 0<=c['kd_coefficient']<=1:raise ValueError('Invalid fixed coefficient')
    if c['arm']=='N' and c['kd_coefficient']!=0:raise ValueError('N must have zero dose')
    validate_method(c)
    return c


def data_output(path):
    p=Path(path).resolve()
    if os.name!='posix' or not str(p).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Output must be a new directory on the project data disk')
    return p

def explicit(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def copy_sources(output,paths):
    dest=output/'source_copies';dest.mkdir(exist_ok=False);rows=[]
    for i,p in enumerate(sorted({Path(p).resolve() for p in paths},key=str)):
        target=dest/('%04d_'%i+p.name);shutil.copyfile(p,target)
        if p.read_bytes()!=target.read_bytes():raise AssertionError('Source copy differs')
        rows.append(dict(stat(p),copy=str(target),byte_identity=True))
    write_new(output/'source_manifest.json',dict(new_hash_computed=False,files=rows))

METHOD_IDENTITY='OBJECT_DFL_FT3_BNFROZEN'
METHOD=dict(temperature=2.0, teacher_anchor='same_reference_index',
    normalization='coarse_reference_base_before_quality', teacher_transport='object_coordinate_full_mass',
    gt_target='sqrt_adjacent_bins_normalized', full_support='reject_any_positive_mass_outside_0_15',
    native_loss_replaced=False, physical_registration_claim=False)

def validate_method(cfg):
    if cfg.get('method_identity')!=METHOD_IDENTITY or cfg.get('object_dfl')!=METHOD:
        raise ValueError('Frozen object DFL method identity differs')
    if cfg.get('classification_coefficient')!=0:
        raise ValueError('Object DFL has no classification KD')
    if cfg.get('localization_coefficient')!=cfg['kd_coefficient']:
        raise ValueError('Single localization coefficient must equal outer KD dose')
    if cfg.get('direction_screen',{}).get('scope')!=METHOD_IDENTITY:
        raise ValueError('Nested method scope differs')


def validate_amp_prior(cfg, prior_path=None, config_path=None):
    prior_path=Path(prior_path) if prior_path is not None else Path(__file__).with_name('validated_amp_prior.json')
    config_path=Path(config_path) if config_path is not None else Path(__file__).with_name('validated_amp_prior_config.yaml')
    prior=read(prior_path);prior_cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if (prior.get('status')!='RAW_DFL_SINGLE_BATCH_COMPLETED' or prior.get('scope')!='RAW_DFL_SINGLE_BATCH'
            or prior.get('actual_amp') is not True or cfg['amp'] is not True
            or prior.get('dataset')!='llvip' or prior.get('seed')!=42):
        raise ValueError('Missing actual same-model LLVIP AMP validation')
    if prior.get('initialization')!={k:stat(cfg[k]) for k in ('model','teacher','reference')}:
        raise ValueError('AMP prior initialization path/stat differs')
    for k in ('torch_version','ultralytics_version','model','teacher','reference','amp'):
        if prior_cfg[k]!=cfg[k]:raise ValueError('AMP prior configuration differs: '+k)
    return dict(actual_amp=True,prior_receipt=stat(prior_path),prior_configuration=stat(config_path),
        prior_current_forward_id=prior['current_forward_id'],initialization=prior['initialization'])


def setup_with_validated_amp(trainer, setup, cfg_amp, prior_amp, trainer_module):
    """Same scoped check_amp binding as the executed raw DFL probe; actual AMP unchanged."""
    if cfg_amp is not True or prior_amp is not True:raise ValueError('Validated AMP must remain enabled')
    original=trainer_module.check_amp;calls=[]
    def validated(model):
        if model is not trainer.model:raise ValueError('Unexpected setup AMP model')
        calls.append(1);return cfg_amp
    trainer_module.check_amp=validated
    try:setup()
    finally:trainer_module.check_amp=original
    if len(calls)!=1 or bool(trainer.amp)!=cfg_amp:raise ValueError('Setup bypass/actual AMP differs')
    return dict(new_protocol_specific_setup_check_amp_bypass=True,validated_value=cfg_amp,calls=len(calls),
        original_binding_restored=True,setup_equivalence_claim=False,unrelated_model_forward=False,
        actual_autocast_unchanged=True)


def calibration_resource_check(resources, allocated, reserved):
    values=list(resources['per_gpu_peak_vram_mib'].values())
    all_values=values+[resources['peak_rss_mib'],allocated,reserved]
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in all_values):
        raise ValueError('Missing finite actual calibration peaks')
    peak=max(values+[allocated,reserved]);rss=resources['peak_rss_mib']
    vram_budget=math.ceil((peak+max(256,.05*peak))/256)*256
    rss_budget=math.ceil((rss+max(2048,.05*rss))/1024)*1024
    passed=vram_budget<=8192 and rss_budget<=32768
    return dict(status='PASS' if passed else 'RESOURCE_LIMIT_EXCEEDED',resources=resources,
        gpu_allocated_peak_mib=allocated,gpu_reserved_peak_mib=reserved,
        measured_with_margin_vram_mib=vram_budget,measured_with_margin_rss_mib=rss_budget,
        limits=dict(vram_mib=8192,rss_mib=32768),new_hash_computed=False)
