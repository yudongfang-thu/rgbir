"""Fixed subset fine-tuning screen. Separate from formal E200 evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import yaml

COEFFICIENTS={'N':0.,'C0':.1,'C1':0.09227393550836771}
ENDPOINT='HOURLY_SCREEN_FT_E3_LAST_EMA'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign'
INITIAL=BASE+'/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def stat(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def load_config(path):
    c=yaml.safe_load(Path(path).read_text(encoding='utf-8'));arm=c.get('arm')
    if arm not in COEFFICIENTS:raise ValueError('Hourly screen only admits N/C0/C1')
    fixed=dict(dataset='dronevehicle',seed=42,epochs=3,expected_nc=5,imgsz=640,batch=32,nbs=64,
        workers=4,expected_train_images=2048,expected_val_images=1469,amp=True,source='paired',
        optimizer='SGD',lr0=.001,lrf=.1,warmup_epochs=0.,model=INITIAL,kd_weight=.1,
        classification_coefficient=COEFFICIENTS[arm],localization_coefficient=0.,formal_training_authorized=False)
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError('Hourly config differs: '+k)
    s=c.get('hourly_screen',{})
    if (s.get('scope')!='HOURLY_SCREEN_FT' or s.get('endpoint')!=ENDPOINT or
        s.get('subset_seed')!=20260908 or s.get('single_seed') is not True):
        raise ValueError('Missing independent FT screen identity')
    if set(c['auxiliary_data_identity'])!={'student_data_yaml','privileged_data_yaml'}:
        raise ValueError('Frozen teacher/reference provenance required')
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
