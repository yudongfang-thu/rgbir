"""Fixed subset fine-tuning screen. Separate from formal E200 evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import yaml

COEFFICIENTS={'N':0.,'C0':.1}
ENDPOINT='LLVIP_CONFIDENCE_FT3_LAST_EMA'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign'
INITIAL=BASE+'/runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def stat(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def load_config(path):
    c=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if c.get('dataset')!='llvip':raise ValueError('Only LLVIP admitted')
    arms=('N','C0')
    if c.get('arm') not in arms:raise ValueError('Arm not admitted')
    fixed=dict(seed=42,epochs=3,imgsz=640,batch=32,nbs=64,workers=4,amp=True,
        source='paired',optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,
        expected_train_images=2048,freeze_bn_running_statistics=True,scope='LLVIP_CONFIDENCE_FT3')
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError('Direction config differs: '+k)
    coefficient=0. if c['arm']=='N' else .1
    for key in ('kd_coefficient','kd_weight','classification_coefficient'):
        if c.get(key)!=coefficient:raise ValueError('Original C0 fixed coefficient differs: '+key)
    if c.get('localization_coefficient')!=0.:raise ValueError('No localization loss permitted')
    if c.get('expected_val_images')!=2406 or c.get('expected_nc')!=1:raise ValueError('Full LLVIP dev identity differs')
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
