"""Verify delivered probe integrity; no inference or new experiment outcomes."""
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime,timezone
import numpy as np
from PIL import Image

p=Path(__file__).resolve().parent
read=lambda f:json.loads(f.read_text(encoding='utf-8'))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
checked=[]
source=p/'remote_execution/probe_rgbir_v2.py'
for d,n in [('dronevehicle',200),('llvip',200),('vedai',121)]:
    base=p/(d+'_full');s=read(base/'summary.json');m=read(base/'input_manifest.json')
    c=read(base/'completion_receipt.json');r=read(base/'prediction_records.json')
    o=read(base/'matched_objects.json');f=read(base/'feature_metrics.json')
    assert c['status']=='probe_completed' and c['n_images']==n and not c['sealed_test_accessed']
    assert s['n_images']==n and m['n']==n and not m['canary']
    assert m['script_sha256']==sha(source)
    assert [x['id'] for x in r]==m['sample_ids'] and len(set(m['sample_ids']))==n
    assert len(o)==s['labels']['matched_n']
    assert sum(s['common_object_hits'].values())==len(o)
    for key,a,b in [('both',1,1),('rgb_only',1,0),('ir_only',0,1),('neither',0,0)]:
        assert sum(x['rgb_hit']==a and x['ir_hit']==b for x in o)==s['common_object_hits'][key]
    names=read(base/'class_names.json');assert names[0]==names[1] and len(names[0])==m['config']['nc']
    e=np.load(base/'energy_maps.npz')
    for level,grid in [('P3',80),('P4',40),('P5',20)]:
        for mod in ('rgb','ir'):assert e[f'{mod}_{level}'].shape==(n,grid,grid)
        for region in ('all','fg','bg'):
            vals=[x['delta_cka'] for x in f if x['level']==level and x['region']==region and x['delta_cka'] is not None]
            v=s['features'][level][region]['delta_cka']
            assert len(vals)==v['n']
            if vals:assert abs(float(np.mean(vals))-v['mean'])<1e-7
    for png in sorted(base.glob('*.png')):
        with Image.open(png) as im:im.verify()
    checked.append({'dataset':d,'images':n,'common_objects':len(o),'script_sha256':m['script_sha256'],
        'class_names':names[0],'completion_sha256':sha(base/'completion_receipt.json'),
        'summary_sha256':sha(base/'summary.json')})
for png in (p/'registration_panels').glob('*.png'):
    with Image.open(png) as im:im.verify()
links=[]
docs=[p/'README.md',p/'特征图册.md',p.parents[1]/'07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md']
for doc in docs:
    for target in re.findall(r'\]\((E:/[^)]+)\)',doc.read_text(encoding='utf-8')):
        assert Path(target).exists(),(str(doc),target)
        links.append(target)
manifest=[]
for f in sorted(p.rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_verification.json':
        manifest.append({'path':f.relative_to(p).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)})
out={'status':'verified','utc':datetime.now(timezone.utc).isoformat(),'datasets':checked,
    'valid_absolute_artifact_links':len(links),'png_integrity':'verified',
    'scope':'Artifact identity and arithmetic integrity; not method efficacy or accepted analyzer.',
    'files':manifest}
(p/'delivery_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'datasets':checked,'files':len(manifest),'links':len(links)},ensure_ascii=True))
