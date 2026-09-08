"""Frozen own-GT/native-pred identities plus original cross-modal GT assignment."""
import ast
import importlib.util
from pathlib import Path
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

CLASSES=['car','freight car','truck','bus','van']
GROUPS={'all_iou50':(None,.5),'all_iou75':(None,.75),'gt025_iou50':(.25,.5),'gt025_iou75':(.25,.75)}
BUCKETS=('both_correct','T_only','N_only','both_unmatched')

def require(ok,message):
    if not ok:raise ValueError(message)

def load_original(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_iou','_match_objects')]
    require(len(nodes)==2,'Exactly two original GT assignment functions required')
    env=dict(torch=torch,Tensor=torch.Tensor,linear_sum_assignment=linear_sum_assignment)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    return env

def tensor_boxes(rows):return torch.tensor(np.asarray(rows,dtype=np.float32).reshape(-1,4))

def gt_pair(n,t,original):
    require(n['canvas_shape']==t['canvas_shape'] and n['original_shape']==t['original_shape'],'Cross-modal canvas/original shape differs; no guessed transform')
    nb,tb=tensor_boxes(n['gt_boxes']),tensor_boxes(t['gt_boxes'])
    nc,tc=torch.tensor(n['gt_classes']),torch.tensor(t['gt_classes'])
    ni,ti=original['_match_objects'](nb,nc,tb,tc,.5)
    overlaps=original['_iou'](nb,tb)
    pairs=[dict(N_gt_row=int(a),T_gt_row=int(b),iou=float(overlaps[a,b])) for a,b in zip(ni,ti)]
    require(len({p['N_gt_row'] for p in pairs})==len(pairs)==len({p['T_gt_row'] for p in pairs}),'GT assignment not one-to-one')
    require(all(n['gt_classes'][p['N_gt_row']]==t['gt_classes'][p['T_gt_row']] and p['iou']>=.5 for p in pairs),'GT eligibility differs')
    return pairs

def bind_images(mapping,models,aliases):
    require(isinstance(mapping,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in mapping.items()),'Explicit image mapping required')
    inverse={};lookup={}
    for name in ('N','T'):
        require(set(aliases[name])==set(models[name]),'Cache alias set differs')
        require(len(set(aliases[name].values()))==len(aliases[name]),'Canonical image aliases not unique')
        inverse[name]={canonical:alias for alias,canonical in aliases[name].items()}
        lookup[name]=dict(aliases[name])
        for canonical in inverse[name]:
            require(canonical not in lookup[name] or lookup[name][canonical]==canonical,'Ambiguous alias/canonical spelling')
            lookup[name][canonical]=canonical
    rows=[]
    for key,value in mapping.items():
        require(key in lookup['N'] and value in lookup['T'],'Mapping contains image outside captured dev rosters')
        nc,tc=lookup['N'][key],lookup['T'][value]
        rows.append(dict(pair_key=nc,rgb_image=inverse['N'][nc],ir_image=inverse['T'][tc],rgb_canonical=nc,ir_canonical=tc))
    require(len(rows)==len(models['N'])==len(models['T']),'Mapping image population differs')
    for key,name in [('rgb_canonical','N'),('ir_canonical','T')]:
        require(len({r[key] for r in rows})==len(rows) and {r[key] for r in rows}==set(inverse[name]),'Mapping not a complete bijection')
    # Use RGB native evaluation order, never mapping JSON insertion order.
    indexed={r['rgb_image']:r for r in rows}
    return [indexed[image] for image in models['N']]

def original_image_fallback(models,aliases,source):
    """Execute the private original constructor's None branch on audited identity carriers.

    No images/transforms are loaded. Remote canonicalization is supplied only by
    the accepted native capture's alias-to-canonical identity, never local resolve.
    """
    all_alias={}
    for name in ('N','T'):
        require(set(models[name])==set(aliases[name]),'Incomplete alias identity')
        canonical=list(aliases[name].values())
        require(len(set(canonical))==len(canonical),'Duplicate canonical image')
        require(len({Path(p).stem for p in canonical})==len(canonical),'Duplicate canonical stem')
        for alias,canon in aliases[name].items():
            require(alias not in all_alias or all_alias[alias]==canon,'Conflicting alias identities')
            all_alias[alias]=canon
        for canon in canonical:
            require(canon not in all_alias or all_alias[canon]==canon,'Ambiguous canonical spelling')
            all_alias[canon]=canon
    spec=importlib.util.spec_from_file_location('private_original_pair_dataset',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # This private module exists only in this CPU analysis process.
    module._canonical=lambda path:all_alias[str(path)]
    Compose=type('Compose',(),{'transforms':[]})
    Carrier=type('AuditedNativeIdentityCarrier',(),{})
    carriers=[]
    for name in ('N','T'):
        c=Carrier();c.im_files=list(models[name]);c.data={'names':dict(enumerate(CLASSES))}
        c.imgsz=640;c.rect=False;c.cache=False;c.ims=[];c.transforms=Compose();carriers.append(c)
    dataset=module.DualLabelRGBIRDataset(carriers[0],carriers[1],strong_by_weak=None)
    mapping=dict(dataset._mapping)
    bind_images(mapping,models,aliases)
    return mapping

def validate_row(row):
    require(isinstance(row['image'],str),'Missing image')
    for k in ('canvas_shape','original_shape'):require(len(row[k])==2 and all(type(x) is int and x>0 for x in row[k]),'Invalid shape')
    require(len(row['gt_boxes'])==len(row['gt_classes']),'GT arrays differ')
    require(len(row['pred_boxes'])==len(row['pred_classes'])==len(row['pred_confidence'])<=300,'Prediction arrays/max_det differ')
    for k in ('gt_boxes','pred_boxes'):
        a=np.asarray(row[k],dtype=np.float32).reshape(-1,4)
        require(np.isfinite(a).all() and (a[:,2:]>=a[:,:2]).all(),'Nonfinite or inverted box')
    for k in ('gt_classes','pred_classes'):require(all(math.isfinite(x) and x==int(x) and 0<=x<5 for x in row[k]),'Invalid five-class identity')
    require(all(math.isfinite(x) and .001<x<=1 for x in row['pred_confidence']),'Confidence outside original native range')
    h,w=row['canvas_shape'];a=np.asarray(row['gt_boxes'],dtype=np.float32).reshape(-1,4)
    require((a>=0).all() and (a[:,[0,2]]<=w).all() and (a[:,[1,3]]<=h).all(),'GT outside canvas')
    a=np.asarray(row['pred_boxes'],dtype=np.float32).reshape(-1,4)
    return int(((a<0).any(1)|(a[:,[0,2]]>w).any(1)|(a[:,[1,3]]>h).any(1)).sum())

def bucket(n,t):return 'both_correct' if n and t else 'T_only' if t else 'N_only' if n else 'both_unmatched'
def gt_id(image,row):return image+'::cached_gt:'+str(row)
