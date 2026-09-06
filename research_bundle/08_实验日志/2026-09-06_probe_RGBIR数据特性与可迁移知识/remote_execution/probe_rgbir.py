"""Frozen descriptive RGB/IR baseline probe. No training or sealed-test access.
Run only through project_resource_guard. Images use native geometry; network
predict uses fixed 640 letterbox. Per-image centered CKA uses 20x20 pooled tokens,
with five independent derangements and union-foreground/content masks.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import yaml

REPO = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
sys.path.insert(0, str(REPO))
from tools.project_resource_guard import require_bound_lease_from_environment
from ultralytics import YOLO

SEED = 20260906
SIZE = 640
POOL = 20
LEVELS = ['P3', 'P4', 'P5']

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def summary(vals):
    a=np.array([v for v in vals if v is not None and np.isfinite(v)],float)
    if not len(a): return {'n':0}
    return clean({'n':len(a),'mean':a.mean(),'sd_images':a.std(ddof=1) if len(a)>1 else None,
                  'p10':np.quantile(a,.1),'median':np.median(a),'p90':np.quantile(a,.9)})

def labels(p):
    rows=[]
    if not p.is_file(): raise FileNotFoundError(p)
    for line in p.read_text().splitlines():
        v=[float(t) for t in line.split()]
        if len(v)!=5: raise ValueError(f'Expected HBB labels: {p}')
        c,x,y,w,h=v
        rows.append([c,x-w/2,y-h/2,x+w/2,y+h/2])
    return np.array(rows,float).reshape(-1,5)

def iou(a,b):
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.clip(hi-lo,0,None).prod(-1)
    aa=np.clip(a[:,2:]-a[:,:2],0,None).prod(-1)
    bb=np.clip(b[:,2:]-b[:,:2],0,None).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-10)

def matched_labels(a,b):
    mat=iou(a[:,1:],b[:,1:]); score=mat*(a[:,None,0]==b[None,:,0])
    if not score.size:return []
    ia,ib=linear_sum_assignment(-score)
    out=[]
    for x,y in zip(ia,ib):
        if score[x,y]<.1:continue
        ca=(a[x,1:3]+a[x,3:])/2;cb=(b[y,1:3]+b[y,3:])/2
        scale=np.sqrt(max(np.prod(a[x,3:]-a[x,1:3]),1e-10))
        out.append({'rgb_gt':int(x),'ir_gt':int(y),'class_id':int(a[x,0]),
                    'iou':float(score[x,y]),'center_shift_norm':float(np.linalg.norm(ca-cb)),
                    'center_shift_over_rgb_sqrt_area':float(np.linalg.norm(ca-cb)/scale),
                    'dx_norm':float(cb[0]-ca[0]),'dy_norm':float(cb[1]-ca[1])})
    return out

def gray_edge(im):
    g=cv2.cvtColor(cv2.resize(im,(320,320)),cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    g=cv2.GaussianBlur(g,(5,5),1)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    edge=(mag>=max(float(np.quantile(mag,.85)),.01))
    return g,mag,edge

def corr(a,b,mask=None):
    a=np.asarray(a);b=np.asarray(b)
    if mask is not None:a=a[mask];b=b[mask]
    a=a.ravel().astype(float);b=b.ravel().astype(float)
    if len(a)<8:return None
    a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-10 else None

def image_metrics(a,b):
    ga,ma,ea=gray_edge(a);gb,mb,eb=gray_edge(b)
    win=cv2.createHanningWindow((320,320),cv2.CV_32F)
    shift,response=cv2.phaseCorrelate(ma.copy(),mb.copy(),win)
    d=[]
    if ea.any() and eb.any():
        d=np.r_[distance_transform_edt(~eb)[ea],distance_transform_edt(~ea)[eb]]/np.sqrt(320**2*2)
    return clean({'rgb_luminance':float(ga.mean()*255),'ir_luminance':float(gb.mean()*255),
        'rgb_gray_sd':float(ga.std()*255),'gradient_corr':corr(ma,mb),
        'edge_distance_median_norm_diag':float(np.median(d)) if len(d) else None,
        'edge_distance_p90_norm_diag':float(np.quantile(d,.9)) if len(d) else None,
        'phase_dx_norm':shift[0]/320,'phase_dy_norm':shift[1]/320,'phase_response':response,
        'phase_shift_norm':float(np.linalg.norm(shift)/320),
        'rgb_shape':list(a.shape[:2]),'ir_shape':list(b.shape[:2])})

def masks(shape,gt):
    h,w=shape[:2];ratio=min(SIZE/h,SIZE/w)
    left=round((SIZE-round(w*ratio))/2-.1);top=round((SIZE-round(h*ratio))/2-.1)
    yy,xx=np.meshgrid((np.arange(POOL)+.5)*SIZE/POOL,(np.arange(POOL)+.5)*SIZE/POOL,indexing='ij')
    valid=(xx>=left)&(xx<left+w*ratio)&(yy>=top)&(yy<top+h*ratio)
    fg=np.zeros((POOL,POOL),bool)
    for _,x1,y1,x2,y2 in gt:
        fg|=(xx>=left+x1*w*ratio)&(xx<=left+x2*w*ratio)&(yy>=top+y1*h*ratio)&(yy<=top+y2*h*ratio)
    return valid,fg

def cka(x,y,mask):
    # x,y are C,H,W CPU tensors; center across corresponding spatial samples.
    ids=torch.from_numpy(mask.reshape(-1))
    a=x.flatten(1).T[ids].float();b=y.flatten(1).T[ids].float()
    if len(a)<8:return None
    a=a-a.mean(0,keepdim=True);b=b-b.mean(0,keepdim=True)
    den=torch.linalg.matrix_norm(a.T@a)*torch.linalg.matrix_norm(b.T@b)
    if float(den)<1e-10:return None
    return float((a.T@b).square().sum()/den)

def gt_errors(gt,pred):
    # pred: normalized x1,y1,x2,y2,confidence,class; one-to-one geometry assignment.
    p=pred[pred[:,4]>=.25] if len(pred) else pred
    mat=iou(gt[:,1:],p[:,:4]);score=mat*(gt[:,None,0]==p[None,:,5])
    best=np.zeros(len(gt));wrong=np.zeros(len(gt),bool)
    if score.size:
        ii,jj=linear_sum_assignment(-score)
        for i,j in zip(ii,jj):best[i]=score[i,j]
        wrong=((mat>=.5)&(gt[:,None,0]!=p[None,:,5])).any(1)
    return [{'iou':float(v),'hit':bool(v>=.5),'wrong_class_overlap':bool(wrong[i])} for i,v in enumerate(best)]

def predict_model(ckpt,paths,out,tag):
    net=YOLO(str(ckpt));det=net.model.model[-1];capture={}
    def hook(mod,args):
        capture['f']=[z.detach().float().clone() for z in args[0][-3:]]
    handle=det.register_forward_pre_hook(hook)
    feats=[];energies=[];preds=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,p in enumerate(paths):
            im=cv2.imread(str(p));assert im is not None,str(p)
            r=net.predict(im,imgsz=SIZE,rect=False,device='0',conf=.05,iou=.7,max_det=300,
                          verbose=False,save=False,save_txt=False,plots=False)[0]
            f=capture['f']
            if len(f)!=3:raise RuntimeError('Expected three feature levels')
            feats.append([F.adaptive_avg_pool2d(z,(POOL,POOL))[0].cpu().half() for z in f])
            energies.append([F.adaptive_avg_pool2d(torch.linalg.vector_norm(z,dim=1,keepdim=True),(POOL,POOL))[0,0].cpu().numpy() for z in f])
            boxes=r.boxes
            if boxes is None or not len(boxes):preds.append(np.empty((0,6)))
            else:
                xy=boxes.xyxyn.cpu().numpy();conf=boxes.conf.cpu().numpy();cls=boxes.cls.cpu().numpy()
                preds.append(np.c_[xy,conf,cls])
            if (i+1)%40==0:print(f'{tag} {i+1}/{len(paths)}',flush=True)
    peak={'allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
    names=net.names
    handle.remove();del net;capture.clear();gc.collect();torch.cuda.empty_cache()
    return feats,energies,preds,peak,names

def energy_overlay(im,e):
    h,w=im.shape[:2];ratio=min(SIZE/h,SIZE/w)
    nw,nh=round(w*ratio),round(h*ratio)
    l=round((SIZE-nw)/2-.1);t=round((SIZE-nh)/2-.1)
    m=cv2.resize(e,(SIZE,SIZE))[t:t+nh,l:l+nw]
    m=cv2.resize(m,(w,h));lo,hi=np.quantile(m,[.02,.98])
    m=np.clip((m-lo)/max(hi-lo,1e-8),0,1)
    heat=plt.get_cmap('magma')(m)[...,:3]
    return .5*cv2.cvtColor(im,cv2.COLOR_BGR2RGB)/255+.5*heat

def draw_boxes(ax,im,gt,pred):
    h,w=im.shape[:2];ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    from matplotlib.patches import Rectangle
    for _,x1,y1,x2,y2 in gt:
        ax.add_patch(Rectangle((x1*w,y1*h),(x2-x1)*w,(y2-y1)*h,fill=False,edgecolor='#00ffb3',lw=.8))
    for x1,y1,x2,y2,c,k in pred:
        if c<.25:continue
        ax.add_patch(Rectangle((x1*w,y1*h),(x2-x1)*w,(y2-y1)*h,fill=False,edgecolor='#ff6845',lw=.7))

def figures(out,items,energy,pred,rows,pairs,feature_rows):
    order=np.argsort([r['rgb_luminance'] for r in rows])
    chosen=[int(order[round(q*(len(order)-1))]) for q in (.1,.35,.65,.9)]
    for z,i in enumerate(chosen):
        fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
        for side in range(2):
            im=cv2.imread(str(items[i]['paths'][side]));gt=items[i]['gt'][side]
            draw_boxes(axes[side,0],im,gt,pred[side][i]);axes[side,0].set_title(('RGB','IR/NIR')[side]+' | GT green; pred orange',fontsize=10)
            for lv in range(3):
                axes[side,lv+1].imshow(energy_overlay(im,energy[side][i][lv]));axes[side,lv+1].set_title(LEVELS[lv]+' activation energy',fontsize=10)
            for ax in axes[side]:ax.axis('off')
        fig.suptitle(f'{out.name}: fixed luminance quantile {(.1,.35,.65,.9)[z]} | ID {items[i]["id"]}\nEach energy map normalized independently; color intensity is not comparable across models.',fontsize=10)
        fig.savefig(out/f'feature_case_{z+1}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    if pairs:
        ax[0,0].hist([p['iou'] for p in pairs],bins=np.linspace(0,1,21),color='#4477aa')
    ax[0,0].set(xlabel='Cross-modal annotation IoU (matched >= 0.1)',ylabel='Object pairs',title='Annotation correspondence, not pixel-registration truth')
    ax[0,1].scatter([r['phase_shift_norm'] for r in rows],[r['phase_response'] for r in rows],s=12,alpha=.6)
    ax[0,1].set(xlabel='Gradient phase shift / image side',ylabel='Phase response',title='Low response means ambiguous shift')
    vals=[];labs=[]
    for lv in LEVELS:
        for reg in ('all','fg'):
            v=[r['delta_cka'] for r in feature_rows if r['level']==lv and r['region']==reg and r['delta_cka'] is not None]
            vals.append(v or [np.nan]);labs.append(lv+' '+reg)
    ax[1,0].boxplot(vals,tick_labels=labs,showfliers=False);ax[1,0].axhline(0,c='gray',ls='--')
    ax[1,0].set(ylabel='Centered CKA paired - mean of 5 donors',title='Spatial structure signal; not KD utility')
    counts=[sum(p['rgb_hit']==a and p['ir_hit']==b for p in pairs) for a,b in [(1,1),(1,0),(0,1),(0,0)]]
    ax[1,1].bar(['Both','RGB only','IR only','Neither'],counts,color=['#228833','#4477aa','#cc6677','#999999'])
    ax[1,1].set(ylabel='Matched annotated objects',title='Baseline hits at conf 0.25, IoU 0.5')
    fig.savefig(out/'diagnostic_overview.png',dpi=180);fig.savefig(out/'diagnostic_overview.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=200);p.add_argument('--canary',action='store_true')
    a=p.parse_args();require_bound_lease_from_environment();torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=json.loads(a.config.read_text())[a.dataset];out=a.out
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    roots=[Path(x) for x in cfg['images']];labroots=[Path(x) for x in cfg['labels']]
    maps=[{q.stem:q for q in root.iterdir() if q.suffix.lower() in ('.jpg','.png','.jpeg','.bmp','.tif','.tiff')} for root in roots]
    ids=sorted(set(maps[0])&set(maps[1]));rng=np.random.default_rng(SEED)
    ids=sorted(rng.choice(ids,min(len(ids),a.n),replace=False).tolist())
    info={'config':cfg,'seed':SEED,'sample_ids':ids,'n':len(ids),'role':cfg['role'],'canary':a.canary,
          'metric_contract':'descriptive, single checkpoint seed; not efficacy or formal AP',
          'checkpoints':[],'script_sha256':digest(__file__)}
    for side,ck in enumerate(cfg['checkpoints']):
        path=Path(ck);arg=path.parents[1]/'args.yaml';train=yaml.safe_load(arg.read_text())
        if str(train['data'])!=cfg['expected_train_data'][side]:raise ValueError('checkpoint training data identity mismatch')
        info['checkpoints'].append({'path':ck,'sha256':digest(path),'args_path':str(arg),'args_sha256':digest(arg),'args':train})
    dump(out/'input_manifest.json',info)
    items=[];rows=[]
    for id in ids:
        paths=[m[id] for m in maps];gt=[labels(r/(id+'.txt')) for r in labroots]
        ims=[cv2.imread(str(q)) for q in paths]
        if any(im is None for im in ims):raise ValueError('unreadable image')
        imet=image_metrics(*ims);va,fa=masks(ims[0].shape,gt[0]);vb,fb=masks(ims[1].shape,gt[1])
        items.append({'id':id,'paths':paths,'gt':gt,'valid':va&vb,'fg':(fa|fb)&va&vb,'match':matched_labels(*gt)})
        rows.append({'id':id,**imet,'rgb_gt_n':len(gt[0]),'ir_gt_n':len(gt[1]),'matched_gt_n':len(items[-1]['match']),
                     'identical_label_bytes':(labroots[0]/(id+'.txt')).read_bytes()==(labroots[1]/(id+'.txt')).read_bytes()})
    feat=[];energy=[];pred=[];peaks=[]
    for side,ck in enumerate(cfg['checkpoints']):
        f,e,p,peak,names=predict_model(ck,[x['paths'][side] for x in items],out,('rgb','ir')[side])
        if len(names)!=cfg['nc']:raise ValueError('model class count mismatch')
        feat.append(f);energy.append(e);pred.append(p);peaks.append(peak)
    permutations=[]
    if len(items)>1:
        for _ in range(5):
            perm=rng.permutation(len(items))
            while (perm==np.arange(len(items))).any():perm=rng.permutation(len(items))
            permutations.append(perm)
    dump(out/'donor_indices.json',clean(permutations))
    fr=[];objects=[]
    for i,it in enumerate(items):
        err=[gt_errors(it['gt'][s],pred[s][i]) for s in range(2)]
        for pair in it['match']:
            x,y=pair['rgb_gt'],pair['ir_gt'];r=err[0][x];t=err[1][y]
            objects.append({'id':it['id'],**pair,'rgb_hit':r['hit'],'ir_hit':t['hit'],
                'rgb_pred_iou':r['iou'],'ir_pred_iou':t['iou'],
                'rgb_wrong_class_overlap':r['wrong_class_overlap'],'ir_wrong_class_overlap':t['wrong_class_overlap'],
                'ir_minus_rgb_loc_iou':t['iou']-r['iou'],'rgb_luminance':rows[i]['rgb_luminance']})
        for li,lv in enumerate(LEVELS):
            for region,mask in [('all',it['valid']),('fg',it['fg']),('bg',it['valid']&~it['fg'])]:
                pc=cka(feat[0][i][li],feat[1][i][li],mask)
                nc=[cka(feat[0][i][li],feat[1][perm[i]][li],mask & items[perm[i]]['valid']) for perm in permutations]
                nc=[v for v in nc if v is not None];nm=float(np.mean(nc)) if nc else None
                ec=corr(energy[0][i][li],energy[1][i][li],mask)
                ne=[corr(energy[0][i][li],energy[1][perm[i]][li],mask & items[perm[i]]['valid']) for perm in permutations]
                ne=[v for v in ne if v is not None];em=float(np.mean(ne)) if ne else None
                fr.append({'id':it['id'],'level':lv,'region':region,'tokens':int(mask.sum()),'cka_paired':pc,
                           'cka_donor_mean':nm,'delta_cka':pc-nm if pc is not None and nm is not None else None,
                           'energy_corr_paired':ec,'energy_corr_donor_mean':em,
                           'delta_energy_corr':ec-em if ec is not None and em is not None else None})
    for name,data in [('image_metrics',rows),('matched_objects',objects),('feature_metrics',fr)]:
        dump(out/(name+'.json'),clean(data))
        if data:
            with (out/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    stats={'dataset':a.dataset,'n_images':len(items),'sample_seed':SEED,'single_checkpoint_seed':42,
        'image':{k:summary([r[k] for r in rows]) for k in ('rgb_luminance','rgb_gray_sd','gradient_corr','phase_shift_norm','phase_response','edge_distance_p90_norm_diag')},
        'labels':{'rgb_n':sum(r['rgb_gt_n'] for r in rows),'ir_n':sum(r['ir_gt_n'] for r in rows),
                  'matched_n':len(objects),'identical_files':sum(r['identical_label_bytes'] for r in rows),
                  'match_iou':summary([r['iou'] for r in objects]),'relative_center_shift':summary([r['center_shift_over_rgb_sqrt_area'] for r in objects])},
        'common_object_hits':{k:sum(r['rgb_hit']==v[0] and r['ir_hit']==v[1] for r in objects) for k,v in {'both':(1,1),'rgb_only':(1,0),'ir_only':(0,1),'neither':(0,0)}.items()},
        'features':{},'gpu_peak_mib':peaks,'duration_seconds':time.time()-started,
        'limitations':['Annotation matching threshold IoU>=0.1 is a proxy; unpaired labels do not establish modality invisibility.',
          'Paired spatial representation correlation does not establish KD benefit.',
          'Phase shifts at low response are ambiguous, not registration failures.',
          'Foreground CKA uses pooled tokens and union GT; <8 tokens is N/A.',
          'Ground-truth hits use modality-specific labels on matched object pairs, not a shared-label AP evaluation.',
          'Per-image SD is not multi-training-seed uncertainty.']}
    for lv in LEVELS:
        stats['features'][lv]={}
        for reg in ('all','fg','bg'):
            sub=[r for r in fr if r['level']==lv and r['region']==reg]
            stats['features'][lv][reg]={k:summary([r[k] for r in sub]) for k in ('tokens','cka_paired','cka_donor_mean','delta_cka','energy_corr_paired','delta_energy_corr')}
    dump(out/'summary.json',clean(stats))
    if not a.canary:
        figures(out,items,energy,pred,rows,objects,fr)
        arrays={f'{side}_{lv}':np.stack([e[li] for e in energy[s]]) for s,side in enumerate(('rgb','ir')) for li,lv in enumerate(LEVELS)}
        np.savez_compressed(out/'energy_maps.npz',**arrays)
    dump(out/'completion_receipt.json',{'status':'canary_completed' if a.canary else 'probe_completed','n_images':len(items),'gpu_peak_mib':peaks,'duration_seconds':time.time()-started,'sealed_test_accessed':False})
    print(json.dumps(clean(stats)),flush=True)

if __name__=='__main__':main()
