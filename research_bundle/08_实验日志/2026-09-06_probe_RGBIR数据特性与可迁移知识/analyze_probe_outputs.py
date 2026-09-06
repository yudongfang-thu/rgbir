"""CPU-only descriptive extension of the frozen baseline probe; no efficacy claims.

Uses saved predictions, without changing inference, matching, or sample selection.
Candidate flags are intentionally not one-to-one recall or physical visibility.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def stats(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    if not len(a): return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),
            'fraction_positive':float((a>0).mean())}

def overlap(box, preds):
    if not len(preds):return np.zeros(0)
    inter=np.maximum(0,np.minimum(box[2:],preds[:,2:4])-np.maximum(box[:2],preds[:,:2])).prod(1)
    areas=np.maximum(0,preds[:,2:4]-preds[:,:2]).prod(1)
    return inter/np.maximum(np.maximum(0,box[2:]-box[:2]).prod()+areas-inter,1e-10)

def masks(shape,gt,grid):
    h,w=shape;ratio=min(640/h,640/w)
    left=round((640-round(w*ratio))/2-.1);top=round((640-round(h*ratio))/2-.1)
    yy,xx=np.meshgrid((np.arange(grid)+.5)*640/grid,(np.arange(grid)+.5)*640/grid,indexing='ij')
    valid=(xx>=left)&(xx<left+w*ratio)&(yy>=top)&(yy<top+h*ratio)
    fg=np.zeros((grid,grid),bool)
    for _,x1,y1,x2,y2 in gt:
        fg|=(xx>=left+x1*w*ratio)&(xx<=left+x2*w*ratio)&(yy>=top+y1*h*ratio)&(yy<=top+y2*h*ratio)
    return valid,fg&valid

def calculate(name):
    d=ROOT/(name+'_full');s=read(d/'summary.json')
    objects=read(d/'matched_objects.json');rows=read(d/'image_metrics.json')
    records=read(d/'prediction_records.json');pred={r['id']:r for r in records}
    rowmap={r['id']:r for r in rows}
    teacher,student=('rgb','ir') if name=='vedai' else ('ir','rgb')
    both=[r for r in objects if r['rgb_hit'] and r['ir_hit']]
    tonly=[r for r in objects if r[teacher+'_hit'] and not r[student+'_hit']]
    sonly=[r for r in objects if r[student+'_hit'] and not r[teacher+'_hit']]
    getdelta=lambda r:r[teacher+'_pred_iou']-r[student+'_pred_iou']
    candidate_rows=[]
    for o in tonly:
        rec=pred[o['id']];gt=np.array(rec['gt_'+student][o[student+'_gt']],float)
        pp=np.array(rec['pred_'+student],float).reshape(-1,6)
        io=overlap(gt[1:],pp);same=pp[:,5]==gt[0]
        best=lambda sel:float(io[sel].max()) if sel.any() else 0.
        candidate_rows.append({'id':o['id'],'gt_index':o[student+'_gt'],
            'same_class_iou50_at_conf05':best(same)>=.5,
            'same_class_iou10_at_conf05':best(same)>=.1,
            'any_class_iou10_at_conf05':best(np.ones(len(pp),bool))>=.1,
            'wrong_class_iou50_at_conf25':best((~same)&(pp[:,4]>=.25))>=.5,
            'best_same_class_iou':best(same),
            'best_any_class_iou':best(np.ones(len(pp),bool))})
    flags={k:sum(r[k] for r in candidate_rows) for k in ('same_class_iou50_at_conf05',
       'same_class_iou10_at_conf05','any_class_iou10_at_conf05','wrong_class_iou50_at_conf25')}
    quartiles=np.quantile([r['rgb_luminance'] for r in rows],[.25,.5,.75])
    bins=[]
    for b in range(4):
        sub=[o for o in objects if int(np.searchsorted(quartiles,o['rgb_luminance'],side='right'))==b]
        bins.append({'brightness_quartile':b+1,'objects':len(sub),
            'teacher_only':sum(o[teacher+'_hit'] and not o[student+'_hit'] for o in sub),
            'student_only':sum(o[student+'_hit'] and not o[teacher+'_hit'] for o in sub),
            'both':sum(o[student+'_hit'] and o[teacher+'_hit'] for o in sub)})
    energy=np.load(d/'energy_maps.npz');emetrics={}
    for mod in ('rgb','ir'):
        emetrics[mod]={}
        for level in ('P3','P4','P5'):
            arr=energy[f'{mod}_{level}'];enrich=[];shares=[];areas=[]
            for i,r in enumerate(records):
                valid,fg=masks(rowmap[r['id']][mod+'_shape'],r['gt_'+mod],arr.shape[-1])
                bg=valid&~fg
                if not fg.any() or not bg.any():continue
                e=arr[i]
                enrich.append(float(e[fg].mean()/max(float(e[bg].mean()),1e-10)))
                shares.append(float(e[fg].sum()/max(float(e[valid].sum()),1e-10)))
                areas.append(float(fg.sum()/valid.sum()))
            emetrics[mod][level]={'grid':list(arr.shape[1:]),'fg_to_bg_mean_l2':stats(enrich),
                                'fg_fraction_total_l2':stats(shares),'fg_area_fraction':stats(areas)}
    high=[r for r in both if r['iou']>=.5]
    out={'dataset':name,'teacher':teacher,'student':student,'n_images':s['n_images'],
         'n_common_objects':len(objects),'teacher_only':len(tonly),'student_only':len(sonly),
         'both':len(both),'neither':s['common_object_hits']['neither'],
         'label_rgb_coverage':len(objects)/s['labels']['rgb_n'],
         'label_ir_coverage':len(objects)/s['labels']['ir_n'],
         'label_iou_below_05':sum(o['iou']<.5 for o in objects),
         'label_iou_exact1':sum(abs(o['iou']-1)<1e-8 for o in objects),
         'both_hit_teacher_minus_student_iou':stats([getdelta(r) for r in both]),
         'both_hit_label_iou_ge05_teacher_minus_student_iou':stats([getdelta(r) for r in high]),
         'teacher_only_student_candidate_flags':flags,
         'teacher_only_student_candidates':candidate_rows,
         'brightness_quartile_edges':quartiles.tolist(),'brightness_groups':bins,
         'activation_l2':emetrics,
         'limitations':['CPU exploratory extension of frozen probe. No new training, inference, or threshold selection.',
            'Candidate flags may overlap; they are per-GT proposals, not one-to-one recall and not physical visibility.',
            'Both-hit localization comparison is conditioned on both detectors succeeding; it is not AP or attainable KD gain.',
            'Brightness groups are descriptive sample quartiles, not official day/night labels.',
            'Foreground activation uses own-modality GT and L2 norm, not gradient attribution.']}
    return out

def main():
    result={d:calculate(d) for d in ('dronevehicle','llvip','vedai')}
    (ROOT/'probe_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for d,s in result.items():
        print(d,json.dumps({k:v for k,v in s.items() if k not in ('teacher_only_student_candidates','activation_l2','limitations')},ensure_ascii=False))

if __name__=='__main__':main()
