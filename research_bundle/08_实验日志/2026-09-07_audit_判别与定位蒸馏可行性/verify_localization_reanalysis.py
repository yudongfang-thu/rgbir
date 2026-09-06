"""Independent scalar IoU check from saved prediction indices, no GPU or inference."""
import csv,json,math
from pathlib import Path
D=Path(__file__).resolve().parent
SOURCE=D.parent/'2026-09-06_probe_RGBIR数据特性与可迁移知识'
def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/max(area_a+area_b-intersection,1e-10)
out={}
for name in ['dronevehicle','llvip']:
    raw={r['id']:r for r in json.loads((SOURCE/f'{name}_full/prediction_records.json').read_text(encoding='utf-8'))}
    rows=list(csv.DictReader((D/f'{name}_localization_objects.csv').open(encoding='utf-8')))
    verified=[];both=[];selected=[];mapped_selected=[]
    for r in rows:
        record=raw[r['id']];g=record['gt_rgb'][int(r['rgb_gt'])];t=record['gt_ir'][int(r['ir_gt'])]
        if not r['rgb_prediction_index'] or not r['ir_prediction_index']:continue
        p=record['pred_rgb'][int(r['rgb_prediction_index'])];q=record['pred_ir'][int(r['ir_prediction_index'])]
        assert int(p[5])==int(q[5])==int(g[0])==int(t[0])
        assert p[4]>=.25 and q[4]>=.25
        ri,ti,mi=overlap(p[:4],g[1:]),overlap(q[:4],t[1:]),overlap(q[:4],g[1:])
        assert abs(ri-float(r['rgb_candidate_iou']))<1e-8
        assert abs(ti-float(r['ir_own_iou']))<1e-8
        assert abs(mi-float(r['ir_direct_rgb_iou']))<1e-8
        verified.append((r,ri,ti,mi))
        if r['rgb_hit']=='True' and r['ir_hit']=='True':both.append((ri,ti,mi))
        if .5<=ri<.75 and ti>=.5 and ti-ri>=.1:selected.append((r,ri,ti,mi))
        if .5<=ri<.75 and ti>=.5 and mi>=.5 and mi-ri>=.1:mapped_selected.append((r,ri,ti,mi))
    mean=lambda x:sum(x)/len(x)
    out[name]={'scalar_rows_verified':len(verified),'both_n':len(both),
        'own_gain_mean':mean([t-r for r,t,m in both]),'direct_gain_mean':mean([m-r for r,t,m in both]),
        'direct_negative_n':sum(m<r-1e-9 for r,t,m in both),'direct_gain_ge_01_n':sum(m-r>=.1 for r,t,m in both),
        'own_gate_n':len(selected),'own_gate_direct_negative_n':sum(m<r-1e-9 for _,r,t,m in selected),
        'own_gate_direct_gain_ge_01_n':sum(m-r>=.1 for _,r,t,m in selected),
        'own_gate_and_label_iou_ge05_n':sum(float(x['label_iou'])>=.5 for x,_,_,_ in selected),
        'mapped_gate_n':len(mapped_selected),
        'mapped_gate_and_label_iou_ge05_n':sum(float(x['label_iou'])>=.5 for x,_,_,_ in mapped_selected),
        'note':'Mapped gate is a retrospective RGB-GT oracle diagnostic with identity mapping; not a trained effect, verified registration or chosen training gate.'}
result={'status':'passed','method':'Independent scalar IoU on original normalized xyxy and CSV-selected prediction indices; all pointwise results agree within 1e-8.','datasets':out}
with (D/'ROOT_RECHECK.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
