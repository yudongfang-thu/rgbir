"""Independent saved-frame CPU review; directly executes pinned matching functions."""
import __future__
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ENTRY = Path(__file__).resolve().parents[1]
LOGS = ENTRY.parent
SOURCE = LOGS / '2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/N42_full_attempt1/sources'
INPUT = LOGS / '2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe'
OUTPUT = ENTRY / 'output_attempt1'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def main():
    torch.set_num_threads(2)
    env = dict(np=np, torch=torch)
    for file, name in [('7_metrics.py', 'box_iou'), ('4_validator.py', 'match_predictions')]:
        fn = next(n for n in ast.walk(ast.parse((SOURCE/file).read_text(encoding='utf-8'))) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE/file), 'exec', flags=__future__.annotations.compiler_flag), env)
    match, iou_fn = env['match_predictions'], env['box_iou']
    orig = rows(INPUT/'witness_objects.jsonl'); ir = {r['ir_global_row']:r for r in rows(INPUT/'witness_ir_objects.jsonl')}
    rgb = {r['rgb_global_row']:r for r in orig}
    frames = rows(INPUT/'witness_frames.jsonl'); saved_frames = {r['frame_id']:r for r in rows(OUTPUT/'frames.jsonl')}
    records = rows(OUTPUT/'objects.jsonl'); summary = read(OUTPUT/'summary.json')
    assert len(rgb)==len(ir)==len(records)==80 and len(frames)==len(saved_frames)==32
    observed = {}; calls=0; pred_bits=0; gt_flags=0; pairs=0; maxerr=0.; pair_changes=0
    for f in frames:
        for m in ('S','R','T'):
            ids=f['ir_global_rows' if m=='T' else 'rgb_global_rows']; source=ir if m=='T' else rgb
            gt=[source[i] for i in ids]; det=f['native_detections'][m]
            assert [d['prediction_index'] for d in det]==list(range(len(det)))
            boxes=torch.tensor([d['box'] for d in det],dtype=torch.float32).reshape(-1,4)
            gb=torch.tensor([g['ir_gt_xyxy' if m=='T' else 'rgb_gt_xyxy'] for g in gt],dtype=torch.float32).reshape(-1,4)
            overlaps=iou_fn(gb,boxes); pc=torch.tensor([d['class'] for d in det]); gc=torch.tensor([g['gt_class'] for g in gt])
            for threshold in (.5,.75):
                key=str(threshold); captures=[]; prior=sys.getprofile()
                def trace(frame,event,arg):
                    if event=='return' and frame.f_code is match.__code__: captures.append(frame.f_locals['matches'].copy())
                try:
                    sys.setprofile(trace)
                    tp=match(SimpleNamespace(iouv=torch.tensor([threshold])),pc,gc,overlaps,use_scipy=False)
                finally: sys.setprofile(prior)
                assert len(captures)==1; calls+=1
                mapping={int(g):int(p) for g,p in captures[0]}
                saved=saved_frames[f['frame_id']]['models'][m][key]
                assert tp[:,0].tolist()==saved['prediction_tp']
                assert set(saved['gt_matches'])==set(str(g) for g in mapping)
                for gi,g in enumerate(gt):
                    gid=ids[gi]; observed[m,gid,key]=None
                    if gi in mapping:
                        pi=mapping[gi]; p=saved['gt_matches'][str(gi)]; d=det[pi]
                        assert p['prediction_id']==p['filtered_prediction_id']==pi and p['confidence']==d['confidence']
                        err=abs(p['iou']-float(overlaps[gi,pi]));maxerr=max(maxerr,err);assert err==0
                        observed[m,gid,key]=dict(p,anchor_index=d['anchor_index'],class_id=d['class'],box=d['box'])
                    if threshold==.5:
                        old=g['detector_T'] if m=='T' else g['detector'][m];gt_flags+=1
                        assert old['native_postnms_matched']==(gi in mapping)
                        if gi in mapping:
                            pairs+=1; w=old['native_witness'];new=observed[m,gid,key]
                            assert (w['prediction_index'],w['anchor_index'],w['box'],w['confidence'])==(new['prediction_id'],new['anchor_index'],new['box'],new['confidence'])
                            assert abs(w['iou']-new['iou'])<1e-6
                if threshold==.5:
                    assert tp[:,0].tolist()==[d['native_tp_iou50'] for d in det];pred_bits+=len(det)
    for r in records:
        old=rgb[r['rgb_global_row']]
        for k in ('frame_id','image_index','stable_rgb_gt_id','stable_ir_gt_id','ir_global_row','rgb_gt_xyxy','ir_gt_xyxy'):assert r[k]==old[k]
        assert r['C_gates']==old['gates']
        for m in ('S','R','T'):
            gid=r['ir_global_row'] if m=='T' else r['rgb_global_row']
            for key in ('0.5','0.75'):assert r['matches'][m][key]==observed[m,gid,key]
            a,b=(r['matches'][m][k] for k in ('0.5','0.75'))
            if a and b and a['prediction_id']!=b['prediction_id']:pair_changes+=1
        s,t=(r['matches'][m]['0.5'] is not None for m in ('S','T'));u,v=(r['matches'][m]['0.75'] is not None for m in ('S','T'))
        expected=('both05_'+('both075' if u and v else 'onlyT075' if v else 'onlyS075' if u else 'neither075')) if s and t else 'T_only05' if t else 'S_only05' if s else 'neither05'
        assert r['bucket']==expected
    def check_coverage(rr,saved):
        assert saved['objects']==len(rr) and saved['unique_images']==len({r['frame_id'] for r in rr})
        assert saved['fraction_all_gt']==len(rr)/80 and saved['stable_rgb_gt_ids']==[r['stable_rgb_gt_id'] for r in rr]
        for gate in ('base','eligible','selected'):
            subset=[r for r in rr if r['C_gates'][gate]];v=saved[gate]
            assert v['objects']==len(subset) and v['unique_images']==len({r['frame_id'] for r in subset})
            assert v['fraction_bucket']==(len(subset)/len(rr) if rr else None)
            assert v['stable_rgb_gt_ids']==[r['stable_rgb_gt_id'] for r in subset]
    check_coverage(records,summary['C_all'])
    for b,v in summary['buckets'].items():check_coverage([r for r in records if r['bucket']==b],v)
    for m in ('S','R','T'):
        assert summary['counts'][m]=={k:sum(r['matches'][m][k] is not None for r in records) for k in ('0.5','0.75')}
    for name in ('analyze_training_localization.py','native_cached_match.py'):assert (ENTRY/name).read_bytes()==(OUTPUT/'source_copies'/name).read_bytes()
    for name in ('3_val.py','4_validator.py','7_metrics.py'):assert (SOURCE/name).read_bytes()==(OUTPUT/'source_copies'/name).read_bytes()
    assert (ENTRY/'native_cached_match.py').read_bytes()==(LOGS/'2026-09-08_probe_开发集真实检出机会/native_cached_match.py').read_bytes()
    proxy=read(ENTRY/'NATIVE_BOX_PROXY_11.json'); wanted=[r for r in records if r['bucket']=='both05_onlyT075'];assert len(wanted)==11
    for p,r in zip(proxy['rows'],wanted):
        s,t=r['matches']['S']['0.5'],r['matches']['T']['0.5']
        assert p['S']==s and p['T']==t and p['stable_rgb_gt_id']==r['stable_rgb_gt_id'] and p['C_selected']==r['C_gates']['selected']
        assert p['S_iou_lt_070']==(s['iou']<.7) and p['T_lead_gt_005']==(t['iou']-s['iou']>.05)
        assert p['T_minus_S_iou']==t['iou']-s['iou']
    counts=[sum(p['S_iou_lt_070'] for p in proxy['rows']),sum(p['T_lead_gt_005'] for p in proxy['rows']),sum(p['S_iou_lt_070'] and p['T_lead_gt_005'] for p in proxy['rows'])]
    assert counts==[proxy[k] for k in ('S_iou_lt_070_count','T_lead_gt_005_count','conjunction_count')]==[4,10,4]
    assert all(p['C_selected'] for p in proxy['rows'] if p['S_iou_lt_070'] and p['T_lead_gt_005']) and pair_changes==0
    assert not torch.cuda.is_initialized()
    result=dict(status='ACCEPTED_FIRST32_CACHED_LOCALIZATION_DESCRIPTION',frames=32,objects=80,native_calls=calls,original_gt_flags=gt_flags,original_positive_pairs=pairs,original_prediction_tp_bits=pred_bits,max_iou_difference=maxerr,all_identity_and_C_gates_exact=True,all_bucket_counts_fractions_images_ids_exact=True,matched_prediction_changes_between_thresholds=pair_changes,proxy_counts=counts,proxy_is_actual_L2=False,source_bytes_exact=True,local_torch=str(torch.__version__),original_GPU_bitwise_reproduction=False,new_GPU=False,new_forward=False,new_hash=False)
    with (ENTRY/'independent_review/ACTUAL_REVIEW_RECEIPT.json').open('x',encoding='utf-8') as fp:json.dump(result,fp,indent=2,ensure_ascii=False)
    print(json.dumps(result))
if __name__=='__main__':main()
