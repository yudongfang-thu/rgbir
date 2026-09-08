"""Independent scalar cross-IoU and fixed prediction/cohort identity review."""
import __future__,ast,gzip,json
from pathlib import Path
from collections import Counter
import torch
E=Path(__file__).resolve().parents[1];P=E/'cpu_analysis/output_attempt2';O=E/'cross_gt_attempt1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def main():
    torch.set_num_threads(2);env={'torch':torch};source=E/'cpu_analysis/frozen_sources/7_metrics.py'
    fn=next(n for n in ast.walk(ast.parse(source.read_text(encoding='utf-8'))) if isinstance(n,ast.FunctionDef) and n.name=='box_iou')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),env)
    def iou(gt,box):return float(env['box_iou'](torch.tensor([gt],dtype=torch.float32),torch.tensor([box],dtype=torch.float32))[0,0])
    def cohort(p):
        low,high=p['groups']['gt025_iou50'],p['groups']['gt025_iou75']
        if not low['N']['correct'] or not low['T']['correct']:return None
        return 'T_own75_only' if high['T']['correct'] and not high['N']['correct'] else 'N_own75_only' if high['N']['correct'] and not high['T']['correct'] else None
    original={p['N_gt_id']:p for p in rows(P/'paired_objects.jsonl.gz') if cohort(p)}
    expected={'T_own75_only':1869,'N_own75_only':950}
    assert dict(Counter(cohort(p) for p in original.values()))==expected
    saved=rows(O/'objects.jsonl.gz');assert len(saved)==2819 and {r['N_gt_id'] for r in saved}==set(original)
    needed={(p['pair_key'],m,p['groups'][g][m]['witness']['prediction_id']) for p in original.values() for m in ('N','T') for g in ('gt025_iou50','gt025_iou75') if p['groups'][g][m]['witness'] is not None}
    predictions={}
    for f in rows(P/'prediction_matches.jsonl.gz'):
        for p in f['predictions']:
            k=f['pair_key'],f['model'],p['prediction_id']
            if k in needed:predictions[k]=p
    assert set(predictions)==needed
    frames={r['pair_key']:r for r in rows(P/'frames.jsonl.gz')};maxerr=0
    for r in saved:
        p=original[r['N_gt_id']];assert r['cohort']==cohort(p)
        for key in ('pair_key','N_gt_id','T_gt_id','N_gt_row','T_gt_row','N_gt_box','T_gt_box','gt_class'):assert r[key]==p[key]
        assert r['label_pair_iou']==p['pair_iou'] and r['stratum']==('label_iou_0.5_to_lt0.8' if p['pair_iou']<.8 else 'label_iou_0.8_to_1')
        assert r['canvas_shape']==frames[p['pair_key']]['canvas_shape']==[544,672] and r['original_shape']==[512,640]
        for m in ('N','T'):
            lo,hi=(p['groups'][g][m]['witness'] for g in ('gt025_iou50','gt025_iou75'));q=predictions[p['pair_key'],m,lo['prediction_id']]
            assert r[m]['prediction_id']==lo['prediction_id'] and r[m]['filtered_prediction_id']==lo['filtered_prediction_id']
            assert r[m]['box']==q['box'] and r[m]['confidence']==q['confidence']==lo['confidence']
            assert r[m]['native_iou75_match']==hi and r[m]['own_iou50_witness']==lo['iou']
            assert q['groups']['gt025_iou50']['gt_id']==p[m+'_gt_id'] and q['confidence']>.25
            state='unmatched' if hi is None else 'same' if hi['prediction_id']==lo['prediction_id'] else 'changed'
            assert r[m]['iou75_identity_status']==state
            if hi:
                alt=predictions[p['pair_key'],m,hi['prediction_id']]
                assert alt['box']==r[m]['native_iou75_box'] and alt['groups']['gt025_iou75']['gt_id']==p[m+'_gt_id']
                assert iou(p[m+'_gt_box'],alt['box'])==hi['iou']
            else:assert r[m]['native_iou75_box'] is None
        calculated={'N_fixed_to_RGB_GT_iou':iou(p['N_gt_box'],r['N']['box']),'T_fixed_to_IR_GT_iou':iou(p['T_gt_box'],r['T']['box']),'T_fixed_to_RGB_GT_iou':iou(p['N_gt_box'],r['T']['box']),'N_fixed_to_IR_GT_iou':iou(p['T_gt_box'],r['N']['box'])}
        for k,v in calculated.items():maxerr=max(maxerr,abs(v-r[k]));assert v==r[k]
        assert r['delta_T_minus_N_on_RGB_GT']==calculated['T_fixed_to_RGB_GT_iou']-calculated['N_fixed_to_RGB_GT_iou']
        assert r['delta_T_own_minus_N_own']==calculated['T_fixed_to_IR_GT_iou']-calculated['N_fixed_to_RGB_GT_iou']
    s=read(O/'summary.json');assert read(O/'completion.json')['objects']==2819 and not (O/'failure.json').exists()
    def check(values,v):
        assert v['objects']==len(values) and v['images']==len({r['pair_key'] for r in values})
        specs={'T_on_RGB_ge050':('T_fixed_to_RGB_GT_iou',.5),'T_on_RGB_ge075':('T_fixed_to_RGB_GT_iou',.75),'N_on_RGB_ge050':('N_fixed_to_RGB_GT_iou',.5),'N_on_RGB_ge075':('N_fixed_to_RGB_GT_iou',.75),'T_fixed_own_ge075':('T_fixed_to_IR_GT_iou',.75)}
        for name,(field,threshold) in specs.items():assert v[name]==sum(r[field]>=threshold for r in values)
        deltas=[r['delta_T_minus_N_on_RGB_GT'] for r in values]
        assert (v['delta_positive'],v['delta_gt005'],v['delta_negative'],v['delta_lt_negative005'],v['delta_zero'])==(sum(x>0 for x in deltas),sum(x>.05 for x in deltas),sum(x<0 for x in deltas),sum(x<-.05 for x in deltas),sum(x==0 for x in deltas))
        assert v['teacher_own_positive_but_cross_nonpositive']==sum(r['delta_T_own_minus_N_own']>0 and r['delta_T_minus_N_on_RGB_GT']<=0 for r in values)
        for m in ('N','T'):
            assert v['native_iou75_identity'][m]=={key:sum(r[m]['iou75_identity_status']==key for r in values) for key in ('same','changed','unmatched')}
            assert sum(v['native_iou75_identity'][m].values())==len(values)
    for name,c in s['cohorts'].items():
        selected=[r for r in saved if r['cohort']==name];assert len(selected)==expected[name];check(selected,c['overall'])
        for label,v in c['by_label_iou'].items():check([r for r in selected if r['stratum']==label],v)
        assert sum(v['objects'] for v in c['by_label_iou'].values())==expected[name]
    copies=list((O/'source_copies').iterdir())
    for src in (E/'cross_gt_source/analyze_cross_gt.py',E/'cross_gt_source/test_cross_gt_cpu.py',E/'CROSS_GT_PLAN.md',E/'cpu_analysis/native_cached_match.py',source):
        dest=next(p for p in copies if p.name.endswith('_'+src.name));assert src.read_bytes()==dest.read_bytes()
    assert not torch.cuda.is_initialized()
    result=dict(status='ACCEPTED_FIXED_IOU50_CROSS_GT_DESCRIPTION',objects=2819,cohorts=expected,all_fixed_native_prediction_IDs_boxes_and_GT_exact=True,all_iou75_same_changed_unmatched_identities_exact=True,all_11276_scalar_cross_IoUs_exact=True,max_abs_IoU_difference=maxerr,all_strata_denominators_images_and_signed_difference_counts_exact=True,source_copies_exact=True,overall={c:v['overall'] for c,v in s['cohorts'].items()},new_GPU=False,new_matching=False,new_NMS=False,new_hash=False,physical_alignment_verified=False,KD_gain_claim=False)
    with (E/'independent_review/CROSS_GT_ACTUAL_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='overall'}))
if __name__=='__main__':main()
