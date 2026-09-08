"""Bounded historical record identity and counter verification, no tensor inference."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import yaml

ENTRY=Path(__file__).resolve().parents[1]
LOGS=ENTRY.parent
CAL=LOGS/'2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip'
WIT=LOGS/'2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe'
OUT=ENTRY/'bridge/output_attempt1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def main():
    source=ENTRY/'bridge/bridge_l2_records.py'
    spec=importlib.util.spec_from_file_location('bridge_review_target',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s,r,stats=m.analyze(CAL,WIT,ENTRY/'output_attempt1')
    assert s==read(OUT/'summary.json') and r==lines(OUT/'objects.jsonl') and stats==read(OUT/'original_first_batch_L2_stats.json')
    assert source.read_bytes()==(OUT/'bridge_l2_records_source.py').read_bytes()
    old=lines(CAL/'calibration_batches.jsonl')[0]; ws=lines(WIT/'witness_objects.jsonl'); core=lines(ENTRY/'output_attempt1/objects.jsonl')
    assert old['files']==read(WIT/'first_batch_stream.json')['im_file']
    records=old['stats']['L2-box']['base_records'];assert records==old['stats']['L2-GT']['base_records']
    byid={v['rgb_gt_index']:v for v in records};assert len(records)==len(byid)==79
    fixed=stats['config'];assert fixed==s['old_L2_config']
    for w,c,b in zip(ws,core,r):
        i=w['rgb_global_row'];original=byid.get(i);assert b['historical_L2_record']==original
        assert {k:v for k,v in b.items() if k not in ('historical_L2_record','historical_L2_gates','raw_dense_overlap')}==c
        assert old['files'][w['image_index']]==w['image'] and b['C_gates']==w['gates']
        if original is None:
            assert b['historical_L2_gates']['exit']=='not_in_historical_L2_base';continue
        for oldkey,newkey in [('batch_index','image_index'),('rgb_gt_index','rgb_global_row'),('ir_gt_index','ir_global_row'),('class_id','gt_class'),('rgb_gt','rgb_gt_xyxy'),('ir_gt','ir_gt_xyxy'),('pair_iou','pair_iou')]:assert original[oldkey]==w[newkey]
        reliable=original['reference_reliable'];gap=reliable and original['reference_iou']<.7
        own=original['teacher_anchor'] is not None if gap else None
        quality=original['mapped_teacher_rgb_iou']>=.6 if own else None
        margin=original['mapped_teacher_rgb_iou']>original['reference_iou']+.05 if quality else None
        selected=bool(gap and own and quality and margin)
        assert original['selected']==selected and b['historical_L2_gates']['selected']==selected
        assert b['historical_L2_gates']['teacher_own_quality'] is own
        assert b['historical_L2_gates']['mapped_rgb_quality'] is quality
        assert b['historical_L2_gates']['localization_margin'] is margin
        if own:assert original['teacher_conf']>=.25 and original['teacher_own_iou']>=.5
    def counts(rr,v):
        assert v['objects']==len(rr) and v['unique_images']==len({x['frame_id'] for x in rr})
        assert v['exits']==dict(Counter(x['historical_L2_gates']['exit'] for x in rr))
        for g in ('base','reference_reliable','reference_gap','teacher_own_quality','mapped_rgb_quality','selected'):
            keep=[x for x in rr if x['historical_L2_gates'][g] is True];cv=v[g]
            assert cv['objects']==len(keep) and cv['unique_images']==len({x['frame_id'] for x in keep})
            assert cv['not_evaluated']==sum(x['historical_L2_gates'][g] is None for x in rr)
            assert cv['stable_rgb_gt_ids']==[x['stable_rgb_gt_id'] for x in keep]
            assert cv['fraction_bucket']==(len(keep)/len(rr) if rr else None)
        for c in (0,1):
            for l in (0,1):assert v['C_selected_cross_L2_selected'][f'C{c}_L2{l}']==sum(x['C_gates']['selected']==bool(c) and x['historical_L2_gates']['selected']==bool(l) for x in rr)
    counts(r,s['all'])
    for b,v in s['by_native_bucket'].items():counts([x for x in r if x['bucket']==b],v)
    overlap={}
    for model,prefix in [('R','reference'),('T','teacher')]:
        same=exact=0
        for w in ws:
            record=byid.get(w['rgb_global_row']);d=w['detector'][model]['dense_witness']
            if record and d and record[prefix+'_anchor']==d['anchor_index']:
                same+=1
                exact+=all((record[prefix+'_box']==d['box'],record[prefix+'_conf']==d['confidence'],record['class_id']==d['class'],record['reference_iou' if model=='R' else 'teacher_own_iou']==d['iou']))
        overlap[model]=dict(same_anchor=same,all_fields_exact=exact)
        assert s['partial_same_anchor_raw_dense_field_overlap'][model]['same_anchor_records']==same
        assert s['partial_same_anchor_raw_dense_field_overlap'][model]['all_reported_fields_exact']==exact
    target=s['by_native_bucket']['both05_onlyT075']
    result=dict(status='ACCEPTED_HISTORICAL_RECORD_BRIDGE_ONLY',objects=80,original_base_records=79,source_and_replay_exact=True,original_identity_and_gate_fields_exact=True,all_counts_and_cross_cells_exact=True,all_C_L2_selected_cross=s['all']['C_selected_cross_L2_selected'],eleven_C_L2_selected_cross=target['C_selected_cross_L2_selected'],eleven_L2_gate_counts={g:target[g]['objects'] for g in ('base','reference_reliable','reference_gap','teacher_own_quality','mapped_rgb_quality','selected')},partial_raw_overlap=overlap,historical_teacher_separate_stat_available=False,current_L2_reexecution=False,full_raw_tensor_identity_claim=False,new_GPU=False,new_forward=False,new_hash=False)
    with (ENTRY/'independent_review/BRIDGE_ACTUAL_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result))
if __name__=='__main__':main()
