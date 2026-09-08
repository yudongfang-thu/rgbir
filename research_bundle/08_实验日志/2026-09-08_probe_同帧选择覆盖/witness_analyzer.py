"""CPU receipt readout: old assigned status, dense witnesses and native NMS matching."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import shutil

SCOPE='SAME_FORWARD_DETECTOR_WITNESS'
DEFINITIONS=('old_assigned','dense_any_correct','native_pre_nms_any_correct','native_postnms_matched')
BUCKETS=('T_correct_S_error','S_correct_T_error','both_correct','both_error','teacher_unknown')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def ratio(n,d):return n/d if d else None


def numeric(x):return type(x) in (int,float) and math.isfinite(x)


def correct(row,model,definition):
    if row['detector'][model] is None:return None
    return row['states'][model]['correct'] if definition=='old_assigned' else row['detector'][model][definition]


def validate_witness(witness,strict=False):
    require(isinstance(witness,dict),'Positive predicate requires witness')
    require(type(witness.get('anchor_index')) is int and witness['anchor_index']>=0,'Invalid witness anchor')
    require(type(witness.get('class')) is int and witness['class']==0,'LLVIP person class must be zero')
    require(numeric(witness.get('confidence')) and witness['confidence']<=1
        and (witness['confidence']>.25 if strict else witness['confidence']>=.25),'Witness confidence differs')
    require(numeric(witness.get('iou')) and .5<=witness['iou']<=1,'Witness own-GT IoU differs')
    require(isinstance(witness.get('box'),list) and len(witness['box'])==4
        and all(numeric(v) for v in witness['box']),'Witness box invalid')


def validate(previous,current,receipt,contract):
    fixed=dict(status=SCOPE+'_COMPLETED',scope=SCOPE,dataset='llvip',seed=42,batches=1,frames=32,objects_n=80,
        training=0,backward=0,optimizer_updates=0,ema_updates=0,actual_batch_size=32,
        previous_objects_exact=True,first_batch_stream_exact=True,student_full_state_unchanged=True,
        auxiliary_gradients_absent=True,new_hash_computed=False,official_test_accessed=False,
        full_dev_evaluated=False,formal_paper_gain_claim=False,global_population_coverage_claim=False)
    for k,v in fixed.items():require(type(receipt.get(k)) is type(v) and receipt[k]==v,'Completion identity differs: '+k)
    require(receipt.get('raw_forward_counts')==dict(student=1,teacher=1,reference=1),'Exactly one model forward each required')
    require(len(previous)==len(current)==80,'Fixed previous 80 objects required')
    require(contract.get('scope')==SCOPE and contract.get('objects_n')==80 and contract.get('native_tp_and_gt_pairs_exact') is True,
        'Native matching contract missing')
    profile=dict(conf=.25,iou=.7,max_det=300,agnostic_nms=False,multi_label=True,matching_iou=.5)
    require(contract.get('native_profile')==profile,'Frozen actual native profile differs')
    provenance=contract.get('native_match_provenance',{})
    require(provenance.get('captured_matches_exact') is True and all(isinstance(provenance.get(k),str) and provenance[k]
        for k in ('native_process_batch','native_match_predictions')),'Actual native matching source missing')
    require(contract.get('dense_conf_operator')=='>=' and contract.get('native_conf_operator')=='>','Confidence-boundary definitions differ')
    seen=set();native_ids=set()
    for old,new in zip(previous,current):
        require({k:v for k,v in new.items() if k!='detector'}==old,'Same80 original state/gates/IDs changed')
        sid=old['stable_rgb_gt_id'];require(sid not in seen,'Duplicate stable RGB GT ID');seen.add(sid)
        require(set(new.get('detector',{}))=={'S','T','R'},'All modalities required')
        for model in ('S','T','R'):
            d=new['detector'][model]
            if model=='T' and old['ir_global_row'] is None:
                require(d is None,'Unpaired teacher must remain unknown');continue
            require(isinstance(d,dict),'Missing modality detector witness')
            require(d.get('global_gt_row')==old['ir_global_row' if model=='T' else 'rgb_global_row']
                and d.get('stable_gt_id')==old['stable_ir_gt_id' if model=='T' else 'stable_rgb_gt_id']
                and d.get('own_gt_scope')==('ir' if model=='T' else 'rgb'),'Witness own-GT identity differs')
            for pred,count,wit,strict in (('dense_any_correct','dense_correct_count','dense_witness',False),
                ('native_pre_nms_any_correct','native_pre_nms_correct_count','native_pre_nms_witness',True)):
                require(type(d.get(pred)) is bool and type(d.get(count)) is int and d[count]>=0
                    and d[pred]==(d[count]>0),'Dense count/predicate differs')
                if d[pred]:validate_witness(d.get(wit),strict)
                else:require(d.get(wit) is None,'False dense predicate with witness')
            require(type(d.get('native_postnms_matched')) is bool,'Native matched boolean required')
            if d['native_postnms_matched']:
                w=d.get('native_witness');validate_witness(w,True)
                require(type(w.get('prediction_index')) is int and w['prediction_index']>=0,'Native prediction index invalid')
                key=(new['frame_id'],model,w['prediction_index'])
                require(key not in native_ids,'Native prediction assigned to multiple GT');native_ids.add(key)
            else:require(d.get('native_witness') is None,'Unmatched native GT with witness')
        if old['gates']['matched']:
            require(new['detector']['T']['dense_any_correct']==old['gates']['teacher_correct_own'],
                'Raw FP32 dense teacher witness differs from actual C teacher gate')


def confusion(records,model,left,right):
    counts=Counter((correct(r,model,left),correct(r,model,right)) for r in records)
    known=sum(v for (a,b),v in counts.items() if a is not None and b is not None)
    return dict(row_definition=left,column_definition=right,known_gt_n=known,unknown_gt_n=len(records)-known,
        cells=[dict(row_correct=a,column_correct=b,n=counts[(a,b)],fraction_known=ratio(counts[(a,b)],known))
            for a in (False,True) for b in (False,True)])


def bucket(row,definition):
    s=correct(row,'S',definition);t=correct(row,'T',definition)
    if t is None:return 'teacher_unknown'
    return 'both_correct' if s and t else 'T_correct_S_error' if t else 'S_correct_T_error' if s else 'both_error'


def buckets(records,definition):
    selected_total=sum(r['gates'].get('selected') is True for r in records);result={}
    for name in BUCKETS:
        r=[x for x in records if bucket(x,definition)==name];selected=[x for x in r if x['gates'].get('selected') is True]
        result[name]=dict(objects=len(r),fraction_all_gt=ratio(len(r),len(records)),selected=len(selected),
            selected_over_bucket=ratio(len(selected),len(r)),fraction_of_all_selected=ratio(len(selected),selected_total),
            unique_images=len({x['frame_id'] for x in r}),selected_images=len({x['frame_id'] for x in selected}),
            stable_rgb_gt_ids=[x['stable_rgb_gt_id'] for x in r],selected_rgb_gt_ids=[x['stable_rgb_gt_id'] for x in selected])
    require(sum(x['objects'] for x in result.values())==len(records),'Bucket partition differs')
    require(sum(x['selected'] for x in result.values())==selected_total,'Selected bucket sum differs')
    return result


def summarize(previous,current,receipt,contract):
    validate(previous,current,receipt,contract)
    return dict(status='SAME_FORWARD_DETECTOR_WITNESS_READOUT_COMPLETED',scope=SCOPE,dataset='llvip',frames=32,objects_n=80,
        selected_total=sum(r['gates'].get('selected') is True for r in current),same_previous_objects_exact=True,
        modalities={m:dict(correct_counts={d:sum(correct(r,m,d) is True for r in current) for d in DEFINITIONS},
            confusion_tables=[confusion(current,m,a,b) for a,b in combinations(DEFINITIONS,2)]) for m in ('S','T','R')},
        coverage_by_definition={d:buckets(current,d) for d in DEFINITIONS},
        native_profile=contract['native_profile'],decode_diagnostics=contract.get('decode_diagnostics'),
        definitions=dict(old_assigned='Previous FP32 GT-assisted one-to-one assigned raw anchor correctness.',
            dense_any_correct='Any FP32 proxy-decode anchor correct to own GT, confidence >= .25.',
            native_pre_nms_any_correct='Any native-head-decode anchor correct to own GT, native confidence > .25.',
            native_postnms_matched='Native NMS(.25,.7,max_det300,multi_label=True) plus captured native one-to-one IoU .5 GT match.',
            T_coordinates='IR own GT; no physical geometric alignment or RGB-remap claim.'),
        AP_estimated=False,training_performed=False,negative_transfer_claim=False,negative_transfer_eliminated_claim=False,
        global_dev_claim=False,old_dev200_combined=False,new_hash_computed=False,
        limitations='Same 32 train frames/80 augmented GT and unchanged original selection only. Matching/state definitions differ; counts are not gradients, AP changes or observed learning effects.')


def markdown(result):
    lines=['# 同帧检测见证：定义转换与实际selected覆盖','',
        '原80对象、状态、C门和selected保持exact；仅改变诊断读出。T始终相对IR own GT。没有新AP、学习更新或全dev结论。','',
        '|模型|旧assigned正确|FP32 dense-any正确|native pre-NMS正确|native post-NMS匹配正确|','|---|---:|---:|---:|---:|']
    for m in ('S','R','T'):lines.append('|'+m+'|'+'|'.join(str(result['modalities'][m]['correct_counts'][d]) for d in DEFINITIONS)+'|')
    lines+=['','## 原生NMS后匹配的机会与风险','',
        '|状态桶|对象数|原selected|selected/桶|占全部selected|','|---|---:|---:|---:|---:|']
    for name,row in result['coverage_by_definition']['native_postnms_matched'].items():
        a=row['selected_over_bucket'];b=row['fraction_of_all_selected']
        lines.append(f"|{name}|{row['objects']}|{row['selected']}|"+('NA' if a is None else f'{a:.6f}')+'|'+('NA' if b is None else f'{b:.6f}')+'|')
    lines+=['','所有模态的两两2×2混淆表、四种定义下的机会/风险与完整ID见[summary.json](summary.json)。原FP32与native decode/置信度边界可能不同；native pre-NMS列用于分开记录这种差异。NMS后的匹配为固定confidence .25的一阈值诊断，不能当作AP评价。风险桶不表示负迁移已经发生，机会桶也不表示已修复。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-probe',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'New output required');require(not (a.input/'failure.json').exists(),'Conflicting probe failure')
    previous_path=a.previous_probe/'objects.jsonl';previous=rows(previous_path);same=rows(a.input/'objects.jsonl')
    require(same==previous,'Current baseline object export differs from previous exact objects')
    receipt=read(a.input/'completion_receipt.json')
    require(receipt.get('previous_objects',{}).get('bytes')==previous_path.stat().st_size,'Prior object file byte count differs')
    current=rows(a.input/'witness_objects.jsonl');contract=read(a.input/'witness_contract.json')
    result=summarize(previous,current,receipt,contract)
    result['input_sources']=[dict(path=str(p.absolute()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        for p in (previous_path,a.input/'objects.jsonl',a.input/'witness_objects.jsonl',a.input/'completion_receipt.json',a.input/'witness_contract.json')]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (a.output/'README.md').write_text(markdown(result),encoding='utf-8');shutil.copyfile(__file__,a.output/'witness_analyzer_source.py')
    print(result['status'])


if __name__=='__main__':main()
