"""CPU-only same-forward object buckets and actual C-gate coverage; no AP."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

SCOPE='SAME_FORWARD_SELECTION_COVERAGE'
STATES=('no_candidate','low_confidence','class_and_localization','class_only','localization_only','correct')
GATE_ORDER=('matched','region_valid','reference_candidate','base','teacher_correct_own','quality_positive','eligible','selected')
BUCKETS=('T_correct_S_error','S_correct_T_error','both_correct','both_error','teacher_unpaired')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fraction(n,d):return n/d if d else None


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def validate_state(state):
    require(isinstance(state,dict) and state.get('state') in STATES,'Unknown object state')
    for k in ('candidate','confidence_ok','class_ok','localization_ok','correct'):
        require(type(state.get(k)) is bool,'Missing state boolean: '+k)
    if not state['candidate']:expected='no_candidate'
    elif not state['confidence_ok']:expected='low_confidence'
    elif not state['class_ok'] and not state['localization_ok']:expected='class_and_localization'
    elif not state['class_ok']:expected='class_only'
    elif not state['localization_ok']:expected='localization_only'
    else:expected='correct'
    require(state['state']==expected and state['correct']==(expected=='correct'),'State priority/flags differ')


def validate_records(records):
    require(isinstance(records,list),'Object list required')
    ids=set();global_rows=set();matched=[];base=[];selected=[]
    for row in records:
        for k in ('frame_id','stable_rgb_gt_id','image'):
            require(isinstance(row.get(k),str) and row[k],'Missing stable object/frame identity: '+k)
        require(row['stable_rgb_gt_id'] not in ids,'Duplicate stable RGB GT ID');ids.add(row['stable_rgb_gt_id'])
        require(type(row.get('rgb_global_row')) is int and row['rgb_global_row']>=0,'Invalid global RGB GT row')
        require(row['rgb_global_row'] not in global_rows,'Duplicate RGB global row');global_rows.add(row['rgb_global_row'])
        require(type(row.get('image_index')) is int and 0<=row['image_index']<32,'Invalid batch image index')
        states=row.get('states',{});validate_state(states.get('S'));validate_state(states.get('R'))
        g=row.get('gates',{});require(type(g.get('matched')) is bool,'Matched flag required')
        if not g['matched']:
            require(states.get('T') is None and states.get('T_to_RGB') is None,'Unpaired teacher state is unknown, not incorrect')
            require(row.get('stable_ir_gt_id') is None and row.get('ir_global_row') is None,'Unpaired IR identity must be null')
            require(all(v is None for k,v in g.items() if k!='matched'),'Unpaired downstream gates must be null')
            continue
        validate_state(states.get('T'));validate_state(states.get('T_to_RGB'))
        require(isinstance(row.get('stable_ir_gt_id'),str) and row['stable_ir_gt_id'],'Missing paired IR stable ID')
        require(type(row.get('ir_global_row')) is int and row['ir_global_row']>=0,'Missing paired IR row')
        for k in GATE_ORDER[1:]:require(type(g.get(k)) is bool,'Missing actual gate: '+k)
        require(isinstance(g.get('valid_levels'),list) and len(g['valid_levels'])==2
            and all(type(v) is bool for v in g['valid_levels']),'P3/P4 validity required')
        require(g['region_valid']==any(g['valid_levels']),'Region validity differs')
        require(finite(g.get('quality_q')) and g['quality_q']>=0 and g['quality_positive']==(g['quality_q']>0),'q predicate differs')
        require(g['base']==(g['region_valid'] and g['reference_candidate']),'Actual base differs from region/ref conjunction')
        require(g['eligible']==(g['base'] and g['teacher_correct_own'] and g['quality_positive']),'Actual eligible differs')
        require(not g['selected'] or g['eligible'],'Selected outside eligible')
        require(type(g.get('matched_row')) is int and g['matched_row']>=0,'Matched row missing');matched.append(row)
        if g['base']:
            require(type(g.get('base_row')) is int and g['base_row']>=0,'Base row missing');base.append(row)
        else:require(g.get('base_row') is None or g.get('base_row')==-1,'Nonbase row must be missing/-1')
        if g['selected']:
            require(type(g.get('selected_rank')) is int and g['selected_rank']>=1,'Selected rank missing');selected.append(row)
        else:require(g.get('selected_rank') is None,'Unselected rank must be null')
    require(sorted(global_rows)==list(range(len(records))),'Global RGB rows must cover actual augmented label table')
    for rows,k in ((matched,'matched_row'),(base,'base_row'),(selected,'selected_rank')):
        start=1 if k=='selected_rank' else 0
        require(sorted(x['gates'][k] for x in rows)==list(range(start,len(rows)+start)),'Incomplete/duplicate '+k)
    require(len({x['stable_ir_gt_id'] for x in matched})==len(matched),'Paired IR GT reused')
    eligible=sorted((x for x in matched if x['gates']['eligible']),key=lambda x:x['gates']['matched_row'])
    expected=sorted(eligible,key=lambda x:-x['gates']['quality_q'])[:math.ceil(.5*len(eligible))]
    observed=sorted(selected,key=lambda x:x['gates']['selected_rank'])
    require([x['stable_rgb_gt_id'] for x in expected]==[x['stable_rgb_gt_id'] for x in observed],
        'Actual whole-batch rho=.5 stable q selection differs')


def bucket(row,teacher_key='T'):
    teacher=row['states'][teacher_key]
    if teacher is None:return 'teacher_unpaired'
    s=row['states']['S']['correct'];t=teacher['correct']
    return 'both_correct' if s and t else 'T_correct_S_error' if t else 'S_correct_T_error' if s else 'both_error'


def support(rows):
    return dict(objects=len(rows),unique_images=len({r['frame_id'] for r in rows}),
        stable_rgb_gt_ids=[r['stable_rgb_gt_id'] for r in rows])


def funnel(rows,total,paired):
    start=len(rows);current=list(rows);out=[]
    for gate in GATE_ORDER:
        before=current;current=[r for r in before if r['gates'].get(gate) is True]
        lost=[r for r in before if r['gates'].get(gate) is not True]
        out.append(dict(gate=gate,entering_n=len(before),retained_n=len(current),lost_n=len(lost),
            retained_over_entering=fraction(len(current),len(before)),retained_over_bucket=fraction(len(current),start),
            retained_over_all_rgb=fraction(len(current),total),retained_over_paired=fraction(len(current),paired),
            retained=support(current),lost=support(lost)))
    require(sum(x['lost_n'] for x in out)+len(current)==start,'Cumulative first-loss accounting differs')
    return out


def selector_counts(records):
    matched=[r for r in records if r['gates']['matched']]
    c=dict(rgb_gt_count=len(records),common_count=len(matched),
        valid_region_count=sum(r['gates']['region_valid'] for r in matched),
        reference_candidate_count=sum(r['gates']['reference_candidate'] for r in matched),
        base_count=sum(r['gates']['base'] for r in matched),
        teacher_correct_base_count=sum(r['gates']['base'] and r['gates']['teacher_correct_own'] for r in matched),
        eligible_count=sum(r['gates']['eligible'] for r in matched),selected_count=sum(r['gates']['selected'] for r in matched))
    c['normalizer']=max(1,c['base_count']);return c


def summarize(records,receipt,frames):
    fixed=dict(status=SCOPE+'_COMPLETED',scope=SCOPE,dataset='llvip',seed=42,frames=32,batches=1,
        training=0,backward=0,optimizer_updates=0,ema_updates=0,new_hash_computed=False,
        official_test_accessed=False,full_dev_evaluated=False,formal_paper_gain_claim=False,global_population_coverage_claim=False,
        actual_batch_size=32,student_full_state_unchanged=True,auxiliary_gradients_absent=True,optimizer_state_empty=True)
    for k,v in fixed.items():require(type(receipt.get(k)) is type(v) and receipt[k]==v,'Completion scope differs: '+k)
    require(receipt.get('objects_n')==len(records),'Object population differs')
    identity=receipt.get('identity_contract',{})
    require(identity.get('status')=='STABLE_GT_IDENTITY_VERIFIED','Stable identity receipt missing')
    require(receipt.get('raw_forward_counts')==dict(student=1,teacher=1,reference=1),'Exactly one shared raw forward per model required')
    require(receipt.get('first_batch_stream_exact') is True,'Frozen first32 real stream proof missing')
    require(len(frames)==32 and len({x['frame_id'] for x in frames})==32,'All 32 frames including empty frames required')
    require(sorted(x['image_index'] for x in frames)==list(range(32)),'Frame indices differ')
    frame_map={x['frame_id']:x for x in frames}
    validate_records(records)
    require(len(identity.get('rgb_rows',[]))==len(records),'Stable RGB identity population differs')
    for row in records:
        require(row['frame_id'] in frame_map and row['image_index']==frame_map[row['frame_id']]['image_index']
            and row['image']==frame_map[row['frame_id']]['image'],'Object/frame binding differs')
        native=identity['rgb_rows'][row['rgb_global_row']]
        require(native['global_gt_row']==row['rgb_global_row'] and native['image_index']==row['image_index']
            and native['stable_gt_id']==row['stable_rgb_gt_id'],'Stable RGB identity mapping differs')
        if row['gates']['matched']:
            native=identity['ir_rows'][row['ir_global_row']]
            require(native['global_gt_row']==row['ir_global_row'] and native['image_index']==row['image_index']
                and native['stable_gt_id']==row['stable_ir_gt_id'],'Stable IR identity mapping differs')
    counts=selector_counts(records)
    counts['teacher_gt_count']=len(identity.get('ir_rows',[]))
    require(all(type(receipt.get('selector_counts',{}).get(k)) is int and receipt['selector_counts'][k]==v
        for k,v in counts.items()),'Per-object gates differ from actual original C0 aggregate counts')
    n=len(records);paired=counts['common_count'];tables={}
    for teacher_key in ('T','T_to_RGB'):
        rows={b:[r for r in records if bucket(r,teacher_key)==b] for b in BUCKETS}
        require(sum(map(len,rows.values()))==n,'Bucket partition differs')
        tables[teacher_key]={b:dict(**support(v),fraction_all_rgb=fraction(len(v),n),
            fraction_paired_rgb=fraction(len(v),paired) if b!='teacher_unpaired' else None,
            student_state_counts=dict(Counter(r['states']['S']['state'] for r in v)),
            reference_state_counts=dict(Counter(r['states']['R']['state'] for r in v)),
            teacher_state_counts=dict(Counter(r['states'][teacher_key]['state'] if r['states'][teacher_key] else 'unpaired' for r in v)),
            actual_gate_funnel=funnel(v,n,paired)) for b,v in rows.items()}
    joint=Counter((r['states']['T']['correct'],r['gates']['teacher_correct_own']) for r in records if r['gates']['matched'])
    return dict(status='SAME_FORWARD_COVERAGE_READOUT_COMPLETED',scope=SCOPE,dataset='llvip',frames=32,batches=1,
        augmented_rgb_gt_objects=n,paired_rgb_gt_objects=paired,empty_frames=32-len({r['frame_id'] for r in records}),
        actual_selector_counts=counts,all_objects_gate_funnel=funnel(records,n,paired),
        teacher_own_gt_primary=tables['T'],teacher_box_to_RGB_secondary=tables['T_to_RGB'],
        assigned_teacher_state_vs_actual_any_candidate_gate=[dict(assigned_T_correct=a,actual_T_gate=b,objects=joint[(a,b)])
            for a in (False,True) for b in (False,True)],
        frame_counts=[dict(frame_id=f['frame_id'],image=f['image'],image_index=f['image_index'],
            rgb_gt_n=sum(r['frame_id']==f['frame_id'] for r in records),
            selected_n=sum(r['frame_id']==f['frame_id'] and r['gates'].get('selected') is True for r in records)) for f in frames],
        denominator_notes=dict(all_rgb='All surviving augmented RGB GT in the fixed 32 train frames; dropped augmentation GT are outside this denominator.',
            paired='RGB GT matched to IR GT by the real selector. Unpaired teacher is unknown, not incorrect.',
            gate='Each cumulative row enters only after earlier gates; isolated original selector counts are separate and may overlap.',
            state='Mutually exclusive one-to-one assigned-box diagnostic; it does not replace the actual any-candidate C gates.'),
        AP_estimated=False,negative_transfer_eliminated_claim=False,dev200_repair_rates_combined=False,
        gradients_measured=False,training_performed=False,new_hash_computed=False)


def markdown(r):
    lines=['# 固定首32图同帧状态与实际C门覆盖','',
        '仅真实首批训练图的一次无梯度前向；对象分母为增强后保留RGB GT。T主状态相对IR own GT，T框对RGB GT另列。状态诊断不代替实际选择门；没有AP、训练效果或负迁移消除结论。','',
        f"32图，增强后RGB GT {r['augmented_rgb_gt_objects']}，实际配对 {r['paired_rgb_gt_objects']}；空图 {r['empty_frames']}。",'',
        '|主状态桶|对象数|占全部RGB GT|实际selected|selected/桶|','|---|---:|---:|---:|---:|']
    for b,v in r['teacher_own_gt_primary'].items():
        selected=v['actual_gate_funnel'][-1]['retained_n'];rate=fraction(selected,v['objects'])
        lines.append('|'+b+'|'+str(v['objects'])+'|'+('NA' if v['fraction_all_rgb'] is None else f"{v['fraction_all_rgb']:.6f}")+'|'+str(selected)+'|'+('NA' if rate is None else f'{rate:.6f}')+'|')
    for name in ('T_correct_S_error','S_correct_T_error'):
        lines+=['','### '+name,'','|实际累计门|进入|保留|本门流失|保留/进入|','|---|---:|---:|---:|---:|']
        for row in r['teacher_own_gt_primary'][name]['actual_gate_funnel']:
            rate=row['retained_over_entering'];lines.append(f"|{row['gate']}|{row['entering_n']}|{row['retained_n']}|{row['lost_n']}|"+('NA' if rate is None else f'{rate:.6f}')+'|')
    lines+=['','完整对象ID、S/R/T错误类型、T对RGB次级状态、原C0计数与全部逐门分母见[summary.json](summary.json)。区域/R候选是并列primitive条件；表按固定顺序作首次流失分解，不能把前后门顺序当因果贡献。32图属于train，不与旧200图dev修复率混用。']
    return '\n'.join(lines)+'\n'


def validate_export_contract(value,objects_n):
    require(value.get('scope')==SCOPE and value.get('objects_n')==objects_n
        and value.get('exact_selector_crosscheck') is True and value.get('full_diagnostics') is True
        and value.get('stable_identity_contract_status')=='STABLE_GT_IDENTITY_VERIFIED','Actual selector export contract missing')
    require(value.get('evidence_config',{}).get('rho')==.5
        and value['evidence_config'].get('levels')==[0,1],'Frozen selector rho/levels differ')
    for k,v in dict(match_iou=.5,reference_conf=.05,reference_iou=.1,teacher_conf=.25,teacher_iou=.5,
                    minimum_foreground=1,minimum_background=4,background_scale=2.).items():
        require(value['evidence_config'].get(k)==v,'Frozen actual selector threshold differs: '+k)
    state=value.get('state_definition',{})
    for k,v in dict(coarse_confidence=.05,coarse_iou=.1,correct_confidence=.25,correct_iou=.5).items():
        require(state.get(k)==v,'Frozen diagnostic thresholds differ')
    require(state.get('state_priority')==list(STATES),'State priority contract differs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'New output required')
    require(not (a.input/'failure.json').exists(),'Conflicting producer failure')
    rows=[json.loads(x) for x in (a.input/'objects.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    frames=[json.loads(x) for x in (a.input/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    validate_export_contract(read(a.input/'export_contract.json'),len(rows))
    result=summarize(rows,read(a.input/'completion_receipt.json'),frames)
    result['input_sources']=[dict(path=str(p.absolute()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        for p in (a.input/'objects.jsonl',a.input/'frames.jsonl',a.input/'completion_receipt.json',a.input/'export_contract.json')]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (a.output/'README.md').write_text(markdown(result),encoding='utf-8');shutil.copyfile(__file__,a.output/'analyzer_source.py')
    print(result['status'])


if __name__=='__main__':main()
