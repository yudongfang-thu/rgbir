"""CPU identity joins only: no model, no matching, no geometry gate rerun."""
import argparse,json,shutil,time,traceback
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCOPE='LLVIP_FIRST32_HISTORICAL_L2_NATIVE_ANCHOR_JOIN'
OPPORTUNITY='both05_onlyT075'
REVERSE='both05_onlyS075'

def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def stat(p):
    p=Path(p);s=p.stat();return dict(path=str(p.absolute()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def point(anchor,shape):
    if anchor is None:return None
    require(type(anchor) is int and anchor>=0,'Invalid anchor ID')
    offset=0
    for level,(feature,stride) in enumerate(zip(shape['feats'],(8,16,32))):
        h,w=feature[-2:];n=h*w
        if anchor<offset+n:
            i=anchor-offset;y,x=divmod(i,w)
            return dict(anchor_index=anchor,level=level,stride=stride,row=y,col=x,center_xy=[(x+.5)*stride,(y+.5)*stride])
        offset+=n
    raise ValueError('Anchor outside recorded raw layout')
def compare(a,b,missing):
    if a is None or b is None:return dict(status='unavailable',reason=missing)
    require(a[0]==b[0],'Cross-frame anchor identity comparison forbidden')
    return dict(status='same' if a[1]==b[1] else 'different',reason=None)
def learning_anchor(record,gates):
    if gates['selected']:
        require(record is not None and record['selected'] is True,'Selected object missing original record')
        return record['reference_anchor']
    return None

def one(core,bridged,witness,frame,stats,shape):
    require(frame['frame_id']==core['frame_id']==witness['frame_id'],'Frame mismatch')
    require({k:v for k,v in bridged.items() if k not in ('historical_L2_record','historical_L2_gates','raw_dense_overlap')}==core,'Core/bridge fields differ')
    for k in ('stable_rgb_gt_id','stable_ir_gt_id','rgb_global_row','ir_global_row','rgb_gt_xyxy','ir_gt_xyxy','image_index'):
        require(core[k]==witness[k],'Stable object binding differs: '+k)
    require(core['C_gates']==witness['gates'],'C gates changed')
    record=bridged['historical_L2_record'];gates=bridged['historical_L2_gates'];bi=core['image_index']
    if record is not None:
        require((record['batch_index'],record['rgb_gt_index'],record['ir_gt_index'])==(bi,core['rgb_global_row'],core['ir_global_row']),'Historical row binding differs')
        require(record['rgb_gt']==core['rgb_gt_xyxy'] and record['ir_gt']==core['ir_gt_xyxy'] and record['class_id']==witness['gt_class'],'Historical dual GT differs')
        require(record in stats['base_records'],'Historical record not in original first batch')
        if not record['reference_gap']:
            require(record['teacher_anchor'] is None and gates['teacher_own_quality'] is None and gates['mapped_rgb_quality'] is None,'Unevaluated teacher gate changed from null')
    ra=None if record is None else record['reference_anchor'];ta=None if record is None else record['teacher_anchor']
    la=learning_anchor(record,gates)
    if la is not None:require([bi,la,ta,core['rgb_global_row'],core['ir_global_row']] in stats['selected_anchors'],'Actual learning tuple not in historical selected_anchors')
    native={};dense={};detections={}
    for name in ('S','R','T'):
        m=core['matches'][name]['0.5'];native[name]=m
        nw=witness['detector'][name]['native_witness']
        require((m is None)==(nw is None),'Existing native correctness differs')
        if m is not None:
            candidates=[d for d in frame['native_detections'][name] if d['prediction_index']==m['prediction_id']]
            require(len(candidates)==1,'Native prediction ID missing or duplicated')
            d=candidates[0]
            for k in ('anchor_index','box','confidence'):require(d[k]==m[k]==nw[k],'Native ID fields differ: '+k)
            require(d['class']==m['class_id']==nw['class'] and nw['prediction_index']==m['prediction_id'],'Native class/prediction identity differs')
        dense[name]=witness['detector'][name]['dense_witness']
        wanted=ta if name=='T' else ra
        at=[d for d in frame['native_detections'][name] if wanted is not None and d['anchor_index']==wanted]
        require(len(at)<=1,'Native same-model anchor duplicated')
        detections[name]=dict(candidate_anchor=wanted,present_in_native_NMS_set=None if wanted is None else bool(at),
            existing_detection=at[0] if at else None,matched_to_this_gt=None if wanted is None else bool(at and m is not None and at[0]['prediction_index']==m['prediction_id']))
    key=lambda a:None if a is None else (bi,a)
    nk=lambda n:None if native[n] is None else key(native[n]['anchor_index'])
    relations={}
    for name in ('S','R'):
        relations['R_candidate_vs_native_'+name]=compare(key(ra),nk(name),'no_historical_base' if ra is None else 'native_GT_unmatched')
        relations['actual_learning_vs_native_'+name]=compare(key(la),nk(name),'not_L2_selected' if la is None else 'native_GT_unmatched')
    relations['T_candidate_vs_native_T']=compare(key(ta),nk('T'),'teacher_not_evaluated_or_no_candidate' if ta is None else 'native_GT_unmatched')
    dw=dense['R'];dk=None if dw is None else key(dw['anchor_index'])
    relations['R_candidate_vs_raw_dense_R']=compare(key(ra),dk,'no_historical_base' if ra is None else 'no_saved_dense_witness')
    exact=None
    if relations['R_candidate_vs_raw_dense_R']['status']=='same':
        exact=dict(box=record['reference_box']==dw['box'],confidence=record['reference_conf']==dw['confidence'],own_iou=record['reference_iou']==dw['iou'],class_id=record['class_id']==dw['class'])
        require(all(exact.values()),'Historical/current FP32 same-anchor known fields differ')
    # C is region pooling and has no single student learning-anchor ID.
    return dict(stable_rgb_gt_id=core['stable_rgb_gt_id'],stable_ir_gt_id=core['stable_ir_gt_id'],frame_id=core['frame_id'],image_index=bi,
        rgb_global_row=core['rgb_global_row'],ir_global_row=core['ir_global_row'],rgb_gt_xyxy=core['rgb_gt_xyxy'],ir_gt_xyxy=core['ir_gt_xyxy'],bucket=core['bucket'],
        C_selected=core['C_gates']['selected'],C_gates=core['C_gates'],C_single_learning_anchor=None,
        historical_L2_record=record,historical_L2_gates=gates,actual_L2_learning_anchor=point(la,shape['S']),
        historical_R_candidate=point(ra,shape['R']),historical_T_candidate=point(ta,shape['T']),
        native_iou50_matches={n:None if m is None else dict(m,anchor_geometry=point(m['anchor_index'],shape[n])) for n,m in native.items()},
        native_iou75_matches={n:core['matches'][n]['0.75'] for n in ('S','R','T')},
        raw_FP32_dense_witnesses=dense,same_R_anchor_FP32_fields_exact=exact,native_set_at_historical_candidate=detections,relations=relations,
        historical_student_DFL_at_learning_anchor=None,historical_student_decoded_box_at_learning_anchor=None)

def aggregate(values):
    rels=list(values[0]['relations']) if values else ['R_candidate_vs_native_S','R_candidate_vs_native_R','actual_learning_vs_native_S','actual_learning_vs_native_R','T_candidate_vs_native_T','R_candidate_vs_raw_dense_R']
    return dict(objects=len(values),images=len({r['frame_id'] for r in values}),C_selected=sum(r['C_selected'] for r in values),L2_base=sum(r['historical_L2_gates']['base'] for r in values),L2_selected=sum(r['historical_L2_gates']['selected'] for r in values),
        relations={k:dict({s:sum(r['relations'][k]['status']==s for r in values) for s in ('same','different','unavailable')},unavailable_reasons=dict(Counter(r['relations'][k]['reason'] for r in values if r['relations'][k]['status']=='unavailable'))) for k in rels},
        C_and_L2_selected=sum(r['C_selected'] and r['historical_L2_gates']['selected'] for r in values),
        teacher_gate_not_evaluated=sum(r['historical_L2_gates']['teacher_own_quality'] is None for r in values))

def run(a):
    require(not a.output.exists(),'New attempt required');a.output.mkdir(parents=True);started=time.perf_counter()
    try:
        require(read(a.core.parent/'independent_review/ACTUAL_REVIEW_RECEIPT.json')['status']=='ACCEPTED_FIRST32_CACHED_LOCALIZATION_DESCRIPTION','Accepted core input required')
        require(read(a.core.parent/'independent_review/BRIDGE_ACTUAL_RECEIPT.json')['status']=='ACCEPTED_HISTORICAL_RECORD_BRIDGE_ONLY','Accepted historical bridge input required')
        core=rows(a.core/'objects.jsonl');bridged=rows(a.bridge/'objects.jsonl');witness=rows(a.witness/'witness_objects.jsonl');frames=rows(a.witness/'witness_frames.jsonl')
        stats=read(a.bridge/'original_first_batch_L2_stats.json');export=read(a.witness/'export_contract.json');identity=read(a.bridge/'summary.json')['identity']
        require(len(core)==len(bridged)==len(witness)==80 and len(frames)==32 and len(stats['base_records'])==79 and stats['selected_count']==7,'Fixed population differs')
        require(export['raw_shapes']['S']==export['raw_shapes']['R']==export['raw_shapes']['T'],'Anchor layout differs')
        require('decode_selected(student, ids[:, 0], ids[:, 1], centers, stride_values)' in a.l2_source.read_text(encoding='utf-8'),'Frozen student learning-anchor source changed')
        by=lambda rs,k:{r[k]:r for r in rs};cr,br,wr=[by(rs,'stable_rgb_gt_id') for rs in (core,bridged,witness)];fr=by(frames,'frame_id')
        require(len(cr)==len(br)==len(wr)==80 and set(cr)==set(br)==set(wr),'Duplicate or missing GT identity')
        values=[one(r,br[k],wr[k],fr[r['frame_id']],stats,export['raw_shapes']) for k,r in cr.items()]
        cohorts={'all80':values,'native_T_localization11':[r for r in values if r['bucket']==OPPORTUNITY],'native_S_reverse1':[r for r in values if r['bucket']==REVERSE], 'L2_selected7':[r for r in values if r['historical_L2_gates']['selected']]}
        require([len(cohorts[k]) for k in cohorts]==[80,11,1,7],'Fixed cohort counts differ')
        cohorts['T_localization_intersection_L2_selected']=[r for r in cohorts['native_T_localization11'] if r['historical_L2_gates']['selected']]
        with (a.output/'objects.jsonl').open('x',encoding='utf-8') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        missing=[dict(field='historical_student_DFL_at_learning_anchor',scope='all 7 selected',status='unavailable',effect='Cannot reconstruct loss gradient at actual student learning point'),
            dict(field='historical_student_decoded_box_at_learning_anchor',scope='all 7 selected',status='unavailable',effect='Do not substitute frozen R box or current native AMP box as historical S decode'),
            dict(field='C_single_learning_anchor',scope='all 80',status='not_defined',effect='C pools multiple foreground/background positions; selected object is not one localization anchor'),
            dict(field='historical_teacher_candidate_after_reference_gap_rejection',scope='per object null remains null',status='not_executed',effect='Cannot infer whether a teacher candidate would pass'),
            dict(field='entire_historical_raw_or_pixels_and_T_checkpoint_stat',scope='across historical/current forwards',status='not_recorded',effect='Identity bridge and same-anchor partial fields do not establish whole-forward bitwise equivalence')]
        summary=dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',scope=SCOPE,cohorts={k:aggregate(v) for k,v in cohorts.items()},frames=32,objects=80,original_bridge_identity=identity,missing_fields=missing,
            actual_student_learning_anchor_source='selected_anchors[:,1] passed to decode_selected(student, ids[:,0], ids[:,1],...)',
            same_anchor_meaning='same image index and original flattened anchor ID on recorded grid; box values are separate',
            no_current_L2_reexecution=True,new_GPU=False,new_matching=False,new_forward=False,new_training=False,new_hash_computed=False,gradient_claim=False,seconds=time.perf_counter()-started)
        write(a.output/'summary.json',summary)
        paths=[a.core/'objects.jsonl',a.bridge/'objects.jsonl',a.bridge/'original_first_batch_L2_stats.json',a.bridge/'summary.json',a.witness/'witness_objects.jsonl',a.witness/'witness_frames.jsonl',a.witness/'export_contract.json',a.l2_source,HERE/'PLAN.md',Path(__file__),HERE/'test_anchor_join_cpu.py']
        write(a.output/'input_manifest.json',dict(files=[stat(p) for p in paths],new_hash_computed=False));snap=a.output/'source_copies';snap.mkdir()
        for i,p in enumerate(paths[-4:]):shutil.copyfile(p,snap/(str(i)+'_'+p.name))
        md=['# 首32图native框与历史L2学习anchor','', '仅CPU连接现有记录，待独立验收。全部80GT、历史base79与selected7保持原分母。','', '|分组|GT|C selected|L2 selected|R候选对S匹配 same/different/unavailable|实际学习对S匹配 same/different/unavailable|','|---|---:|---:|---:|---|---|']
        for k,v in summary['cohorts'].items():
            rel=lambda key:'/'.join(str(v['relations'][key][s]) for s in ('same','different','unavailable'))
            md.append(f"|{k}|{v['objects']}|{v['C_selected']}|{v['L2_selected']}|{rel('R_candidate_vs_native_S')}|{rel('actual_learning_vs_native_S')}|")
        md+=['', '逐对象保留真实frame/GT/prediction/anchor ID、level/row/col/center、native框与FP32历史/dense字段。same来自ID而不是框相等。C是区域池化，不能把C selected冒称L2学习位置。未通过reference gap而未执行教师门的null全部保留。', '', '缺字段：历史S在学习anchor的DFL和decode未保存；不能用R框或当前native AMP框补成历史S状态。两次forward仅沿用旧桥接已证明的配置/流/GT/初始化和部分FP32重叠，不声称整张raw或像素exact。此项不重跑任何匹配/selector，不提供梯度或增益结论。']
        (a.output/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
        write(a.output/'completion.json',dict(status=summary['status'],scope=SCOPE,objects=80,seconds=summary['seconds'],new_GPU=False,new_hash_computed=False));print(json.dumps(summary['cohorts'],ensure_ascii=False))
    except BaseException as e:
        write(a.output/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('core','bridge','witness','l2-source','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
