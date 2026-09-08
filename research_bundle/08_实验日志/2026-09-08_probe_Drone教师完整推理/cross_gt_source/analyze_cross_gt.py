"""Fixed native IoU .5 witnesses cross independent GT; CPU only, no rematching."""
import argparse,gzip,json,shutil,sys,time,traceback
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
CPU=HERE.parent/'cpu_analysis'
sys.path.insert(0,str(CPU))
from native_cached_match import load_native

SCOPE='DRONE_FIXED_NATIVE_IOU50_WITNESS_CROSS_GT'
EXPECTED={'T_own75_only':1869,'N_own75_only':950}
STRATA=('label_iou_0.5_to_lt0.8','label_iou_0.8_to_1')

def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def stat(p):
    p=Path(p);s=p.stat();return dict(path=str(p.absolute()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def cohort(pair):
    low,high=pair['groups']['gt025_iou50'],pair['groups']['gt025_iou75']
    if not (low['N']['correct'] and low['T']['correct']):return None
    n,t=high['N']['correct'],high['T']['correct']
    return 'T_own75_only' if t and not n else 'N_own75_only' if n and not t else None
def stratum(iou):
    require(.5<=iou<=1,'Invalid original label-pair IoU')
    return STRATA[0] if iou<.8 else STRATA[1]
def matrix(ngt,tgt,npred,tpred,box_iou):
    return box_iou(torch.tensor([ngt,tgt],dtype=torch.float32),torch.tensor([npred,tpred],dtype=torch.float32))

def make_record(pair,frame,predictions,box_iou):
    c=cohort(pair);require(c in EXPECTED,'Object outside fixed cohorts')
    require(frame['pair_key']==pair['pair_key'],'Frame identity differs')
    require(frame['canvas_shape']==[544,672] and frame['original_shape']==[512,640],'Frozen native/original canvas differs')
    selected={};statuses={}
    for n in ('N','T'):
        low=pair['groups']['gt025_iou50'][n]['witness'];high=pair['groups']['gt025_iou75'][n]['witness']
        p=predictions[(pair['pair_key'],n,low['prediction_id'])]
        require(p['prediction_id']==low['prediction_id'] and p['confidence']>.25 and p['pred_class']==pair['gt_class'],'Fixed native prediction identity differs')
        require(p['groups']['gt025_iou50']['included'] is True and p['groups']['gt025_iou50']['gt_id']==pair[n+'_gt_id'],'Fixed .5 native GT mapping differs')
        require(p['confidence']==low['confidence'],'Native witness confidence differs')
        p75=None
        if high is not None:
            p75=predictions[(pair['pair_key'],n,high['prediction_id'])]
            require(p75['groups']['gt025_iou75']['gt_id']==pair[n+'_gt_id'],'Native .75 witness identity differs')
        statuses[n]='unmatched' if high is None else 'same' if high['prediction_id']==low['prediction_id'] else 'changed'
        selected[n]=dict(prediction_id=low['prediction_id'],filtered_prediction_id=low['filtered_prediction_id'],box=p['box'],confidence=p['confidence'],
            own_iou50_witness=low['iou'],native_iou75_match=high,native_iou75_box=None if p75 is None else p75['box'],iou75_identity_status=statuses[n])
    v=matrix(pair['N_gt_box'],pair['T_gt_box'],selected['N']['box'],selected['T']['box'],box_iou)
    nn,tn,nt,tt=float(v[0,0]),float(v[0,1]),float(v[1,0]),float(v[1,1])
    require(nn==selected['N']['own_iou50_witness'] and tt==selected['T']['own_iou50_witness'],'Fixed .5 own IoU not exact to accepted native witness')
    for n,gt in [('N',pair['N_gt_box']),('T',pair['T_gt_box'])]:
        s=selected[n]
        if s['native_iou75_match'] is not None:
            measured=float(box_iou(torch.tensor([gt],dtype=torch.float32),torch.tensor([s['native_iou75_box']],dtype=torch.float32))[0,0])
            require(measured==s['native_iou75_match']['iou'],'Native .75 alternative witness IoU differs')
    return dict(cohort=c,pair_key=pair['pair_key'],N_gt_id=pair['N_gt_id'],T_gt_id=pair['T_gt_id'],N_gt_row=pair['N_gt_row'],T_gt_row=pair['T_gt_row'],
        N_gt_box=pair['N_gt_box'],T_gt_box=pair['T_gt_box'],gt_class=pair['gt_class'],label_pair_iou=pair['pair_iou'],stratum=stratum(pair['pair_iou']),
        canvas_shape=frame['canvas_shape'],original_shape=frame['original_shape'],N=selected['N'],T=selected['T'],
        N_fixed_to_RGB_GT_iou=nn,T_fixed_to_IR_GT_iou=tt,T_fixed_to_RGB_GT_iou=tn,N_fixed_to_IR_GT_iou=nt,
        delta_T_minus_N_on_RGB_GT=tn-nn,delta_T_own_minus_N_own=tt-nn)

def count(values):
    n=len(values);delta=[r['delta_T_minus_N_on_RGB_GT'] for r in values]
    return dict(objects=n,images=len({r['pair_key'] for r in values}),
        T_on_RGB_ge050=sum(r['T_fixed_to_RGB_GT_iou']>=.5 for r in values),T_on_RGB_ge075=sum(r['T_fixed_to_RGB_GT_iou']>=.75 for r in values),
        N_on_RGB_ge050=sum(r['N_fixed_to_RGB_GT_iou']>=.5 for r in values),N_on_RGB_ge075=sum(r['N_fixed_to_RGB_GT_iou']>=.75 for r in values),
        T_fixed_own_ge075=sum(r['T_fixed_to_IR_GT_iou']>=.75 for r in values),
        delta_positive=sum(d>0 for d in delta),delta_gt005=sum(d>.05 for d in delta),delta_negative=sum(d<0 for d in delta),delta_lt_negative005=sum(d<-.05 for d in delta),delta_zero=sum(d==0 for d in delta),
        teacher_own_positive_but_cross_nonpositive=sum(r['delta_T_own_minus_N_own']>0 and r['delta_T_minus_N_on_RGB_GT']<=0 for r in values),
        native_iou75_identity={m:{status:sum(r[m]['iou75_identity_status']==status for r in values) for status in ('same','changed','unmatched')} for m in ('N','T')})

def summarize(values):
    result={}
    for name,expected in EXPECTED.items():
        selected=[r for r in values if r['cohort']==name];require(len(selected)==expected,'Frozen cohort denominator differs')
        result[name]=dict(overall=count(selected),by_label_iou={s:count([r for r in selected if r['stratum']==s]) for s in STRATA})
        require(sum(v['objects'] for v in result[name]['by_label_iou'].values())==expected,'Strata do not close')
    return result

def run(a):
    require(not a.output.exists(),'New immutable output required');a.output.mkdir(parents=True);started=time.perf_counter()
    try:
        accepted=read(a.acceptance);require(accepted['status']=='ACCEPTED_DRONE_PAIRED_CPU_TABLES' and accepted['all_native_pairs_TP_and_original_prediction_IDs_exact'] is True,'Accepted parent readout required')
        parent=read(a.input/'summary.json');require(parent['scope']=='DRONE_FULL_DEV_CACHED_NATIVE_PAIRED_OPPORTUNITIES' and parent['paired_GT']==21568,'Parent scope differs')
        pairs=[r for r in rows(a.input/'paired_objects.jsonl.gz') if cohort(r) is not None]
        require(len({r['N_gt_id'] for r in pairs})==sum(EXPECTED.values()),'Duplicate/missing cohort GT')
        for c,n in EXPECTED.items():require(sum(cohort(r)==c for r in pairs)==n,'Fixed cohort selection differs')
        needed=set()
        for p in pairs:
            for m in ('N','T'):
                for g in ('gt025_iou50','gt025_iou75'):
                    witness=p['groups'][g][m]['witness']
                    if witness is not None:needed.add((p['pair_key'],m,witness['prediction_id']))
        predictions={}
        for frame in rows(a.input/'prediction_matches.jsonl.gz'):
            for p in frame['predictions']:
                key=(frame['pair_key'],frame['model'],p['prediction_id'])
                if key in needed:require(key not in predictions,'Duplicate needed native prediction');predictions[key]=p
        require(set(predictions)==needed,'Missing original prediction witness')
        frames={r['pair_key']:r for r in rows(a.input/'frames.jsonl.gz')}
        box_iou=load_native(CPU/'frozen_sources')[2]
        values=[make_record(p,frames[p['pair_key']],predictions,box_iou) for p in pairs]
        with gzip.open(a.output/'objects.jsonl.gz','xt',encoding='utf-8') as f:
            for r in values:f.write(json.dumps(r,allow_nan=False)+'\n')
        summary=dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',scope=SCOPE,cohorts=summarize(values),
            primary_boxes='Original native IoU .5 matched prediction IDs, never substituted by .75 matches',native_canvas_HW=[544,672],original_image_HW=[512,640],
            physical_alignment_verified=False,approximate_identity_coordinate_readout=True,new_matching=False,new_GPU=False,new_NMS=False,new_hash_computed=False,checkpoint_loaded=False,AP_estimated=False,KD_gain_claim=False,seconds=time.perf_counter()-started)
        write(a.output/'summary.json',summary)
        source=[Path(__file__),HERE/'test_cross_gt_cpu.py',HERE.parent/'CROSS_GT_PLAN.md',CPU/'native_cached_match.py']+[CPU/'frozen_sources'/name for name in ('3_val.py','4_validator.py','7_metrics.py')]
        inputs=[a.acceptance,a.input/'summary.json',a.input/'paired_objects.jsonl.gz',a.input/'prediction_matches.jsonl.gz',a.input/'frames.jsonl.gz']
        write(a.output/'input_manifest.json',dict(inputs=[stat(p) for p in inputs],sources=[stat(p) for p in source],new_hash_computed=False))
        snap=a.output/'source_copies';snap.mkdir()
        for i,p in enumerate(source):shutil.copyfile(p,snap/(str(i)+'_'+p.name))
        md=['# 固定native .5匹配框的跨GT读出','', '固定1869教师own@.75优势对及950反向对照；全部坐标保留原native画布544×672，原图512×640。待独立验收。','', '|cohort/标签IoU|GT|T→RGB≥.5|T→RGB≥.75|T−N>0|>0.05|<0|<−0.05|', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,c in summary['cohorts'].items():
            for label,v in [('all',c['overall'])]+list(c['by_label_iou'].items()):
                md.append('|'+name+'/'+label+'|'+'|'.join(str(v[k]) for k in ('objects','T_on_RGB_ge050','T_on_RGB_ge075','delta_positive','delta_gt005','delta_negative','delta_lt_negative005'))+'|')
        md+=['', '原IoU.75的same/changed/unmatched预测ID计数单列于summary；主统计始终使用IoU.5原匹配框。逐对象保存双方GT/框/ID/own与cross IoU。未clip、未换最佳框、未重匹配。近似恒等坐标比较不是物理配准准入，也不是可蒸馏收益；不改变现有L门。']
        (a.output/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
        write(a.output/'completion.json',dict(status=summary['status'],scope=SCOPE,objects=len(values),seconds=summary['seconds'],new_GPU=False,new_hash_computed=False))
        print(json.dumps(summary['cohorts'],ensure_ascii=False))
    except BaseException as e:
        write(a.output/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    torch.set_num_threads(2);run(p.parse_args())
