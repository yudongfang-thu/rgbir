"""Independent full saved-cache matching identities and paired/unpaired tables."""
import __future__,ast,gzip,json,sys,time
from collections import Counter
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
E=Path(__file__).resolve().parents[1]
A=E/'cpu_analysis';O=A/'output_attempt2';F=A/'frozen_sources'
N=E.parent/'2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/N_s42_attempt1'
T=E/'evidence_1401/T42_full_attempt1'
GROUPS={'all_iou50':(None,.5),'all_iou75':(None,.75),'gt025_iou50':(.25,.5),'gt025_iou75':(.25,.75)}
B=('both_correct','T_only','N_only','both_unmatched')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def buck(n,t):return 'both_correct' if n and t else 'T_only' if t else 'N_only' if n else 'both_unmatched'
def gid(image,i):return image+'::cached_gt:'+str(i)
def main():
    start=time.perf_counter();torch.set_num_threads(2)
    env=dict(np=np,torch=torch,Tensor=torch.Tensor,linear_sum_assignment=linear_sum_assignment)
    for fn,name in [('7_metrics.py','box_iou'),('4_validator.py','match_predictions'),('object_evidence_loss.py','_iou'),('object_evidence_loss.py','_match_objects')]:
        node=next(n for n in ast.walk(ast.parse((F/fn).read_text(encoding='utf-8'))) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(F/fn),'exec',flags=__future__.annotations.compiler_flag),env)
    match=env['match_predictions'];iou_fn=env['box_iou'];pair_fn=env['_match_objects']
    models={'N':{r['image']:r for r in rows(N/'predictions/objects.jsonl.gz')},'T':{r['image']:r for r in rows(T/'capture/objects.jsonl.gz')}}
    alias={'N':{p:p for p in models['N']},'T':read(T/'capture_contract.json')['alias_to_canonical']}
    mapping=read(O/'image_mapping.json');frames=rows(O/'frames.jsonl.gz');paired=rows(O/'paired_objects.jsonl.gz');allgt=rows(O/'all_gt_objects.jsonl.gz');pred=rows(O/'prediction_matches.jsonl.gz');s=read(O/'summary.json')
    assert read(O/'completion.json')['status']=='COMPLETED_PENDING_INDEPENDENT_REVIEW' and not (O/'failure.json').exists()
    assert len(frames)==len(mapping)==1469 and len({r['rgb_image'] for r in frames})==1469 and len({r['ir_image'] for r in frames})==1469
    assert [r['rgb_image'] for r in frames]==list(models['N'])
    canonical_t={PurePosixPath(p).stem:p for p in alias['T'].values()}
    assert len(canonical_t)==1469
    assert mapping=={p:canonical_t[PurePosixPath(p).stem] for p in models['N']}
    pg={(r['model'],r['image'],r['gt_row']):r for r in allgt};pp={(r['model'],r['image']):r for r in pred};po={(r['pair_key'],r['N_gt_row']):r for r in paired}
    assert len(pg)==len(allgt)==46952 and len(pp)==len(pred)==2938 and len(po)==len(paired)==21568
    conditions={};metrics={g:{n:Counter(tp=0,fp=0,fn=0,predictions=0) for n in ('N','T')} for g in GROUPS};calls=0;pred_n=0;maxerr=0.;pair_n=0
    for frame in frames:
        rr={n:models[n][frame['rgb_image' if n=='N' else 'ir_image']] for n in ('N','T')}
        assert mapping[frame['rgb_canonical']]==frame['ir_canonical'] and alias['T'][frame['ir_image']]==frame['ir_canonical'] and frame['pair_key']==frame['rgb_canonical']
        assert all(rr[n]['canvas_shape']==frame['canvas_shape'] and rr[n]['original_shape']==frame['original_shape'] for n in ('N','T'))
        nb=torch.tensor(rr['N']['gt_boxes'],dtype=torch.float32).reshape(-1,4);tb=torch.tensor(rr['T']['gt_boxes'],dtype=torch.float32).reshape(-1,4)
        nc=torch.tensor(rr['N']['gt_classes']);tc=torch.tensor(rr['T']['gt_classes']);ni,ti=pair_fn(nb,nc,tb,tc,.5);overlap=env['_iou'](nb,tb)
        found=[dict(N_gt_row=int(i),T_gt_row=int(j),iou=float(overlap[i,j])) for i,j in zip(ni,ti)]
        assert found==frame['GT_pairs'] and frame['paired_GT']==len(found) and frame['N_GT']==len(nb) and frame['T_GT']==len(tb)
        pair_n+=len(found);partners={'N':dict(zip(ni.tolist(),ti.tolist())),'T':dict(zip(ti.tolist(),ni.tolist()))}
        for pair in found:
            p=po[frame['pair_key'],pair['N_gt_row']];i,j=pair['N_gt_row'],pair['T_gt_row']
            assert p['T_gt_row']==j and p['N_gt_box']==rr['N']['gt_boxes'][i] and p['T_gt_box']==rr['T']['gt_boxes'][j]
            assert p['N_gt_id']==gid(rr['N']['image'],i) and p['T_gt_id']==gid(rr['T']['image'],j)
            assert p['gt_class']==int(nc[i])==int(tc[j]) and p['pair_iou']==pair['iou']
        for n in ('N','T'):
            row=rr[n];other='T' if n=='N' else 'N';save=pp[n,row['image']];assert len(save['predictions'])==len(row['pred_boxes']);assert save['pair_key']==frame['pair_key']
            gt=torch.tensor(row['gt_boxes'],dtype=torch.float32).reshape(-1,4);gc=torch.tensor(row['gt_classes']);boxes=torch.tensor(row['pred_boxes'],dtype=torch.float32).reshape(-1,4);pc=torch.tensor(row['pred_classes']);conf=torch.tensor(row['pred_confidence'],dtype=torch.float32)
            for pi,p in enumerate(save['predictions']):
                assert p['prediction_id']==pi and p['box']==row['pred_boxes'][pi] and p['pred_class']==row['pred_classes'][pi] and p['confidence']==row['pred_confidence'][pi];pred_n+=1
            for gi,box in enumerate(row['gt_boxes']):
                r=pg[n,row['image'],gi];assert r['gt_id']==gid(row['image'],gi) and r['gt_box']==box and r['gt_class']==row['gt_classes'][gi]
                partner=partners[n].get(gi);assert r['partner_gt_id']==(None if partner is None else gid(rr[other]['image'],partner))
            for g,(cut,threshold) in GROUPS.items():
                keep=torch.arange(len(conf)) if cut is None else torch.nonzero(conf>cut,as_tuple=False).flatten();ov=iou_fn(gt,boxes[keep]);trace_out=[];prior=sys.getprofile()
                def trace(f,event,arg):
                    if event=='return' and f.f_code is match.__code__:trace_out.append(f.f_locals['matches'].copy())
                try:
                    sys.setprofile(trace);bits=match(SimpleNamespace(iouv=torch.tensor([threshold])),pc[keep],gc,ov,use_scipy=False)
                finally:sys.setprofile(prior)
                assert len(trace_out)==1;calls+=1;matched={int(i):int(j) for i,j in trace_out[0]};reverse={int(keep[j]):i for i,j in matched.items()}
                assert set(np.nonzero(bits[:,0].numpy())[0])==set(matched.values())
                included=set(keep.tolist())
                for pi,p in enumerate(save['predictions']):
                    state=p['groups'][g];assert state['included']==(pi in included) and state['gt_row']==reverse.get(pi)
                    assert state['gt_id']==(gid(row['image'],reverse[pi]) if pi in reverse else None)
                for gi in range(len(gt)):
                    state=pg[n,row['image'],gi]['groups'][g];assert state['correct']==(gi in matched)
                    conditions[n,row['image'],gi,g]=gi in matched
                    if gi not in matched:assert state['witness'] is None;continue
                    j=matched[gi];w=state['witness'];assert w['prediction_id']==int(keep[j]) and w['filtered_prediction_id']==j and w['confidence']==float(conf[keep[j]])
                    err=abs(w['iou']-float(ov[gi,j]));maxerr=max(maxerr,err);assert err==0
                metrics[g][n].update(tp=len(matched),fp=len(keep)-len(matched),fn=len(gt)-len(matched),predictions=len(keep))
    assert pair_n==21568
    for r in allgt:
        for g,st in r['groups'].items():
            if r['partner_gt_id'] is None:assert st['partner_correct'] is None
            else:
                other='T' if r['model']=='N' else 'N';image,i=r['partner_gt_id'].rsplit('::cached_gt:',1)
                assert st['partner_correct']==conditions[other,image,int(i),g]
    for r in paired:
        ni_image,ni=r['N_gt_id'].rsplit('::cached_gt:',1);ti_image,ti=r['T_gt_id'].rsplit('::cached_gt:',1)
        for g,st in r['groups'].items():
            for n,im,i in [('N',ni_image,int(ni)),('T',ti_image,int(ti))]:
                own=pg[n,im,i]['groups'][g];assert st[n]==dict(correct=own['correct'],witness=own['witness'])
            assert st['bucket']==buck(st['N']['correct'],st['T']['correct'])
    def check_buckets(rr,g,v):
        for b in B:
            selected=[r for r in rr if r['groups'][g]['bucket']==b];x=v[b]
            assert x==dict(objects=len(selected),fraction=len(selected)/len(rr) if rr else None,images=len({r['pair_key'] for r in selected}))
    unpaired={n:[r for r in allgt if r['model']==n and r['partner_gt_id'] is None] for n in ('N','T')}
    assert s['unpaired_GT']=={n:len(rr) for n,rr in unpaired.items()}=={'N':894,'T':2922}
    for g,v in s['groups'].items():
        check_buckets(paired,g,v['buckets']);assert v['whole_model_metrics']=={n:dict(x) for n,x in metrics[g].items()}
        for n,rr in unpaired.items():assert v['unpaired'][n]==dict(objects=len(rr),images=len({r['image'] for r in rr}),correct=sum(r['groups'][g]['correct'] for r in rr),unmatched=sum(not r['groups'][g]['correct'] for r in rr))
        for cl in v['per_class']:
            sub=[r for r in paired if r['gt_class']==cl['class_id']];assert cl['paired_GT']==len(sub);check_buckets(sub,g,cl['buckets'])
            for n,rr in unpaired.items():assert cl['unpaired'][n]==dict(GT=sum(r['gt_class']==cl['class_id'] for r in rr),correct=sum(r['gt_class']==cl['class_id'] and r['groups'][g]['correct'] for r in rr))
        assert sum(cl['paired_GT'] for cl in v['per_class'])==len(paired)
    for suffix,table in s['low_to_gt025_transitions'].items():
        c=Counter((r['groups']['all_'+suffix]['bucket'],r['groups']['gt025_'+suffix]['bucket']) for r in paired)
        for cell in table:assert cell['objects']==c[cell['low_threshold_bucket'],cell['gt025_bucket']]
    bitkeys=('N_iou50','N_iou75','T_iou50','T_iou75')
    def joint(r):return tuple(r['groups']['gt025_iou'+th][n]['correct'] for n,th in [('N','50'),('N','75'),('T','50'),('T','75')])
    for cell in s['gt025_joint_iou50_iou75']:
        subset=[r for r in paired if joint(r)==tuple(cell[k] for k in bitkeys)]
        assert cell['objects']==len(subset) and cell['images']==len({r['pair_key'] for r in subset})
    for file in ('analyze_drone_dev.py','pairing_core.py','native_cached_match.py'):
        copy=next(p for p in (O/'source_copies').iterdir() if p.name.endswith('_'+file));assert copy.read_bytes()==(A/file).read_bytes()
    assert not torch.cuda.is_initialized()
    result=dict(status='ACCEPTED_DRONE_PAIRED_CPU_TABLES',images=1469,own_GT_checked=len(allgt),paired_GT=pair_n,unpaired_GT=s['unpaired_GT'],native_calls=calls,prediction_identity_rows=pred_n,all_native_pairs_TP_and_original_prediction_IDs_exact=True,max_witness_IoU_difference=maxerr,all_GT_pair_partner_links_exact=True,all_four_bucket_and_five_class_tables_exact=True,all_transitions_and_16_joint_cells_exact=True,source_snapshots_exact=True,torch_local=str(torch.__version__),native_GPU_bitwise_reproduction=False,GT_pairing_is_physical_truth=False,new_GPU=False,new_hash=False,new_NMS=False,checkpoint_loaded=False,seconds=time.perf_counter()-start)
    with (E/'independent_review/DRONE_TABLES_ACTUAL_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
