"""Independent full-cache CPU identity/match/table recheck; no model or NMS."""
import __future__
import ast
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ENTRY=Path(__file__).resolve().parents[1]
CAMPAIGN=ENTRY.parent/'2026-09-07_probe_双数据集证据优先推进/llvip_full_eval'
OUTPUT=ENTRY/'output_attempt1'
SOURCE=CAMPAIGN/'remote_completed_attempt2/N42_full_attempt1/sources'
GROUPS={'all_iou50':(None,.5),'all_iou75':(None,.75),'gt025_iou50':(.25,.5),'gt025_iou75':(.25,.75)}


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt',encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def stat(p):s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def load_functions():
    env=dict(np=np,torch=torch)
    for file,name in [('7_metrics.py','box_iou'),('4_validator.py','match_predictions')]:
        tree=ast.parse((SOURCE/file).read_text(encoding='utf-8'))
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE/file),'exec',flags=__future__.annotations.compiler_flag),env)
    return env['box_iou'],env['match_predictions']


def main():
    started=time.perf_counter();torch.set_num_threads(2)
    originals={name:{r['image']:r for r in rows(CAMPAIGN/('remote_completed_attempt2/'+name+'42_full_attempt1/capture/objects.jsonl.gz'))} for name in ('N','T')}
    pairs=rows(CAMPAIGN/'verified_pair_manifest.jsonl');objs=rows(OUTPUT/'objects.jsonl.gz');pr=rows(OUTPUT/'prediction_matches.jsonl.gz')
    summary=read(OUTPUT/'summary.json');completion=read(OUTPUT/'completion.json')
    assert not (OUTPUT/'failure.json').exists() and completion['objects']==7879 and completion['images']==2406
    assert len(pairs)==len(originals['N'])==len(originals['T'])==2406 and len(objs)==7879 and len(pr)==4812
    bygt={(r['pair_key'],r['gt_row']):r for r in objs};bypred={(r['pair_key'],r['model']):r for r in pr}
    assert len(bygt)==7879 and len(bypred)==4812
    snapshots=list((OUTPUT/'source_copies').iterdir())
    for file in ('analyze_dev_cache.py','native_cached_match.py'):
        copy=next(p for p in snapshots if p.name.endswith('_'+file));assert copy.read_bytes()==(ENTRY/file).read_bytes()
    for file in ('3_val.py','4_validator.py','7_metrics.py'):
        copy=next(p for p in snapshots if p.name.endswith('_'+file));assert copy.read_bytes()==(SOURCE/file).read_bytes()
    iou_fn,match=load_functions();conditions={};metrics={g:{n:Counter() for n in ('N','T')} for g in GROUPS}
    gt_checked=0;calls=0;max_iou_abs=0.;pred_identities=0
    for pair in pairs:
        key=pair['pair_key'];n=originals['N'][pair['rgb_image']];t=originals['T'][pair['ir_image']]
        assert all(n[f]==t[f] for f in ('gt_boxes','gt_classes','canvas_shape','original_shape'))
        assert pair['gt_count']==len(n['gt_boxes']) and pair['canvas_shape']==n['canvas_shape'] and pair['original_shape']==n['original_shape']
        for gi,box in enumerate(n['gt_boxes']):
            r=bygt[key,gi];assert r['gt_box']==box and r['gt_class']==n['gt_classes'][gi]
            assert r['gt_id']==key+'::cached_gt:'+str(gi) and r['rgb_image']==pair['rgb_image'] and r['ir_image']==pair['ir_image']
            gt_checked+=1
        for model,row in [('N',n),('T',t)]:
            saved=bypred[key,model];assert saved['image']==row['image'] and len(saved['predictions'])==len(row['pred_boxes'])
            for pi,pred in enumerate(saved['predictions']):
                assert pred['prediction_id']==pi and pred['box']==row['pred_boxes'][pi]
                assert pred['class']==row['pred_classes'][pi] and pred['confidence']==row['pred_confidence'][pi]
                pred_identities+=1
            gtboxes=torch.tensor(row['gt_boxes'],dtype=torch.float32).reshape(-1,4);gtclass=torch.tensor(row['gt_classes'],dtype=torch.float32)
            confidence=torch.tensor(row['pred_confidence'],dtype=torch.float32)
            allboxes=torch.tensor(row['pred_boxes'],dtype=torch.float32).reshape(-1,4);allclass=torch.tensor(row['pred_classes'],dtype=torch.float32)
            for group,(cut,threshold) in GROUPS.items():
                keep=torch.arange(len(confidence)) if cut is None else torch.nonzero(confidence>cut,as_tuple=False).flatten()
                overlaps=iou_fn(gtboxes,allboxes[keep]);observed=[];prior=sys.getprofile()
                def trace(frame,event,arg):
                    if event=='return' and frame.f_code is match.__code__:observed.append(frame.f_locals['matches'].copy())
                try:
                    sys.setprofile(trace)
                    tp=match(SimpleNamespace(iouv=torch.tensor([threshold])),allclass[keep],gtclass,overlaps,use_scipy=False)
                finally:sys.setprofile(prior)
                assert len(observed)==1;calls+=1
                actual={int(g):int(p) for g,p in observed[0]};reverse={int(keep[p]):g for g,p in actual.items()}
                fullbits=torch.zeros(len(keep),dtype=torch.bool)
                for gi,pi in actual.items():fullbits[pi]=True
                assert torch.equal(tp[:,0],fullbits)
                included=set(keep.tolist())
                for pi,pred in enumerate(saved['predictions']):
                    state=pred['groups'][group];assert state['included']==(pi in included)
                    assert state['gt_row']==reverse.get(pi)
                    assert state['gt_id']==(None if pi not in reverse else key+'::cached_gt:'+str(reverse[pi]))
                for gi in range(len(gtboxes)):
                    state=bygt[key,gi]['groups'][group][model]
                    assert state['correct']==(gi in actual)
                    conditions[(key,gi,group,model)]=gi in actual
                    if gi not in actual:assert state['witness'] is None;continue
                    pi=actual[gi];w=state['witness'];assert w['prediction_id']==int(keep[pi]) and w['filtered_prediction_id']==pi
                    assert w['confidence']==float(confidence[keep[pi]])
                    error=abs(w['iou']-float(overlaps[gi,pi]));max_iou_abs=max(max_iou_abs,error);assert error==0
                metrics[group][model].update(tp=len(actual),fp=len(keep)-len(actual),fn=len(gtboxes)-len(actual),predictions=len(keep))
    tables={};images={}
    def bucket(n,t):return 'both_correct' if n and t else 'T_only' if t else 'N_only' if n else 'both_unmatched'
    for group in GROUPS:
        tally=Counter();covered={b:set() for b in ('both_correct','T_only','N_only','both_unmatched')}
        for key,gi in bygt:
            b=bucket(conditions[key,gi,group,'N'],conditions[key,gi,group,'T']);tally[b]+=1;covered[b].add(key)
            assert bygt[key,gi]['groups'][group]['bucket']==b
        tables[group]=dict(tally);images[group]={k:len(v) for k,v in covered.items()}
        assert {n:dict(c) for n,c in metrics[group].items()}==summary['groups'][group]['models']
        for b,v in summary['groups'][group]['buckets'].items():
            assert v['objects']==tally[b] and v['fraction']==tally[b]/7879 and v['images']==len(covered[b])
    joint=Counter();joint_images={}
    for key,gi in bygt:
        bits=tuple(conditions[key,gi,g,n] for n,g in [('N','gt025_iou50'),('N','gt025_iou75'),('T','gt025_iou50'),('T','gt025_iou75')])
        joint[bits]+=1;joint_images.setdefault(bits,set()).add(key)
    for r in summary['gt025_joint_iou50_iou75']:
        bits=tuple(r[k] for k in ('N_iou50','N_iou75','T_iou50','T_iou75'))
        assert r['objects']==joint[bits] and r['fraction']==joint[bits]/7879 and r['images']==len(joint_images.get(bits,set()))
    for suffix,output in summary['low_to_gt025_transitions'].items():
        transition=Counter();image_sets={}
        for key,gi in bygt:
            state=tuple(bucket(conditions[key,gi,prefix+'_'+suffix,'N'],conditions[key,gi,prefix+'_'+suffix,'T']) for prefix in ('all','gt025'))
            transition[state]+=1;image_sets.setdefault(state,set()).add(key)
        for r in output:
            state=(r['low_threshold_bucket'],r['gt025_bucket'])
            assert r['objects']==transition[state] and r['images']==len(image_sets.get(state,set()))
    result=dict(status='ACCEPTED_LOCAL_CPU_CACHED_REMATCH',images=2406,paired_GT=gt_checked,prediction_rows=pred_identities,
        independent_native_match_calls=calls,all_gt_and_original_prediction_ids_exact=True,all_tp_bits_exact_within_local_CPU=True,
        witness_iou_max_abs_difference=max_iou_abs,four_bucket_counts=tables,four_bucket_image_counts=images,
        all_joint_and_confidence_transition_cells_exact=True,source_copies_byte_exact=True,
        local_torch=str(torch.__version__),native_source='accepted Ultralytics8.4.115 function bodies',
        original_GPU_TP_bitwise_reproduction_claim=False,new_GPU=False,new_NMS=False,new_model_forward=False,
        checkpoint_loaded=False,new_hash_computed=False,seconds=time.perf_counter()-started,
        inputs=[stat(p) for p in (OUTPUT/'summary.json',OUTPUT/'objects.jsonl.gz',OUTPUT/'prediction_matches.jsonl.gz',ENTRY/'analyze_dev_cache.py',ENTRY/'native_cached_match.py')])
    with (ENTRY/'independent_review/ACTUAL_REVIEW_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('status','paired_GT','prediction_rows','independent_native_match_calls','witness_iou_max_abs_difference','seconds')}))


if __name__=='__main__':main()
