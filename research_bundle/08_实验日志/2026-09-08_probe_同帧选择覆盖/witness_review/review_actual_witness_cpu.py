"""Bounded independent CPU recheck of collected native GT witnesses and counts."""
import __future__
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ENTRY=Path(__file__).resolve().parents[1]
ROOT=ENTRY.parents[1]
PROBE=ENTRY/'witness_evidence_1315_final/probe'
ANALYSIS=ENTRY/'witness_analysis_final'
NATIVE=ROOT/'08_实验日志/2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/N42_full_attempt1/sources'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def stat(p):s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def native_functions():
    env=dict(np=np,torch=torch)
    for file,name in [('7_metrics.py','box_iou'),('4_validator.py','match_predictions')]:
        tree=ast.parse((NATIVE/file).read_text(encoding='utf-8'))
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(NATIVE/file),'exec',flags=__future__.annotations.compiler_flag),env)
    return env['box_iou'],env['match_predictions']


def run():
    assert (ENTRY/'witness_analyzer.py').read_bytes()==(ANALYSIS/'witness_analyzer_source.py').read_bytes()
    current=rows(PROBE/'witness_objects.jsonl');ir=rows(PROBE/'witness_ir_objects.jsonl');frames=rows(PROBE/'witness_frames.jsonl')
    previous=rows(ENTRY/'evidence_1255_final/probe/objects.jsonl')
    assert rows(PROBE/'objects.jsonl')==previous
    assert [{k:v for k,v in r.items() if k!='detector'} for r in current]==previous
    receipt=read(PROBE/'completion_receipt.json');contract=read(PROBE/'witness_contract.json');published=read(ANALYSIS/'summary.json')
    spec=importlib.util.spec_from_file_location('reviewed_witness_analyzer',ENTRY/'witness_analyzer.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    recomputed=analyzer.summarize(previous,current,receipt,contract)
    assert {k:v for k,v in published.items() if k!='input_sources'}==recomputed
    box_iou,match=native_functions();max_iou_abs=0.;match_counts=Counter();detections=Counter();frames_checked=0
    for frame in frames:
        fid=frame['frame_id']
        for model in ('S','R','T'):
            source=ir if model=='T' else current
            gr=[r for r in source if r['frame_id']==fid]
            gr.sort(key=lambda r:r['ir_global_row' if model=='T' else 'rgb_global_row'])
            gb=torch.tensor([r['ir_gt_xyxy' if model=='T' else 'rgb_gt_xyxy'] for r in gr],dtype=torch.float32).reshape(-1,4)
            gc=torch.tensor([r['gt_class'] for r in gr])
            dr=frame['native_detections'][model]
            assert [r['prediction_index'] for r in dr]==list(range(len(dr)))
            pb=torch.tensor([r['box'] for r in dr],dtype=torch.float32).reshape(-1,4)
            pc=torch.tensor([r['class'] for r in dr])
            assert all(r['confidence']>.25 and r['class']==0 for r in dr) and len(dr)<=300
            overlap=box_iou(gb,pb);captured=[];old=sys.getprofile()
            def profile(f,event,arg):
                if event=='return' and f.f_code is match.__code__:captured.append(f.f_locals['matches'].copy())
            try:
                sys.setprofile(profile);tp=match(SimpleNamespace(iouv=torch.tensor([.5])),pc,gc,overlap,use_scipy=False)
            finally:sys.setprofile(old)
            pairs=captured[0];actual={int(g):int(p) for g,p in pairs}
            assert tp[:,0].tolist()==[r['native_tp_iou50'] for r in dr]
            for j,r in enumerate(gr):
                d=r['detector_T'] if model=='T' else r['detector'][model]
                assert d['native_postnms_matched']==(j in actual)
                if j in actual:
                    k=actual[j];w=d['native_witness']
                    assert w['prediction_index']==k
                    assert {key:value for key,value in w.items() if key!='iou'}==dr[k]
                    error=abs(w['iou']-float(overlap[j,k]));max_iou_abs=max(max_iou_abs,error)
                    assert error<=1e-6
            match_counts[model]+=len(pairs);detections[model]+=len(dr);frames_checked+=1
    names=('old_assigned','dense_any_correct','native_pre_nms_any_correct','native_postnms_matched')
    counts={};changes={};buckets={}
    for model in ('S','R','T'):
        states={name:[r['states'][model]['correct'] if name=='old_assigned' else r['detector'][model][name] for r in current] for name in names}
        counts[model]={name:sum(v) for name,v in states.items()}
        assert counts[model]==published['modalities'][model]['correct_counts']
        changes[model]=dict(fp32_dense_vs_native_pre=sum(a!=b for a,b in zip(states[names[1]],states[names[2]])),
            native_pre_vs_post=sum(a!=b for a,b in zip(states[names[2]],states[names[3]])),
            fp32_vs_native_dense_count_changed=sum(r['detector'][model]['dense_correct_count']!=r['detector'][model]['native_pre_nms_correct_count'] for r in current),
            old_error_to_dense_correct=sum(not a and b for a,b in zip(states[names[0]],states[names[1]])))
    for definition in names:
        tally=Counter();selected=Counter()
        for r in current:
            s=r['states']['S']['correct'] if definition=='old_assigned' else r['detector']['S'][definition]
            t=r['states']['T']['correct'] if definition=='old_assigned' else r['detector']['T'][definition]
            key=('both_correct' if s and t else 'T_correct_S_error' if t else 'S_correct_T_error' if s else 'both_error')
            tally[key]+=1
            if r['gates']['selected']:selected[key]+=1
        buckets[definition]={key:dict(objects=tally[key],selected=selected[key]) for key in published['coverage_by_definition'][definition]}
        assert sum(tally.values())==80 and sum(selected.values())==32
        for key,v in buckets[definition].items():
            assert all(published['coverage_by_definition'][definition][key][field]==n for field,n in v.items())
    return dict(status='ACCEPTED',scope='fixed_same_batch_descriptive_witness_readout',frames=32,objects_n=80,
        native_role_frames_recomputed=frames_checked,native_match_counts=dict(match_counts),native_detections=dict(detections),
        native_gt_pairs_and_full_tp_bits_exact=True,native_witness_iou_max_abs_cpu_difference=max_iou_abs,
        exact_previous_objects_and_analyzer_source=True,summary_recomputation_exact=True,
        independent_correct_counts=counts,definition_changes=changes,independent_buckets=buckets,
        recorded_decode_diagnostics=contract['decode_diagnostics'],
        raw_dense_false_negatives_not_rederived_without_full_raw=True,GPU_used=False,new_hash_computed=False,
        inputs=[stat(p) for p in [PROBE/'witness_objects.jsonl',PROBE/'witness_ir_objects.jsonl',PROBE/'witness_frames.jsonl',
            PROBE/'witness_contract.json',ANALYSIS/'summary.json',ENTRY/'witness_analyzer.py']])


if __name__=='__main__':
    torch.set_num_threads(1);result=run()
    with (ENTRY/'witness_review/ACTUAL_WITNESS_REVIEW_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:result[k] for k in ('status','native_role_frames_recomputed','native_match_counts','native_detections','native_witness_iou_max_abs_cpu_difference','definition_changes')},ensure_ascii=True))
