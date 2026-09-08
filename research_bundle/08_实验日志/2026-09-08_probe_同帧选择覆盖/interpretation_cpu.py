"""Read existing same-forward JSON only; distinguish assignment from any-candidate gates."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-7)


def bucket(r):
    s,t=r['states']['S']['correct'],r['states']['T']['correct']
    return 'both_correct' if s and t else 'T_correct_S_error' if t else 'S_correct_T_error' if s else 'both_error'


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--accepted-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[json.loads(s) for s in (a.input/'objects.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    summary=read(a.accepted_summary);receipt=read(a.input/'completion_receipt.json')
    assert receipt['status']=='SAME_FORWARD_SELECTION_COVERAGE_COMPLETED' and len(rows)==80
    assert receipt['raw_forward_counts']==dict(student=1,teacher=1,reference=1)
    assert summary['augmented_rgb_gt_objects']==len(rows)
    max_iou_error=0.
    for r in rows:
        assert r['gates']['matched']
        for name,gt in (('S',r['rgb_gt_xyxy']),('R',r['rgb_gt_xyxy']),('T',r['ir_gt_xyxy'])):
            st=r['states'][name];pred=st['assigned']
            if pred is None:
                assert st['state']=='no_candidate';continue
            actual=iou(gt,pred['box']);max_iou_error=max(max_iou_error,abs(actual-st['iou']))
            assert abs(actual-st['iou'])<1e-6
            assert st['correct']==(pred['confidence']>=.25 and pred['class']==r['gt_class'] and actual>=.5)
    counts=Counter(bucket(r) for r in rows)
    for b,n in counts.items():assert summary['teacher_own_gt_primary'][b]['objects']==n
    details=[]
    for r in rows:
        b=bucket(r)
        if b not in ('T_correct_S_error','S_correct_T_error'):continue
        alternatives=[]
        # A strict subset of predictions: only teacher boxes already assigned to another GT and exported.
        # No witness here means unknown, never no correct dense candidate.
        own=r['states']['T']['assigned']
        for q in rows:
            pred=q['states']['T']['assigned']
            if q['frame_id']!=r['frame_id'] or pred is None:continue
            if own and pred['anchor_index']==own['anchor_index']:continue
            overlap=iou(r['ir_gt_xyxy'],pred['box'])
            if pred['class']==r['gt_class'] and pred['confidence']>=.25 and overlap>=.5:
                alternatives.append(dict(anchor_index=pred['anchor_index'],assigned_to_global_row=q['rgb_global_row'],
                    confidence=pred['confidence'],iou_to_this_ir_gt=overlap,box=pred['box']))
        g=r['gates'];s=r['states']['S'];t=r['states']['T']
        details.append(dict(bucket=b,global_row=r['rgb_global_row'],image_index=r['image_index'],
            stable_rgb_gt_id=r['stable_rgb_gt_id'],stable_ir_gt_id=r['stable_ir_gt_id'],
            rgb_gt_xyxy=r['rgb_gt_xyxy'],ir_gt_xyxy=r['ir_gt_xyxy'],S=s,T=t,gates=g,
            observed_saved_assigned_alternative_witnesses=alternatives,
            complete_candidate_witnesses_available=False,
            dense_correct_alternative_exists_by_original_gate=(g['teacher_correct_own'] and not t['correct']),
            highest_iou_per_object_not_proven=True))
    bybucket={}
    for b,n in counts.items():
        subset=[r for r in rows if bucket(r)==b]
        bybucket[b]=dict(objects=n,selected=sum(r['gates']['selected'] for r in subset),
            eligible=sum(r['gates']['eligible'] for r in subset),quality_positive=sum(r['gates']['quality_positive'] for r in subset),
            teacher_any_correct=sum(r['gates']['teacher_correct_own'] for r in subset),
            S_states=dict(Counter(r['states']['S']['state'] for r in subset)),T_states=dict(Counter(r['states']['T']['state'] for r in subset)),
            rows=[r['rgb_global_row'] for r in subset])
    selected=sorted((r for r in rows if r['gates']['selected']),key=lambda r:r['gates']['selected_rank'])
    out=dict(status='COMPLETED_CPU_ASSIGNMENT_GATE_INTERPRETATION',scope=receipt['scope'],
        frames=receipt['frames'],objects=len(rows),buckets=bybucket,
        teacher_any_correct=sum(r['gates']['teacher_correct_own'] for r in rows),
        teacher_gate_false_rows=[dict(row=r['rgb_global_row'],stable_id=r['stable_rgb_gt_id'],S=r['states']['S'],T=r['states']['T'],gates=r['gates']) for r in rows if not r['gates']['teacher_correct_own']],
        rgb_ir_augmented_gt_exact=sum(r['rgb_gt_xyxy']==r['ir_gt_xyxy'] for r in rows),
        S_R_states_exact=sum(r['states']['S']==r['states']['R'] for r in rows),
        S_R_scalar_evidence_exact=sum(r['gates']['student_evidence']==r['gates']['reference_evidence'] for r in rows),
        independent_iou_max_abs_error=max_iou_error,
        selected_last=dict(row=selected[-1]['rgb_global_row'],q=selected[-1]['gates']['quality_q']),
        assigned_alternative_witness_objects=sum(bool(r['observed_saved_assigned_alternative_witnesses']) for r in details),
        complete_raw_outputs_saved=False,full_candidate_ids_saved=False,details=details,
        source_files=[dict(path=str(path.absolute()),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns) for path in (a.input/'objects.jsonl',a.input/'completion_receipt.json',a.accepted_summary)],
        GPU_used=False,model_forward=False,training=False,new_hash_computed=False,
        AP_detection_error_claim=False,negative_transfer_claim=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k not in ('details','teacher_gate_false_rows','source_files')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
