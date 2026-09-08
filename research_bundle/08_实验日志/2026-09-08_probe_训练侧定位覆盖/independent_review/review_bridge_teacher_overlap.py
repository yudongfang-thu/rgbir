import json
from pathlib import Path
ENTRY=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
w={r['rgb_global_row']:r for r in rows(ENTRY.parent/'2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/witness_objects.jsonl')}
selected=[r for r in rows(ENTRY/'bridge/output_attempt1/objects.jsonl') if r['historical_L2_gates']['selected']]
matches=[]
for row in selected:
    record=row['historical_L2_record']; a=w[row['rgb_global_row']]['states']['T']['assigned']
    if a is not None and a['anchor_index']==record['teacher_anchor']:
        fields=dict(box=a['box']==record['teacher_box'],confidence=a['confidence']==record['teacher_conf'],own_gt_iou=a['own_gt_iou']==record['teacher_own_iou'],class_id=a['class']==record['class_id'])
        assert all(fields.values())
        matches.append(dict(stable_rgb_gt_id=row['stable_rgb_gt_id'],anchor=record['teacher_anchor'],exact_fields=fields))
assert len(selected)==7 and len(matches)==5
result=dict(status='PASS_PARTIAL_TEACHER_OUTPUT_OVERLAP',historical_selected=7,same_assigned_anchor=5,all_four_fields_exact=5,rows=matches,historical_teacher_checkpoint_stat_verified=False,full_raw_identity_claim=False,new_GPU=False,new_hash=False)
with (ENTRY/'independent_review/BRIDGE_TEACHER_OVERLAP.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
