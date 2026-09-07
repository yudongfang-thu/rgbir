"""Read-only payload inventory; no geometry acceptance, AP or new GPU work."""
import gzip
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'2026-09-07_probe_TaskConditional机会诊断'
rows={}
for name in ('drone_train','llvip_train'):
    path=BASE/name/'d2_anchors.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        selected=[r for r in map(json.loads,stream) if r.get('selected')]
    rows[name]=dict(source=str(path),selected_rows=len(selected),
        observed_fields=sorted({key for row in selected for key in row}),
        payload_scope='Selected rows from old static unverified-geometry diagnostic, not current natural-augmentation L1 admission',
        complete_raw_teacher_dfl_present=False,complete_raw_reference_dfl_present=False,
        decoded_teacher_boxes_present=False,decoded_reference_boxes_present=False,
        finding='Observed fields contain GT distances and scalar output/gradient summaries; cannot reconstruct complete prediction distributions or decoded boxes from them')
with (HERE/'local_payload_inventory.json').open('x',encoding='utf-8') as stream:
    json.dump(rows,stream,ensure_ascii=False,indent=2)
print(json.dumps({key:value['selected_rows'] for key,value in rows.items()}))
