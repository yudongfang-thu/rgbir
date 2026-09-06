"""Independent CPU audit of saved candidate flags; no model inference."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-10)


out = {}
for ds in ['dronevehicle', 'llvip', 'vedai']:
    d = ROOT / (ds + '_full')
    objects = json.loads((d / 'matched_objects.json').read_text())
    predictions = {r['id']: r for r in json.loads((d / 'prediction_records.json').read_text())}
    teacher, student = ('rgb', 'ir') if ds == 'vedai' else ('ir', 'rgb')
    both = [o for o in objects if o[teacher + '_hit'] and o[student + '_hit']]
    delta = [o[teacher + '_pred_iou'] - o[student + '_pred_iou'] for o in both]
    rows = []
    for o in objects:
        if not o[teacher + '_hit'] or o[student + '_hit']:
            continue
        p = predictions[o['id']]
        gt = p['gt_' + student][o[student + '_gt']]
        candidates = p['pred_' + student]
        low = any(q[5] == gt[0] and .05 <= q[4] < .25 and iou(gt[1:], q[:4]) >= .5 for q in candidates)
        high = any(q[5] == gt[0] and q[4] >= .25 and iou(gt[1:], q[:4]) >= .5 for q in candidates)
        wrong = any(q[5] != gt[0] and q[4] >= .25 and iou(gt[1:], q[:4]) >= .5 for q in candidates)
        coarse = any(q[4] >= .05 and iou(gt[1:], q[:4]) >= .1 for q in candidates)
        rows.append({'id': o['id'], 'student_gt': o[student + '_gt'], 'low_correct': low,
                     'high_correct_despite_one_to_one_miss': high, 'wrong_class_at_conf25': wrong,
                     'any_coarse': coarse})
    out[ds] = {
        'n_teacher_only': len(rows), 'both_hit_n': len(delta),
        'both_hit_loc_mean': statistics.mean(delta), 'both_hit_loc_median': statistics.median(delta),
        'both_hit_teacher_better_n': sum(x > 0 for x in delta),
        'candidate_same_class_iou50_conf_ge05': sum(r['low_correct'] or r['high_correct_despite_one_to_one_miss'] for r in rows),
        'candidate_strict_low_correct_no_high_correct': sum(r['low_correct'] and not r['high_correct_despite_one_to_one_miss'] for r in rows),
        'candidate_high_correct_but_one_to_one_miss': sum(r['high_correct_despite_one_to_one_miss'] for r in rows),
        'candidate_high_wrong_class': sum(r['wrong_class_at_conf25'] for r in rows),
        'low_correct_and_high_wrong_overlap': sum(r['low_correct'] and r['wrong_class_at_conf25'] for r in rows),
        'no_any_class_coarse_candidate_conf_ge05_iou_ge10': sum(not r['any_coarse'] for r in rows),
        'rows': rows,
        'scope': 'post-NMS saved proposals at conf>=.05; per-GT flags, may overlap or reuse one proposal; not attainable KD gain',
    }
(ROOT / 'candidate_flag_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for ds, r in out.items():
    print(ds, json.dumps({k: v for k, v in r.items() if k != 'rows'}))
