# coding: utf-8
"""CPU derivative algebra on saved initial-reference boxes; no parameter-gradient claim."""
import csv
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'raw_attempt1/screen_attempt1/calibration/llvip/calibration_batches.jsonl'


def relative(box, gt):
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    assert w > 0 and h > 0
    return [(box[0] - gt[0]) / w, (box[1] - gt[1]) / h,
            (box[2] - gt[0]) / w, (box[3] - gt[1]) / h]


def derivative(s, target):
    return [min(1., max(-1., (a - b) / .1)) for a, b in zip(s, target)]


def cosine(a, b):
    na, nb = sum(x*x for x in a), sum(x*x for x in b)
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(na*nb) if na > 0 and nb > 0 else None


def describe(values):
    v = [x for x in values if x is not None]
    return dict(n=len(v), mean=statistics.mean(v) if v else None,
                median=statistics.median(v) if v else None, min=min(v) if v else None, max=max(v) if v else None)


def main():
    rows, gt_target = [], [0., 0., 1., 1.]
    for batch in [json.loads(x) for x in SOURCE.read_text(encoding='utf-8').splitlines()]:
        stats = batch['stats']['L2-box']
        assert stats['config']['smooth_l1_beta'] == .1
        for record in stats['base_records']:
            if not record['selected']:
                continue
            s = relative(record['reference_box'], record['rgb_gt'])
            t = relative(record['mapped_teacher_box'], record['rgb_gt'])
            a, b = derivative(s, t), derivative(s, gt_target)
            row = dict(batch=batch['batch'], batch_index=record['batch_index'], rgb_gt_index=record['rgb_gt_index'],
                       image=batch['files'][record['batch_index']], base=stats['base_count'],
                       output_derivative_cosine=cosine(a, b),
                       teacher_target_vs_GT_normalized_mean_L1=sum(abs(x-y) for x, y in zip(t, gt_target))/4,
                       teacher_target_vs_GT_normalized_max_abs=max(abs(x-y) for x, y in zip(t, gt_target)),
                       relative_reference=s, relative_teacher=t, derivative_teacher_target=a, derivative_GT_target=b,
                       exact_derivative_equal=a == b, exact_target_equal=t == gt_target)
            rows.append(row)
    assert len(rows) == 30
    stack_a = [v for r in rows for v in r['derivative_teacher_target']]
    stack_b = [v for r in rows for v in r['derivative_GT_target']]
    weighted_a = [32 / (4 * r['base']) * v for r in rows for v in r['derivative_teacher_target']]
    weighted_b = [32 / (4 * r['base']) * v for r in rows for v in r['derivative_GT_target']]
    summary = dict(status='CPU_INITIAL_REFERENCE_OUTPUT_DERIVATIVE_ONLY', selected_objects=len(rows), beta=.1,
                   derivative_formula='clip((s-target)/0.1,-1,1), derivative wrt GT-normalized reference xyxy',
                   s_source='Saved frozen RGB reference box, not an autograd read of student parameters',
                   unweighted_stacked_derivative_cosine=cosine(stack_a, stack_b),
                   batch_loss_scaled_stacked_derivative_cosine=cosine(weighted_a, weighted_b),
                   batch_loss_scale='B=32 * lambda=1 / (4 * batch base_count); excludes pixel-coordinate Jacobian and model Jacobian',
                   per_object_cosine=describe([r['output_derivative_cosine'] for r in rows]),
                   target_vs_GT_normalized_mean_L1=describe([r['teacher_target_vs_GT_normalized_mean_L1'] for r in rows]),
                   target_vs_GT_normalized_max_abs=describe([r['teacher_target_vs_GT_normalized_max_abs'] for r in rows]),
                   exact_derivative_equal_objects=sum(r['exact_derivative_equal'] for r in rows),
                   exact_target_equal_objects=sum(r['exact_target_equal'] for r in rows),
                   parameter_gradient_cosine_measured=False, student_output_exact_to_reference_claim=False,
                   dataset_initial_checkpoint_equals_reference_identity=True, initial_BN_buffers_frozen=True,
                   new_GPU_or_AP=False, new_hash_computed=False, source=str(SOURCE))
    with (HERE / 'OUTPUT_COORDINATE_SUMMARY.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (HERE / 'OUTPUT_COORDINATE_OBJECTS.json').open('x', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, allow_nan=False)
    fields = ['batch', 'batch_index', 'rgb_gt_index', 'base', 'output_derivative_cosine',
              'teacher_target_vs_GT_normalized_mean_L1', 'teacher_target_vs_GT_normalized_max_abs',
              'exact_derivative_equal', 'exact_target_equal', 'image']
    with (HERE / 'OUTPUT_COORDINATE_TABLE.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    # Exact known-truth formula checks, independent of observed values.
    assert derivative([0.,0.,1.,1.], gt_target) == [0.]*4
    assert derivative([.2,-.2,1.05,.95], gt_target)[:2] == [1.,-1.]
    assert cosine([1.,0.],[0.,1.]) == 0. and cosine([1.,2.],[2.,4.]) == 1.
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
