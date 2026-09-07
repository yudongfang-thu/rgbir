"""CPU-only diagnostic of frozen D2 records; never modifies experiment masks.

Counterfactual counts are post-hoc descriptions, not threshold selection or AP
evidence. Existing E/anchor identities and unverified-geometry status remain.
"""
import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '2026-09-07_probe_TaskConditional机会诊断'
GATES = ('r_reliable', 't_reliable', 'r_iou_cap', 't_rgb_iou', 't_own_iou', 'margin')


def gate_values(r, cap=.70, t_rgb=.60, margin=.05):
    return {
        'r_reliable': r['reference_correct'],
        't_reliable': r['teacher_correct'],
        'r_iou_cap': r['reference_iou'] < cap,
        't_rgb_iou': r['teacher_rgb_iou'] >= t_rgb,
        't_own_iou': r['teacher_ir_iou'] >= .50,
        'margin': r['teacher_rgb_iou'] - r['reference_iou'] > margin,
    }


def stats(rows, label, rgb_gt_count, variant, cap=.70, t_rgb=.60, margin=.05, omit=None):
    chosen = [r for r in rows if all(v for k, v in gate_values(r, cap, t_rgb, margin).items() if k != omit)]
    n = len(chosen)
    return dict(run=label, geometry_status='UNVERIFIED_DIAGNOSTIC', variant=variant,
                base_count=len(rows), rgb_gt_count=rgb_gt_count, selected_count=n,
                selected_pct_gt=100*n/rgb_gt_count, selected_pct_base=100*n/len(rows),
                teacher_dfl_ce_worse_count=sum(r['teacher_gt_dfl_ce'] > r['reference_gt_dfl_ce'] for r in chosen),
                teacher_dfl_ce_worse_pct=(100*sum(r['teacher_gt_dfl_ce'] > r['reference_gt_dfl_ce'] for r in chosen)/n) if n else None,
                dfl_ce_delta_mean=mean(r['teacher_gt_dfl_ce']-r['reference_gt_dfl_ce'] for r in chosen) if n else None,
                local_logit_cosine_negative_count=sum(r['reference_logit_kd_native_cosine'] is not None and r['reference_logit_kd_native_cosine'] < 0 for r in chosen))


def main():
    output, checks = [], []
    for label in ('drone_train', 'drone_val', 'llvip_train', 'llvip_val'):
        rows = [json.loads(x) for x in (SOURCE/label/'d2_anchors.jsonl').read_text(encoding='utf-8').splitlines() if x]
        summary = json.loads((SOURCE/label/'summary.json').read_text(encoding='utf-8'))
        assert summary['status'] == 'completed' and summary['geometry_verified'] is False
        rgb_gt_count = summary['totals']['rgb_gt_count']
        assert len(rows) == summary['totals']['base_count']
        for r in rows:
            assert all(gate_values(r).values()) == r['selected'] == r['quality_gate']
        checks.append(dict(run=label, base_records=len(rows), selected=sum(r['selected'] for r in rows),
                           exact_existing_selection_reconstructed=True))
        assert checks[-1]['selected'] == summary['totals']['selected_count']
        output.append(stats(rows, label, rgb_gt_count, 'frozen_v1'))
        output.extend(stats(rows, label, rgb_gt_count, 'omit_' + gate, omit=gate) for gate in GATES)
        output.extend(stats(rows, label, rgb_gt_count, name, **kwargs) for name, kwargs in (
            ('r_iou_cap_0.80', dict(cap=.80)),
            ('r_iou_cap_0.90', dict(cap=.90)),
            ('margin_0.00', dict(margin=0)),
            ('teacher_rgb_iou_0.50', dict(t_rgb=.50)),
        ))
    with (HERE/'gate_sensitivity.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    key_variants = [r for r in output if r['variant'] in ('frozen_v1', 'r_iou_cap_0.80', 'omit_r_iou_cap', 'margin_0.00')]
    with (HERE/'gate_sensitivity_key_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(key_variants)
    receipt = dict(status='completed_cpu_readonly', source=str(SOURCE), checks=checks,
                   caution='Post-existing-outcome exploratory eligibility sensitivity on existing unverified E/anchors. This is not an outcome-blind protocol or AP evidence. No training, no geometry waiver, no anchor reselection, no protocol modification.')
    (HERE/'gate_sensitivity_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(checks=checks, variants=[x for x in output if x['variant'] in ('frozen_v1','omit_r_iou_cap','r_iou_cap_0.80','margin_0.00')]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
