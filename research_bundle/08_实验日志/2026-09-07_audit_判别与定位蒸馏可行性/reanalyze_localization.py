"""CPU-only descriptive localization audit of frozen 2026-09-06 probe outputs.

Before computing results, fix confidence >= .25 (.5 sensitivity), protected
same-class one-to-one RGB matches at IoU >= .5 then residual matches >= .3,
RGB bands [.3,.5), [.5,.75), teacher own-GT IoU >= .5 / .75, and gain >= .1.
These exploratory cutoffs are not a trained gate or outcome-blind train protocol.
No images, checkpoints, GPU, additional inference, or original files are changed.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0}
    return {'n': len(a), 'mean': float(a.mean()),
            **{'p' + str(q): float(np.percentile(a, q)) for q in (0, 10, 25, 50, 75, 90, 95, 99, 100)},
            'positive_n': int((a > 1e-9).sum()), 'negative_n': int((a < -1e-9).sum()),
            'zero_n': int((np.abs(a) <= 1e-9).sum())}


def iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(-1)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-10)


def assign(score, threshold):
    if not score.size:
        return {}
    eligible = score >= threshold
    ii, jj = linear_sum_assignment(-(eligible * (min(score.shape) + 1 + score)))
    return {int(i): int(j) for i, j in zip(ii, jj) if eligible[i, j]}


def predictions(gt, pred, confidence=.25, allow_rough=False):
    """Preserve IoU .5 TP assignment, then allocate unused predictions to rough GT."""
    ids = np.flatnonzero(pred[:, 4] >= confidence)
    pp = pred[ids]
    score = iou(gt[:, 1:], pp[:, :4]) * (gt[:, None, 0] == pp[None, :, 5])
    matched = assign(score, .5)
    if allow_rough:
        gi = np.asarray([i for i in range(len(gt)) if i not in matched], dtype=int)
        used = set(matched.values())
        pi = np.asarray([j for j in range(len(pp)) if j not in used], dtype=int)
        residual = assign(score[np.ix_(gi, pi)], .3)
        matched.update({int(gi[i]): int(pi[j]) for i, j in residual.items()})
    assert len(matched.values()) == len(set(matched.values()))
    return {i: (int(ids[j]), float(score[i, j])) for i, j in matched.items()}


def naive_reuse(gt, pred, threshold):
    pp = pred[pred[:, 4] >= .25]
    score = iou(gt[:, 1:], pp[:, :4]) * (gt[:, None, 0] == pp[None, :, 5])
    if not score.size:
        return {'qualifying_gt_n': 0, 'gt_in_reused_prediction_groups_n': 0, 'extra_assignments_n': 0}
    best = score.argmax(1)
    valid = score.max(1) >= threshold
    _, count = np.unique(best[valid], return_counts=True)
    return {'qualifying_gt_n': int(valid.sum()),
            'gt_in_reused_prediction_groups_n': int(count[count > 1].sum()),
            'extra_assignments_n': int((count[count > 1] - 1).sum())}


def count_by(rows, key):
    return sum(bool(r[key]) for r in rows)


def direct_stats(rows):
    valid = [r for r in rows if r['rgb_candidate_iou'] is not None and r['ir_own_iou'] is not None]
    return {'n': len(valid),
            'ir_own_minus_rgb_own': stats([r['ir_own_iou'] - r['rgb_candidate_iou'] for r in valid]),
            'ir_box_in_rgb_minus_rgb_own': stats([r['ir_direct_rgb_iou'] - r['rgb_candidate_iou'] for r in valid]),
            'ir_box_in_rgb_minus_ir_own': stats([r['ir_direct_rgb_iou'] - r['ir_own_iou'] for r in valid]),
            'direct_improves_ge_01_n': sum(r['ir_direct_rgb_iou'] - r['rgb_candidate_iou'] >= .1 for r in valid),
            'direct_harms_ge_01_n': sum(r['rgb_candidate_iou'] - r['ir_direct_rgb_iou'] >= .1 for r in valid),
            'direct_harms_any_n': sum(r['ir_direct_rgb_iou'] < r['rgb_candidate_iou'] - 1e-9 for r in valid)}


def analyze(source, name):
    records = read(source / (name + '_full') / 'prediction_records.json')
    prior = read(source / (name + '_full') / 'matched_objects.json')
    old = {(r['id'], r['rgb_gt'], r['ir_gt']): r for r in prior}
    rows = []
    conf50_rows = []
    annotation = {'rgb_gt_n': 0, 'ir_gt_n': 0, 'matched_n': 0}
    reuse = {side + '_iou' + str(th): {'qualifying_gt_n': 0, 'gt_in_reused_prediction_groups_n': 0,
                                   'extra_assignments_n': 0}
             for side in ('rgb', 'ir') for th in (.3, .5)}
    for rec in records:
        gr = np.asarray(rec['gt_rgb'], dtype=float).reshape(-1, 5)
        gt = np.asarray(rec['gt_ir'], dtype=float).reshape(-1, 5)
        pr = np.asarray(rec['pred_rgb'], dtype=float).reshape(-1, 6)
        pt = np.asarray(rec['pred_ir'], dtype=float).reshape(-1, 6)
        annotation['rgb_gt_n'] += len(gr)
        annotation['ir_gt_n'] += len(gt)
        gs = iou(gr[:, 1:], gt[:, 1:]) * (gr[:, None, 0] == gt[None, :, 0])
        pairs = assign(gs, .1)
        annotation['matched_n'] += len(pairs)
        hr, ht = predictions(gr, pr), predictions(gt, pt)
        cr = predictions(gr, pr, allow_rough=True)
        cr50 = predictions(gr, pr, confidence=.5, allow_rough=True)
        ct50 = predictions(gt, pt, confidence=.5)
        for side, gg, pp in [('rgb', gr, pr), ('ir', gt, pt)]:
            for th in (.3, .5):
                result = naive_reuse(gg, pp, th)
                for k, v in result.items():
                    reuse[side + '_iou' + str(th)][k] += v
        for ri, ti in pairs.items():
            oldrow = old[(rec['id'], ri, ti)]
            assert oldrow['rgb_hit'] == (ri in hr) and oldrow['ir_hit'] == (ti in ht)
            if ri in hr:
                assert abs(oldrow['rgb_pred_iou'] - hr[ri][1]) < 1e-6
            if ti in ht:
                assert abs(oldrow['ir_pred_iou'] - ht[ti][1]) < 1e-6
            rgb_match = cr.get(ri)
            ir_match = ht.get(ti)
            own = ir_match[1] if ir_match else None
            direct = float(iou(gr[ri:ri + 1, 1:], pt[ir_match[0]:ir_match[0] + 1, :4])[0, 0]) if ir_match else None
            rgb_degree = int((gs[ri, :] >= .1).sum())
            ir_degree = int((gs[:, ti] >= .1).sum())
            rgb_alt = np.delete(gs[ri, :], ti)
            ir_alt = np.delete(gs[:, ti], ri)
            margin = gs[ri, ti] - max(float(rgb_alt.max()) if len(rgb_alt) else 0.,
                                      float(ir_alt.max()) if len(ir_alt) else 0.)
            all_overlap = iou(gr[ri:ri + 1, 1:], pr[:, :4])[0]
            same = pr[:, 5] == gr[ri, 0]
            rr = {'id': rec['id'], 'rgb_gt': ri, 'ir_gt': ti, 'class_id': int(gr[ri, 0]),
                  'label_iou': float(gs[ri, ti]),
                  'center_shift_over_rgb_sqrt_area': oldrow['center_shift_over_rgb_sqrt_area'],
                  'rgb_gt_candidate_degree_at_01': rgb_degree, 'ir_gt_candidate_degree_at_01': ir_degree,
                  'unique_gt_pair_at_01': rgb_degree == 1 and ir_degree == 1,
                  'label_assignment_margin': float(margin),
                  'rgb_hit': ri in hr, 'ir_hit': ti in ht,
                  'rgb_prediction_index': rgb_match[0] if rgb_match else None,
                  'ir_prediction_index': ir_match[0] if ir_match else None,
                  'rgb_confidence': float(pr[rgb_match[0], 4]) if rgb_match else None,
                  'ir_confidence': float(pt[ir_match[0], 4]) if ir_match else None,
                  'rgb_candidate_iou': rgb_match[1] if rgb_match else None,
                  'ir_own_iou': own, 'ir_direct_rgb_iou': direct,
                  'any_same_class_rough_conf05': bool(((all_overlap >= .1) & same).any()),
                  'any_any_class_rough_conf05': bool((all_overlap >= .1).any()),
                  'any_same_class_iou50_conf05': bool(((all_overlap >= .5) & same).any()),
                  'rgb_luminance': oldrow['rgb_luminance']}
            rows.append(rr)
            rr50 = dict(rr)
            m, t = cr50.get(ri), ct50.get(ti)
            rr50['rgb_candidate_iou'] = m[1] if m else None
            rr50['ir_own_iou'] = t[1] if t else None
            conf50_rows.append(rr50)
    assert len(rows) == len(prior)
    common = len(rows)
    hits = {key: sum(r['rgb_hit'] == a and r['ir_hit'] == b for r in rows)
            for key, (a, b) in {'both': (1, 1), 'rgb_only': (1, 0), 'ir_only': (0, 1), 'neither': (0, 0)}.items()}
    both = [r for r in rows if r['rgb_hit'] and r['ir_hit']]
    tonly = [r for r in rows if not r['rgb_hit'] and r['ir_hit']]
    annotation.update({'iou': stats([r['label_iou'] for r in rows]),
                       'center_shift_over_rgb_sqrt_area': stats([r['center_shift_over_rgb_sqrt_area'] for r in rows]),
                       'label_iou_lt_05_n': sum(r['label_iou'] < .5 for r in rows),
                       'label_iou_lt_075_n': sum(r['label_iou'] < .75 for r in rows),
                       'unique_gt_pair_at_01_n': count_by(rows, 'unique_gt_pair_at_01'),
                       'ambiguous_gt_pair_at_01_n': common - count_by(rows, 'unique_gt_pair_at_01'),
                       'assignment_margin': stats([r['label_assignment_margin'] for r in rows]),
                       'nonpositive_assignment_margin_n': sum(r['label_assignment_margin'] <= 0 for r in rows)})
    candidates = []
    for lower, upper in ((.3, .5), (.5, .75)):
        student = [r for r in rows if r['rgb_candidate_iou'] is not None and lower <= r['rgb_candidate_iou'] < upper]
        student50 = [r for r in conf50_rows if r['rgb_candidate_iou'] is not None and lower <= r['rgb_candidate_iou'] < upper]
        for tq in (.5, .75):
            select = lambda r: r['ir_own_iou'] is not None and r['ir_own_iou'] >= tq and r['ir_own_iou'] - r['rgb_candidate_iou'] >= .1
            selected = [r for r in student if select(r)]
            selected50 = [r for r in student50 if select(r)]
            robust = [r for r in selected if r['label_iou'] >= .5 and r['unique_gt_pair_at_01']]
            candidates.append({'rgb_iou_band': [lower, upper], 'teacher_iou_min': tq, 'gain_min': .1,
                               'common_gt_denominator': common, 'student_band_denominator': len(student),
                               'candidate_n': len(selected), 'candidate_fraction_of_common': len(selected) / common,
                               'candidate_image_n': len({r['id'] for r in selected}),
                               'label_iou_ge_05_n': sum(r['label_iou'] >= .5 for r in selected),
                               'label_iou_ge_075_n': sum(r['label_iou'] >= .75 for r in selected),
                               'label_iou_ge_05_and_unique_gt_pair_n': len(robust),
                               'confidence_05_sensitivity': {'student_band_denominator': len(student50), 'candidate_n': len(selected50)},
                               'direct_transfer': direct_stats(selected),
                               'direct_transfer_label_iou_ge_05_and_unique': direct_stats(robust)})
    results = {'n_images': len(records), 'common_objects': common, 'annotation': annotation,
               'strict_hits_conf25_iou50': hits, 'naive_best_prediction_reuse': reuse,
               'both_hit_localization': direct_stats(both),
               'both_hit_label_iou_ge_05': direct_stats([r for r in both if r['label_iou'] >= .5]),
               'teacher_only_candidate_presence_not_one_to_one': {
                   'denominator': len(tonly),
                   **{k: count_by(tonly, k) for k in ('any_same_class_rough_conf05', 'any_any_class_rough_conf05', 'any_same_class_iou50_conf05')},
                   'no_any_class_rough_at_conf05_n': sum(not r['any_any_class_rough_conf05'] for r in tonly)},
               'localization_candidates': candidates,
               'all_common_ir_matched': {'n': sum(r['ir_own_iou'] is not None for r in rows),
                   'own_to_rgb_coordinate_quality_difference': stats([r['ir_direct_rgb_iou'] - r['ir_own_iou'] for r in rows if r['ir_own_iou'] is not None])}}
    return results, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent / '2026-09-06_probe_RGBIR数据特性与可迁移知识')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = {'status': 'descriptive_cpu_reanalysis', 'source': str(args.source.resolve()),
              'datasets': {}, 'contract': {
                'gt_pairing': 'same class, normalized box IoU >= .1; one-to-one maximum cardinality then IoU',
                'saved_predictions': 'post-NMS conf >= .05, NMS IoU .7, max_det 300; no raw logits or DFL distributions',
                'strict_hit': 'same-class conf >= .25 and IoU >= .5, one-to-one within all image GT including unmatched cross-modal GT',
                'rough_candidate': 'preserve strict hits then assign remaining GT/predictions one-to-one at IoU >= .3',
                'candidate_cutoffs_fixed_before_reanalysis': 'RGB bands [.3,.5), [.5,.75); teacher own IoU >= .5/.75; teacher advantage >= .1; conf .25 with .5 sensitivity',
                'confidence_interpretation': 'GT-class correctness at fixed score threshold is a retrospective proxy, not calibrated semantic reliability',
                'direct_copy': 'IR normalized xyxy box copied unchanged to RGB normalized coordinates; no fitted registration or GT-derived geometry transform',
                'claim_limit': 'single seed42 baseline pair, 200 development images per dataset; candidate counts and oracle quality gates do not establish trainability, KD gain, or novelty'}}
    for name in ('dronevehicle', 'llvip'):
        result, rows = analyze(args.source, name)
        output['datasets'][name] = result
        with (args.output / (name + '_localization_objects.csv')).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(output['datasets'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
