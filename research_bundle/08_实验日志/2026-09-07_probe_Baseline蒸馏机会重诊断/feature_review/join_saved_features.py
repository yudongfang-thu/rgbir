"""CPU-only join of immutable 2026-09-06 baseline artifacts; no new inference/hash.

Energy is a within-model response proxy, NOT semantic information or a KD gain.
Image CKA is deliberately kept separate from object measurements in summaries.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def stats(values):
    x = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)])
    return {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
            'median': float(np.median(x)) if len(x) else None}


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = np.prod(wh)
    return float(inter / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter, 1e-10))


def xy_grid(shape, grid):
    h, w = shape
    ratio = min(640 / h, 640 / w)
    left = round((640 - round(w * ratio)) / 2 - .1)
    top = round((640 - round(h * ratio)) / 2 - .1)
    yy, xx = np.meshgrid((np.arange(grid) + .5) * 640 / grid,
                         (np.arange(grid) + .5) * 640 / grid, indexing='ij')
    x, y = (xx - left) / (w * ratio), (yy - top) / (h * ratio)
    return x, y, (x >= 0) & (x < 1) & (y >= 0) & (y < 1)


def in_box(x, y, box):
    return (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])


def energy_response(energy, shape, gt_all, target_index):
    x, y, valid = xy_grid(shape, energy.shape[0])
    box = np.asarray(gt_all[target_index][1:], dtype=float)
    fg = valid & in_box(x, y, box)
    all_fg = np.zeros_like(valid)
    for gt in gt_all:
        all_fg |= in_box(x, y, gt[1:])
    center, half = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]) * .875
    ring = valid & in_box(x, y, np.r_[center - half, center + half]) & ~all_fg
    # No interpolation or fallback to unrelated global background.
    f, b = energy[fg], energy[ring]
    return {'fg_tokens': int(fg.sum()), 'ring_tokens': int(ring.sum()),
            'fg_mean_energy': float(f.mean()) if len(f) else None,
            'ring_mean_energy': float(b.mean()) if len(b) else None,
            'fg_over_ring': float(f.mean() / b.mean()) if len(f) and len(b) and b.mean() > 1e-12 else None}


def candidate_flags(pred, gt):
    low = high = wrong = coarse = False
    for q in pred:
        overlap = iou(gt[1:], q[:4])
        same = int(q[5]) == int(gt[0])
        low |= same and .05 <= q[4] < .25 and overlap >= .5
        high |= same and q[4] >= .25 and overlap >= .5
        wrong |= not same and q[4] >= .25 and overlap >= .5
        coarse |= q[4] >= .05 and overlap >= .1
    return {'student_strict_low_correct': bool(low and not high),
            'student_high_correct_candidate': bool(high),
            'student_high_wrong_class': bool(wrong), 'student_no_coarse_saved_candidate': bool(not coarse)}


def correlations(rows, xkeys, ykeys):
    result = []
    for x in xkeys:
        for y in ykeys:
            subset = [(r[x], r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
            rho = None
            if len(subset) >= 8:
                a, b = np.asarray(subset).T
                if np.ptp(a) > 0 and np.ptp(b) > 0:
                    rho = float(spearmanr(a, b).statistic)
            result.append({'feature': x, 'detection_variable': y, 'n_images': len(subset), 'spearman_rho': rho})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / '2026-09-06_probe_RGBIR数据特性与可迁移知识')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'cpu_join_v2')
    a = p.parse_args()
    a.out.mkdir(exist_ok=False, parents=True)
    object_rows, image_rows, group_rows, corr_rows = [], [], [], []
    summaries = {}
    for ds in ('dronevehicle', 'llvip', 'vedai'):
        d = a.source / (ds + '_full')
        teacher, student = ('rgb', 'ir') if ds == 'vedai' else ('ir', 'rgb')
        objects = read(d / 'matched_objects.json')
        predictions = {r['id']: r for r in read(d / 'prediction_records.json')}
        images = {r['id']: r for r in read(d / 'image_metrics.json')}
        features = {(r['id'], r['level'], r['region']): r for r in read(d / 'feature_metrics.json')}
        ids = read(d / 'input_manifest.json')['sample_ids']
        with np.load(d / 'energy_maps.npz', allow_pickle=False) as stored:
            energies = {key: stored[key] for key in stored.files}
        id_index = {x: i for i, x in enumerate(ids)}
        by_image, dataset_rows = defaultdict(list), []
        for obj in objects:
            ident = obj['id']
            pr, im = predictions[ident], images[ident]
            th, sh = obj[teacher + '_hit'], obj[student + '_hit']
            status = 'both_hit' if th and sh else 'teacher_only' if th else 'student_only' if sh else 'neither'
            flags = candidate_flags(pr['pred_' + student], pr['gt_' + student][obj[student + '_gt']])
            delta = obj[teacher + '_pred_iou'] - obj[student + '_pred_iou'] if th and sh else None
            base = {'dataset': ds, 'id': ident, 'rgb_gt': obj['rgb_gt'], 'ir_gt': obj['ir_gt'],
                    'class_id': obj['class_id'], 'status': status, 'label_pair_iou': obj['iou'],
                    'teacher': teacher, 'student': student, 'both_hit_own_gt_iou_delta': delta,
                    'rgb_luminance': im['rgb_luminance'], **flags}
            by_image[ident].append(base)
            for level in ('P3', 'P4', 'P5'):
                f = features[(ident, level, 'fg')]
                row = {**base, 'level': level, 'image_fg_cka_paired': f['cka_paired'],
                       'image_fg_cka_donor': f['cka_donor_mean'], 'image_fg_delta_cka': f['delta_cka']}
                for side in ('rgb', 'ir'):
                    response = energy_response(energies[side + '_' + level][id_index[ident]],
                                               im[side + '_shape'], pr['gt_' + side], obj[side + '_gt'])
                    row.update({side + '_' + k: v for k, v in response.items()})
                object_rows.append(row)
                dataset_rows.append(row)
        ds_images = []
        for ident in ids:
            group = by_image[ident]
            n = len(group)
            if not n:
                continue
            row = {'dataset': ds, 'id': ident, 'n_common_objects': n, 'rgb_luminance': images[ident]['rgb_luminance'],
                   'teacher_only_fraction': sum(o['status'] == 'teacher_only' for o in group) / n,
                   'student_only_fraction': sum(o['status'] == 'student_only' for o in group) / n,
                   'teacher_only_strict_low_fraction': sum(o['status'] == 'teacher_only' and o['student_strict_low_correct'] for o in group) / n,
                   'both_hit_loc_delta_mean': stats([o['both_hit_own_gt_iou_delta'] for o in group])['mean']}
            for level in ('P3', 'P4', 'P5'):
                for region in ('fg', 'bg'):
                    row[level + '_' + region + '_delta_cka'] = features[(ident, level, region)]['delta_cka']
            ds_images.append(row)
        image_rows.extend(ds_images)
        cs = correlations(ds_images, [lv + '_fg_delta_cka' for lv in ('P3', 'P4', 'P5')],
                          ['teacher_only_fraction', 'teacher_only_strict_low_fraction', 'both_hit_loc_delta_mean', 'rgb_luminance'])
        corr_rows.extend({'dataset': ds, **r} for r in cs)
        common_layer_ids = [i for i in ids if all(features[(i, lv, 'fg')]['delta_cka'] is not None for lv in ('P3', 'P4', 'P5'))]
        summaries[ds] = {'n_images': len(ids), 'n_common_objects': len(objects),
                         'all_three_layers_valid_same_image_n': len(common_layer_ids),
                         'fg_cka_on_all_layers_valid_images': {lv: stats([features[(i, lv, 'fg')]['delta_cka'] for i in common_layer_ids]) for lv in ('P3', 'P4', 'P5')},
                         'teacher_only_strict_low_n': sum(o['status'] == 'teacher_only' and o['student_strict_low_correct'] for g in by_image.values() for o in g)}
        for lv in ('P3', 'P4', 'P5'):
            for status in ('both_hit', 'teacher_only', 'student_only', 'neither'):
                group = [r for r in dataset_rows if r['level'] == lv and r['status'] == status]
                paired = [r for r in group if r[teacher + '_fg_over_ring'] is not None and r[student + '_fg_over_ring'] is not None]
                t = stats([r[teacher + '_fg_over_ring'] for r in paired])
                s = stats([r[student + '_fg_over_ring'] for r in paired])
                group_rows.append({'dataset': ds, 'level': lv, 'status': status, 'n_objects': len(group),
                                   'n_both_valid_fg_ring': len(paired), 'teacher_ratio_median': t['median'], 'student_ratio_median': s['median']})
    table(a.out / 'object_level_join.csv', object_rows)
    table(a.out / 'image_level_join.csv', image_rows)
    table(a.out / 'energy_group_summary.csv', group_rows)
    table(a.out / 'image_descriptive_correlations.csv', corr_rows)
    save(a.out / 'summary.json', summaries)
    save(a.out / 'receipt.json', {'status': 'completed_cpu_only', 'source': str(a.source.resolve()),
                                 'object_layer_rows': len(object_rows), 'image_rows': len(image_rows),
                                 'new_inference': False, 'new_hash_computation': False,
                                 'limitations': ['Energy is within-model foreground/background response, not information quality.',
                                                 'Image CKA cannot be interpreted as an object feature.',
                                                 'Correlation unit is image; scene/luminance/scale confounding remains.',
                                                 'Localization delta compares each model with its own modality GT.',
                                                 'Post-NMS conf>=.05 predictions cannot recover full logits or DFL.',
                                                 'No object ROI vectors or same-modal independent baseline feature cache exists here.',
                                                 'Tiny-object grid-center sampling misses some foreground; paired valid denominators are reported.',
                                                 'Descriptive, one checkpoint seed; no efficacy or statistical significance claim.']})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
