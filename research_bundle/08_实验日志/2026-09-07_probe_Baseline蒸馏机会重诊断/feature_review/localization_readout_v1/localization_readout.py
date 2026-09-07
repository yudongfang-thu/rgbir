"""Frozen CPU localization readout. See PROTOCOL.md; no hash or GPU imports."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve

SEED = 20260907
ALPHA = 1.0
PROJECTION_DIM = 128


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def dump_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def target_distances(boxes, centers, strides):
    return np.concatenate((centers - boxes[:, :2], boxes[:, 2:] - centers), axis=1) / strides[:, None]


def decode_distances(distances, centers, strides):
    pixels = distances * strides[:, None]
    return np.concatenate((centers - pixels[:, :2], centers + pixels[:, 2:]), axis=1)


def native_expectation(logits):
    values = logits.astype(np.float64)
    probability = np.exp(values - values.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    return (probability * np.arange(16)[None, None, :]).sum(axis=-1)


def prediction_metrics(prediction, target, centers, strides, gt_boxes):
    assert prediction.shape == target.shape and np.isfinite(prediction).all()
    invalid = (prediction < 0).any(axis=1)
    decoded = decode_distances(prediction, centers, strides)
    wh = np.maximum(0, np.minimum(decoded[:, 2:], gt_boxes[:, 2:]) - np.maximum(decoded[:, :2], gt_boxes[:, :2]))
    intersection = wh.prod(axis=1)
    pred_area = np.maximum(0, decoded[:, 2:] - decoded[:, :2]).prod(axis=1)
    gt_area = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1)
    ious = intersection / np.maximum(pred_area + gt_area - intersection, 1e-12)
    ious[invalid] = 0.0
    err = np.square(prediction - target)
    summary = {'n_objects': len(target), 'four_edge_mse': float(err.mean()),
               **{edge + '_mse': float(err[:, i].mean()) for i, edge in enumerate(('left', 'top', 'right', 'bottom'))},
               'mean_iou_rgb_gt': float(ious.mean()), 'invalid_negative_distance_n': int(invalid.sum())}
    return summary, ious, invalid


def standardize(values, fit):
    mean = values[fit].mean(axis=0)
    scale = values[fit].std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale, mean, scale


def fit_ridge(x, y):
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(x)
    gram.flat[::gram.shape[0] + 1] += ALPHA
    coef = solve(gram, xc.T @ yc / len(x), assume_a='pos', check_finite=True)
    intercept = ym - xm @ coef
    return coef, intercept


def fixed_donors(ids, splits):
    donor = np.full(len(ids), -1, dtype=np.int64)
    for split in ('train', 'val'):
        indices = np.array(sorted(np.flatnonzero(splits == split), key=lambda i: ids[i]))
        if len(indices) < 2:
            raise ValueError('At least two cohort objects per split are required')
        order = np.random.default_rng(SEED).permutation(indices)
        donor[order] = np.roll(order, 1)
        assert len(np.unique(donor[indices])) == len(indices)
        assert np.all(donor[indices] != indices)
    assert (donor >= 0).all()
    return donor


def known_truth_tests():
    checks = {}
    center = np.array([[10., 10.], [20., 20.]])
    stride = np.array([2., 4.])
    box = np.array([[6., 4., 18., 20.], [16., 12., 32., 36.]])
    target = target_distances(box, center, stride)
    assert np.array_equal(target[0], [2., 3., 4., 5.])
    assert np.array_equal(decode_distances(target, center, stride), box)
    checks['ltrb_roundtrip_known_box'] = 'passed'
    s, ious, invalid = prediction_metrics(target, target, center, stride, box)
    assert s['four_edge_mse'] == 0 and np.all(ious == 1) and not invalid.any()
    negative = target.copy(); negative[0, 0] = -0.1
    s, ious, invalid = prediction_metrics(negative, target, center, stride, box)
    assert s['invalid_negative_distance_n'] == 1 and ious[0] == 0 and ious[1] == 1
    checks['negative_distance_is_invalid_not_clamped'] = 'passed'
    assert np.allclose(native_expectation(np.zeros((2, 4, 16))), 7.5)
    peaked = np.full((1, 4, 16), -1000.); peaked[:, :, 3] = 0
    assert np.array_equal(native_expectation(peaked), np.full((1, 4), 3.))
    checks['native_dfl_bin_expectation'] = 'passed'
    x = np.array([[-1.], [1.]])
    coef, intercept = fit_ridge(x, 2*x+3)
    assert np.allclose(coef, 1) and np.allclose(intercept, 3)
    checks['ridge_mean_objective_alpha1_closed_form'] = 'passed'
    values = np.array([[0.], [2.], [100.]])
    z, mean, scale = standardize(values, np.array([True, True, False]))
    assert np.array_equal(mean, [1.]) and np.array_equal(scale, [1.]) and z[2, 0] == 99
    checks['standardizer_train_only'] = 'passed'
    ids = np.array(['c', 'b', 'a', 'e', 'd', 'f'])
    split = np.array(['train']*3 + ['val']*3)
    donor = fixed_donors(ids, split)
    assert np.all(donor != np.arange(6)) and np.array_equal(np.sort(donor), np.arange(6)) and np.array_equal(split[donor], split)
    checks['split_shuffle_bijection_no_self_pair'] = 'passed'
    return checks


def read_completed_source(source):
    receipt = source / 'collection_receipt.json'
    if not receipt.is_file():
        receipt = source.parents[1] / (source.name.split('_full_attempt')[0] + '_collection_receipt.json')
    if not receipt.is_file():
        raise RuntimeError('Collection is not complete: ' + str(receipt))
    collection = json.loads(receipt.read_text(encoding='utf-8'))
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed' and summary['reg_max'] == 16
    assert summary['head_input_verified_against_raw_feats']
    rows = [json.loads(line) for line in (source / 'objects.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == summary['objects_including_background']
    assert len({r['object_id'] for r in rows}) == len(rows)
    return rows, summary, collection


def run_dataset(dataset, source, out):
    started = time.time()
    rows, source_summary, collection = read_completed_source(source)
    counts = {sp: Counter() for sp in ('train', 'val')}
    selected = []
    for i, row in enumerate(rows):
        c = counts[row['split']]
        c['all_exported_including_background'] += 1
        if row['is_background']:
            continue
        c['real_gt_objects'] += 1
        if row['paired_gt_iou'] is None or row['paired_gt_iou'] < .5:
            continue
        c['paired_gt_iou_ge_0_5'] += 1
        if not row['anchor_has_reference_candidate']:
            continue
        c['with_reference_candidate'] += 1
        b, center, stride = np.asarray(row['gt_box_input']), np.asarray(row['anchor_center']), row['anchor_stride']
        distances = target_distances(b[None], center[None], np.array([stride]))[0]
        if not np.all((distances >= 0) & (distances <= 14.99)):
            continue
        c['all_four_gt_distances_in_0_14_99'] += 1
        selected.append(i)
    selected = np.asarray(selected, dtype=np.int64)
    cohort = [rows[i] for i in selected]
    ids = np.asarray([r['object_id'] for r in cohort])
    splits = np.asarray([r['split'] for r in cohort])
    fit, dev = splits == 'train', splits == 'val'
    assert fit.sum() >= 2 and dev.sum() >= 2
    gt = np.asarray([r['gt_box_input'] for r in cohort], dtype=np.float64)
    centers = np.asarray([r['anchor_center'] for r in cohort], dtype=np.float64)
    strides = np.asarray([r['anchor_stride'] for r in cohort], dtype=np.float64)
    target = target_distances(gt, centers, strides)
    donor = fixed_donors(ids, splits)
    models = ['N42', 'T42'] + (['N0'] if 'N0' in source_summary['models'] else [])
    blocks, transformations, projections, dfl = {}, {}, {}, {}
    with np.load(source / 'logits.npz', allow_pickle=False) as arrays:
        for model in models:
            full = arrays[model + '_dfl']
            assert full.shape == (len(rows), 4, 16)
            dfl[model] = full[selected].astype(np.float64)
            assert np.isfinite(dfl[model]).all()
            values = dfl[model].reshape(len(cohort), 64)
            key = model + '_dfl'
            blocks[key], transformations[key + '_mean'], transformations[key + '_scale'] = standardize(values, fit)
    with np.load(source / 'features.npz', allow_pickle=False) as arrays:
        for model in models:
            level_values = []
            for level in ('P3', 'P4'):
                key = model + '_anchor_' + level
                full = arrays[key]
                assert len(full) == len(rows) and full.ndim == 2
                values = full[selected].astype(np.float64)
                assert np.isfinite(values).all()
                projection_key = 'input_' + str(values.shape[1])
                if projection_key not in projections:
                    projections[projection_key] = np.random.default_rng(SEED).normal(size=(values.shape[1], PROJECTION_DIM)) / np.sqrt(PROJECTION_DIM)
                projected = values @ projections[projection_key]
                z, transformations[key + '_mean'], transformations[key + '_scale'] = standardize(projected, fit)
                level_values.append(z)
            blocks[model + '_feature'] = np.concatenate(level_values, axis=1)
    blocks['T42_feature_shuffled'] = blocks['T42_feature'][donor]
    meta = np.c_[centers / 640., np.log(strides)]
    blocks['meta'], transformations['meta_mean'], transformations['meta_scale'] = standardize(meta, fit)
    base = ['N42_dfl', 'N42_feature']
    arms = {'ridge_N_dfl': ['N42_dfl'], 'ridge_N_dfl_T_dfl': ['N42_dfl', 'T42_dfl'],
            'ridge_N_dfl_N_feature': base,
            'ridge_N_dfl_N_feature_T_feature': base + ['T42_feature'],
            'ridge_N_dfl_N_feature_shuffled_T_feature': base + ['T42_feature_shuffled'],
            'ridge_meta_only': ['meta']}
    if 'N0' in models:
        arms['ridge_N_dfl_N0_dfl'] = ['N42_dfl', 'N0_dfl']
        arms['ridge_N_dfl_N_feature_N0_feature'] = base + ['N0_feature']
    predictions = {'native_N_dfl_expectation': native_expectation(dfl['N42'])}
    fitted = {}
    for name, keys in arms.items():
        x = np.concatenate([blocks[key] for key in keys], axis=1)
        coef, intercept = fit_ridge(x[fit], target[fit])
        predictions[name] = x @ coef + intercept
        fitted[name + '_coef'], fitted[name + '_intercept'] = coef, intercept
    metrics, outputs = [], {'cohort_source_row': selected, 'object_id': ids, 'split': splits,
                           'target_ltrb_stride': target, 'anchor_center': centers, 'anchor_stride': strides,
                           'gt_box_input': gt, 'shuffled_teacher_donor_cohort_index': donor}
    for name, prediction in predictions.items():
        outputs[name + '_distance'] = prediction
        for split, mask in (('train', fit), ('val', dev)):
            metric, ious, invalid = prediction_metrics(prediction[mask], target[mask], centers[mask], strides[mask], gt[mask])
            metrics.append({'dataset': dataset, 'split': split, 'arm': name,
                            'input_dim': 0 if name.startswith('native') else sum(blocks[k].shape[1] for k in arms[name]), **metric})
        _, ious, invalid = prediction_metrics(prediction, target, centers, strides, gt)
        outputs[name + '_iou'] = ious
        outputs[name + '_invalid'] = invalid
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'predictions.npz', **outputs)
    np.savez_compressed(out / 'fitted_models.npz', **fitted, **transformations, **{'projection_' + k: v for k, v in projections.items()})
    metadata = []
    for i, row in enumerate(cohort):
        metadata.append({'cohort_index': i, 'source_row': int(selected[i]), 'object_id': row['object_id'],
                         'split': row['split'], 'image': row['image'], 'source_group': row['source_group'],
                         'class': row['class'], 'scale_bin': row['scale_bin'],
                         'paired_gt_iou': row['paired_gt_iou'], 'anchor_index': row['anchor_index'],
                         'shuffled_teacher_donor_object_id': cohort[donor[i]]['object_id'],
                         'donor_same_image': row['image'] == cohort[donor[i]]['image']})
    dump_csv(out / 'cohort_and_donors.csv', metadata)
    dump_csv(out / 'metrics.csv', metrics)
    diagnostic = {'dataset': dataset, 'status': 'completed_cpu_only', 'source': str(source.resolve()),
                  'filter_counts': {sp: dict(c) for sp, c in counts.items()},
                  'models': source_summary['models'], 'arms': {'native_N_dfl_expectation': [], **arms},
                  'missing_arms': [] if 'N0' in models else ['ridge_N_dfl_N0_dfl', 'ridge_N_dfl_N_feature_N0_feature'],
                  'shuffle_same_image_count_by_split': {sp: sum(r['split'] == sp and r['donor_same_image'] for r in metadata) for sp in ('train', 'val')},
                  'metrics': metrics, 'alpha': ALPHA, 'projection_dim_per_level': PROJECTION_DIM, 'seed': SEED,
                  'new_gpu_inference': False, 'new_hash_computation': False, 'test_accessed': False,
                  'gt_privileged_association': True, 'deployable_detector': False, 'kd_gain_or_ap': False,
                  'collection_receipt': collection, 'seconds': time.time() - started}
    save_json(out / 'summary.json', diagnostic)
    print(json.dumps({'dataset': dataset, 'filter_counts': diagnostic['filter_counts'], 'dev_metrics': [m for m in metrics if m['split'] == 'val']}), flush=True)
    return diagnostic


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2] / 'remote_exports')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    p.add_argument('--test-only', action='store_true')
    a = p.parse_args()
    tests = known_truth_tests()
    if a.test_only:
        save_json(Path(__file__).resolve().parent / 'known_truth_test_receipt.json', {'status': 'passed', 'tests': tests})
        print(json.dumps(tests)); return
    if a.out.exists():
        raise RuntimeError('Output exists; preserve prior attempt and choose a new output path')
    # Both receipt barriers checked before any data-dependent fit.
    for ds in ('dronevehicle', 'llvip'):
        if not ((a.source_root / (ds + '_full_attempt1') / 'collection_receipt.json').is_file()
                or (a.source_root.parent / (ds + '_collection_receipt.json')).is_file()):
            raise RuntimeError('Wait for collection_receipt.json: ' + ds)
    a.out.mkdir(parents=True, exist_ok=False)
    save_json(a.out / 'known_truth_tests.json', {'status': 'passed', 'tests': tests})
    summaries = [run_dataset(ds, a.source_root / (ds + '_full_attempt1'), a.out / ds) for ds in ('dronevehicle', 'llvip')]
    metrics = [m for s in summaries for m in s['metrics']]
    dump_csv(a.out / 'all_metrics.csv', metrics)
    save_json(a.out / 'completion_receipt.json', {'status': 'completed_cpu_only', 'datasets': [s['dataset'] for s in summaries],
                                               'known_truth_tests': tests, 'test_accessed': False, 'new_hash_computation': False})


if __name__ == '__main__':
    main()
