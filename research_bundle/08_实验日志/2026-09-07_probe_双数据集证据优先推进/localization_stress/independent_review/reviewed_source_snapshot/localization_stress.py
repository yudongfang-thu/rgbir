"""CPU, frozen-cohort translation stress diagnostic. See PROTOCOL.md."""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / '2026-09-07_probe_Baseline蒸馏机会重诊断'
SIDES = ('left', 'top', 'right', 'bottom')
SHIFTS = [(0, 0)] + [(dx, dy) for a in (1, 2, 4) for dx, dy in
                    ((a, 0), (-a, 0), (0, a), (0, -a), (a, a), (a, -a), (-a, a), (-a, -a))]


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def dump_csv(path, rows):
    if not rows:
        raise ValueError('Empty CSV')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def logsoftmax(z):
    a = np.asarray(z, dtype=np.float64)
    a = a - a.max(-1, keepdims=True)
    return a - np.log(np.exp(a).sum(-1, keepdims=True))


def softmax(z):
    a = np.asarray(z, dtype=np.float64)
    a = a - a.max(-1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(-1, keepdims=True)


def distances(box, center, stride):
    return np.c_[center - box[:, :2], box[:, 2:] - center] / stride[:, None]


def decode(d, center, stride):
    p = d * stride[:, None]
    return np.c_[center - p[:, :2], center + p[:, 2:]]


def iou(a, b):
    wh = np.maximum(np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0)
    inter = wh.prod(-1)
    union = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1) + np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def in_support(d):
    return np.isfinite(d).all(-1) & (d >= 0).all(-1) & (d <= 14.99).all(-1)


def gt_distribution(d):
    if not in_support(d).all():
        raise ValueError('Unclamped GT distance outside native support')
    q = np.zeros(d.shape + (16,), dtype=np.float64)
    lo = np.floor(d).astype(int)
    frac = d - lo
    row, side = np.indices(d.shape)
    q[row, side, lo] += 1 - frac
    q[row, side, lo + 1] += frac
    return q


def transport(p, delta):
    """Translate bin masses, track escape, condition remaining in-domain mass."""
    p = np.asarray(p, dtype=np.float64)
    delta = np.broadcast_to(np.asarray(delta, dtype=np.float64), p.shape[:-1])
    if np.all(delta == 0):
        return p.copy(), np.zeros(p.shape[:-1]), np.zeros(p.shape[:-1])
    flat = p.reshape(-1, p.shape[-1])
    move = delta.reshape(-1)
    indices = np.arange(len(flat))
    bins = p.shape[-1]
    out = np.zeros_like(flat)
    under = np.zeros(len(flat))
    over = np.zeros(len(flat))
    for k in range(bins):
        location = k + move
        lower = np.floor(location).astype(int)
        fraction = location - lower
        for dest, weight in ((lower, 1 - fraction), (lower + 1, fraction)):
            mass = flat[:, k] * weight
            below, above = dest < 0, dest >= bins
            valid = ~(below | above)
            under[below] += mass[below]
            over[above] += mass[above]
            out[indices[valid], dest[valid]] += mass[valid]
    retained = out.sum(-1)
    if np.any(retained <= 0):
        raise ValueError('All probability escaped native bins')
    np.testing.assert_allclose(retained + under + over, flat.sum(-1), atol=2e-14, rtol=0)
    out /= retained[:, None]
    return out.reshape(p.shape), under.reshape(p.shape[:-1]), over.reshape(p.shape[:-1])


def cross_entropy(q, logp):
    terms = np.zeros_like(q)
    use = q > 0
    terms[use] = -q[use] * logp[use]
    return terms.sum(-1).mean(-1)


def kl_value(q, logq, logp, temperature=2.):
    terms = np.zeros_like(q)
    use = q > 0
    terms[use] = q[use] * (logq[use] - logp[use])
    return terms.sum(-1).mean(-1) * temperature ** 2


def gradient_metrics(pn1, pn2, pt2, qgt):
    kd = (2 * (pn2 - pt2) / 4).reshape(len(pn1), -1)
    gt = ((pn1 - qgt) / 4).reshape(len(pn1), -1)
    kn, gn = np.linalg.norm(kd, axis=1), np.linalg.norm(gt, axis=1)
    dot = (kd * gt).sum(1)
    denom = kn * gn
    cosine = np.divide(dot, denom, out=np.full(len(kd), np.nan), where=denom > 1e-14)
    ratio = np.divide(kn, gn, out=np.full(len(kd), np.nan), where=gn > 1e-14)
    return cosine, dot, kn, gn, ratio


def quality_gate(n_correct, t_correct, niou, trgb, tir):
    return n_correct & t_correct & (niou < .70) & (trgb >= .60) & (tir >= .50) & ((trgb - niou) > .05)


def statistics(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    y = x[finite]
    out = {'n': int(x.size), 'finite_n': int(finite.sum()), 'nan_n': int(np.isnan(x).sum()),
           'posinf_n': int(np.isposinf(x).sum()), 'neginf_n': int(np.isneginf(x).sum())}
    for key, fn in (('mean', np.mean), ('sd_population', np.std), ('min', np.min), ('max', np.max)):
        out[key] = float(fn(y)) if len(y) else None
    for key, quantile in (('p05', .05), ('p50', .5), ('p95', .95)):
        out[key] = float(np.quantile(y, quantile)) if len(y) else None
    # Any infinities stay explicit; finite mean is always accompanied by counts.
    out['mean_scope'] = 'all_objects' if finite.all() else 'finite_only_with_nonfinite_counts'
    return out


def add_metric(summary, name, values, mask):
    summary.update({name + '_' + k: v for k, v in statistics(values[mask]).items()})


def add_worst_metrics(summary, values, mask):
    """Boolean successes and statistical denominators use distinct field names."""
    for name, value in values.items():
        if np.issubdtype(value.dtype, np.bool_):
            summary[name + '_success_n'] = int((mask & value).sum())
            summary[name + '_denominator_n'] = int(mask.sum())
            summary[name + '_fraction'] = float(value[mask].mean()) if mask.any() else None
        else:
            add_metric(summary, name, value, mask)


def true_test_suite():
    checks = {}
    center = np.array([[10., 20.]])
    stride = np.array([8.])
    box = np.array([[2., 4., 34., 60.]])
    d = distances(box, center, stride)
    np.testing.assert_array_equal(decode(d, center, stride), box)
    np.testing.assert_array_equal(decode(d + np.array([[-.25, .125, .25, -.125]]), center, stride), box + [2, -1, 2, -1])
    checks['box_distance_inverse_and_xy_translation_signs'] = 'passed'
    p = np.zeros((1, 4, 16)); p[:, :, 5] = 1
    q, lo, hi = transport(p, np.zeros((1, 4)))
    assert np.array_equal(q, p) and not lo.any() and not hi.any()
    checks['zero_shift_probability_bit_exact'] = 'passed'
    q, lo, hi = transport(p, np.ones((1, 4)))
    assert np.all(q[:, :, 6] == 1) and not lo.any() and not hi.any()
    q, lo, hi = transport(p, np.full((1, 4), .25))
    assert np.all(q[:, :, 5] == .75) and np.all(q[:, :, 6] == .25) and not lo.any() and not hi.any()
    checks['interior_integer_and_fractional_mass_transport'] = 'passed'
    p = np.zeros((1, 4, 16)); p[:, :, 0] = .4; p[:, :, 15] = .6
    q, lo, hi = transport(p, np.full((1, 4), -.5))
    np.testing.assert_allclose(lo, .2); assert not hi.any()
    np.testing.assert_allclose(q[:, :, 0], .25)
    np.testing.assert_allclose(q[:, :, 14], .375)
    np.testing.assert_allclose(q[:, :, 15], .375)
    q2, lo2, hi2 = transport(p, np.full((1, 4), .5))
    np.testing.assert_allclose(hi2, .3); assert not lo2.any()
    np.testing.assert_allclose(q2[:, :, 15], 3 / 7)
    checks['boundary_under_over_mass_conservation_no_clamp'] = 'passed'
    d = np.array([[0., 1.25, 2.5, 14.99]])
    qgt = gt_distribution(d)
    np.testing.assert_allclose(qgt.sum(-1), 1)
    assert qgt[0, 1, 1] == .75 and qgt[0, 1, 2] == .25
    assert qgt[0, 0, 0] == 1 and not in_support(np.array([[0., 1., 2., 15.]]))[0]
    checks['two_bin_gt_and_unclamped_native_support'] = 'passed'
    rng = np.random.default_rng(7)
    zn = rng.normal(size=(1, 4, 16)); zt = rng.normal(size=(1, 4, 16))
    pn1, pn2, pt2 = softmax(zn), softmax(zn / 2), softmax(zt / 2)
    shifted1, _, _ = transport(softmax(zt), np.full((1, 4), .5))
    shifted2, _, _ = transport(pt2, np.full((1, 4), .5))
    post_temper = np.sqrt(shifted1); post_temper /= post_temper.sum(-1, keepdims=True)
    assert not np.allclose(post_temper, shifted2)
    checks['temperature_transport_noncommutation_explicit'] = 'passed'
    np.testing.assert_allclose(kl_value(pn2, logsoftmax(zn / 2), logsoftmax(zn / 2)), 0, atol=0)
    assert kl_value(pt2, logsoftmax(zt / 2), logsoftmax(zn / 2))[0] > 0
    checks['kl_equal_zero_and_known_distinct_positive'] = 'passed'
    gradkd = 2 * (pn2 - shifted2) / 4
    gradgt = (pn1 - qgt) / 4
    numerical_kd = np.zeros_like(zn); numerical_gt = np.zeros_like(zn)
    h = 1e-5
    for index in np.ndindex(zn.shape):
        plus, minus = zn.copy(), zn.copy()
        plus[index] += h; minus[index] -= h
        numerical_kd[index] = (kl_value(shifted2, np.log(shifted2), logsoftmax(plus / 2))[0] - kl_value(shifted2, np.log(shifted2), logsoftmax(minus / 2))[0]) / (2 * h)
        numerical_gt[index] = (cross_entropy(qgt, logsoftmax(plus))[0] - cross_entropy(qgt, logsoftmax(minus))[0]) / (2 * h)
    np.testing.assert_allclose(gradkd, numerical_kd, atol=1e-9, rtol=0)
    np.testing.assert_allclose(gradgt, numerical_gt, atol=1e-9, rtol=0)
    checks['all_64_kd_and_gt_gradients_central_finite_difference'] = 'passed'
    assert np.isnan(gradient_metrics(pn1, pn2, pn2, qgt)[0][0])
    checks['zero_norm_cosine_is_nan'] = 'passed'
    truth = np.ones(4, dtype=bool)
    g = quality_gate(truth, truth, np.array([.7, .69, .5, .5]), np.array([.9, .9, .55, .60]), np.full(4, .8))
    assert np.array_equal(g, [False, True, False, True])
    saved = g.copy()
    perturbed = quality_gate(truth, truth, np.array([.7, .69, .5, .5]), np.full(4, .4), np.full(4, .4))
    assert np.array_equal(saved, g) and saved.sum() == 2 and not perturbed.any()
    assert not quality_gate(np.array([True]), np.array([True]), np.array([.55]), np.array([.60]), np.array([.8]))[0]
    checks['strict_070_and_fixed_mask_survival_without_reselection'] = 'passed'
    mask = np.array([True, True, True, True, True, False])
    success = np.array([True, False, True, False, False, True])
    record = {}
    add_worst_metrics(record, {'all_directions_gate': success,
                              'min_iou_advantage': np.arange(6, dtype=float)}, mask)
    assert record['all_directions_gate_success_n'] == 2
    assert record['all_directions_gate_denominator_n'] == 5
    assert record['all_directions_gate_fraction'] == .4
    assert 'all_directions_gate_n' not in record
    assert record['min_iou_advantage_n'] == 5
    checks['worst_boolean_success_count_cannot_be_overwritten_by_sample_denominator'] = 'passed'
    return checks


def load_data(dataset):
    source = PRIOR / 'remote_exports' / (dataset + '_full_attempt1')
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed' and summary['reg_max'] == 16
    rows = [json.loads(s) for s in (source / 'objects.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(rows) == summary['objects_including_background']
    assert len(set(r['object_id'] for r in rows)) == len(rows)
    counts = {s: Counter() for s in ('train', 'val')}
    selected = []
    for i, r in enumerate(rows):
        c = counts[r['split']]; c['all_exported'] += 1
        if r['is_background']: continue
        c['real_rgb_gt'] += 1
        if r['paired_gt_iou'] is None or r['paired_gt_iou'] < .5: continue
        c['pair_iou_ge_05'] += 1
        if not r['anchor_has_reference_candidate']: continue
        c['with_reference_candidate'] += 1
        d = distances(np.array([r['gt_box_input']]), np.array([r['anchor_center']]), np.array([r['anchor_stride']]))
        if not in_support(d)[0]: continue
        c['base_cohort'] += 1
        selected.append(i)
    with (PRIOR / 'feature_review' / 'localization_readout_v1' / 'outputs' / dataset / 'cohort_and_donors.csv').open(encoding='utf-8-sig', newline='') as h:
        previous = list(csv.DictReader(h))
    assert selected == [int(r['source_row']) for r in previous]
    assert [rows[i]['object_id'] for i in selected] == [r['object_id'] for r in previous]
    cohort = [rows[i] for i in selected]
    data = {'source_row': np.array(selected), 'object_id': np.array([r['object_id'] for r in cohort]),
            'split': np.array([r['split'] for r in cohort]), 'image': np.array([r['image'] for r in cohort]),
            'source_group': np.array([r['source_group'] for r in cohort]),
            'class': np.array([r['class'] for r in cohort]), 'anchor_index': np.array([r['anchor_index'] for r in cohort]),
            'center': np.array([r['anchor_center'] for r in cohort], dtype=float),
            'stride': np.array([r['anchor_stride'] for r in cohort], dtype=float),
            'rgb_gt': np.array([r['gt_box_input'] for r in cohort], dtype=float),
            'ir_gt': np.array([r['ir_gt_box_input'] for r in cohort], dtype=float),
            'paired_gt_iou': np.array([r['paired_gt_iou'] for r in cohort], dtype=float)}
    assert np.isin(data['stride'], [8, 16]).all()
    with np.load(source / 'logits.npz', allow_pickle=False) as f:
        for m in ('N42', 'T42'):
            data[m + '_dfl'] = f[m + '_dfl'][selected].astype(float)
            data[m + '_cls'] = f[m + '_cls'][selected].astype(float)
            assert data[m + '_dfl'].shape == (len(selected), 4, 16)
            assert np.isfinite(data[m + '_dfl']).all() and np.isfinite(data[m + '_cls']).all()
    # Reconstruct native geometry owner count from every exported foreground GT.
    images = defaultdict(list)
    for r in rows:
        if not r['is_background']: images[r['image']].append(r['gt_box_input'])
    owner = np.zeros(len(cohort), dtype=np.int64)
    for i, r in enumerate(cohort):
        boxes = np.asarray(images[r['image']], dtype=float)
        xy = (boxes[:, :2] + boxes[:, 2:]) / 2
        wh = boxes[:, 2:] - boxes[:, :2]
        wh = np.where(wh < 8, 16., wh)
        lower, upper = xy - wh / 2, xy + wh / 2
        owner[i] = ((data['center'][i] - lower > 1e-9) & (upper - data['center'][i] > 1e-9)).all(-1).sum()
    data['native_owner_count_reconstructed'] = owner
    context = {'source': str(source.resolve()), 'models': summary['models'], 'filter_counts': {s: dict(c) for s, c in counts.items()},
               'cohort_matches_previous_exact': True, 'source_files': {f: {'bytes': (source/f).stat().st_size, 'mtime_ns': (source/f).stat().st_mtime_ns} for f in ('objects.jsonl', 'logits.npz', 'summary.json', 'model_identity.json')}}
    return data, cohort, context


def summarize_condition(dataset, data, fixed_masks, values, perturb_gate, j, dx, dy):
    rows = []
    for split in ('train', 'val'):
        for name, fixed in fixed_masks.items():
            mask = fixed & (data['split'] == split)
            n = int(mask.sum())
            original_gate = mask & fixed_masks['quality_gate070']
            survived = original_gate & perturb_gate
            row = {'dataset': dataset, 'split': split, 'fixed_stratum': name, 'condition_index': j, 'dx_input_px': dx, 'dy_input_px': dy,
                   'linf_px': max(abs(dx), abs(dy)), 'l2_px': math.hypot(dx, dy), 'fixed_n': n,
                   'fixed_images': len(set(data['image'][mask])), 'fixed_source_groups': len(set(data['source_group'][mask])),
                   'teacher_unclamped_support_n': int((mask & values['teacher_support']).sum()),
                   'teacher_unclamped_support_fraction': float(values['teacher_support'][mask].mean()) if n else None,
                   'perturbed_quality_gate_n': int((mask & perturb_gate).sum()),
                   'original_quality_gate_n': int(original_gate.sum()), 'original_quality_gate_survived_n': int(survived.sum()),
                   'original_quality_gate_survival_fraction': float(survived.sum()/original_gate.sum()) if original_gate.any() else None,
                   'new_quality_gate_members_n': int((mask & ~fixed_masks['quality_gate070'] & perturb_gate).sum()),
                   'teacher_iou_better_n': int((mask & (values['iou_advantage'] > 0)).sum()),
                   'teacher_iou_better_fraction': float((values['iou_advantage'][mask] > 0).mean()) if n else None,
                   'teacher_ce_better_n': int((mask & (values['teacher_minus_native_ce'] < 0)).sum()),
                   'teacher_ce_better_fraction': float((values['teacher_minus_native_ce'][mask] < 0).mean()) if n else None,
                   'positive_cosine_n': int((mask & (values['cosine'] > 0)).sum()),
                   'cosine_defined_n': int((mask & np.isfinite(values['cosine'])).sum()),
                   'positive_cosine_fraction_fixed': float((values['cosine'][mask] > 0).mean()) if n else None,
                   'positive_cosine_fraction_defined': float((values['cosine'][mask & np.isfinite(values['cosine'])] > 0).mean()) if (mask & np.isfinite(values['cosine'])).any() else None}
            for metric, v in values.items():
                if metric != 'teacher_support': add_metric(row, metric, v, mask)
            rows.append(row)
    return rows


def run_dataset(dataset, out):
    started = time.time()
    data, cohort, context = load_data(dataset)
    n = len(cohort)
    zn, zt = data.pop('N42_dfl'), data.pop('T42_dfl')
    pn1, pn2, pt1, pt2 = softmax(zn), softmax(zn/2), softmax(zt), softmax(zt/2)
    ln1, ln2, lt1, lt2 = logsoftmax(zn), logsoftmax(zn/2), logsoftmax(zt), logsoftmax(zt/2)
    bins = np.arange(16)
    dn, dt = (pn1 * bins).sum(-1), (pt1 * bins).sum(-1)
    bn, bt = decode(dn, data['center'], data['stride']), decode(dt, data['center'], data['stride'])
    niou = iou(bn, data['rgb_gt'])
    gt_d = distances(data['rgb_gt'], data['center'], data['stride'])
    ir_d = distances(data['ir_gt'], data['center'], data['stride'])
    qgt = gt_distribution(gt_d)
    ce_n = cross_entropy(qgt, ln1)
    correct = {}
    agreement = {}
    for model, box in (('N42', bn), ('T42', bt)):
        cls = data.pop(model + '_cls')
        conf = 1/(1+np.exp(-cls.max(-1)))
        pred = cls.argmax(-1)
        correct[model] = (pred == data['class']) & (conf >= .25)
        data[model + '_confidence'] = conf
        data[model + '_pred_class'] = pred
        source_box = np.asarray([r[model]['same_anchor']['box'] for r in cohort])
        error = float(np.max(np.abs(box-source_box)))
        assert error <= 1e-3, (model, error)
        assert np.array_equal(pred, [r[model]['same_anchor']['pred_class'] for r in cohort])
        np.testing.assert_allclose(conf, [r[model]['same_anchor']['confidence'] for r in cohort], atol=2e-7, rtol=0)
        agreement[model + '_decode_vs_json_max_abs_px'] = error
    original_gate = quality_gate(correct['N42'], correct['T42'], niou, iou(bt, data['rgb_gt']), iou(bt, data['ir_gt']))
    fixed = {'base': np.ones(n, dtype=bool), 'n_iou_lt_070': niou < .7, 'quality_gate070': original_gate.copy()}
    outputs = dict(data)
    outputs.update({'gt_ltrb_stride': gt_d, 'ir_gt_ltrb_stride': ir_d, 'native_box': bn, 'teacher_box_zero': bt,
                    'native_iou': niou, 'native_gt_ce': ce_n, 'shifts_xy_input_px': np.array(SHIFTS),
                    **{'fixed_' + key: val for key, val in fixed.items()}})
    all_values = defaultdict(list)
    rows = []
    for j, (dx, dy) in enumerate(SHIFTS):
        delta = np.array([-dx, -dy, dx, dy])[None] / data['stride'][:, None]
        shifted_box = bt + np.array([dx, dy, dx, dy])
        shifted_d = dt + delta
        p1, u1, o1 = transport(pt1, delta)
        p2, u2, o2 = transport(pt2, delta)
        if j == 0:
            lp1, lp2 = lt1, lt2
            assert np.array_equal(p1, pt1) and np.array_equal(p2, pt2)
        else:
            with np.errstate(divide='ignore'):
                lp1, lp2 = np.log(p1), np.log(p2)
        trgb, tir = iou(shifted_box, data['rgb_gt']), iou(shifted_box, data['ir_gt'])
        ce_t = cross_entropy(qgt, lp1)
        kl = kl_value(p2, lp2, ln2)
        cosine, dot, kn, gn, ratio = gradient_metrics(pn1, pn2, p2, qgt)
        approximate_box = decode((p1*bins).sum(-1), data['center'], data['stride'])
        values = {'teacher_support': in_support(shifted_d), 'teacher_rgb_iou': trgb, 'teacher_ir_iou': tir,
                  'iou_advantage': trgb-niou, 'native_ce': ce_n, 'teacher_ce': ce_t,
                  'teacher_minus_native_ce': ce_t-ce_n, 'kl_t2_mean4': kl, 'cosine': cosine, 'gradient_dot': dot,
                  'kd_gradient_norm': kn, 'gt_gradient_norm': gn, 'gradient_norm_ratio': ratio,
                  'lost_mass_t1_mean4': (u1+o1).mean(-1), 'lost_mass_t1_max_side': (u1+o1).max(-1),
                  'lost_mass_t2_mean4': (u2+o2).mean(-1), 'lost_mass_t2_max_side': (u2+o2).max(-1),
                  'conditional_vs_exact_box_max_abs_px': np.max(np.abs(approximate_box-shifted_box), axis=-1),
                  'conditional_box_rgb_iou': iou(approximate_box, data['rgb_gt'])}
        gate = quality_gate(correct['N42'], correct['T42'], niou, trgb, tir)
        if j == 0: assert np.array_equal(gate, original_gate)
        rows += summarize_condition(dataset, data, fixed, values, gate, j, dx, dy)
        for key, val in values.items(): all_values[key].append(val)
        for key, val in [('gate_after', gate), ('underflow_t1', u1), ('overflow_t1', o1), ('underflow_t2', u2), ('overflow_t2', o2)]:
            all_values[key].append(val)
    arrays = {k: np.stack(v, axis=1) for k, v in all_values.items()}
    outputs.update(arrays)
    # Exact zero-shift replication of the old CE/KL/local head-gradient definitions.
    old_ce_n = -(qgt * logsoftmax(zn)).sum(-1).mean(-1)
    old_ce_t = -(qgt * logsoftmax(zt)).sum(-1).mean(-1)
    old_kl = (pt2 * (logsoftmax(zt/2)-logsoftmax(zn/2))).sum(-1).mean(-1)*4
    old_gk = (2*(pn2-pt2)).reshape(n,-1); old_gg = (pn1-qgt).reshape(n,-1)
    denom = np.linalg.norm(old_gk, axis=1)*np.linalg.norm(old_gg, axis=1)
    old_cos = np.divide((old_gk*old_gg).sum(-1), denom, out=np.full(n,np.nan), where=denom>1e-14)
    zero_check = {}
    for key, actual, expected in [('native_ce', ce_n, old_ce_n), ('teacher_ce', arrays['teacher_ce'][:,0], old_ce_t), ('kl', arrays['kl_t2_mean4'][:,0], old_kl), ('cosine', arrays['cosine'][:,0], old_cos)]:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)
        zero_check[key + '_max_abs'] = float(np.nanmax(np.abs(actual-expected)))
    worst_rows = []
    for amplitude in (1,2,4):
        indices = np.array([j for j,(x,y) in enumerate(SHIFTS) if max(abs(x),abs(y)) == amplitude])
        assert len(indices) == 8
        # Ordinary min deliberately propagates undefined cosine to this stress statistic.
        worst = {'min_iou_advantage': arrays['iou_advantage'][:,indices].min(1),
                 'max_teacher_minus_native_ce': arrays['teacher_minus_native_ce'][:,indices].max(1),
                 'min_cosine': arrays['cosine'][:,indices].min(1),
                 'max_lost_mass_t1_mean4': arrays['lost_mass_t1_mean4'][:,indices].max(1),
                 'max_lost_mass_t2_mean4': arrays['lost_mass_t2_mean4'][:,indices].max(1),
                 'all_directions_teacher_support': arrays['teacher_support'][:,indices].all(1),
                 'all_directions_gate': arrays['gate_after'][:,indices].all(1)}
        outputs.update({'worst_' + str(amplitude) + '_' + key: val for key,val in worst.items()})
        for split in ('train','val'):
            for name, f in fixed.items():
                m = f & (data['split'] == split); count = int(m.sum())
                row = {'dataset':dataset,'split':split,'fixed_stratum':name,'linf_px':amplitude,'directions':8,'fixed_n':count,
                       'all_directions_iou_better_fraction':float((worst['min_iou_advantage'][m]>0).mean()) if count else None,
                       'all_directions_ce_better_fraction':float((worst['max_teacher_minus_native_ce'][m]<0).mean()) if count else None,
                       'all_directions_positive_cosine_fraction':float((worst['min_cosine'][m]>0).mean()) if count else None}
                add_worst_metrics(row,worst,m)
                worst_rows.append(row)
    context['zero_shift_matches_previous_definitions'] = zero_check
    context['raw_vs_json_agreement'] = agreement
    context['stratum_context'] = []
    cohort_rows = []
    for i, r in enumerate(cohort):
        cohort_rows.append({'cohort_index':i,'source_row':int(data['source_row'][i]),'object_id':r['object_id'],'split':r['split'],
                            'image':r['image'],'source_group':r['source_group'],'class':r['class'],'scale_bin':r['scale_bin'],
                            'anchor_index':r['anchor_index'],'stride':r['anchor_stride'],'paired_gt_iou':r['paired_gt_iou'],
                            'ir_gt_support':bool(in_support(ir_d[i:i+1])[0]),'native_owner_count':int(data['native_owner_count_reconstructed'][i]),
                            'n_iou_lt_070':bool(fixed['n_iou_lt_070'][i]),'quality_gate070':bool(original_gate[i])})
    for split in ('train','val'):
        for name,f in fixed.items():
            m = f & (data['split'] == split)
            context['stratum_context'].append({'split':split,'fixed_stratum':name,'n':int(m.sum()),'unique_images':len(set(data['image'][m])),
                                               'source_groups':dict(Counter(data['source_group'][m].tolist())),
                                               'stride_counts':dict(Counter(str(x) for x in data['stride'][m])),
                                               'pair_iou_ge_08_n':int((m & (data['paired_gt_iou']>=.8)).sum()),
                                               'ir_gt_support_n':int((m & in_support(ir_d)).sum()),
                                               'native_owner_count_eq1_n':int((m & (data['native_owner_count_reconstructed']==1)).sum())})
    out.mkdir(parents=True, exist_ok=False)
    dump_csv(out/'metrics.csv', rows)
    dump_csv(out/'worst_case.csv', worst_rows)
    dump_csv(out/'cohort.csv', cohort_rows)
    np.savez_compressed(out/'per_object.npz', **outputs)
    context.update({'status':'COMPLETED_PENDING_INDEPENDENT_REVIEW','dataset':dataset,'cpu_only':True,'new_network_inference':False,
                    'new_training':False,'test_accessed':False,'l1_admission_changed':False,'ap_or_kd_gain_claim':False,
                    'conditions':[{'dx':dx,'dy':dy,'linf':max(abs(dx),abs(dy)),'l2':math.hypot(dx,dy)} for dx,dy in SHIFTS],
                    'dfl_translation':'linear mass transport at T1/T2 separately; discarded exterior mass recorded; conditional retained distribution for losses',
                    'seconds':time.time()-started,'per_object_npz_bytes':(out/'per_object.npz').stat().st_size})
    dump_json(out/'summary.json', context)
    print(json.dumps({'dataset':dataset,'status':context['status'],'n':n,'seconds':context['seconds'],'out':str(out)},ensure_ascii=False), flush=True)
    return rows, worst_rows, context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'outputs_attempt2')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    started = time.time()
    checks = true_test_suite()
    if args.test_only:
        print(json.dumps({'status':'PASSED','checks':checks},ensure_ascii=False,indent=2))
        return
    args.out.mkdir(parents=True,exist_ok=False)
    dump_json(args.out/'known_truth_tests.json',{'status':'PASSED','checks':checks})
    all_rows, all_worst, contexts = [], [], []
    for dataset in ('llvip','dronevehicle'):
        rows,worst,context = run_dataset(dataset,args.out/dataset)
        all_rows += rows; all_worst += worst; contexts.append(context)
    dump_csv(args.out/'all_metrics.csv',all_rows)
    dump_csv(args.out/'all_worst_case.csv',all_worst)
    dump_json(args.out/'completion_receipt.json',{'status':'COMPLETED_PENDING_INDEPENDENT_REVIEW','seconds':time.time()-started,
                                                'dataset_execution_order':['llvip','dronevehicle'],'datasets':contexts,
                                                'protocol_path':str(HERE/'PROTOCOL.md'),'script_path':str(Path(__file__).resolve())})


if __name__ == '__main__':
    main()
