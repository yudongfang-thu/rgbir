"""Independent source-to-array reconstruction; does not import executor formulas."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT.parents[1] / '2026-09-07_probe_Baseline蒸馏机会重诊断'


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def probs(z, temperature):
    v = z / temperature
    logp = v - np.max(v, axis=2, keepdims=True)
    logp -= np.log(np.exp(logp).sum(axis=2, keepdims=True))
    return np.exp(logp), logp


def boxes(d, c, s):
    return np.concatenate((c - d[:, :2] * s[:, None], c + d[:, 2:] * s[:, None]), axis=1)


def overlap(x, y):
    intersection = np.clip(np.minimum(x[:, 2:], y[:, 2:]) - np.maximum(x[:, :2], y[:, :2]), 0, None).prod(axis=1)
    union = (x[:, 2:] - x[:, :2]).prod(axis=1) + (y[:, 2:] - y[:, :2]).prod(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def transported(p, offsets):
    # Independent triangular interpolation matrix, grouped by stride-derived offset.
    out = np.empty_like(p)
    lower = np.empty(p.shape[:2]); upper = np.empty(p.shape[:2])
    src = np.arange(16, dtype=float)
    dst = np.arange(16, dtype=float)
    for side in range(4):
        for delta in np.unique(offsets[:, side]):
            take = offsets[:, side] == delta
            location = src + delta
            kernel = np.maximum(1 - np.abs(location[:, None] - dst[None, :]), 0)
            full = p[take, side] @ kernel
            lower[take, side] = p[take, side] @ np.clip(-location, 0, 1)
            upper[take, side] = p[take, side] @ np.clip(location - 15, 0, 1)
            np.testing.assert_allclose(full.sum(axis=1) + lower[take, side] + upper[take, side], 1, atol=2e-14, rtol=0)
            out[take, side] = full / full.sum(axis=1, keepdims=True)
    return out, lower, upper


def main():
    watched = [ROOT / n for n in ('PROTOCOL.md', 'localization_stress.py', 'verify_saved.py', 'write_readout.py', 'README.md')]
    watched += list((ROOT / 'outputs_attempt1').rglob('*'))
    watched += list((ROOT / 'outputs_attempt2').rglob('*'))
    watched = [p for p in watched if p.is_file()]
    before = {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in watched}
    snapshot = HERE / 'reviewed_source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for p in watched[:5]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    r = subprocess.run([sys.executable, str(ROOT / 'localization_stress.py'), '--test-only'], capture_output=True, text=True, encoding='utf-8')
    assert r.returncode == 0, r.stderr
    (HERE / 'author_truth_reexecution.json').write_text(r.stdout, encoding='utf-8')
    # Reexecute author's verifier with its original source roots, redirect only receipt.
    script = (ROOT / 'verify_saved.py').read_text(encoding='utf-8')
    needle = "(HERE/'saved_output_self_check.json').write_text("
    assert script.count(needle) == 1
    script = script.replace(needle, "(Path(" + repr(str(HERE / 'author_verify_saved_reexecution.json')) + ")).write_text(")
    exec(compile(script, str(ROOT / 'verify_saved.py'), 'exec'), {'__file__': str(ROOT / 'verify_saved.py'), '__name__': '__independent_reexecution__'})
    report = {'status': 'PENDING', 'cpu_only': True, 'hash_computed': False, 'datasets': {}, 'reporting_defects': []}
    for dataset in ('llvip', 'dronevehicle'):
        folder = ROOT / 'outputs_attempt1' / dataset
        with np.load(folder / 'per_object.npz', allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        with np.load(ROOT / 'outputs_attempt2' / dataset / 'per_object.npz', allow_pickle=False) as z:
            assert set(z.files) == set(a)
            for key, value in a.items():
                np.testing.assert_array_equal(z[key], value)
        source = OLD / 'remote_exports' / (dataset + '_full_attempt1')
        raw = [json.loads(x) for x in (source / 'objects.jsonl').read_text(encoding='utf-8').splitlines() if x]
        indices = []
        for i, row in enumerate(raw):
            if row['is_background'] or row['paired_gt_iou'] is None or row['paired_gt_iou'] < .5 or not row['anchor_has_reference_candidate']:
                continue
            center = np.array(row['anchor_center']); b = np.array(row['gt_box_input'])
            d = np.concatenate((center - b[:2], b[2:] - center)) / row['anchor_stride']
            if np.isfinite(d).all() and np.all(d >= 0) and np.all(d <= 14.99):
                indices.append(i)
        np.testing.assert_array_equal(a['source_row'], indices)
        np.testing.assert_array_equal(a['object_id'], [raw[i]['object_id'] for i in indices])
        prior_roster = readcsv(OLD / 'feature_review/localization_readout_v1/outputs' / dataset / 'cohort_and_donors.csv')
        assert indices == [int(r['source_row']) for r in prior_roster]
        rows = [raw[i] for i in indices]
        c = np.array([r['anchor_center'] for r in rows]); s = np.array([r['anchor_stride'] for r in rows])
        gt = np.array([r['gt_box_input'] for r in rows]); irgt = np.array([r['ir_gt_box_input'] for r in rows])
        split = np.array([r['split'] for r in rows])
        for key, expected in [('center', c), ('stride', s), ('rgb_gt', gt), ('ir_gt', irgt), ('split', split)]:
            np.testing.assert_array_equal(a[key], expected)
        with np.load(source / 'logits.npz', allow_pickle=False) as z:
            zn = z['N42_dfl'][indices].astype(float); zt = z['T42_dfl'][indices].astype(float)
            ncls = z['N42_cls'][indices].astype(float); tcls = z['T42_cls'][indices].astype(float)
        pn, ln = probs(zn, 1); pn2, ln2 = probs(zn, 2); pt, lt = probs(zt, 1); pt2, lt2 = probs(zt, 2)
        dn = pn @ np.arange(16); dt = pt @ np.arange(16)
        bn = boxes(dn, c, s); bt = boxes(dt, c, s)
        niou = overlap(bn, gt)
        dist = np.concatenate((c - gt[:, :2], gt[:, 2:] - c), axis=1) / s[:, None]
        qgt = np.maximum(1 - np.abs(np.arange(16)[None, None] - dist[:, :, None]), 0)
        np.testing.assert_allclose(qgt.sum(axis=2), 1, atol=2e-15, rtol=0)
        cn = -(qgt * ln).sum(axis=2).mean(axis=1)
        correct = ((ncls.argmax(axis=1) == a['class']) & (tcls.argmax(axis=1) == a['class']) &
                   (ncls.max(axis=1) >= np.log(.25 / .75)) & (tcls.max(axis=1) >= np.log(.25 / .75)))
        def gate(tbox):
            rgb = overlap(tbox, gt); infrared = overlap(tbox, irgt)
            return correct & (niou < .7) & (rgb >= .6) & (infrared >= .5) & ((rgb - niou) > .05)
        fixed = {'base': np.ones(len(indices), dtype=bool), 'n_iou_lt_070': niou < .7, 'quality_gate070': gate(bt)}
        for key, m in fixed.items():
            np.testing.assert_array_equal(a['fixed_' + key], m)
        expected_shifts = [(0, 0)] + [(x, y) for amp in (1, 2, 4) for x, y in ((amp, 0), (-amp, 0), (0, amp), (0, -amp), (amp, amp), (amp, -amp), (-amp, amp), (-amp, -amp))]
        np.testing.assert_array_equal(a['shifts_xy_input_px'], expected_shifts)
        errors = {}
        def compare(key, actual, expected):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11, equal_nan=True, err_msg=key)
            finite = np.isfinite(expected)
            error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.
            errors[key] = max(errors.get(key, 0.), error)
        compare('native_iou', a['native_iou'], niou)
        compare('native_gt_ce', a['native_gt_ce'], cn)
        with np.load(OLD / 'feature_review/localization_readout_v1/outputs' / dataset / 'predictions.npz', allow_pickle=False) as p:
            compare('prior_native_distance', p['native_N_dfl_expectation_distance'], dn)
        for j, (dx, dy) in enumerate(expected_shifts):
            delta = np.array([-dx, -dy, dx, dy])[None] / s[:, None]
            translated = bt + [dx, dy, dx, dy]
            if j == 0:
                p1, l1, p2, l2 = pt, lt, pt2, lt2
                u1 = o1 = u2 = o2 = np.zeros((len(indices), 4))
            else:
                p1, u1, o1 = transported(pt, delta); p2, u2, o2 = transported(pt2, delta)
                with np.errstate(divide='ignore'):
                    l1 = np.log(p1); l2 = np.log(p2)
            ce = np.zeros_like(qgt); mask = qgt > 0; ce[mask] = -qgt[mask] * l1[mask]
            ce = ce.sum(axis=2).mean(axis=1)
            kl = np.zeros_like(p2); mask = p2 > 0; kl[mask] = p2[mask] * (l2[mask] - ln2[mask])
            kl = kl.sum(axis=2).mean(axis=1) * 4
            gk = ((pn2 - p2) / 2).reshape(len(indices), 64); gg = ((pn - qgt) / 4).reshape(len(indices), 64)
            kn = np.linalg.norm(gk, axis=1); gn = np.linalg.norm(gg, axis=1); dot = (gk * gg).sum(axis=1)
            cos = np.divide(dot, kn * gn, out=np.full(len(indices), np.nan), where=kn * gn > 1e-14)
            ratio = np.divide(kn, gn, out=np.full(len(indices), np.nan), where=gn > 1e-14)
            approx = boxes(p1 @ np.arange(16), c, s)
            expected = {'teacher_rgb_iou': overlap(translated, gt), 'teacher_ir_iou': overlap(translated, irgt),
                'iou_advantage': overlap(translated, gt) - niou, 'native_ce': cn, 'teacher_ce': ce,
                'teacher_minus_native_ce': ce - cn, 'kl_t2_mean4': kl, 'cosine': cos, 'gradient_dot': dot,
                'kd_gradient_norm': kn, 'gt_gradient_norm': gn, 'gradient_norm_ratio': ratio,
                'lost_mass_t1_mean4': (u1 + o1).mean(axis=1), 'lost_mass_t1_max_side': (u1 + o1).max(axis=1),
                'lost_mass_t2_mean4': (u2 + o2).mean(axis=1), 'lost_mass_t2_max_side': (u2 + o2).max(axis=1),
                'conditional_vs_exact_box_max_abs_px': np.abs(approx - translated).max(axis=1),
                'conditional_box_rgb_iou': overlap(approx, gt), 'underflow_t1': u1, 'overflow_t1': o1,
                'underflow_t2': u2, 'overflow_t2': o2}
            for key, value in expected.items():
                compare(key, a[key][:, j], value)
            np.testing.assert_array_equal(a['gate_after'][:, j], gate(translated))
            np.testing.assert_array_equal(a['teacher_support'][:, j], ((dt + delta >= 0) & (dt + delta <= 14.99)).all(axis=1))
        wrong = []
        for r in readcsv(folder / 'worst_case.csv'):
            amp = int(r['linf_px']); ix = np.flatnonzero(np.max(np.abs(a['shifts_xy_input_px']), axis=1) == amp)
            m = fixed[r['fixed_stratum']] & (split == r['split'])
            actual = int(a['gate_after'][m][:, ix].all(axis=1).sum())
            saved = int(r['all_directions_gate_n'])
            if saved != actual:
                wrong.append({'split': r['split'], 'stratum': r['fixed_stratum'], 'amplitude': amp, 'saved_gate_n': saved, 'actual_gate_success_n': actual, 'denominator': int(m.sum())})
            assert abs(float(r['all_directions_gate_fraction']) - actual / m.sum()) <= 1e-15
        for r in readcsv(ROOT / 'outputs_attempt2' / dataset / 'worst_case.csv'):
            amp = int(r['linf_px']); ix = np.flatnonzero(np.max(np.abs(a['shifts_xy_input_px']), axis=1) == amp)
            m = fixed[r['fixed_stratum']] & (split == r['split'])
            for kind, key in [('gate', 'gate_after'), ('teacher_support', 'teacher_support')]:
                expected_n = int(a[key][m][:, ix].all(axis=1).sum())
                assert expected_n == int(r['all_directions_' + kind + '_success_n'])
                assert m.sum() == int(r['all_directions_' + kind + '_denominator_n'])
                assert 'all_directions_' + kind + '_n' not in r
        contextual = []
        for sp in ('train', 'val'):
            for name, f in fixed.items():
                m = f & (split == sp)
                record = {'split': sp, 'stratum': name, 'n': int(m.sum()), 'images': len(set(a['image'][m])), 'groups': sorted(set(a['source_group'][m])), 'stride_counts': {str(x): int((m & (s == x)).sum()) for x in np.unique(s[m])}}
                if name == 'quality_gate070':
                    record['gate_survival_all8'] = {str(amp): int(a['gate_after'][m][:, np.max(np.abs(a['shifts_xy_input_px']), axis=1) == amp].all(axis=1).sum()) for amp in (1, 2, 4)}
                contextual.append(record)
        report['datasets'][dataset] = {'cohort_n': len(indices), 'source_row_object_id_gt_split_exact': True,
            'all_25_raw_mass_metrics_reconstructed': True, 'independent_max_abs_errors': errors,
            'attempt2_arrays_exact_vs_attempt1': True, 'attempt2_fixed_success_and_denominator_counts_exact': True,
            'attempt1_worst_case_gate_count_defects': wrong, 'contexts': contextual,
            'source_model_paths': json.loads((source / 'summary.json').read_text(encoding='utf-8'))['models']}
        report['reporting_defects'] += [{'dataset': dataset, **r} for r in wrong]
    after = {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in watched}
    report['watched_original_inputs_changed_during_review'] = [p for p in before if before[p] != after[p]]
    report['status'] = 'PASS_ATTEMPT2_RAW_METRICS_AND_CORRECTED_COUNTS'
    save('input_size_mtime_manifest.json', before)
    save('independent_cpu_receipt.json', report)
    print(json.dumps({'status': report['status'], 'reporting_defects': len(report['reporting_defects']), 'datasets': list(report['datasets']), 'watched_changes': report['watched_original_inputs_changed_during_review']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
