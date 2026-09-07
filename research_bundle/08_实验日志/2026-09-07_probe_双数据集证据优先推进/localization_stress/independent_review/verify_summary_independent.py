"""Recompute every emitted continuous statistic and worst-case vector on CPU."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'outputs_attempt2'
checks = {}
for dataset in ('llvip', 'dronevehicle'):
    with np.load(OUT / dataset / 'per_object.npz', allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    comparison_n = 0
    max_error = 0.
    def compare(actual, expected):
        global comparison_n, max_error
        comparison_n += 1
        if expected is None:
            assert actual == ''
        else:
            error = abs(float(actual) - float(expected))
            assert error <= 2e-12, (actual, expected)
            max_error = max(max_error, error)
    for filename in ('metrics.csv', 'worst_case.csv'):
        with (OUT / dataset / filename).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            mask = a['fixed_' + row['fixed_stratum']] & (a['split'] == row['split'])
            compare(row['fixed_n'], mask.sum())
            for field in row:
                if not field.endswith('_mean'):
                    continue
                metric = field[:-5]
                if filename == 'metrics.csv':
                    value = a[metric][mask, int(row['condition_index'])]
                else:
                    amplitude = int(row['linf_px'])
                    key = 'worst_' + str(amplitude) + '_' + metric
                    value = a[key][mask]
                    columns = np.max(np.abs(a['shifts_xy_input_px']), axis=1) == amplitude
                    operation, raw_key = {'min_iou_advantage': (np.min, 'iou_advantage'),
                        'max_teacher_minus_native_ce': (np.max, 'teacher_minus_native_ce'),
                        'min_cosine': (np.min, 'cosine'), 'max_lost_mass_t1_mean4': (np.max, 'lost_mass_t1_mean4'),
                        'max_lost_mass_t2_mean4': (np.max, 'lost_mass_t2_mean4')}[metric]
                    np.testing.assert_array_equal(value, operation(a[raw_key][mask][:, columns], axis=1))
                finite = value[np.isfinite(value)]
                expected = {'n': len(value), 'finite_n': len(finite), 'nan_n': np.isnan(value).sum(),
                    'posinf_n': np.isposinf(value).sum(), 'neginf_n': np.isneginf(value).sum()}
                for suffix, function in [('mean', np.mean), ('sd_population', np.std), ('min', np.min), ('max', np.max)]:
                    expected[suffix] = function(finite) if len(finite) else None
                for suffix, q in [('p05', .05), ('p50', .5), ('p95', .95)]:
                    expected[suffix] = np.quantile(finite, q) if len(finite) else None
                for suffix, v in expected.items():
                    compare(row[metric + '_' + suffix], v)
                scope = 'all_objects' if len(finite) == len(value) else 'finite_only_with_nonfinite_counts'
                assert row[metric + '_mean_scope'] == scope
    checks[dataset] = {'all_emitted_continuous_statistics_recomputed': True,
                       'all_emitted_worst_continuous_vectors_recomputed': True,
                       'numeric_comparisons': comparison_n, 'maximum_absolute_error': max_error}
receipt = {'status': 'PASSED', 'cpu_only': True, 'output_attempt': 2, 'checks': checks}
(HERE / 'independent_summary_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
