"""CPU-only descriptive E8 comparison; no E200 selection or GPU evaluation."""
import copy
import json
import math
from pathlib import Path

ARMS = ('N', 'C0', 'C1')
METRICS = ('mAP50_95', 'AP50', 'AP75', 'precision', 'recall')
AP = METRICS[:3]
PAIRS = (('C1', 'N'), ('C1', 'C0'), ('C0', 'N'))
ENDPOINT_DIRS = {
    'N': 'endpoint_N_20260908_043812',
    'C0': 'endpoint_C0_20260908_053913',
    'C1': 'endpoint_C1_20260908_063705',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(receipts):
    require(set(receipts) == set(ARMS), 'missing or extra arm')
    common = ('seed', 'dataset', 'endpoint', 'epochs', 'independent_lr_horizon',
              'full_dev_images', 'full_dev_gt_objects', 'native_profile_binding')
    reference = receipts['N']
    classes = None
    values = {}
    for arm in ARMS:
        r = receipts[arm]
        require(r['arm'] == arm, 'arm identity')
        require(r['status'] == 'SHORT_SCREEN_EVALUATION_COMPLETED', 'incomplete evaluation')
        require(r['scope'] == 'SHORT_SCREEN' and r['single_seed'] is True, 'scope')
        require(r['metric_units'] == 'fraction_0_to_1', 'metric units')
        require(r['seed'] == 42 and r['epochs'] == 8 and r['independent_lr_horizon'] == 8, 'fixed seed/horizon')
        require(r['endpoint'] == 'SHORT_SCREEN_E8_LAST_EMA', 'fixed endpoint')
        require(r['dataset'] == 'dronevehicle', 'dataset')
        require(r['full_dev_images'] == 1469 and r['full_dev_gt_objects'] == 22462, 'full dev population')
        require(not r['formal_e200_complete'] and not r['formal_paper_gain_claim'], 'claim scope')
        require(not r['official_test_accessed'], 'sealed test')
        require(all(r[k] == reference[k] for k in common), 'unmatched protocol')
        for k in METRICS:
            require(isinstance(r[k], (int, float)) and math.isfinite(r[k]) and 0 <= r[k] <= 1, 'invalid metric')
        pc = sorted(r['per_class'], key=lambda item: item['class_id'])
        names = [(p['class_id'], p['name']) for p in pc]
        require([p['class_id'] for p in pc] == list(range(5)), 'five unique classes')
        if classes is None:
            classes = names
        require(names == classes, 'unmatched class identity')
        for k in AP:
            require(all(math.isfinite(p[k]) and 0 <= p[k] <= 1 for p in pc), 'invalid class metric')
            require(math.isclose(sum(p[k] for p in pc) / 5, r[k], abs_tol=1e-12, rel_tol=0), 'class mean mismatch')
        values[arm] = {
            'percent': {k: 100 * r[k] for k in METRICS},
            'per_class_percent': [{**{'class_id': p['class_id'], 'name': p['name']},
                                   **{k: 100 * p[k] for k in AP}} for p in pc],
        }
    differences = {}
    for a, b in PAIRS:
        ca, cb = values[a]['per_class_percent'], values[b]['per_class_percent']
        differences[a + '-' + b] = {
            'pp': {k: 100 * (receipts[a][k] - receipts[b][k]) for k in METRICS},
            'relative_mAP_percent': (100 * (receipts[a]['mAP50_95'] / receipts[b]['mAP50_95'] - 1)
                                     if receipts[b]['mAP50_95'] else None),
            'per_class_pp': [{'class_id': p['class_id'], 'name': p['name'],
                              **{k: p[k] - q[k] for k in AP}} for p, q in zip(ca, cb)],
        }
    return {'scope': 'DESCRIPTIVE_SINGLE_SEED_E8_ONLY', 'seed': 42, 'n_seeds': 1,
            'sample_sd': None, 'formal_gain_claim': False, 'automatic_expansion': False,
            'values': values, 'differences': differences}


def self_test():
    checks = {}
    fixture = {}
    for arm, v in zip(ARMS, (.2, .3, .25)):
        fixture[arm] = dict(arm=arm, status='SHORT_SCREEN_EVALUATION_COMPLETED',
            scope='SHORT_SCREEN', single_seed=True, metric_units='fraction_0_to_1',
            seed=42, epochs=8, independent_lr_horizon=8, endpoint='SHORT_SCREEN_E8_LAST_EMA',
            dataset='dronevehicle', full_dev_images=1469, full_dev_gt_objects=22462,
            native_profile_binding='known_fixture', formal_e200_complete=False,
            formal_paper_gain_claim=False, official_test_accessed=False,
            **{k: v for k in METRICS})
        fixture[arm]['per_class'] = [dict(class_id=i, name=str(i), **{k: v for k in AP}) for i in range(5)]
    out = analyze(fixture)
    checks['fraction_to_percent'] = out['values']['C0']['percent']['mAP50_95'] == 30
    checks['signed_pp'] = all(math.isclose(out['differences'][k]['pp']['mAP50_95'], v, abs_tol=1e-12)
                              for k, v in [('C1-N', 5), ('C1-C0', -5), ('C0-N', 10)])
    checks['no_single_seed_sd_or_expansion'] = out['sample_sd'] is None and not out['automatic_expansion']
    reorder = copy.deepcopy(fixture)
    reorder['C1']['per_class'].reverse()
    checks['class_id_alignment'] = analyze(reorder) == out
    def reject(label, mutate):
        bad = copy.deepcopy(fixture)
        mutate(bad)
        try:
            analyze(bad)
        except (ValueError, KeyError, TypeError):
            checks[label] = True
        else:
            checks[label] = False
    reject('missing_arm', lambda x: x.pop('C1'))
    reject('wrong_seed', lambda x: x['C1'].update(seed=0))
    reject('incomplete_receipt', lambda x: x['C1'].update(status='FAILED'))
    reject('missing_metric', lambda x: x['C1'].pop('AP75'))
    reject('nan_metric', lambda x: x['C1'].update(mAP50_95=float('nan')))
    reject('wrong_units', lambda x: x['C1'].update(metric_units='percent'))
    reject('mismatched_population', lambda x: x['C1'].update(full_dev_images=1468))
    reject('mismatched_evaluator', lambda x: x['C1'].update(native_profile_binding='other'))
    reject('class_mean_error', lambda x: x['C1']['per_class'][0].update(AP50=.9))
    reject('class_identity_error', lambda x: x['C1']['per_class'][0].update(name='other'))
    reject('duplicate_class', lambda x: x['C1']['per_class'][0].update(class_id=1))
    require(all(checks.values()), 'self-test failed: ' + repr(checks))
    return {'status': 'PASS', 'checks': checks, 'known_truth_only': True}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    import sys
    here = Path(__file__).parent
    test = self_test()
    (here / 'analyzer_self_test.json').write_text(json.dumps(test, indent=2) + '\n', encoding='utf-8')
    if '--self-test-only' in sys.argv:
        print(json.dumps(test))
        return
    receipts = {}
    sources = {}
    roster = None
    for arm, folder in ENDPOINT_DIRS.items():
        root = here.parent / folder
        review = read_json(root / 'endpoint_review_receipt.json')
        require(review['status'].startswith('PASS') and review['checks'] and all(review['checks'].values()), 'endpoint review not passed')
        train = read_json(root / 'runs' / arm / 'short_training_receipt.json')
        evpath = root / 'evaluations' / arm / 'short_evaluation_receipt.json'
        ev = read_json(evpath)
        require(train['status'] == 'SHORT_SCREEN_TRAINING_COMPLETED', 'incomplete training')
        require(train['arm'] == arm and train['seed'] == ev['seed'], 'train/eval identity')
        require(train['checkpoint'] == ev['checkpoint'], 'train/eval checkpoint mismatch')
        require(train['last_epoch'] == 8 and train['batches'] == 4504, 'training duration')
        require(train['optimizer_updates'] + train['amp_skips'] == train['attempts'], 'counter closure')
        rb = (root / 'evaluations' / arm / 'development_roster.txt').read_bytes()
        if roster is None:
            roster = rb
        require(rb == roster, 'dev roster mismatch')
        receipts[arm] = ev
        sources[arm] = str(evpath.relative_to(here.parent))
    result = analyze(receipts)
    result['sources_relative_to_experiment'] = sources
    (here / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# E8 三臂原值和差值（单 seed 描述性统计）', '',
             '百分制原值；差值单位 pp。未生成跨 seed SD、显著性结论或自动扩展决策。', '',
             '|实验|mAP50–95|AP50|AP75|Precision|Recall|', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append('|' + arm + '|' + '|'.join(f'{result["values"][arm]["percent"][k]:.6f}' for k in METRICS) + '|')
    lines += ['', '|比较|ΔmAP50–95|ΔAP50|ΔAP75|ΔPrecision|ΔRecall|', '|---|---:|---:|---:|---:|---:|']
    for pair, diff in result['differences'].items():
        lines.append('|' + pair + '|' + '|'.join(f'{diff["pp"][k]:+.6f}' for k in METRICS) + '|')
    for k in AP:
        lines += ['', '## 逐类 ' + k, '', '|类别|N|C0|C1|C1−N|C1−C0|C0−N|', '|---|---:|---:|---:|---:|---:|---:|']
        for i in range(5):
            row = result['values']['N']['per_class_percent'][i]['name']
            raw = [result['values'][a]['per_class_percent'][i][k] for a in ARMS]
            diff = [result['differences'][a + '-' + b]['per_class_pp'][i][k] for a, b in PAIRS]
            lines.append('|' + row + '|' + '|'.join(f'{v:.6f}' for v in raw) + '|' + '|'.join(f'{v:+.6f}' for v in diff) + '|')
    (here / 'COMPARISON_TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(here), 'differences': result['differences']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
