"""Read saved endpoint records; no model execution or new metric estimation."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / '08_实验日志/2026-09-08_audit_项目与94最新全景/independent_outcomes/recomputed_results.json'
records = json.loads(SOURCE.read_text(encoding='utf-8'))
names = {'weight0': 'N', 'paired': 'C0', 'paired_random': 'random'}
seeds = (0, 42, 123)
metrics = ('mAP50_95', 'AP50', 'AP75')
arms = {name: {} for name in names.values()}
evidence = []
for run in records['runs']:
    ev, completion = run.get('evaluation'), run.get('completion')
    if not ev or ev.get('arm') not in names:
        continue
    assert completion['status'] == 'training_completed'
    assert completion['last_epoch'] == 200
    assert ev['endpoint'] == 'fixed_budget_last_ema'
    assert ev['metric_units'] == 'fraction_0_to_1'
    assert ev['official_test_accessed'] is False
    name, seed = names[ev['arm']], ev['seed']
    assert seed not in arms[name], 'duplicate endpoint'
    arms[name][seed] = {metric: 100 * ev[metric] for metric in metrics}
    evidence.append({'arm': name, 'seed': seed, 'source_record': run['run'],
                     'checkpoint_path': ev['checkpoint'], 'endpoint': ev['endpoint']})
assert all(set(rows) == set(seeds) for rows in arms.values())
assert statistics.stdev([1, 2, 3]) == 1
def aggregate(values):
    return {'values_seed0_42_123': values, 'mean': statistics.mean(values),
            'sample_sd_ddof1': statistics.stdev(values),
            'positive_count': sum(value > 0 for value in values)}
arm_tables = {arm: {metric: aggregate([rows[s][metric] for s in seeds])
                   for metric in metrics} for arm, rows in arms.items()}
comparisons = {}
for left, right in (('C0', 'N'), ('C0', 'random'), ('random', 'N')):
    comparisons[f'{left}-{right}'] = {
        metric: aggregate([arms[left][s][metric] - arms[right][s][metric] for s in seeds])
        for metric in metrics}
assert math.isclose(comparisons['C0-N']['mAP50_95']['mean'], 0.266655,
                    abs_tol=0.0000005)
result = {'status': 'ARITHMETIC_RECHECK_PASS', 'source': str(SOURCE),
          'scope': 'saved accepted endpoint arithmetic; no AP recomputation',
          'raw_metric_units': 'fraction_0_to_1', 'display_units': 'percent; deltas pp',
          'seed_order': seeds, 'evidence': evidence,
          'arms': arm_tables, 'paired_comparisons': comparisons}
destination = OUT / 'paired_statistics.json'
assert not destination.exists(), 'keep earlier output immutable'
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'comparisons': comparisons}, ensure_ascii=True))
