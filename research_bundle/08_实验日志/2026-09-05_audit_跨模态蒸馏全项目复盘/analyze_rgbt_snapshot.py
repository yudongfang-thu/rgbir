"""Recompute descriptive statistics only; does not grant accepted evidence status."""
import csv
import io
import json
import statistics
from pathlib import Path

D = Path(__file__).resolve().parent
snap = json.loads((D/'rgbt_readonly_snapshot_v2.json').read_text(encoding='utf-8-sig'))
extra = json.loads((D/'rgbt_eval_and_code_snapshot.json').read_text(encoding='utf-8-sig'))
rows = []
for f in extra['files']:
    if '/eval_records/' not in f['path'] and not f['path'].endswith('metrics_record.json'): continue
    x = json.loads(f['text'])
    x['arm'] = {'h2':'h2_shuffled','h3':'h3_same_modal'}.get(x['arm'],x['arm'])
    rows.append({'source': f['path'], 'sha256': f['sha256'], 'dataset': x['dataset'],
                 'arm': x['arm'], 'seed': x['seed'], 'checkpoint': x['checkpoint'],
                 'role': x['evaluation_role'], **{k: 100*v for k,v in x['metrics'].items()},
                 'use': 'provenance_check_needed' if '_best.json' in f['path'] else 'descriptive_last'})
rows.sort(key=lambda x: (x['dataset'], x['arm'], x['seed'], x['source']))
with (D/'rgbt_eval_rows.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

def stats(xs):
    return {'n': len(xs), 'mean': statistics.mean(xs), 'sample_sd': statistics.stdev(xs) if len(xs)>1 else None}

out = {'units': 'percentage_points; AP values multiplied by 100',
       'evidence_grade': 'read_only_descriptive_reanalysis_not_accepted_analyzer',
       'arms': [], 'paired_deltas': [], 'p3_core_terminal_csv': [], 'notes': [
           'Native seed42 *_best eval uses /tmp/best_probe/last.pt and is excluded from paired multiseed comparison.',
           'Hnewa native comparisons use only shared seeds0/123; h1/h2/h3 comparisons use0/42/123.',
           'Pairing null may be harmful; paired>shuffled alone does not establish net KD usefulness.',
           'CMD adapted_v2 has external metrics_record; zero/malformed CSV metric fields must not be treated as zero detector performance.'
       ]}
for dataset in ('dronevehicle','llvip'):
    ds = [r for r in rows if r['dataset']==dataset and r['use']=='descriptive_last']
    for arm in sorted(set(r['arm'] for r in ds)):
        a = [r for r in ds if r['arm']==arm]
        out['arms'].append({'dataset':dataset,'arm':arm,'seeds':[r['seed'] for r in a],
                            'mAP50_95':stats([r['mAP50_95'] for r in a]),
                            'AP50':stats([r['AP50'] for r in a]),
                            'per_seed':[{k:r[k] for k in ('seed','AP50','mAP50_95')} for r in a]})
    pa = {r['seed']:r for r in ds if r['arm']=='h1_paired'}
    for control in ('h2_shuffled','h3_same_modal','native'):
        co = {r['seed']:r for r in ds if r['arm']==control}
        seeds = sorted(pa.keys() & co.keys())
        vals = [pa[s]['mAP50_95']-co[s]['mAP50_95'] for s in seeds]
        out['paired_deltas'].append({'dataset':dataset,'contrast':'h1_paired - '+control,
                                     'seeds':seeds, 'per_seed_mAP50_95_pp':vals,
                                     'positive_count':sum(v>0 for v in vals), **stats(vals)})

for r in snap['runs']:
    if '/rgbt_p3_causal_v1/core_seed42/' in r['path'] or '/rgbt_p3_causal_v1/formal_native/' in r['path']:
        out['p3_core_terminal_csv'].append({'path':r['path'],'epoch':r['last']['epoch'],
                                          'mAP50_95':100*float(r['last']['metrics/mAP50-95(B)'])})
(D/'rgbt_descriptive_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
