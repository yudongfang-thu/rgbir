"""Descriptive tables from completed accepted C0/N object-analysis artifacts.

Reuses the accepted independent analyzer's percent conversion and sample-SD
arithmetic; does not construct accepted endpoints, sign an AP bridge, or upgrade
claims. Raw posthoc per-class AP is read from the CLI's input-evidence copies.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

SEEDS=(0,42,123)
AP_KEYS=('mAP50_95','AP50','AP75')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(args):
    sys.path.insert(0,str(args.analyzer_root))
    spec=importlib.util.spec_from_file_location('independent_descriptive_statistics',args.analyzer_root/'analyze_independent.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    if not analyzer.verify_analyzer_acceptance(args.accepted_analyzer):raise ValueError('Independent arithmetic source acceptance is stale')
    rows=[];classes=None
    for seed in SEEDS:
        folder=args.inputs/f'seed{seed}';receipt=read(folder/'analysis_receipt.json');summary=read(folder/'pair_summary.json')
        if (receipt.get('status')!='COMPLETED' or receipt.get('analyzer_acceptance')!='ACCEPTED'
            or summary.get('analyzer_status')!='ACCEPTED' or summary.get('contract')!=analyzer.ERROR_CONTRACT):
            raise ValueError('Require actually completed accepted fixed-threshold analysis')
        metrics={role:read(folder/'input_evidence'/role/'000_evaluation_val.json') for role in ('baseline','candidate')}
        for role,actual in (('baseline','weight0'),('candidate','paired')):
            metric=metrics[role]
            if (metric.get('seed')!=seed or metric.get('arm')!=actual or metric.get('source')!='paired'
                or metric.get('evaluation_kind')!='posthoc_legacy_checkpoint_diagnostics'
                or metric.get('historical_five_metrics_exact') is not True or metric.get('official_test_accessed') is not False):
                raise ValueError('Wrong actual old endpoint identity')
        inventory={str(row['class_id']):row['name'] for row in metrics['baseline']['per_class']}
        other={str(row['class_id']):row['name'] for row in metrics['candidate']['per_class']}
        if set(inventory)!={'0','1','2','3','4'} or inventory!=other or classes is not None and classes!=inventory:
            raise ValueError('Class inventory differs')
        classes=inventory
        converted={role:{str(row['class_id']):{key:analyzer.to_percent(row[key],metric['metric_units']) for key in AP_KEYS}
            for row in metric['per_class']} for role,metric in metrics.items()}
        s=summary['summary']
        rows.append(dict(seed=seed,summary=s,baseline=summary['baseline'],candidate=summary['candidate'],
            net_correct=s['repaired']-s['damaged'],correct_fraction_delta_pp=100*(s['candidate_correct']-s['baseline_correct'])/s['gt_objects'],
            per_class_ap=converted,object_groups=summary['object_groups'],background_groups=summary['background_groups'],
            source_summary=str(folder/'pair_summary.json'),source_receipt=str(folder/'analysis_receipt.json')))
    def stats(values):
        if len(values)!=3:raise ValueError('Three paired seeds required')
        return dict(analyzer.summarize(values),values_by_seed={str(seed):v for seed,v in zip(SEEDS,values)})
    if any(row['summary']['gt_objects']!=rows[0]['summary']['gt_objects'] for row in rows):raise ValueError('GT population differs across seeds')
    overall={key:stats([row['summary'][key] for row in rows]) for key in
        ('baseline_correct','candidate_correct','repaired','damaged','baseline_incorrect')}
    for key in ('repair_rate','damage_rate'):
        overall[key+'_percent']=stats([100*row['summary'][key] for row in rows])
    overall['net_correct']=stats([row['net_correct'] for row in rows])
    overall['correct_fraction_delta_pp']=stats([row['correct_fraction_delta_pp'] for row in rows])
    overall['background_delta_per_image']=stats([row['candidate']['background_fp_per_image']-row['baseline']['background_fp_per_image'] for row in rows])
    class_table={}
    for ci,name in classes.items():
        table=dict(name=name,gt_objects=rows[0]['object_groups']['class_id'][ci]['gt_objects'],metrics={})
        for metric in AP_KEYS:
            a=[row['per_class_ap']['baseline'][ci][metric] for row in rows]
            b=[row['per_class_ap']['candidate'][ci][metric] for row in rows]
            table['metrics'][metric]=dict(N_percent=stats(a),C0_percent=stats(b),delta_pp=stats([y-x for x,y in zip(a,b)]))
        table['mAP_decreases_all_three_seeds']=all(x<0 for x in table['metrics']['mAP50_95']['delta_pp']['values_by_seed'].values())
        class_table[ci]=table
    group_tables={}
    for group_key in ('class_id','scale','source_group','brightness_bin'):
        keys=set(rows[0]['object_groups'][group_key])
        if any(set(row['object_groups'][group_key])!=keys for row in rows):raise ValueError('Fixed metadata groups differ across seeds')
        group_tables[group_key]={}
        for group in sorted(keys):
            values=[row['object_groups'][group_key][group] for row in rows]
            if any(v['gt_objects']!=values[0]['gt_objects'] for v in values):raise ValueError('Group GT denominator differs')
            group_tables[group_key][group]=dict(gt_objects=values[0]['gt_objects'],
                repaired=stats([v['repaired'] for v in values]),damaged=stats([v['damaged'] for v in values]),
                net_correct=stats([v['repaired']-v['damaged'] for v in values]),
                per_seed={str(seed):value for seed,value in zip(SEEDS,values)})
    review_flags=dict(classes_with_map_decrease_all_three=[ci for ci,t in class_table.items() if t['mAP_decreases_all_three_seeds']],
        background_increase_all_three=all(v>0 for v in overall['background_delta_per_image']['values_by_seed'].values()),
        meaning='Descriptive review signals only; no formal bridge/claim acceptance and no training early stop')
    result=dict(scope='Old C0 versus N supplementary dev diagnosis; not C1',seeds=list(SEEDS),
        full_dev_images=1469,gt_objects=rows[0]['summary']['gt_objects'],class_names=classes,overall=overall,
        per_class=class_table,object_groups=group_tables,raw_per_seed=rows,review_signals=review_flags,
        claim_upgrade=False,ap_bridge_accepted=False,stop_training=False,
        statistics='Mean and sample SD (ddof=1) across exactly three paired seeds; no pooling of rates across seeds',
        statistics_source=str(args.analyzer_root/'analyze_independent.py'))
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'summary_tables.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'DESCRIPTIVE_TABLES_WRITTEN','output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','analyzer-root','accepted-analyzer','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
