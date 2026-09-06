"""Recompute existing comparators; never label legacy receipts as new-contract runs."""
import json
import statistics
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
SNAP=HERE/'snapshots/2026-09-07T022337.461405_0800'
RAW=HERE/'comparator_snapshot'
KEYS=('epochs','imgsz','batch','nbs','workers','optimizer','lr0','lrf','momentum','weight_decay',
      'warmup_epochs','translate','scale','fliplr','mosaic','mixup','hsv_h','hsv_s','hsv_v','amp','deterministic')
METHODS=('cmdistill','historical_native','cclkd_train')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def stats(values):return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'ddof':1}

def main():
 baseline=yaml.safe_load((SNAP/'raw/N42/args.yaml').read_text(encoding='utf-8-sig'))
 summary={'methods':{},'recipe_reference':'same-code OEv1 N42 args.yaml','protocol_comparison':{},'evidence_level':'descriptive historical comparison; not same-code causal controls'}
 roster=(SNAP/'raw/N42/evaluation_val_roster.txt').read_text().splitlines()
 for method in METHODS:
  rows=[]
  for seed in (0,42,123):
   folder=RAW/method/f's{seed}'
   completion=read(folder/'completion_receipt.json')
   args=yaml.safe_load((folder/'args.yaml').read_text(encoding='utf-8-sig'))
   if method=='cclkd_train':
    evaluation=read(RAW/'cclkd_eval'/f's{seed}'/'evaluation_val.json')
    receipt=read(RAW/'cclkd_eval'/f's{seed}'/'eval_evidence/run_receipt.json')
    assert receipt['terminal_status']=='COMPLETED' and receipt['seed']==seed
    assert evaluation['metric_units']=='fraction_0_to_1'
    bound=read(RAW/'cclkd_eval'/f's{seed}'/'eval_evidence'/receipt['metric_snapshots'][0])
    assert bound==evaluation
    eval_rosters=receipt['source_snapshots']['split_roster']
    eval_roster=(RAW/'cclkd_eval'/f's{seed}'/'eval_evidence'/eval_rosters[0]).read_text().splitlines()
    roster_same=eval_roster==roster
    metrics={key:evaluation[key]*100 for key in ('mAP50_95','AP50','AP75')}
   else:
    evaluation=read(folder/'metrics_record.json')
    assert evaluation['seed']==seed and evaluation['split']=='val' and evaluation['checkpoint'].endswith('/weights/last.pt')
    metrics={key:evaluation['metrics'][key]*100 for key in ('mAP50_95','AP50','AP75')}
    roster_same=None
   assert completion['seed']==seed
   if method!='historical_native':assert completion['status']=='completed'
   assert args['epochs']==200
   rows.append({'seed':seed,'metrics_percent':metrics,'ordered_val_roster_equals_oev1':roster_same,
                'recipe_difference':{key:{'oev1':baseline.get(key),'comparator':args.get(key)} for key in KEYS if baseline.get(key)!=args.get(key)},
                'initial_model':args.get('model'),'dataset_yaml':args.get('data'),
                'legacy_completion_status':completion.get('status'),
                'completion_limitation':'legacy native receipt has no terminal status; not admitted by new analyzer' if method=='historical_native' else None})
  summary['methods'][method]={'rows':rows,'stats_percent':{key:stats([row['metrics_percent'][key] for row in rows]) for key in ('mAP50_95','AP50','AP75')}}
 native=summary['methods']['historical_native']['rows']
 for method in ('cmdistill','cclkd_train'):
  rows=summary['methods'][method]['rows']
  differences=[{'seed':a['seed'],'delta_pp':a['metrics_percent']['mAP50_95']-b['metrics_percent']['mAP50_95']} for a,b in zip(rows,native)]
  summary['methods'][method]['minus_historical_native']={'rows':differences,'stats':stats([row['delta_pp'] for row in differences])}
 out=HERE/'comparator_analysis.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
