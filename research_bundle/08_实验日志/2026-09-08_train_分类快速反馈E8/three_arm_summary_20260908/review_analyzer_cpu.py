"""Independent known-value checks only; never call analyzer.main()."""
import copy
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).parent
SOURCE=HERE/'analyze_e8.py'
before=SOURCE.read_bytes()
spec=importlib.util.spec_from_file_location('e8_descriptive_under_review',SOURCE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
author=m.self_test()
base=[.1,.2,.3,.4,.5]
offsets={'N':[0]*5,'C0':[.04,-.02,.03,0,.05],'C1':[.02,0,0,-.03,.01]}
fixture={}
for arm in m.ARMS:
    vals=[v+d for v,d in zip(base,offsets[arm])]
    fixture[arm]=dict(arm=arm,status='SHORT_SCREEN_EVALUATION_COMPLETED',scope='SHORT_SCREEN',single_seed=True,metric_units='fraction_0_to_1',seed=42,epochs=8,independent_lr_horizon=8,endpoint='SHORT_SCREEN_E8_LAST_EMA',dataset='dronevehicle',full_dev_images=1469,full_dev_gt_objects=22462,native_profile_binding='synthetic_binding',formal_e200_complete=False,formal_paper_gain_claim=False,official_test_accessed=False,**{k:sum(vals)/5 for k in m.METRICS},per_class=[dict(class_id=i,name=str(i),**{k:v for k in m.AP}) for i,v in enumerate(vals)])
out=m.analyze(fixture)
close=lambda x,y:math.isclose(x,y,rel_tol=0,abs_tol=1e-12)
checks={}
checks['nonuniform_macro_percent']=all(close(out['values'][a]['percent']['mAP50_95'],v) for a,v in [('N',30),('C0',32),('C1',30)])
checks['nonuniform_signed_pair_pp']=all(close(out['differences'][p]['pp']['mAP50_95'],v) for p,v in [('C1-N',0),('C1-C0',-2),('C0-N',2)])
checks['per_class_pp_alignment']=all(close(r['mAP50_95'],v) for r,v in zip(out['differences']['C1-N']['per_class_pp'],[2,0,0,-3,1]))
checks['macro_delta_closes_class_delta']=all(close(sum(r['mAP50_95'] for r in d['per_class_pp'])/5,d['pp']['mAP50_95']) for d in out['differences'].values())
checks['scope_no_sd_no_gain_no_expansion']=out['scope']=='DESCRIPTIVE_SINGLE_SEED_E8_ONLY' and out['n_seeds']==1 and out['sample_sd'] is None and out['formal_gain_claim'] is False and out['automatic_expansion'] is False
checks['relative_percent_separate_from_pp']=close(out['differences']['C0-N']['relative_mAP_percent'],100/15)
zero=copy.deepcopy(fixture)
zero['N'].update({k:0. for k in m.METRICS})
for r in zero['N']['per_class']:r.update({k:0. for k in m.AP})
checks['zero_relative_denominator_null']=m.analyze(zero)['differences']['C1-N']['relative_mAP_percent'] is None
for label,mutate in [('reject_e200_claim',lambda x:x['C1'].update(formal_e200_complete=True)),('reject_paper_gain_flag',lambda x:x['C1'].update(formal_paper_gain_claim=True)),('reject_test_access',lambda x:x['C1'].update(official_test_accessed=True)),('reject_nonfinite_class',lambda x:x['C1']['per_class'][0].update(AP75=float('inf')))]:
    bad=copy.deepcopy(fixture);mutate(bad)
    try:m.analyze(bad)
    except (ValueError,KeyError,TypeError):checks[label]=True
    else:checks[label]=False
checks['source_bytes_unchanged']=SOURCE.read_bytes()==before
result=dict(status='PASS_FOR_FIXED_E8_DESCRIPTIVE_ANALYZER' if all(checks.values()) and author['status']=='PASS' else 'FAIL',author_known_truth_count=len(author['checks']),author_known_truth=author,independent_checks=checks,source_bytes=len(before),scope='Known synthetic fixtures and static review only; main() and actual endpoint analysis NOT executed.',new_hashes=False,gpu=False,formal_outputs_written=False)
with (HERE/'analyzer_review_receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=True,indent=2))
