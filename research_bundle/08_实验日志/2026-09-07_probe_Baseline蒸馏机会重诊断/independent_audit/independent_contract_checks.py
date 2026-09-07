"""Additional CPU counterexamples on a requested frozen analyzer source copy."""
from pathlib import Path
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import numpy as np

source = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('audited_probe', source)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
rng = np.random.default_rng(8293)
rows=[]
for i in range(36):
    row=dict(object_id=str(i), image=f'img_{i//3}', split='train' if i<24 else 'val',
        **{'class':i%3},is_background=i%3==2,source_group=str(i//9),scale_bin='small',
        mean_rgb_luma=float(i),paired_gt_iou=.9,gt_box_input=[0,0,32,32],
        anchor_center=[16,16],anchor_stride=8,anchor_has_reference_candidate=i%4!=0)
    for model in ('N42','T42','N0'):
        row[model]={'regions':{lv:{'valid':i not in [1,25]} for lv in ['P3','P4']}}
    rows.append(row)
logits={m+'_cls':rng.normal(size=(36,2)) for m in ('N42','T42','N0')}
logits.update({m+'_region_cls':rng.normal(size=(36,2,2)) for m in ('N42','T42','N0')})
features={m+'_'+lv:rng.normal(size=(36,6)) for m in ('N42','T42','N0') for lv in ('P3','P4','anchor_P3','anchor_P4')}
train=np.arange(36)<24
groups,_=a.group_columns(rows,train)
checks=[]
with tempfile.TemporaryDirectory(dir=source.parent) as tmp:
    p=Path(tmp);(p/'base').mkdir();(p/'perturbed').mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        first=a.probe_analysis(copy.deepcopy(rows),features,logits,groups,p/'base')
        changed={k:v.copy() for k,v in features.items()}
        for key,value in changed.items():
            if '_anchor_' not in key: value[[1,25],:]=1e9
        second=a.probe_analysis(copy.deepcopy(rows),changed,logits,groups,p/'perturbed')
    b=np.load(p/'base'/'probe_predictions.npz');c=np.load(p/'perturbed'/'probe_predictions.npz')
    valid=b['roi_common_valid'];assert valid.sum()==34
    for name,record in first['arms'].items():
        if record['cohort']=='auxiliary_gt_roi_common_valid':
            assert np.array_equal(b[name][valid],c[name][valid]),name
            assert record['train_n']==23 and record['val_n']==11
    checks.append('Invalid ROI values cannot alter valid-cohort predictions; all ROI arms use 23 train/11 val')
    for levels in ['P3','P4','P3_P4']:
        keys=['N_logits+T_anchor_feature_'+levels,'N_logits+N0_anchor_feature_'+levels,'N_logits+shuffled_T_anchor_feature_'+levels]
        dims=[first['arms'][key]['fit']['input_dim'] for key in keys]
        assert len(set(dims))==1,(keys,dims)
    checks.append('IR, independent RGB and shuffled IR fixed-anchor controls have identical projected dimensions per level')
    for key in ['roi_shuffle_source_row','anchor_shuffle_source_row']:
        ix=b[key];assert np.all(train[ix[train]]) and np.all(~train[ix[~train]])
    assert np.all(valid[b['roi_shuffle_source_row'][valid]])
    checks.append('Both donor maps stay within train/val; ROI donors additionally stay within common valid cohort')
    reference=first['arms']['N_logits']['val']['by_group']['anchor_reference']
    assert reference['annotation_background']['all']['n']+reference['native_candidate']['all']['n']+reference['gt_center_fallback']['all']['n']==12
    checks.append('Main readout reports disjoint annotation-background, reference-candidate and GT-center-fallback groups')
    x=rng.normal(size=(36,7));y=np.arange(36)%3
    pred1,fit1=a.ridge_fit_predict(x,y,train,3);x[~train]=rng.normal(1e8,100,size=x[~train].shape)
    pred2,fit2=a.ridge_fit_predict(x,y,train,3)
    assert np.array_equal(pred1[train],pred2[train]);assert np.array_equal(fit1['train_mean'],fit2['train_mean']);assert np.array_equal(fit1['train_std'],fit2['train_std'])
    checks.append('Arbitrary held-out feature shift leaves fitted training transforms and training predictions unchanged')
    b.close();c.close()
result={'status':'pass','source':str(source),'checks':checks,'n_checks':len(checks),'gpu_used':False,'hashes_computed':False}
(source.parent/'independent_contract_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
