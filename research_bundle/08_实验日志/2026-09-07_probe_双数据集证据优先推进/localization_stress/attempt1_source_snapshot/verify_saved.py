"""Check saved frozen output against prior per-object records and table reducers."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parents[1]/'2026-09-07_probe_Baseline蒸馏机会重诊断'
OUT=HERE/'outputs_attempt1'
checks={}
for dataset in ('llvip','dronevehicle'):
    with np.load(OUT/dataset/'per_object.npz',allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    summary=json.loads((OUT/dataset/'summary.json').read_text(encoding='utf-8'))
    source=Path(summary['source'])
    for name,stat in summary['source_files'].items():
        current=(source/name).stat()
        assert current.st_size==stat['bytes'] and current.st_mtime_ns==stat['mtime_ns']
    with (PRIOR/'new_probe_analysis'/(dataset+'_analysis_v1')/'dfl_per_object.csv').open(encoding='utf-8-sig',newline='') as handle:
        old={int(r['row']):r for r in csv.DictReader(handle)}
    deltas={}
    for key,oldkey in [('teacher_ce','teacher_gt_ce'),('native_ce','native_gt_ce'),('kl_t2_mean4','kl_T2'),('cosine','gradient_cosine')]:
        expected=np.array([float(old[int(i)][oldkey]) for i in a['source_row']])
        actual=a[key][:,0]
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0,equal_nan=True)
        deltas[key]=float(np.nanmax(np.abs(actual-expected)))
    assert a['teacher_support'].shape==(len(a['source_row']),25)
    assert a['underflow_t1'].shape==(len(a['source_row']),25,4)
    assert np.array_equal(a['gate_after'][:,0],a['fixed_quality_gate070'])
    for key in ('underflow_t1','overflow_t1','underflow_t2','overflow_t2'):
        assert not a[key][:,0].any()
        assert (a[key]>=0).all() and (a[key]<=1).all()
    with (OUT/dataset/'metrics.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    max_summary_error=0.
    nonfinite={k:int((~np.isfinite(a[k])).sum()) for k in ('teacher_ce','native_ce','kl_t2_mean4','cosine')}
    for r in rows:
        j=int(r['condition_index'])
        m=a['fixed_'+r['fixed_stratum']] & (a['split']==r['split'])
        assert int(m.sum())==int(r['fixed_n'])
        for key in ('teacher_ce','iou_advantage','cosine','lost_mass_t1_mean4','conditional_vs_exact_box_max_abs_px'):
            y=a[key][m,j]; finite=y[np.isfinite(y)]
            if len(finite):
                error=abs(float(finite.mean())-float(r[key+'_mean']))
                max_summary_error=max(max_summary_error,error)
                assert error<=1e-12
        original=m & a['fixed_quality_gate070']
        assert int((original & a['gate_after'][:,j]).sum())==int(r['original_quality_gate_survived_n'])
    checks[dataset]={'status':'PASSED','prior_saved_zero_rowwise_max_abs':deltas,'all_saved_table_reducer_max_abs':max_summary_error,
                     'source_size_mtime_unchanged':True,'nonfinite_arrays':nonfinite,'fixed_masks_and_all_25_conditions_verified':True}
receipt={'status':'PASSED_SELF_CHECK_NOT_INDEPENDENT_ACCEPTANCE','datasets':checks}
(HERE/'saved_output_self_check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
