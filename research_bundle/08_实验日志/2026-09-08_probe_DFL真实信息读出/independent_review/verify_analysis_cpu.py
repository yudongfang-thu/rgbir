"""Independent stdlib recomputation of all saved distribution/GT descriptors and summaries."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def flatten(v):
    for x in v:
        if isinstance(x,list):yield from flatten(x)
        else:yield x
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k,v in a.items():compare(v,b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+'/'+str(i))
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(path,a,b)
    else:assert a==b,(path,a,b)
def describe(values):
    a=list(flatten(values))
    return dict(n=len(a),mean=statistics.mean(a) if a else None,median=statistics.median(a) if a else None,
                minimum=min(a) if a else None,maximum=max(a) if a else None)
def metrics(logits):
    out=dict(probabilities=[],log_probabilities=[],expectation_bin=[],entropy_nat=[],variance_bin2=[])
    for side in logits:
        m=max(side);z=math.log(sum(math.exp(v-m) for v in side));lp=[v-m-z for v in side];p=[math.exp(v) for v in lp]
        mean=sum(i*v for i,v in enumerate(p));var=sum(v*(i-mean)**2 for i,v in enumerate(p));entropy=-sum(v*l for v,l in zip(p,lp))
        for k,v in zip(out,[p,lp,mean,entropy,var]):out[k].append(v)
    return out
def gt_metrics(desc,dist):
    edges=[]
    for i,d in enumerate(dist):
        reason='nonfinite_GT_distance' if not math.isfinite(d) else 'negative_GT_distance' if d<0 else 'GT_distance_ge_15' if d>=15 else None
        if reason:
            edges.append(dict(valid=False,reason=reason,DFL_CE_nat=None,expectation_minus_GT_bin=None,absolute_expectation_error_bin=None));continue
        lo=math.floor(d);hi=lo+1;wl=hi-d;wr=d-lo;ce=-wl*desc['log_probabilities'][i][lo]-wr*desc['log_probabilities'][i][hi]
        err=desc['expectation_bin'][i]-d
        edges.append(dict(valid=True,reason=None,DFL_CE_nat=ce,expectation_minus_GT_bin=err,absolute_expectation_error_bin=abs(err),left_bin=lo,right_bin=hi,left_weight=wl,right_weight=wr))
    valid=all(e['valid'] for e in edges)
    return dict(edges=edges,all_four_edges_valid=valid,four_edge_mean_DFL_CE_nat=sum(e['DFL_CE_nat'] for e in edges)/4 if valid else None,
                four_edge_mean_absolute_expectation_error_bin=sum(e['absolute_expectation_error_bin'] for e in edges)/4 if valid else None,
                target_clamped=False,training_loss_reconstructed=False)
def summarize(rr,total):
    present=[r for r in rr if r['distribution_id'] is not None];valid=[r for r in present if r['GT_readout']['all_four_edges_valid']]
    ve=[e for r in present for e in r['GT_readout']['edges'] if e['valid']]
    return dict(GT_objects=total,anchor_present=len(present),anchor_missing=total-len(present),unique_distribution_ids=len({r['distribution_id'] for r in present}),
                all_four_GT_edges_valid_objects=len(valid),invalid_GT_objects=len(present)-len(valid),valid_GT_edges=len(ve),invalid_GT_edges=4*len(present)-len(ve),
                invalid_reasons=dict(Counter(e['reason'] for r in present for e in r['GT_readout']['edges'] if not e['valid'])),
                entropy_nat_all_present_edges=describe([r['distribution_metrics']['entropy_nat'] for r in present]),
                variance_bin2_all_present_edges=describe([r['distribution_metrics']['variance_bin2'] for r in present]),
                expectation_bin_all_present_edges=describe([r['distribution_metrics']['expectation_bin'] for r in present]),
                DFL_CE_nat_valid_edges=describe([e['DFL_CE_nat'] for e in ve]),mean4_DFL_CE_nat_valid_objects=describe([r['GT_readout']['four_edge_mean_DFL_CE_nat'] for r in valid]),
                mean4_absolute_mean_error_bin_valid_objects=describe([r['GT_readout']['four_edge_mean_absolute_expectation_error_bin'] for r in valid]))

def run(a):
    obj=rows(a.probe/'objects.jsonl');raw=rows(a.probe/'anchor_distributions.jsonl');contract=read(a.probe/'dfl_contract.json')
    actual=rows(a.analysis/'object_role_readout.jsonl');dd=rows(a.analysis/'distribution_readout.jsonl');summary=read(a.analysis/'summary.json')
    assert len(actual)==720 and len(dd)==len(raw)==317
    objects={o['stable_rgb_gt_id']:o for o in obj};descs={d['distribution_id']:metrics(d['raw_logits']) for d in raw};fid=contract['current_forward_id']
    assert len({(r['stable_rgb_gt_id'],r['model'],r['role']) for r in actual})==720
    compared=0
    for d in dd:
        assert d['current_forward_id']==fid;compare(descs[d['distribution_id']],d['metrics'],'distribution/'+d['distribution_id']);compared+=1
    expected=[]
    for row in actual:
        o=objects[row['stable_rgb_gt_id']];m=row['model'];role=row['role'];ref=o['roles'][m][role]
        assert row['current_forward_id']==fid
        for k in ['stable_ir_gt_id','frame_id','image_index','bucket','C_selected','historical_L2_gates']:assert row[k]==o[k],k
        if ref is None:
            assert row['distribution_id'] is None and row['distribution_metrics'] is None and row['GT_readout'] is None
            expected.append(dict(row));continue
        assert row['distribution_id']==ref['distribution_id'] and row['own_GT']==ref
        desc=descs[ref['distribution_id']];gt=gt_metrics(desc,ref['unclamped_gt_distance_bins'])
        compare(desc,row['distribution_metrics'],'role descriptors');compare(gt,row['GT_readout'],'role GT')
        expected.append(dict(row,distribution_metrics=desc,GT_readout=gt))
    groups={'all80':obj,'both05_onlyT075':[o for o in obj if o['bucket']=='both05_onlyT075'],'both05_onlyS075':[o for o in obj if o['bucket']=='both05_onlyS075']}
    count=0
    for name,oo in groups.items():
        ids={o['stable_rgb_gt_id'] for o in oo}
        for m,roles in contract['roles'].items():
            for role in roles:
                rr=[r for r in expected if r['stable_rgb_gt_id'] in ids and r['model']==m and r['role']==role]
                compare(summarize(rr,len(oo)),summary['by_group'][name][m][role],name+'/'+m+'/'+role);count+=1
    assert summary['current_forward_id']==fid and summary['objects']==80 and summary['unique_distributions']==317
    result=dict(status='PASS_INDEPENDENT_ALL_DERIVED_CPU',auditor='/root/dfl_readout_review',model_identity='unavailable',
                distributions_recomputed=compared,object_role_rows=720,group_model_role_summaries=count,
                formula='Independent stdlib logsumexp/moments/adjacent-bin CE, no import of analyzer or producer',
                tolerance=dict(absolute=1e-10,relative=1e-10),missing_and_invalid_stay_null=True,
                current_forward_id=fid,new_GPU=False,new_model_forward=False,new_hash_computed=False,
                audited_input_hashes='not computed; task boundary',scope='Descriptive own-GT own-anchor metrics only; no KD effect claim')
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(result['status'],compared,720,count)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
