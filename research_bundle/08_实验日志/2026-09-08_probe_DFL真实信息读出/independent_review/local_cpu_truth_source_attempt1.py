"""Independent stdlib-only synthetic semantics and fixed historical identity read."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


def load_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def moment(p):
    mean=sum(i*x for i,x in enumerate(p))
    return dict(mean=mean,variance_bin2=sum(x*(i-mean)**2 for i,x in enumerate(p)),
                entropy_nats=-sum(x*math.log(x) for x in p if x))


def geometry(index):
    offset=0
    for level,(width,stride) in enumerate([(80,8),(40,16),(20,32)]):
        n=width*width
        if offset<=index<offset+n:
            row,col=divmod(index-offset,width)
            return dict(anchor_index=index,level=level,stride=stride,row=row,col=col,
                        center_xy=[(col+.5)*stride,(row+.5)*stride])
        offset+=n
    raise ValueError('anchor out of bounds')


def main(a):
    cases=[]
    def checked(name,actual,expected):
        if actual!=expected:raise AssertionError((name,actual,expected))
        cases.append(dict(name=name,status='pass',actual=actual,expected=expected))
    p=[1/16]*16
    q=[1/32]*16;q[0]+=1/4;q[15]+=1/4
    pm,qm=moment(p),moment(q)
    checked('same_mean_distinct_shape', [pm['mean'],qm['mean'],pm['variance_bin2'],qm['variance_bin2']], [7.5,7.5,21.25,38.75])
    checked('same_mean_not_equal_probabilities',p==q,False)
    # A physical box [0,0,80,80] read from two centers/strides has different bin semantics.
    checked('same_box_different_anchor_distances',[[32/8]*2+[48/8]*2,[40/16]*4],[[4,4,6,6],[2.5,2.5,2.5,2.5]])
    distances=[-0.01,0,2.25,14.99,15,15.01]
    checked('unclamped_support_vs_native_no_change_domain',
            [[0<=d<=15,0<=d<=14.99] for d in distances],
            [[False,False],[True,True],[True,True],[True,True],[True,False],[False,False]])
    # Uniform probabilities give log(16) at any legal two-bin interpolation target.
    d=2.25;tl=math.floor(d);tr=tl+1
    ce=-(tr-d)*math.log(p[tl])-(d-tl)*math.log(p[tr])
    checked('unclamped_two_bin_gt_readout_uniform',math.isclose(ce,math.log(16),abs_tol=1e-15),True)
    checked('level_row_major_boundaries',[geometry(i)['center_xy'] for i in [0,79,80,6399,6400,7999,8000,8399]],
            [[4,4],[636,4],[4,12],[636,636],[8,8],[632,632],[16,16],[624,624]])
    root=a.workspace/'08_实验日志'
    anchor=root/'2026-09-08_probe_定位学习位置与目标/anchor_join/output_attempt3/objects.jsonl'
    witness=root/'2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/witness_objects.jsonl'
    rows=load_rows(anchor);old=load_rows(witness);by_id={r['stable_rgb_gt_id']:r for r in old}
    checked('historical_full_population',[len(rows),len(by_id),len(set(r['stable_rgb_gt_id'] for r in rows)),len(set(r['image_index'] for r in rows))],[80,80,80,32])
    for r in rows:
        w=by_id[r['stable_rgb_gt_id']]
        for key in ['stable_ir_gt_id','frame_id','image_index','rgb_global_row','ir_global_row','rgb_gt_xyxy','ir_gt_xyxy']:
            if r[key]!=w[key]:raise AssertionError(('identity mismatch',key,r['stable_rgb_gt_id']))
        for c in [r['historical_R_candidate'],r['historical_T_candidate']]:
            if c is not None and c!=geometry(c['anchor_index']):raise AssertionError(('geometry mismatch',c))
        for name,c in r['native_iou50_matches'].items():
            ow=w['detector'][name]['native_witness']
            if (c is None)!=(ow is None):raise AssertionError('native match missingness mismatch')
            if c is not None:
                if c['anchor_index']!=ow['anchor_index'] or c['prediction_id']!=ow['prediction_index'] or c['box']!=ow['box']:
                    raise AssertionError('native prediction identity mismatch')
    checked('all80_original_identity_anchor_geometry_native_id',True,True)
    result=dict(status='PASS_INDEPENDENT_STDLIB_CPU_TRUTHS',auditor='/root/dfl_readout_review',model_identity='unavailable',
                date=datetime.now(timezone.utc).isoformat(),python=sys.version,checks=cases,
                same_mean_example=dict(p=p,q=q,p_stats=pm,q_stats=qm),
                fixed_population_bucket_counts=dict(Counter(r['bucket'] for r in rows)),
                inputs=[str(anchor),str(witness)],new_GPU=False,new_model_forward=False,training=False,
                official_test_accessed=False,audited_input_hashes='not computed; task boundary',
                limits=['Synthetic truths verify coordinate/statistical semantics only, not producer execution.',
                        'Historical exact identity is across saved identity records, not an assertion of historical pixel/logit bitwise equality.',
                        'Geometric bin support [0,15] and native clamp-invariant domain [0,14.99] are deliberately distinct.'])
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(result['status'],len(cases),result['fixed_population_bucket_counts'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
