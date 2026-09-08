"""Read producer-completed DFL records; never infer utility or compare cross-anchor KL."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np
from dfl_metrics import distribution,own_anchor_readout


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def stat(p):
    p=Path(p);s=p.stat();return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def describe(v):
    a=np.asarray(v,dtype=np.float64).ravel()
    if not len(a):return dict(n=0,mean=None,median=None,minimum=None,maximum=None)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()))


def role_summary(rows,total):
    present=[r for r in rows if r['distribution_id'] is not None]
    valid=[r for r in present if r['GT_readout']['all_four_edges_valid']]
    valid_edges=[e for r in present for e in r['GT_readout']['edges'] if e['valid']]
    return dict(GT_objects=total,anchor_present=len(present),anchor_missing=total-len(present),
                unique_distribution_ids=len({r['distribution_id'] for r in present}),
                all_four_GT_edges_valid_objects=len(valid),invalid_GT_objects=len(present)-len(valid),
                valid_GT_edges=len(valid_edges),invalid_GT_edges=4*len(present)-len(valid_edges),
                invalid_reasons=dict(Counter(e['reason'] for r in present for e in r['GT_readout']['edges'] if not e['valid'])),
                entropy_nat_all_present_edges=describe([r['distribution_metrics']['entropy_nat'] for r in present]),
                variance_bin2_all_present_edges=describe([r['distribution_metrics']['variance_bin2'] for r in present]),
                expectation_bin_all_present_edges=describe([r['distribution_metrics']['expectation_bin'] for r in present]),
                DFL_CE_nat_valid_edges=describe([e['DFL_CE_nat'] for e in valid_edges]),
                mean4_DFL_CE_nat_valid_objects=describe([r['GT_readout']['four_edge_mean_DFL_CE_nat'] for r in valid]),
                mean4_absolute_mean_error_bin_valid_objects=describe([r['GT_readout']['four_edge_mean_absolute_expectation_error_bin'] for r in valid]))


def check_distribution(row):
    assert row['model'] in ('S','R','T') and row['bins']==16
    assert row['side_order']==['left','top','right','bottom']
    assert row['fp32_decode_exact'] and row['native_decode_exact']
    assert row['stride'] in (8,16,32) and row['level'] in (0,1,2)
    shape=row['raw_feature_shapes'];offset=0;derived=None
    for level,f in enumerate(shape):
        h,w=f[-2:]
        if offset<=row['anchor_index']<offset+h*w:
            y,x=divmod(row['anchor_index']-offset,w);s=(8,16,32)[level]
            derived=(level,s,[(x+.5)*s,(y+.5)*s]);break
        offset+=h*w
    assert derived==(row['level'],row['stride'],row['center_xy'])
    desc=distribution(row['raw_logits']);probs=np.asarray(row['probabilities_fp32'],dtype=float)
    native=np.asarray(row['probabilities_native'],dtype=float)
    assert probs.shape==native.shape==(4,16) and np.isfinite(probs).all() and np.isfinite(native).all()
    assert (probs>=0).all() and (probs<=1).all() and (native>=0).all() and (native<=1).all()
    p_error=float(np.max(abs(probs-np.asarray(desc['probabilities']))))
    mean_error=float(np.max(abs(np.asarray(row['expectation_fp32_bins'])-np.asarray(desc['expectation_bin']))))
    assert p_error<=2e-6 and mean_error<=1e-5
    # Report native rounding separately; do not renormalize native low-precision mass.
    native_mean=(native*np.arange(16)[None,:]).sum(1)
    center=np.asarray(row['center_xy']);dist=np.asarray(row['expectation_fp32_bins'])*row['stride']
    reconstructed=np.r_[center-dist[:2],center+dist[2:]]
    box_delta=float(np.max(abs(reconstructed-np.asarray(row['box_fp32_xyxy']))))
    assert box_delta<=1e-4
    return desc,dict(FP64_recomputed_vs_saved_FP32_probability_max_abs=p_error,
                     FP64_recomputed_vs_saved_FP32_expectation_max_abs=mean_error,
                     saved_FP32_expectation_box_reconstruction_max_abs_px=box_delta,
                     native_probability_mass_error_max_abs=float(np.max(abs(native.sum(1)-1))),
                     native_probability_float64_mean_vs_native_conv_max_abs_bin=float(np.max(abs(native_mean-np.asarray(row['native_dfl_distances_bins'])))),
                     native_vs_FP32_box_max_abs_px=float(np.max(abs(np.asarray(row['box_native_xyxy'])-np.asarray(row['box_fp32_xyxy'])))))


def analyze(input_dir):
    p=Path(input_dir);completion=read(p/'completion_receipt.json');contract=read(p/'dfl_contract.json')
    assert completion['status']=='RAW_DFL_SINGLE_BATCH_COMPLETED' and completion['scope']=='RAW_DFL_SINGLE_BATCH'
    assert contract['scope']=='RAW_DFL_SINGLE_BATCH'
    forward_id=completion['current_forward_id']
    assert isinstance(forward_id,str) and forward_id and contract['current_forward_id']==forward_id
    assert contract['full_probability_vectors'] and not contract['GT_distances_clamped'] and not contract['cross_anchor_KL']
    for field in ('new_NMS','new_matching','new_loss'):assert contract[field] is False
    assert completion['dataset']=='llvip' and completion['seed']==42
    assert completion['batches']==1 and completion['frames']==32 and completion['objects_n']==80
    assert completion['first_batch_stream_exact'] and completion['student_full_state_unchanged']
    assert completion['identity_exact'] and completion['auxiliary_full_state_unchanged'] and completion['all_gradients_absent']
    assert completion['initialization']==contract['initialization']
    assert not completion['historical_same_forward_claim'] and not contract['historical_same_forward_claim']
    assert completion['optimizer_updates']==completion['ema_updates']==completion['backward']==0
    assert completion['raw_forward_counts']==dict(student=1,teacher=1,reference=1)
    objects=lines(p/'objects.jsonl');raws=lines(p/'anchor_distributions.jsonl')
    assert all(row['current_forward_id']==forward_id for row in objects+raws)
    assert len(objects)==80 and len({o['stable_rgb_gt_id'] for o in objects})==80
    assert len({o['stable_ir_gt_id'] for o in objects})==80
    groups={'all80':objects,'both05_onlyT075':[o for o in objects if o['bucket']=='both05_onlyT075'],
            'both05_onlyS075':[o for o in objects if o['bucket']=='both05_onlyS075']}
    assert len(groups['both05_onlyT075'])==11 and len(groups['both05_onlyS075'])==1
    distributions={r['distribution_id']:r for r in raws};assert len(distributions)==len(raws)
    assert len(raws)==completion['unique_distributions']==contract['unique_distributions']
    assert len({(r['model'],r['image_index'],r['anchor_index']) for r in raws})==len(raws)
    descs={};numeric={}
    for did,row in distributions.items():descs[did],numeric[did]=check_distribution(row)
    roles=contract['roles'];assert set(roles)=={'S','R','T'}
    assert all(set(o['roles'])==set(roles) for o in objects)
    role_rows=[];used=set()
    for o in objects:
        for model,expected_roles in roles.items():
            assert set(o['roles'][model])==set(expected_roles)
            for role in expected_roles:
                ref=o['roles'][model][role]
                row=dict(current_forward_id=forward_id,stable_rgb_gt_id=o['stable_rgb_gt_id'],stable_ir_gt_id=o['stable_ir_gt_id'],
                         frame_id=o['frame_id'],image_index=o['image_index'],bucket=o['bucket'],model=model,role=role,
                         C_selected=o['C_selected'],historical_L2_gates=o['historical_L2_gates'],distribution_id=None,
                         distribution_metrics=None,GT_readout=None)
                if ref is not None:
                    did=ref['distribution_id'];d=distributions[did];used.add(did)
                    assert (d['model'],d['image_index'],d['anchor_index'])==(model,o['image_index'],ref['anchor_index'])
                    gt=o['ir_gt_xyxy' if model=='T' else 'rgb_gt_xyxy'];own_id=o['stable_ir_gt_id' if model=='T' else 'stable_rgb_gt_id']
                    assert ref['own_gt_xyxy']==gt and ref['own_gt_id']==own_id
                    assert ref['own_gt_modality']==('IR' if model=='T' else 'RGB')
                    x,y=d['center_xy'];s=d['stride'];target=[(x-gt[0])/s,(y-gt[1])/s,(gt[2]-x)/s,(gt[3]-y)/s]
                    assert target==ref['unclamped_gt_distance_bins']
                    validity=[0<=v<15 for v in target]
                    assert validity==ref['gt_distance_in_range_per_side'] and all(validity)==ref['gt_distance_all_in_range']
                    row.update(distribution_id=did,own_GT=ref,distribution_metrics=descs[did],GT_readout=own_anchor_readout(descs[did],target))
                role_rows.append(row)
    assert used==set(distributions)
    by_group={}
    for group,objs in groups.items():
        ids={o['stable_rgb_gt_id'] for o in objs}
        by_group[group]={m:{role:role_summary([r for r in role_rows if r['stable_rgb_gt_id'] in ids and r['model']==m and r['role']==role],len(objs)) for role in roles[m]} for m in roles}
    summary=dict(status='RAW_DFL_CPU_READOUT_COMPLETED',scope='OWN_ANCHOR_UNCLAMPED_DFL_DIAGNOSTIC',current_forward_id=forward_id,
                 objects=80,unique_distributions=len(raws),model_roles=roles,by_group=by_group,
                 numeric_closure_maxima={k:max(v[k] for v in numeric.values()) for k in next(iter(numeric.values()))},
                 numerical_contract=dict(primary='float64 stable softmax of original-valued raw logits',entropy='nat',variance='bin squared',
                                         GT_interval='0 <= distance < 15, no clamp',CE='Adjacent-bin interpolation; not reconstructed training DFLoss',
                                         native_probabilities='Retained and rounding reported; never renormalized'),
                 inputs=[stat(p/f) for f in ('completion_receipt.json','dfl_contract.json','objects.jsonl','anchor_distributions.jsonl')],
                 GPU_used=False,new_forward=False,new_hash_computed=False,cross_anchor_KL=False,AP_estimated=False,
                 parameter_gradient_comparison=False,DFL_utility_proven=False,full_probability_vectors_available=True)
    return summary,role_rows,[dict(current_forward_id=forward_id,distribution_id=k,metrics=descs[k],numeric_closure=numeric[k]) for k in distributions]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    summary,rows,dist=analyze(a.input)
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for name,values in [('object_role_readout.jsonl',rows),('distribution_readout.jsonl',dist)]:
        (a.output/name).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in values),encoding='utf-8')
    for x in [Path(__file__),Path(__file__).with_name('dfl_metrics.py')]:shutil.copyfile(x,a.output/x.name);assert x.read_bytes()==(a.output/x.name).read_bytes()
    print(json.dumps({k:summary[k] for k in ['status','objects','unique_distributions','numeric_closure_maxima']}))
