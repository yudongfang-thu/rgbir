"""Readout identity/denominator appendix only; no new metric or model execution."""
import argparse
import json
from pathlib import Path


def build(probe,analysis):
    readlines=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=readlines(analysis/'object_role_readout.jsonl');raw=readlines(probe/'anchor_distributions.jsonl')
    lookup={(r['model'],r['image_index'],r['anchor_index']):r for r in raw}
    fields=['raw_logits','probabilities_fp32','probabilities_native','expectation_fp32_bins','native_dfl_distances_bins','box_fp32_xyxy','box_native_xyxy']
    sr=[]
    for key,r in lookup.items():
        if key[0]!='S':continue
        other=lookup[('R',key[1],key[2])]
        assert all(r[f]==other[f] for f in fields)
        sr.append(dict(image_index=key[1],anchor_index=key[2]))
    pairs=[(('S','historical_R_candidate'),('T','same_historical_R_index')),
           (('S','own_native_iou50'),('T','own_native_iou50')),
           (('S','own_native_iou50'),('T','same_S_native_iou50_index'))]
    common=[]
    for group in ['all80','both05_onlyT075','both05_onlyS075']:
        rr=[r for r in rows if group=='all80' or r['bucket']==group]
        for left,right in pairs:
            def selected(spec):
                return {r['stable_rgb_gt_id']:r for r in rr if (r['model'],r['role'])==spec and r['GT_readout'] is not None and r['GT_readout']['all_four_edges_valid']}
            a,b=selected(left),selected(right);ids=sorted(a.keys()&b.keys())
            if group=='all80' and left==('S','historical_R_candidate'):
                assert a.keys()==b.keys() and len(ids)==79
                assert all(a[i]['own_GT']['own_gt_xyxy']==b[i]['own_GT']['own_gt_xyxy'] for i in ids)
                assert all(a[i]['own_GT']['unclamped_gt_distance_bins']==b[i]['own_GT']['unclamped_gt_distance_bins'] for i in ids)
            common.append(dict(group=group,left=list(left),right=list(right),left_valid=len(a),right_valid=len(b),common_valid=len(ids),same_valid_set=a.keys()==b.keys(),
                               common_stable_rgb_gt_ids=ids,
                               common_GT_boxes_exact=all(a[i]['own_GT']['own_gt_xyxy']==b[i]['own_GT']['own_gt_xyxy'] for i in ids),
                               common_GT_distances_exact=all(a[i]['own_GT']['unclamped_gt_distance_bins']==b[i]['own_GT']['unclamped_gt_distance_bins'] for i in ids)))
    invalid=[dict(model=r['model'],role=r['role'],stable_rgb_gt_id=r['stable_rgb_gt_id'],own_GT=r['own_GT'],GT_readout=r['GT_readout']) for r in rows if r['GT_readout'] is not None and not r['GT_readout']['all_four_edges_valid']]
    return dict(status='READOUT_DENOMINATOR_APPENDIX_COMPLETED',current_forward_id=rows[0]['current_forward_id'],
                S_R_same_anchor_raw_fields_exact_pairs=sr,S_R_compared_fields=fields,common_legality=common,invalid_object_roles=invalid,
                distinct_invalid_RGB_GT=len({r['stable_rgb_gt_id'] for r in invalid}),new_CE_comparison=False,new_GPU=False,new_forward=False,new_hash_computed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(build(a.probe,a.analysis),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
