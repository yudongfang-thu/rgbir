"""Read author artifacts and compare to independent cached-data arithmetic."""
import json
import math
from pathlib import Path
import recompute_independent as own

HERE = Path(__file__).resolve().parent
ENTRY = HERE.parent


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14), (a,b)


def main():
    independent=own.read(HERE/'CPU_RECOMPUTATION.json')
    targetdir=ENTRY/'target_audit/output_attempt1'
    target=own.read(targetdir/'summary.json')
    objs=own.lines(targetdir/'objects.jsonl')
    pairs = [
        ('selected_object_occurrences','selected_occurrences'),
        ('base_objects',None),
        ('unique_images','unique_image_paths'),
        ('selected_unique_images','selected_unique_image_paths'),
        ('stack_unscaled_output_derivative_cosine','R_output_derivative_stack_cosine'),
        ('stack_loss_scaled_output_derivative_cosine','R_output_derivative_B_lambda_base_scaled_stack_cosine'),
        ('stack_pixel_B_lambda_output_derivative_cosine','R_pixel_derivative_B_lambda_base_scaled_stack_cosine'),
    ]
    for a,b in pairs:
        close(target[a],independent[b] if b else independent['eight_batch_stage_totals']['base_count'])
    assert target['regimes']==dict(both_saturated_same_sign=47,both_quadratic=58,one_saturated=15)
    assert target['opposite_sign_edges']==10 and target['derivative_exactly_equal_edges']==47
    assert target['mapped_IR_box_nonexact_records']==[
        dict(batch=r['batch'],rgb_gt_index=r['row'],selected=r['selected'],max_abs_pixel_difference=r['max_abs_pixel_error'])
        for r in independent['teacher_mapping_float_nonexact_records']]
    close(target['teacher_target_absolute_residual']['mean'],independent['target_gt_mean_abs'])
    close(target['teacher_target_absolute_residual']['maximum'],independent['target_gt_max_abs'])
    by={(r['batch'],r['row']):r for r in independent['target_rows']}
    quantities_compared=0
    support=[]
    for obj in objs:
        r=obj['historical_record']; q=obj['quantities']; raw=by[(obj['batch'],r['rgb_gt_index'])]
        for x,y in zip(q['teacher_target_minus_GT'],raw['target_gt_residual']):close(x,y)
        close(q['output_derivative_cosine'],raw['output_derivative_cosine'])
        close(q['proxy_teacher_loss_contribution'],raw['reference_point_loss_teacher']/obj['base_count'])
        close(q['proxy_GT_loss_contribution'],raw['reference_point_loss_gt']/obj['base_count'])
        s,t,g=own.rel(r['reference_box'],r['rgb_gt']),own.rel(r['mapped_teacher_box'],r['rgb_gt']),[0.,0.,1.,1.]
        wh=[r['rgb_gt'][2]-r['rgb_gt'][0],r['rgb_gt'][3]-r['rgb_gt'][1]]*2
        for role,v in [('teacher',t),('GT',g)]:
            d=[own.derivative(a-b) for a,b in zip(s,v)]
            for x,y in zip(q['derivative_'+role+'_unscaled'],d):close(x,y)
            for x,y in zip(q['derivative_'+role+'_pixel_B_lambda'],[x/w*32/(4*obj['base_count']) for x,w in zip(d,wh)]):close(x,y)
        quantities_compared+=1
        ai=r['reference_anchor'];stride,n,start=(8,80,0) if ai<6400 else (16,40,6400)
        j=ai-start;x,y=((j%n+.5)*stride,(j//n+.5)*stride);box=r['mapped_teacher_box']
        dist=[(x-box[0])/stride,(y-box[1])/stride,(box[2]-x)/stride,(box[3]-y)/stride]
        support.append(dict(batch=obj['batch'],rgb_gt_index=r['rgb_gt_index'],reference_anchor=ai,target_ltrb_in_stride_units=dist))
    candidates=[p for p in (ENTRY/'anchor_join').glob('output_attempt*') if (p/'completion.json').exists()]
    assert len(candidates)==1, 'Expected one retained completed anchor attempt'
    anchorpath=candidates[0]
    anchors=own.lines(anchorpath/'objects.jsonl');summ=own.read(anchorpath/'summary.json')
    original=own.lines(own.BRIDGE/'objects.jsonl')
    amap={r['rgb_global_row']:r for r in anchors}
    count=0
    for old in original:
        r=amap[old['rgb_global_row']];record=old['historical_L2_record']
        assert r['historical_L2_record']==record and r['historical_L2_gates']==old['historical_L2_gates']
        assert r['stable_rgb_gt_id']==old['stable_rgb_gt_id'] and r['C_gates']==old['C_gates']
        selected=old['historical_L2_gates']['selected']
        assert (r['actual_L2_learning_anchor'] is None)==(not selected)
        if selected:assert r['actual_L2_learning_anchor']['anchor_index']==record['reference_anchor']
        for model in ('S','R','T'):
            m=r['native_iou50_matches'][model];oldm=old['matches'][model]['0.5']
            assert (m is None)==(oldm is None)
            if m:assert {k:v for k,v in m.items() if k!='anchor_geometry'}==oldm
        count+=1
    c=summ['cohorts']
    assert c['all80']['objects']==80 and c['all80']['L2_base']==79 and c['all80']['L2_selected']==7
    assert c['native_T_localization11']['objects']==11 and c['native_T_localization11']['L2_selected']==4
    assert c['native_T_localization11']['relations']['R_candidate_vs_native_S']['same']==11
    assert c['T_localization_intersection_L2_selected']['relations']['T_candidate_vs_native_T']['different']==4
    assert c['native_S_reverse1']['L2_selected']==0
    source_names=['localization_box_v2.py','direction_criterion.py','calibrate_direction.py']
    source_checked=[]
    for name in source_names:
        captured=ENTRY/'source_identity'/name
        inspected=own.LOG/'2026-09-08_probe_快速方向筛选/newentry/release'/name
        assert captured.read_bytes()==inspected.read_bytes()
        source_checked.append(name)
    assert (targetdir/'localization_box_v2.py').read_bytes()==(ENTRY/'source_identity/localization_box_v2.py').read_bytes()
    result=dict(status='PASS_AUTHOR_OUTPUT_COMPARISON',auditor='/root/loc_target_review',model='unavailable',
                target_objects_compared=quantities_compared,anchor_rows_compared=count,anchor_output=str(anchorpath),
                source_byte_comparisons=source_checked,
                teacher_targets_within_reference_DFL_expectation_range=sum(all(0<=d<=15 for d in r['target_ltrb_in_stride_units']) for r in support),
                teacher_target_stride_distance_min=min(min(r['target_ltrb_in_stride_units']) for r in support),
                teacher_target_stride_distance_max=max(max(r['target_ltrb_in_stride_units']) for r in support),
                representability_scope='Cached 30 selected target boxes and recorded 640-grid only; not native TAL assignment or physical registration.',
                support_rows=support,hash='not computed: explicit task constraint',GPU_used=False,model_forward=False,new_training=False)
    out=HERE/'AUTHOR_COMPARISON.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='support_rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
