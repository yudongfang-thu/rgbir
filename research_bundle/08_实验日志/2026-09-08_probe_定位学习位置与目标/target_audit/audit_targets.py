"""L2 target audit: saved reference-output proxy, explicitly not student/parameter gradients."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np

BETA = .1


def relative(box, gt):
    box, gt = np.asarray(box, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    wh = gt[2:] - gt[:2]
    if not np.isfinite(box).all() or not np.isfinite(gt).all() or not (wh > 0).all(): raise ValueError('Invalid box/GT')
    return (box - np.tile(gt[:2], 2)) / np.tile(wh, 2)


def smooth(error):
    a = np.abs(error)
    return np.where(a < BETA, error**2 / (2 * BETA), a - BETA / 2)


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else None


def quantities(record, base, batch_size, coefficient):
    q = np.array([0., 0., 1., 1.]); g = np.asarray(record['rgb_gt'], dtype=np.float64)
    r = relative(record['reference_box'], g)
    t = relative(record['mapped_teacher_box'], g)
    e_gt, e_teacher, delta = r - q, r - t, t - q
    d_gt, d_teacher = np.clip(e_gt / BETA, -1, 1), np.clip(e_teacher / BETA, -1, 1)
    scale = 1 / (4 * max(1, base)); wh = np.tile(g[2:] - g[:2], 2)
    sat_g, sat_t = np.abs(e_gt) >= BETA, np.abs(e_teacher) >= BETA
    regimes = []
    for j in range(4):
        if sat_g[j] and sat_t[j]: label = 'both_saturated_same_sign' if d_gt[j] * d_teacher[j] > 0 else 'both_saturated_opposite_sign'
        elif sat_g[j] or sat_t[j]: label = 'one_saturated'
        else: label = 'both_quadratic'
        regimes.append(label)
    return dict(reference_relative=r.tolist(), teacher_target_relative=t.tolist(), GT_target=q.tolist(),
                teacher_target_minus_GT=delta.tolist(), reference_minus_GT=e_gt.tolist(),
                reference_minus_teacher=e_teacher.tolist(), derivative_GT_unscaled=d_gt.tolist(),
                derivative_teacher_unscaled=d_teacher.tolist(),
                derivative_GT_normalized_loss=(d_gt * scale).tolist(),
                derivative_teacher_normalized_loss=(d_teacher * scale).tolist(),
                derivative_GT_pixel_loss=(d_gt * scale / wh).tolist(),
                derivative_teacher_pixel_loss=(d_teacher * scale / wh).tolist(),
                derivative_GT_pixel_B_lambda=(d_gt * scale / wh * batch_size * coefficient).tolist(),
                derivative_teacher_pixel_B_lambda=(d_teacher * scale / wh * batch_size * coefficient).tolist(),
                proxy_GT_loss_contribution=float(smooth(e_gt).mean() / max(1, base)),
                proxy_teacher_loss_contribution=float(smooth(e_teacher).mean() / max(1, base)),
                output_derivative_cosine=cosine(d_gt, d_teacher),
                target_error_norm_over_reference_error=float(np.linalg.norm(delta) / np.linalg.norm(e_gt)) if np.linalg.norm(e_gt) else None,
                regimes=regimes, reference_is_measured_student=False)


def describe(values):
    a = np.asarray(values, dtype=np.float64).ravel()
    if not len(a): return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), minimum=float(a.min()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def summarize_objects(rows):
    target = np.asarray([r['quantities']['teacher_target_minus_GT'] for r in rows])
    ref = np.asarray([r['quantities']['reference_minus_GT'] for r in rows])
    gt = np.asarray([r['quantities']['derivative_GT_unscaled'] for r in rows])
    teacher = np.asarray([r['quantities']['derivative_teacher_unscaled'] for r in rows])
    same_nonzero = (gt * teacher) > 0; opposite = (gt * teacher) < 0
    return dict(objects=len(rows), edges=int(target.size),
                teacher_target_minus_GT_signed_by_edge={name:describe(target[:,i]) for i,name in enumerate(('x1','y1','x2','y2'))},
                teacher_target_minus_GT_abs_by_edge={name:describe(abs(target[:,i])) for i,name in enumerate(('x1','y1','x2','y2'))},
                teacher_target_absolute_residual=describe(abs(target)), reference_absolute_GT_error=describe(abs(ref)),
                teacher_target_exact_GT_edges=int((target==0).sum()), teacher_target_residual_lt_beta_edges=int((abs(target)<BETA).sum()),
                teacher_target_residual_lt_reference_error_edges=int((abs(target)<abs(ref)).sum()),
                target_error_norm_over_reference_error=describe([r['quantities']['target_error_norm_over_reference_error'] for r in rows]),
                derivative_exactly_equal_edges=int((gt==teacher).sum()), same_nonzero_sign_edges=int(same_nonzero.sum()),
                opposite_sign_edges=int(opposite.sum()), both_zero_edges=int(((gt==0)&(teacher==0)).sum()),
                exactly_one_zero_edge=int(((gt==0)^(teacher==0)).sum()),
                same_sign_different_magnitude_edges=int((same_nonzero&(gt!=teacher)).sum()),
                objects_with_any_opposite_edge=int(opposite.any(1).sum()),
                GT_derivative_saturated_edges=int((abs(gt)==1).sum()), teacher_derivative_saturated_edges=int((abs(teacher)==1).sum()),
                regimes=dict(Counter(label for r in rows for label in r['quantities']['regimes'])),
                per_object_cosine=describe([r['quantities']['output_derivative_cosine'] for r in rows]),
                stack_unscaled_output_derivative_cosine=cosine(gt,teacher),
                stack_loss_scaled_output_derivative_cosine=cosine([r['quantities']['derivative_GT_normalized_loss'] for r in rows],[r['quantities']['derivative_teacher_normalized_loss'] for r in rows]),
                stack_pixel_B_lambda_output_derivative_cosine=cosine([r['quantities']['derivative_GT_pixel_B_lambda'] for r in rows],[r['quantities']['derivative_teacher_pixel_B_lambda'] for r in rows]),
                opposite_edge_records=[dict(occurrence_id=r['occurrence_id'],image=r['image'],edge=('x1','y1','x2','y2')[j],reference_minus_GT=ref[i,j],teacher_minus_GT=target[i,j],GT_derivative=gt[i,j],teacher_derivative=teacher[i,j]) for i,r in enumerate(rows) for j in range(4) if opposite[i,j]])


def run(calibration):
    p = Path(calibration)
    batches = [json.loads(x) for x in (p/'calibration_batches.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    receipt = json.loads((p/'calibration_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='DIRECTION_CALIBRATION_COMPLETED' and receipt['dataset']=='llvip' and receipt['seed']==42
    assert receipt['batches']==len(batches)==8 and receipt['reset_all_parameters_buffers_each_batch']
    assert receipt['bn_running_buffers_unchanged'] and receipt['full_initial_checkpoint_state_verified']
    assert receipt['coefficients']['L2-box']==receipt['coefficients']['L2-GT']==1
    rows=[]; batch_results=[]; all_files=[]; missing_teacher_targets=0; stored_student_box_fields=0
    all_base_GT_equal=0; mapping_differences=[]
    for i,b in enumerate(batches,1):
        assert b['batch']==i and len(b['files'])==32
        all_files += b['files']; st,control=b['stats']['L2-box'],b['stats']['L2-GT']
        assert st['config']['smooth_l1_beta']==BETA
        assert st['base_records']==control['base_records'] and st['selected_anchors']==control['selected_anchors']
        assert st['normalizer']==max(1,st['base_count'])==control['normalizer']
        chosen=[r for r in st['base_records'] if r['selected']]
        assert len(chosen)==st['selected_count']
        missing_teacher_targets += sum(r['mapped_teacher_box'] is None for r in st['base_records'])
        stored_student_box_fields += sum('student_box' in r for r in st['base_records'])
        all_base_GT_equal += sum(r['rgb_gt']==r['ir_gt'] for r in st['base_records'])
        for rec in st['base_records']:
            if rec['mapped_teacher_box'] is not None:
                difference=float(np.max(np.abs(np.asarray(rec['mapped_teacher_box'])-np.asarray(rec['teacher_box']))))
                mapping_differences.append(dict(batch=i,rgb_gt_index=rec['rgb_gt_index'],selected=rec['selected'],max_abs_pixel_difference=difference))
        current=[]
        for r in chosen:
            assert r['teacher_box'] is not None and r['mapped_teacher_box'] is not None
            x=dict(occurrence_id=f"batch{i}:rgb{r['rgb_gt_index']}:ir{r['ir_gt_index']}",batch=i,
                   image=b['files'][r['batch_index']],base_count=st['base_count'],batch_size=len(b['files']),
                   historical_record=r, quantities=quantities(r,st['base_count'],len(b['files']),1))
            current.append(x);rows.append(x)
        batch_results.append(dict(batch=i,base=st['base_count'],selected=len(chosen),selected_image_occurrences=len({r['image'] for r in current}),
                                  reference_proxy_teacher_loss=sum(r['quantities']['proxy_teacher_loss_contribution'] for r in current),
                                  reference_proxy_GT_loss=sum(r['quantities']['proxy_GT_loss_contribution'] for r in current),
                                  actual_recorded_student_teacher_loss=st['loss_unweighted'],actual_recorded_student_GT_loss=control['loss_unweighted'],
                                  actual_recorded_native_cosine=b['native_cosines'],actual_recorded_unit_B_kd_norms=b['unit_B_kd_norms']))
    assert len(rows)==30
    summary=summarize_objects(rows)
    file_counts=Counter(all_files);selected_counts=Counter(r['image'] for r in rows)
    unique_batch_image={(r['batch'],r['image']) for r in rows}
    summary.update(status='L2_TARGET_AUDIT_COMPLETED',scope='REFERENCE_OUTPUT_PROXY_NOT_PARAMETER_GRADIENT',
                   batches=8,base_objects=sum(x['base'] for x in batch_results),image_occurrences=len(all_files),unique_images=len(file_counts),
                   repeated_image_occurrences=len(all_files)-len(file_counts),selected_object_occurrences=len(rows),
                   selected_image_occurrences=len(unique_batch_image),selected_unique_images=len(selected_counts),
                   selected_images_repeated_across_batches=len(unique_batch_image)-len(selected_counts),
                   selected_multiple_objects_same_image={k:v for k,v in selected_counts.items() if v>1},
                   stable_original_GT_identity_available_all8=False,
                   selected_RGB_IR_GT_exact_count=sum(r['historical_record']['rgb_gt']==r['historical_record']['ir_gt'] for r in rows),
                   selected_mapped_teacher_equals_IR_box_count=sum(r['historical_record']['mapped_teacher_box']==r['historical_record']['teacher_box'] for r in rows),
                   all_base_RGB_IR_GT_exact_count=all_base_GT_equal,non_null_teacher_mapping_count=len(mapping_differences),
                   mapped_IR_box_max_abs_pixel_difference=max(r['max_abs_pixel_difference'] for r in mapping_differences),
                   mapped_IR_box_nonexact_records=[r for r in mapping_differences if r['max_abs_pixel_difference']!=0],
                   selected_reference_teacher_same_anchor_count=sum(r['historical_record']['reference_anchor']==r['historical_record']['teacher_anchor'] for r in rows),
                   stored_base_records_without_teacher_target=missing_teacher_targets,stored_base_records_with_student_box=stored_student_box_fields,
                   batch_results=batch_results,physical_registration_claim=False,student_output_exact_claim=False,
                   parameter_gradient_reconstructed=False,DFL_distribution_reconstructed=False,GPU_used=False,new_forward=False,new_hash_computed=False)
    return summary,rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    s,r=run(a.calibration)
    s['inputs']=[dict(path=str(x.resolve()),bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns) for x in [a.calibration/'calibration_batches.jsonl',a.calibration/'calibration_receipt.json',a.source]]
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (a.output/'objects.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in r),encoding='utf-8')
    for x in [Path(__file__),a.source]:
        shutil.copyfile(x,a.output/x.name);assert x.read_bytes()==(a.output/x.name).read_bytes()
    print(json.dumps({k:s[k] for k in ['status','selected_object_occurrences','selected_unique_images','regimes','same_nonzero_sign_edges','opposite_sign_edges','derivative_exactly_equal_edges']}))
