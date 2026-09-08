"""Bounded independent CPU arithmetic; no tensors, models, new selection, or hashes."""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / '08_实验日志'
HERE = Path(__file__).resolve().parent
CAL = LOG / '2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip'
BRIDGE = LOG / '2026-09-08_probe_训练侧定位覆盖/bridge/output_attempt1'
CORE = LOG / '2026-09-08_probe_训练侧定位覆盖/output_attempt1'
WITNESS = LOG / '2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def rel(box, gt):
    wh = [gt[2] - gt[0], gt[3] - gt[1]] * 2
    return [(a-b)/c for a, b, c in zip(box, gt[:2]*2, wh)]


def sl1(e):
    return e*e/.2 if abs(e) < .1 else abs(e)-.05


def derivative(e):
    return max(-1., min(1., e/.1))


def cosine(a, b):
    den = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b))/den if den else None


def main():
    batches = lines(CAL/'calibration_batches.jsonl')
    bridge = lines(BRIDGE/'objects.jsonl')
    core = lines(CORE/'objects.jsonl')
    witness = lines(WITNESS/'witness_objects.jsonl')
    frames = lines(WITNESS/'witness_frames.jsonl')
    receipt = read(CAL/'calibration_receipt.json')
    assert len(batches) == 8 and len(bridge) == len(core) == len(witness) == 80 and len(frames) == 32
    assert receipt['optimizer_updates'] == receipt['ema_updates'] == 0
    assert len(receipt['parameter_names']) == 24
    first = {r['rgb_gt_index']: r for r in batches[0]['stats']['L2-box']['base_records']}
    identities, matches, opp, reverse, distinct = 0, 0, [], [], []
    byrow = {r['rgb_global_row']: r for r in witness}
    for row in bridge:
        w = byrow[row['rgb_global_row']]
        old = first.get(row['rgb_global_row'])
        assert old == row['historical_L2_record']
        assert row['matches'] == core[row['rgb_global_row']]['matches']
        assert row['stable_rgb_gt_id'] == w['stable_rgb_gt_id']
        assert row['stable_ir_gt_id'] == w['stable_ir_gt_id']
        if old:
            for a, b in [('rgb_gt', 'rgb_gt_xyxy'), ('ir_gt', 'ir_gt_xyxy'), ('batch_index', 'image_index'), ('ir_gt_index', 'ir_global_row'), ('class_id', 'gt_class')]:
                assert old[a] == w[b]
            identities += 1
        for model in ('S', 'R', 'T'):
            for threshold in ('0.5', '0.75'):
                m = row['matches'][model][threshold]
                if m:
                    det = frames[row['image_index']]['native_detections'][model][m['prediction_id']]
                    for a, b in [('anchor_index', 'anchor_index'), ('box', 'box'), ('confidence', 'confidence'), ('class_id', 'class')]:
                        assert m[a] == det[b]
                    matches += 1
        s, t = row['matches']['S'], row['matches']['T']
        is_opp = s['0.5'] is not None and t['0.5'] is not None and s['0.75'] is None and t['0.75'] is not None
        is_rev = s['0.5'] is not None and t['0.5'] is not None and s['0.75'] is not None and t['0.75'] is None
        assert is_opp == (row['bucket'] == 'both05_onlyT075')
        assert is_rev == (row['bucket'] == 'both05_onlyS075')
        if is_opp or is_rev:
            anchor = old['reference_anchor'] if old else None
            compact = dict(row=row['rgb_global_row'], image=row['image_index'], selected=bool(old and old['selected']),
                           reference_anchor=anchor, native_S_anchor=s['0.5']['anchor_index'],
                           native_R_anchor=row['matches']['R']['0.5']['anchor_index'],
                           native_T_anchor=t['0.5']['anchor_index'], teacher_anchor=old['teacher_anchor'] if old else None,
                           same_student_anchor=anchor == s['0.5']['anchor_index'])
            (opp if is_opp else reverse).append(compact)
        if old and row['matches']['S']['0.5'] and old['reference_anchor'] != row['matches']['S']['0.5']['anchor_index']:
            distinct.append(row['rgb_global_row'])

    targetrows, norms, selected_paths, ratios, cosines = [], [], [], [], []
    stage_totals = Counter()
    all_g_t, all_g_gt, weighted_t, weighted_gt, pixels_t, pixels_gt = [], [], [], [], [], []
    edge_types, edge_signs = Counter(), Counter()
    max_map_error, max_loss_error = 0., 0.
    mapping_float_nonexact = []
    for b in batches:
        st, gtst = b['stats']['L2-box'], b['stats']['L2-GT']
        assert st['base_records'] == gtst['base_records']
        assert st['selected_anchors'] == gtst['selected_anchors']
        assert st['base_count'] == st['normalizer'] == len(st['base_records'])
        keep = [r for r in st['base_records'] if r['selected']]
        assert len(keep) == st['selected_count']
        assert [[r[k] for k in ('batch_index', 'reference_anchor', 'teacher_anchor', 'rgb_gt_index', 'ir_gt_index')] for r in keep] == st['selected_anchors']
        for k in ('rgb_gt_count', 'base_count', 'selected_count', 'reference_reliable_count', 'reference_gap_count', 'teacher_own_quality_count', 'mapped_rgb_quality_count'):
            stage_totals[k] += st[k]
        batch_lt, batch_lgt = 0., 0.
        for r in st['base_records']:
            assert r['rgb_gt'] == r['ir_gt']
            if not r['reference_gap']:
                assert r['teacher_anchor'] is None and r['mapped_teacher_box'] is None
            if r['teacher_box']:
                map_error = max(abs(x-y) for x,y in zip(r['mapped_teacher_box'],r['teacher_box']))
                max_map_error = max(max_map_error, map_error)
                if map_error:
                    mapping_float_nonexact.append(dict(batch=b['batch'],row=r['rgb_gt_index'],selected=r['selected'],max_abs_pixel_error=map_error))
        for r in keep:
            s, t, q = rel(r['reference_box'], r['rgb_gt']), rel(r['mapped_teacher_box'], r['rgb_gt']), [0., 0., 1., 1.]
            et, eg = [x-y for x,y in zip(s,t)], [x-y for x,y in zip(s,q)]
            dt, dg = list(map(derivative, et)), list(map(derivative, eg))
            residual = [x-y for x,y in zip(t,q)]
            scale = 32./(4*st['normalizer'])
            wh = [r['rgb_gt'][2]-r['rgb_gt'][0], r['rgb_gt'][3]-r['rgb_gt'][1]]*2
            all_g_t += dt; all_g_gt += dg
            weighted_t += [v*scale for v in dt]; weighted_gt += [v*scale for v in dg]
            pixels_t += [v*scale/w for v,w in zip(dt,wh)]; pixels_gt += [v*scale/w for v,w in zip(dg,wh)]
            for x,y in zip(et,eg):
                xt, yg = abs(x) >= .1, abs(y) >= .1
                key = ('both_saturated_same' if x*y > 0 else 'both_saturated_opposite') if xt and yg else 'both_linear' if not xt and not yg else 'one_saturated'
                edge_types[key] += 1
                dx,dy = derivative(x), derivative(y)
                edge_signs['equal' if dx == dy else 'opposite' if dx*dy < 0 else 'one_zero' if dx*dy == 0 else 'same_direction_different'] += 1
            lt, lg = sum(map(sl1,et))/4, sum(map(sl1,eg))/4
            batch_lt += lt/st['normalizer']; batch_lgt += lg/st['normalizer']
            selected_paths.append(b['files'][r['batch_index']])
            targetrows.append(dict(batch=b['batch'], row=r['rgb_gt_index'], mean_abs_target_gt=sum(map(abs,residual))/4,
                                   max_abs_target_gt=max(map(abs,residual)), output_derivative_cosine=cosine(dt,dg),
                                   target_gt_residual=residual, reference_point_loss_teacher=lt,
                                   reference_point_loss_gt=lg))
        max_loss_error = max(max_loss_error, abs(batch_lt-st['loss_unweighted']), abs(batch_lgt-gtst['loss_unweighted']))
        ratio = b['unit_B_kd_norms']['L2-box']/b['native_norm']
        ratios.append(ratio)
        if b['native_cosines']['L2-box'] is not None:
            cosines.append(b['native_cosines']['L2-box'])
        norms.append(dict(batch=b['batch'], base=st['base_count'], selected=len(keep),
                          teacher_loss_at_R=batch_lt, cached_student_loss=st['loss_unweighted'],
                          ratio_shared_parameters=ratio, native_cosine=b['native_cosines']['L2-box']))
    file_paths = [p for b in batches for p in b['files']]
    # Two exact toy obligations: saturation hides unequal targets; nearby GT can oppose teacher.
    assert derivative(1.-.2) == derivative(1.-0.) == 1.
    assert derivative(.02-.04) < 0 < derivative(.02-0.)
    # Base normalization and image scaling are independent of number selected.
    assert (sl1(.2)*4/4)/10 == .015000000000000003
    result = dict(status='PASS_BOUNDED_INDEPENDENT_CPU_RECOMPUTATION', auditor='/root/loc_target_review', model='unavailable',
                  python_runtime='D:/Anaconda/envs/KGJ_proj/python.exe', inspected_original_bridge_gt_records=identities,
                  linked_native_matches=matches, original_GT_count=80, opportunity_objects=opp, reverse_objects=reverse,
                  native_S_matched_but_R_anchor_different_rows=distinct,
                  eight_batch_stage_totals=dict(stage_totals), image_occurrences=len(file_paths), unique_image_paths=len(set(file_paths)),
                  selected_occurrences=len(targetrows), selected_unique_image_paths=len(set(selected_paths)),
                  stable_source_object_deduplication='unavailable for later 7 batches; batch GT rows are not stable physical object IDs',
                  target_gt_mean_abs=statistics.mean(r['mean_abs_target_gt'] for r in targetrows),
                  target_gt_median_abs=statistics.median(r['mean_abs_target_gt'] for r in targetrows),
                  target_gt_max_abs=max(r['max_abs_target_gt'] for r in targetrows),
                  R_output_derivative_stack_cosine=cosine(all_g_t,all_g_gt),
                  R_output_derivative_B_lambda_base_scaled_stack_cosine=cosine(weighted_t,weighted_gt),
                  R_pixel_derivative_B_lambda_base_scaled_stack_cosine=cosine(pixels_t,pixels_gt),
                  edge_types=dict(edge_types), edge_signs=dict(edge_signs),
                  active_batch_shared_gradient_ratio_median=statistics.median(r for r in ratios if r),
                  all_batch_shared_gradient_ratio_median=statistics.median(ratios),
                  native_shared_parameter_cosine_signs=dict(positive=sum(c>0 for c in cosines),negative=sum(c<0 for c in cosines)),
                  maximum_R_proxy_vs_cached_student_scalar_loss_abs_error=max_loss_error,
                  max_teacher_mapping_float_reconstruction_error_px=max_map_error,
                  teacher_mapping_float_nonexact_records=mapping_float_nonexact,
                  scalar_agreement_does_not_prove_student_raw_tensor_identity=True,
                  batches=norms, target_rows=targetrows,
                  toy_checks_passed=3, GPU_used=False, model_loaded=False, model_forward=False, new_training=False,
                  official_test_accessed=False, audited_input_hashes='not computed: explicit no-hash task boundary')
    destination = HERE/'CPU_RECOMPUTATION.json'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('target_rows','opportunity_objects','reverse_objects')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
