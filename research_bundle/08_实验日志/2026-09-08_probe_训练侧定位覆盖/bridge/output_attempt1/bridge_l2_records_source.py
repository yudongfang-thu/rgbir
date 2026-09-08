"""Read-only bridge from historical L2 calibration records to stable first32 GT IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import yaml


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def stat(path):
    p = Path(path); s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def bind(record, witness):
    assert record['batch_index'] == witness['image_index']
    assert record['rgb_gt_index'] == witness['rgb_global_row']
    assert record['ir_gt_index'] == witness['ir_global_row']
    assert record['class_id'] == witness['gt_class']
    assert record['rgb_gt'] == witness['rgb_gt_xyxy']
    assert record['ir_gt'] == witness['ir_gt_xyxy']
    assert record['pair_iou'] == witness['pair_iou']


def gates(record, fixed):
    if record is None:
        return dict(base=False, reference_reliable=None, reference_gap=None, teacher_own_quality=None,
                    mapped_rgb_quality=None, localization_margin=None, selected=False, exit='not_in_historical_L2_base')
    reliable, gap = record['reference_reliable'], record['reference_gap']
    assert gap == (reliable and record['reference_iou'] < fixed['reference_iou_max'])
    own = (record['teacher_anchor'] is not None) if gap else None
    mapped = (record['mapped_teacher_rgb_iou'] >= fixed['mapped_rgb_iou_min']) if own else None
    margin = (record['mapped_teacher_rgb_iou'] > record['reference_iou'] + fixed['localization_margin']) if mapped else None
    selected = record['selected']
    assert selected == bool(gap and own and mapped and margin)
    if own:
        assert record['teacher_conf'] >= fixed['reliable_conf']
        assert record['teacher_own_iou'] >= fixed['teacher_ir_iou_min']
    reason = ('selected' if selected else 'reference_unreliable' if not reliable else
              'reference_iou_not_below_070' if not gap else 'no_teacher_own_quality_candidate' if not own else
              'mapped_rgb_quality_failed' if not mapped else 'localization_margin_failed')
    return dict(base=True, reference_reliable=reliable, reference_gap=gap, teacher_own_quality=own,
                mapped_rgb_quality=mapped, localization_margin=margin, selected=selected, exit=reason)


def coverage(rows, all_n):
    n = len(rows)
    out = dict(objects=n, unique_images=len({r['frame_id'] for r in rows}), fraction_all_gt=n / all_n,
               stable_rgb_gt_ids=[r['stable_rgb_gt_id'] for r in rows],
               exits=dict(Counter(r['historical_L2_gates']['exit'] for r in rows)))
    for g in ('base', 'reference_reliable', 'reference_gap', 'teacher_own_quality', 'mapped_rgb_quality', 'selected'):
        keep = [r for r in rows if r['historical_L2_gates'][g] is True]
        out[g] = dict(objects=len(keep), unique_images=len({r['frame_id'] for r in keep}),
                      fraction_bucket=len(keep)/n if n else None,
                      not_evaluated=sum(r['historical_L2_gates'][g] is None for r in rows),
                      stable_rgb_gt_ids=[r['stable_rgb_gt_id'] for r in keep])
    out['C_selected_cross_L2_selected'] = {f'C{int(c)}_L2{int(l)}': sum(r['C_gates']['selected'] == c and r['historical_L2_gates']['selected'] == l for r in rows) for c in (False, True) for l in (False, True)}
    return out


def analyze(calibration, witness, core):
    c, w, p = map(Path, (calibration, witness, core))
    receipt = read(c/'calibration_receipt.json'); completion = read(w/'completion_receipt.json')
    cfg = yaml.safe_load((c/'direction_config.yaml').read_text(encoding='utf-8'))
    wcfg = yaml.safe_load((w/'inherited_training_config.yaml').read_text(encoding='utf-8'))
    differing = {k: [cfg.get(k), wcfg.get(k)] for k in set(cfg) | set(wcfg) if cfg.get(k) != wcfg.get(k)}
    assert set(differing) == {'calibration_receipt'} and cfg['calibration_receipt'] is None
    assert cfg['dataset'] == 'llvip' and cfg['seed'] == 42 and cfg['batch'] == 32
    assert cfg['freeze_bn_running_statistics'] and cfg['model'] == cfg['reference']
    assert receipt['status'] == 'DIRECTION_CALIBRATION_COMPLETED' and receipt['scope'] == 'EXPLORATORY_FIXED8_BNFROZEN'
    assert receipt['batches'] == 8 and receipt['dataset'] == 'llvip' and receipt['seed'] == 42
    assert all(receipt[k] for k in ('reset_all_parameters_buffers_each_batch', 'bn_running_buffers_unchanged', 'full_initial_checkpoint_state_verified'))
    assert receipt['optimizer_updates'] == receipt['ema_updates'] == 0
    assert completion['status'] == 'SAME_FORWARD_DETECTOR_WITNESS_COMPLETED'
    assert completion['student_full_state_unchanged'] and completion['first_batch_stream_exact']
    assert receipt['student_initial_checkpoint'] == completion['initialization']['model']
    for k in ('model', 'teacher', 'reference'): assert cfg[k] == completion['initialization'][k]['path']
    batches = lines(c/'calibration_batches.jsonl'); first = batches[0]
    assert first['batch'] == 1 and len(first['files']) == 32
    stream = read(w/'first_batch_stream.json')
    assert first['files'] == stream['im_file']
    old = lines(w/'witness_objects.jsonl'); core_rows = lines(p/'objects.jsonl')
    assert len(old) == len(core_rows) == 80
    wr = {r['rgb_global_row']: r for r in old}; cr = {r['rgb_global_row']: r for r in core_rows}
    assert len(wr) == len(cr) == 80 and set(wr) == set(cr)
    for i, r in wr.items():
        assert first['files'][r['image_index']] == r['image']
        assert cr[i]['C_gates'] == r['gates']
        for k in ('stable_rgb_gt_id', 'stable_ir_gt_id', 'rgb_gt_xyxy', 'ir_gt_xyxy', 'image_index', 'ir_global_row'):
            assert cr[i][k] == r[k]
    source_stats = first['stats']['L2-box']; records = source_stats['base_records']
    assert records == first['stats']['L2-GT']['base_records']
    assert len(records) == source_stats['base_count'] == 79
    by_id = {r['rgb_gt_index']: r for r in records}; assert len(by_id) == 79
    fixed = source_stats['config']
    assert fixed['reference_iou_max'] == .7 and fixed['localization_margin'] == .05
    for r in records: bind(r, wr[r['rgb_gt_index']])
    actual_selected = [r for r in records if r['selected']]
    assert [[r['batch_index'], r['rgb_gt_index'], r['ir_gt_index']] for r in actual_selected] == source_stats['selected_object_ids']
    assert [[r['batch_index'], r['reference_anchor'], r['teacher_anchor'], r['rgb_gt_index'], r['ir_gt_index']] for r in actual_selected] == source_stats['selected_anchors']
    bridged = []
    for i, r in wr.items():
        record = by_id.get(i)
        out = dict(cr[i], historical_L2_record=record, historical_L2_gates=gates(record, fixed), raw_dense_overlap={})
        for model, prefix in [('R', 'reference'), ('T', 'teacher')]:
            dense = r['detector'][model]['dense_witness']
            present = record is not None and record[prefix+'_anchor'] is not None
            same = present and dense is not None and record[prefix+'_anchor'] == dense['anchor_index']
            fields = {k: record[prefix+'_'+k] == dense[k] for k in ('box',)} if same else None
            if same:
                fields['confidence'] = record[prefix+'_conf'] == dense['confidence']
                fields['class'] = record['class_id'] == dense['class']
                fields['iou'] = record['reference_iou' if model == 'R' else 'teacher_own_iou'] == dense['iou']
            out['raw_dense_overlap'][model] = dict(historical_anchor_present=present,
                current_dense_witness_present=dense is not None, same_anchor=same, exact_fields=fields)
        bridged.append(out)
    totals = coverage(bridged, 80)
    for gate, field in [('base','base_count'), ('reference_reliable','reference_reliable_count'), ('reference_gap','reference_gap_count'),
                        ('teacher_own_quality','teacher_own_quality_count'), ('mapped_rgb_quality','mapped_rgb_quality_count'), ('selected','selected_count')]:
        assert totals[gate]['objects'] == source_stats[field]
    buckets = sorted({r['bucket'] for r in bridged})
    overlap = {}
    for m in ('R', 'T'):
        selected = [r['raw_dense_overlap'][m] for r in bridged if r['raw_dense_overlap'][m]['same_anchor']]
        overlap[m] = dict(same_anchor_records=len(selected),
                          all_reported_fields_exact=sum(all(x['exact_fields'].values()) for x in selected),
                          field_exact_counts={k:sum(x['exact_fields'][k] for x in selected) for k in ('box','confidence','class','iou')})
    summary = dict(status='HISTORICAL_L2_FIRST_BATCH_RECORD_BRIDGE_COMPLETED', scope='HISTORICAL_CALIBRATION_RECORD_AUDIT_ONLY',
                   frames=32, objects=80, historical_base_records=79, full_config_equal_except=differing,
                   identity=dict(first32_files_exact=True, historical_base_dual_GT_rows_boxes_classes_exact=True,
                                 C_masks_exact=True, initial_student_checkpoint_stat_exact=True,
                                 teacher_reference_config_paths_exact=True, separate_historical_teacher_checkpoint_stat_available=False,
                                 calibration_receipt_reset_all_parameters_buffers_each_batch=True, calibration_BN_unchanged=True,
                                 historical_pixel_tensor_comparison=False, entire_raw_tensor_exact_claim=False),
                   shared_L2_box_GT_base_records_exact=True, old_L2_config=fixed, all=totals,
                   by_native_bucket={b: coverage([r for r in bridged if r['bucket']==b],80) for b in buckets},
                   partial_same_anchor_raw_dense_field_overlap=overlap,
                   inputs=[stat(x) for x in (c/'calibration_receipt.json',c/'calibration_batches.jsonl',c/'direction_config.yaml',
                                             w/'completion_receipt.json',w/'first_batch_stream.json',w/'witness_objects.jsonl',w/'inherited_training_config.yaml',p/'objects.jsonl')],
                   new_forward=False, GPU_used=False, new_training=False, new_hash_computed=False,
                   current_witness_L2_reexecution=False, native_proxy_substituted_for_actual_L2=False)
    return summary, bridged, source_stats


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('calibration','witness','core','output'): p.add_argument('--'+k,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    s,r,original = analyze(a.calibration,a.witness,a.core)
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (a.output/'objects.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in r),encoding='utf-8')
    (a.output/'original_first_batch_L2_stats.json').write_text(json.dumps(original,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    shutil.copyfile(__file__,a.output/'bridge_l2_records_source.py')
    assert Path(__file__).read_bytes()==(a.output/'bridge_l2_records_source.py').read_bytes()
    print(json.dumps(dict(status=s['status'],all_counts={g:s['all'][g]['objects'] for g in ['base','reference_reliable','reference_gap','teacher_own_quality','mapped_rgb_quality','selected']},partial_overlap=s['partial_same_anchor_raw_dense_field_overlap'])))
