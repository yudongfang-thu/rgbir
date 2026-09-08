"""CPU-only, fixed first32 cached post-NMS matching at independently fixed .5/.75."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from native_cached_match import load_native, match_row

BUCKETS = ('both05_both075', 'both05_onlyT075', 'both05_onlyS075',
           'both05_neither075', 'T_only05', 'S_only05', 'neither05')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def stat(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)


def bucket(row):
    s5, t5 = (row['matches'][m]['0.5'] is not None for m in ('S', 'T'))
    s75, t75 = (row['matches'][m]['0.75'] is not None for m in ('S', 'T'))
    if s5 and t5:
        return 'both05_' + ('both075' if s75 and t75 else 'onlyT075' if t75 else 'onlyS075' if s75 else 'neither075')
    return 'T_only05' if t5 else 'S_only05' if s5 else 'neither05'


def coverage(rows, denominator):
    n = len(rows)
    result = dict(objects=n, unique_images=len({r['frame_id'] for r in rows}),
                  fraction_all_gt=n / denominator if denominator else None,
                  stable_rgb_gt_ids=[r['stable_rgb_gt_id'] for r in rows])
    for gate in ('base', 'eligible', 'selected'):
        chosen = [r for r in rows if r['C_gates'][gate]]
        result[gate] = dict(objects=len(chosen), unique_images=len({r['frame_id'] for r in chosen}),
                            fraction_bucket=len(chosen) / n if n else None,
                            stable_rgb_gt_ids=[r['stable_rgb_gt_id'] for r in chosen])
    return result


def analyze(input_dir, source_dir):
    p = Path(input_dir)
    completion = read_json(p / 'completion_receipt.json')
    assert completion['status'] == 'SAME_FORWARD_DETECTOR_WITNESS_COMPLETED'
    assert completion['scope'] == 'SAME_FORWARD_DETECTOR_WITNESS'
    assert (completion['dataset'], completion['seed'], completion['frames'], completion['objects_n']) == ('llvip', 42, 32, 80)
    assert completion['previous_objects_exact'] and completion['student_full_state_unchanged']
    assert completion['first_batch_stream_exact']
    assert all(completion[k] == 0 for k in ('training', 'backward', 'optimizer_updates', 'ema_updates'))
    contract = read_json(p / 'witness_contract.json')
    assert contract['native_profile'] == dict(conf=.25, iou=.7, max_det=300, multi_label=True, agnostic_nms=False, matching_iou=.5)
    assert contract['native_tp_and_gt_pairs_exact']
    originals = read_lines(p / 'objects.jsonl')
    objects = read_lines(p / 'witness_objects.jsonl')
    ir_objects = read_lines(p / 'witness_ir_objects.jsonl')
    frames = read_lines(p / 'witness_frames.jsonl')
    assert [{k: v for k, v in r.items() if k != 'detector'} for r in objects] == originals
    rgb = {r['rgb_global_row']: r for r in objects}
    ir = {r['ir_global_row']: r for r in ir_objects}
    assert len(rgb) == len(ir) == 80 and len(frames) == 32
    assert len({r['stable_rgb_gt_id'] for r in objects}) == 80
    assert len({r['stable_ir_gt_id'] for r in ir_objects}) == 80
    assert len({f['frame_id'] for f in frames}) == 32
    native = load_native(source_dir)
    found = {m: {} for m in ('S', 'R', 'T')}
    frame_results = []
    closure = dict(gt_flags_checked=0, positive_gt_prediction_pairs_checked=0, frame_prediction_tp_checked=0)
    seen = {'rgb': [], 'ir': []}
    for f in frames:
        seen['rgb'] += f['rgb_global_rows']; seen['ir'] += f['ir_global_rows']
        fr = dict(frame_id=f['frame_id'], image_index=f['image_index'], models={})
        for m in ('S', 'R', 'T'):
            is_ir = m == 'T'
            ids = f['ir_global_rows' if is_ir else 'rgb_global_rows']
            gt = [ir[i] if is_ir else rgb[i] for i in ids]
            assert all(g['frame_id'] == f['frame_id'] for g in gt)
            det = f['native_detections'][m]
            assert [d['prediction_index'] for d in det] == list(range(len(det)))
            assert all(d['confidence'] > .25 for d in det)
            row = dict(pred_confidence=[d['confidence'] for d in det], pred_boxes=[d['box'] for d in det],
                       pred_classes=[d['class'] for d in det], gt_classes=[g['gt_class'] for g in gt],
                       gt_boxes=[g['ir_gt_xyxy' if is_ir else 'rgb_gt_xyxy'] for g in gt])
            fr['models'][m] = {}
            for threshold in (.5, .75):
                key = str(threshold)
                # Separate native calls: .75 is never a threshold applied to .5 pairs.
                result = match_row(row, None, threshold, native)
                fr['models'][m][key] = result
                if threshold == .5:
                    assert result['prediction_tp'] == [d['native_tp_iou50'] for d in det]
                    closure['frame_prediction_tp_checked'] += len(det)
                for gi, g in enumerate(gt):
                    pair = result['gt_matches'].get(gi)
                    if pair is not None:
                        d = det[pair['prediction_id']]
                        pair = dict(pair, anchor_index=d['anchor_index'], class_id=d['class'], box=d['box'])
                    found[m].setdefault(ids[gi], {})[key] = pair
                    if threshold == .5:
                        previous = g['detector_T'] if is_ir else g['detector'][m]
                        assert (pair is not None) == previous['native_postnms_matched']
                        closure['gt_flags_checked'] += 1
                        if pair is not None:
                            w = previous['native_witness']
                            assert pair['prediction_id'] == w['prediction_index']
                            assert pair['anchor_index'] == w['anchor_index']
                            assert pair['confidence'] == w['confidence'] and pair['box'] == w['box']
                            assert abs(pair['iou'] - w['iou']) < 1e-6
                            closure['positive_gt_prediction_pairs_checked'] += 1
        frame_results.append(fr)
    assert sorted(seen['rgb']) == sorted(rgb) and sorted(seen['ir']) == sorted(ir)
    records = []
    for r in objects:
        i = r['ir_global_row']
        assert i is not None and ir[i]['stable_ir_gt_id'] == r['stable_ir_gt_id']
        assert ir[i]['ir_gt_xyxy'] == r['ir_gt_xyxy']
        out = {k: r[k] for k in ('frame_id', 'image_index', 'rgb_global_row', 'ir_global_row',
                                  'stable_rgb_gt_id', 'stable_ir_gt_id', 'rgb_gt_xyxy', 'ir_gt_xyxy')}
        out['C_gates'] = r['gates']
        out['matches'] = {m: found[m][i if m == 'T' else r['rgb_global_row']] for m in ('S', 'R', 'T')}
        out['bucket'] = bucket(out)
        records.append(out)
    summary = dict(status='TRAINING_LOCALIZATION_CACHE_DIAGNOSTIC_COMPLETED', dataset='llvip', seed=42,
                   frames=len(frames), paired_rgb_objects=len(records), empty_rgb_frames=sum(f['rgb_gt_count'] == 0 for f in frames),
                   native_iou_thresholds=[.5, .75], confidence='>0.25, existing post-NMS order preserved',
                   original_native_nms=contract['native_profile'], coordinate_scope=dict(S='RGB own GT', R='RGB own GT', T='IR own GT'),
                   closure_at_iou05=closure, C_all=coverage(records, len(records)),
                   buckets={b: coverage([r for r in records if r['bucket'] == b], len(records)) for b in BUCKETS},
                   counts={m: {k: sum(r['matches'][m][k] is not None for r in records) for k in ('0.5', '0.75')} for m in ('S', 'R', 'T')},
                   higher_iou_gt_not_matched_at_lower={m: sum(r['matches'][m]['0.75'] is not None and r['matches'][m]['0.5'] is None for r in records) for m in ('S', 'R', 'T')},
                   native_source_files=[stat(Path(source_dir) / f) for f in ('3_val.py', '4_validator.py', '7_metrics.py')],
                   inputs=[stat(p / f) for f in ('completion_receipt.json', 'objects.jsonl', 'witness_objects.jsonl', 'witness_ir_objects.jsonl', 'witness_frames.jsonl', 'witness_contract.json')],
                   GPU_used=False, new_forward=False, training=False, new_hash_computed=False, AP_estimated=False,
                   actual_L_selector_evaluated=False, historical_pixel_tensor_comparison=False)
    assert sum(v['objects'] for v in summary['buckets'].values()) == len(records)
    return summary, records, frame_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--native-source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    summary, rows, frames = analyze(args.input, args.native_source_dir)
    assert not torch.cuda.is_initialized()
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for name, data in [('objects.jsonl', rows), ('frames.jsonl', frames)]:
        (args.output / name).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in data), encoding='utf-8')
    dest = args.output / 'source_copies'; dest.mkdir()
    for source in [Path(__file__), Path(__file__).with_name('native_cached_match.py')] + [args.native_source_dir / f for f in ('3_val.py', '4_validator.py', '7_metrics.py')]:
        shutil.copyfile(source, dest / source.name)
        assert source.read_bytes() == (dest / source.name).read_bytes()
    print(json.dumps({k: summary[k] for k in ('status', 'frames', 'paired_rgb_objects', 'counts', 'closure_at_iou05')}))


if __name__ == '__main__': main()
