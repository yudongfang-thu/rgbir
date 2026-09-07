"""Read existing local diagnostics only. No source execution, network, GPU or hash."""
from pathlib import Path
import collections
import gzip
import json
import statistics
import numpy as np

ROOT = Path(r'E:\SHARE\光sar\08_实验日志')
PROBE = ROOT/'2026-09-06_probe_RGBIR数据特性与可迁移知识'
D1 = ROOT/'2026-09-07_probe_TaskConditional机会诊断'
END = ROOT/'2026-09-07_train_IndependentKD实施'/'legacy_diagnostics_snapshot_20260907_161459'
OUT = Path(__file__).parent
INPUTS = {}

def register(path):
    st = path.stat()
    INPUTS[str(path)] = dict(size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)

def read(path):
    register(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    register(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def reliable_teacher(row):
    t = row['teacher']
    return bool(t and t['class'] == row['class'] and t['confidence'] >= .25 and t['iou'] >= .5)

report = dict(date='2026-09-07', auditor='/root/baseline_opportunity_audit',
    hashes_computed=False, hash_policy='Explicit user task prohibits hash computation.',
    evaluation_type='real_gt + self_supervised_proxy for CKA/energy and local logit gradients',
    probe={}, d1d2={}, endpoint={})

for name, short in [('dronevehicle_full', 'drone'), ('llvip_full', 'llvip')]:
    path = PROBE/name
    manifest = read(path/'input_manifest.json')
    summary = read(path/'summary.json')
    pred = read(path/'prediction_records.json')
    imet = {r['id']:r for r in read(path/'image_metrics.json')}
    features = read(path/'feature_metrics.json')
    objects = read(path/'matched_objects.json')
    old = {r['paths'][0]:r for r in pred}
    energies = np.load(path/'energy_maps.npz')
    register(path/'energy_maps.npz')
    report['probe'][short] = dict(
        config=manifest['config'], image_count=len(pred),
        rgb_gt=sum(len(r['gt_rgb']) for r in pred), ir_gt=sum(len(r['gt_ir']) for r in pred),
        matched=len(objects), hits=dict(collections.Counter((str(bool(r['rgb_hit']))+'_'+str(bool(r['ir_hit']))) for r in objects)),
        sample_seed=manifest['seed'], checkpoint_args=[{k:x['args'].get(k) for k in ('seed','epochs','model','batch','imgsz','optimizer','workers','data')} for x in manifest['checkpoints']],
        energy_shapes={key:list(energies[key].shape) for key in energies.files},
        stored_feature_channel_tensors=False,
        foreground_delta_cka={lv:dict(n=sum(r['delta_cka'] is not None for r in features if r['level']==lv and r['region']=='fg'),mean=statistics.mean(r['delta_cka'] for r in features if r['level']==lv and r['region']=='fg' and r['delta_cka'] is not None)) for lv in ['P3','P4','P5']})
    for split in ['train','val']:
        path = D1/f'{short}_{split}'
        summary = read(path/'summary.json')
        roster = read(path/'frozen_roster.json')
        objects = rows(path/'d1_objects.jsonl')
        anchors = rows(path/'d2_anchors.jsonl')
        paths = {r['rgb_path'] for r in roster['roster']}
        shared = sorted(paths & old.keys())
        max_error = 0.; compared = 0; class_mismatch = 0
        for row in objects:
            if row['image'] not in old: continue
            source = old[row['image']]
            gt = source['gt_rgb'][row['rgb_gt_local_index']]
            shape = imet[source['id']]['rgb_shape']
            h,w = shape; scale = min(640/h,640/w)
            left = round((640-round(w*scale))/2-.1); top = round((640-round(h*scale))/2-.1)
            transformed = [gt[1]*w*scale+left,gt[2]*h*scale+top,gt[3]*w*scale+left,gt[4]*h*scale+top]
            max_error = max(max_error, max(abs(a-b) for a,b in zip(transformed,row['gt_box_input'])))
            class_mismatch += int(gt[0] != row['class']); compared += 1
        states = {}
        for state in ['class_error','low_confidence','localization_gap','well_localized','no_coarse_candidate']:
            sub = [r for r in objects if r['reference_state']==state]
            useful = [r for r in sub if reliable_teacher(r)]
            states[state] = dict(n=len(sub), reference_iou_ge_05=sum(r['reference'] is not None and r['reference']['iou']>=.5 for r in sub),
                teacher_own_hit=len(useful), teacher_rgb_hit=sum(r['teacher_to_rgb_iou']>=.5 for r in useful),
                teacher_rgb_hit_and_reference_iou_ge_05=sum(r['teacher_to_rgb_iou']>=.5 and r['reference'] is not None and r['reference']['iou']>=.5 for r in useful))
        selected = [r for r in anchors if r['selected']]
        report['d1d2'][f'{short}_{split}'] = dict(image_count=len(paths),population=roster['population_images'],sampling=roster['sampling'],
            image_overlap_with_probe_full_path=len(shared), overlapping_paths=shared, overlap_gt_checked=compared,
            overlap_gt_max_input_px_difference=max_error,overlap_gt_class_mismatches=class_mismatch,
            d1_row_count=len(objects),d1_unique_objects=len({(r['image'],r['rgb_gt_local_index']) for r in objects}),
            d2_row_count=len(anchors),d2_unique_objects=len({r['object_id'] for r in anchors}), d1_states=states,
            summary_totals=summary['totals'], d1_totals=summary['d1_totals'],
            selected_dfl_ce_teacher_minus_reference_mean=statistics.mean(r['teacher_gt_dfl_ce']-r['reference_gt_dfl_ce'] for r in selected),
            teacher=summary['teacher'],reference=summary['reference'],augmentation=summary['augmentation'],
            d1_fields=list(objects[0]),d2_fields=list(anchors[0]))

endpoint_gt = None
for arm in ['N','C0']:
    for seed in [0,42,123]:
        name=f'{arm}_s{seed}_attempt1'; path=END/name
        ev=read(path/'evaluation_val.json'); con=read(path/'evaluation_contract.json')
        pred=rows(path/'predictions/objects.jsonl.gz')
        gt={r['image']:(r['gt_boxes'],r['gt_classes'],r['canvas_shape']) for r in pred}
        if endpoint_gt is None: endpoint_gt=gt
        report['endpoint'][name]=dict(checkpoint=ev['checkpoint'],mAP50_95=ev['mAP50_95'],
            historical_five_metrics_exact=ev['historical_five_metrics_exact'],image_count=len(pred),gt_count=sum(len(r['gt_boxes']) for r in pred),
            predictions_stored=sum(len(r['pred_boxes']) for r in pred),canvas_shapes=sorted({tuple(r['canvas_shape']) for r in pred}),
            exact_gt_equal_first_endpoint=gt==endpoint_gt,kwargs=con['effective_kwargs'],row_fields=list(pred[0]))

report['declared_inputs'] = INPUTS
report['audited_input_hashes'] = []
(OUT/'inventory_recomputed.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for key,value in report['d1d2'].items():
    print(key,json.dumps({k:v for k,v in value.items() if k in ['image_count','population','image_overlap_with_probe_full_path','overlap_gt_checked','overlap_gt_max_input_px_difference','overlap_gt_class_mismatches','d1_states','selected_dfl_ce_teacher_minus_reference_mean']},ensure_ascii=False))
print('output',str(OUT/'inventory_recomputed.json'))
