# coding: utf-8
"""Descriptive fixed-eight calibration readout; no AP, GPU, weights or hashes."""
import csv
from collections import Counter
import json
import math
from pathlib import Path, PurePosixPath
import statistics

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw_attempt1/screen_attempt1'
GROUPS = HERE.parent / 'subset_llvip/remote_subset_llvip_v1/source_groups.tsv'
GATES = ('rgb_gt_count', 'teacher_gt_count', 'common_count', 'pair_iou_count',
         'reference_support_count', 'reference_unique_owner_count', 'base_count',
         'reference_reliable_count', 'reference_gap_count', 'teacher_own_quality_count',
         'mapped_rgb_quality_count', 'selected_count')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def describe(values):
    values = [x for x in values if x is not None]
    assert all(math.isfinite(x) for x in values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def concentration(counts):
    denominator = sum(counts.values())
    ranked = counts.most_common()
    return dict(counts=dict(ranked), total=denominator,
                top1_fraction=sum(x[1] for x in ranked[:1]) / denominator if denominator else None,
                top5_fraction=sum(x[1] for x in ranked[:5]) / denominator if denominator else None)


def main():
    receipt = read(RAW / 'calibration/llvip/calibration_receipt.json')
    assert receipt['status'] == 'DIRECTION_CALIBRATION_COMPLETED' and receipt['batches'] == 8
    rows = [json.loads(x) for x in (RAW / 'calibration/llvip/calibration_batches.jsonl').read_text().splitlines()]
    assert len(rows) == 8 and [x['batch'] for x in rows] == list(range(1, 9))
    with GROUPS.open(encoding='utf-8') as f:
        groups = {x['stem']: x['sequence_prefix'] for x in csv.DictReader(f, delimiter='\t')}
    images = []
    bases, selected = [], []
    batch_rows = []
    all_gate_counts = Counter()
    shared_masks_exact = True
    for row in rows:
        assert len(row['files']) == 32
        images += row['files']
        a, b = row['stats']['L2-box'], row['stats']['L2-GT']
        shared_masks_exact &= a['base_records'] == b['base_records'] and a['selected_anchors'] == b['selected_anchors']
        assert len(a['base_records']) == a['base_count']
        assert sum(bool(x['selected']) for x in a['base_records']) == a['selected_count']
        assert a['normalizer'] == max(1, a['base_count'])
        for key in GATES:
            all_gate_counts[key] += a[key]
        for record in a['base_records']:
            path = row['files'][record['batch_index']]
            group = groups[PurePosixPath(path).stem]
            out = dict(batch=row['batch'], image=path, group=group, **record)
            bases.append(out)
            if record['selected']:
                selected.append(out)
        native = row['native_norm']
        assert math.isfinite(native) and native > 0
        for arm in ('L2-box', 'L2-GT'):
            unit = row['unit_B_kd_norms'][arm]
            coefficient = receipt['coefficients'][arm]
            batch_rows.append(dict(batch=row['batch'], arm=arm, base=a['base_count'], selected=a['selected_count'],
                selected_over_base=a['selected_count'] / max(1, a['base_count']), native_gradient_norm=native,
                B_times_unweighted_kd_gradient_norm=unit, coefficient=coefficient,
                actual_weighted_KD_over_native_norm=coefficient * unit / native if unit is not None else None,
                native_cosine=row['native_cosines'][arm]))
    assert shared_masks_exact
    full_images = set(images)
    base_images = {x['image'] for x in bases}
    selected_images = {x['image'] for x in selected}
    base_groups = Counter(x['group'] for x in bases)
    selected_groups = Counter(x['group'] for x in selected)
    selected_image_groups = Counter(groups[PurePosixPath(x).stem] for x in selected_images)
    gradient = {}
    for arm in ('L2-box', 'L2-GT'):
        values = [x for x in batch_rows if x['arm'] == arm]
        active = [x for x in values if x['B_times_unweighted_kd_gradient_norm'] and x['B_times_unweighted_kd_gradient_norm'] > 0]
        raw_ratios = [.1 * x['native_gradient_norm'] / x['B_times_unweighted_kd_gradient_norm'] for x in active]
        assert raw_ratios == receipt['details'][arm]['ratios']
        assert statistics.median(raw_ratios) == receipt['details'][arm]['raw_median']
        cosine = [x['native_cosine'] for x in active]
        gradient[arm] = dict(coefficient=receipt['coefficients'][arm], raw_calibration_median=statistics.median(raw_ratios),
            active_batches=len(active), weighted_KD_over_native_all8=describe([x['actual_weighted_KD_over_native_norm'] for x in values]),
            weighted_KD_over_native_active=describe([x['actual_weighted_KD_over_native_norm'] for x in active]),
            native_cosine_active=describe(cosine), positive_cosine_batches=sum(x > 0 for x in cosine),
            negative_cosine_batches=sum(x < 0 for x in cosine), zero_cosine_batches=sum(x == 0 for x in cosine))
    resources = []
    for arm, stage in [('N', 'calibration'), ('N', 'canary'), ('L2-box', 'canary'), ('L2-GT', 'canary')]:
        root = RAW / ('calibration/llvip' if stage == 'calibration' else 'canaries/llvip/' + arm)
        rp = root / ('calibration_receipt.json' if stage == 'calibration' else 'canary.json')
        if not rp.is_file():
            continue
        r = read(rp)
        prefix = 'direction_screen_attempt1_llvip_' + arm + '_' + stage
        job = read(RAW / 'queue' / (prefix + '_job.json'))
        profile_path = RAW / 'queue' / (prefix + '_resource_profile.json')
        profile = read(profile_path) if profile_path.exists() else {}
        peak = max(r['resources']['per_gpu_peak_vram_mib'].values())
        resources.append(dict(arm=arm, stage=stage, status=r['status'], seconds=r['seconds'],
            nvml_peak_mib=peak, vram_reserved_mib=job['vram_mib'], excess_above_vram_reservation_mib=max(0, peak - job['vram_mib']),
            allocated_peak_mib=r['gpu_allocated_peak_mib'], cuda_reserved_peak_mib=r['gpu_reserved_peak_mib'],
            rss_peak_mib=r['resources']['peak_rss_mib'], rss_reservation_mib=job['rss_mib'],
            monitor_status=profile.get('status'), monitor_errors=profile.get('monitor_errors'),
            full_card_minimum_free_mib=profile.get('minimum_free_mib'),
            successful_updates=r.get('successful_updates'), attempts=r.get('attempts'), amp_skips=r.get('amp_skips'),
            ema_updates=r.get('ema_updates'), batches=r.get('batches'), selected_objects=r.get('selected_objects')))
    # These are batch-object occurrences and observed source exposures, not stable physical-object identities.
    summary = dict(status='DESCRIPTIVE_FIXED8_CALIBRATION_ONLY', collected_at=read(HERE / 'collection_receipt.json')['recorded_at'],
        dataset='llvip', batches=8, image_exposures=len(images), unique_images=len(full_images),
        all_source_groups=dict(Counter(groups[PurePosixPath(x).stem] for x in images)),
        gates=dict(all_gate_counts), base_unique_images=len(base_images), selected_unique_images=len(selected_images),
        selected_batches=sum(rows[i]['stats']['L2-box']['selected_count'] > 0 for i in range(8)),
        selected_over_base=len(selected) / len(bases), base_group_object_occurrences=concentration(base_groups),
        selected_group_object_occurrences=concentration(selected_groups), selected_group_unique_images=concentration(selected_image_groups),
        selected_image_object_occurrences=concentration(Counter(x['image'] for x in selected)),
        controls_base_records_and_selection_exact=shared_masks_exact,
        all_base_rgb_ir_GT_coordinates_equal=all(x['rgb_gt'] == x['ir_gt'] for x in bases),
        selected_mapped_iou_minus_teacher_own_iou=describe([x['mapped_teacher_rgb_iou'] - x['teacher_own_iou'] for x in selected]),
        gradients=gradient, gradient_parameter_scope=receipt['parameter_names'], resources=resources,
        scope='GT-conditioned object-relative coordinate regression; fixed 8 initial-state resets, shared P3/P4 feature-parameter gradient, no optimizer updates during calibration.',
        omitted_canaries=read(HERE / 'collection_receipt.json')['canary_status'],
        AP_or_KD_gain_claim=False, L1_geometry_admitted=False, new_hash_computed=False,
        inputs=[str(RAW / 'calibration/llvip/calibration_receipt.json'), str(RAW / 'calibration/llvip/calibration_batches.jsonl'), str(GROUPS)])
    with (HERE / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (HERE / 'BATCH_TABLE.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(batch_rows[0]));w.writeheader();w.writerows(batch_rows)
    with (HERE / 'GROUP_TABLE.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f);w.writerow(['group', 'all_image_exposures', 'base_object_occurrences', 'selected_object_occurrences', 'selected_unique_images'])
        for group in sorted(summary['all_source_groups']):
            w.writerow([group, summary['all_source_groups'][group], base_groups[group], selected_groups[group], selected_image_groups[group]])
    compact = {k: summary[k] for k in ('batches', 'image_exposures', 'unique_images', 'gates', 'base_unique_images',
        'selected_unique_images', 'selected_batches', 'selected_over_base', 'selected_group_object_occurrences', 'gradients', 'resources')}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
