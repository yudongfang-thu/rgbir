"""Forward-only fixed natural-flow selection diagnostic; no training admission."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

STATUS = 'UNVERIFIED_GEOMETRY_DIAGNOSTIC'
SEED = 20260907
COUNTS = ('rgb_gt_count', 'teacher_gt_count', 'common_count', 'pair_iou_count', 'geometry_count',
          'inside_count', 'support_count', 'unique_owner_count', 'reference_candidate_count',
          'base_count', 'reference_reliable_count', 'both_reliable_count',
          'reference_localization_gap_count', 'teacher_rgb_quality_count', 'teacher_own_quality_count',
          'eligible_count', 'selected_count', 'selected_quality_count')
METADATA_KEYS = ('seed', 'generator_seed', 'worker_init', 'shuffle', 'replacement', 'drop_last',
                 'prefetch_factor', 'workers', 'batch_size', 'dataset_size', 'dataset',
                 'yaml_sources', 'paired_mapping', 'train_manifests', 'augmentation', 'test_accessed')


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def append(path, obj):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def stat(path):
    value = Path(path).stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def compare_trace(actual, expected, batch_index):
    actual = json.loads(json.dumps(actual, allow_nan=False))
    if actual != expected:
        if len(actual) != len(expected):
            raise ValueError('Natural trace image count differs in batch ' + str(batch_index))
        for i, (a, e) in enumerate(zip(actual, expected)):
            differences = sorted(key for key in set(a) | set(e) if a.get(key) != e.get(key))
            if differences:
                raise ValueError('Natural trace mismatch batch=%s position=%s keys=%s' % (batch_index, i, differences))
        raise ValueError('Natural trace differs')


def metadata_matches(actual, expected):
    actual = json.loads(json.dumps(actual, allow_nan=False))
    for key in METADATA_KEYS:
        if actual[key] != expected[key]:
            raise ValueError('Frozen loader metadata differs: ' + key)


def selected_context(records, image_rows, selected_only=False):
    selected = [r for r in records if not selected_only or r['selected']]
    positions = sorted({int(r['batch_index']) for r in selected})
    groups = sorted({image_rows[i]['governed_group'] for i in positions
                     if image_rows[i]['governed_group'] is not None})
    return {'objects': len(selected), 'class_counts': dict(Counter(str(r['class']) for r in selected)),
            'image_positions': positions, 'unique_images': len(positions), 'source_groups': groups,
            'unknown_group_images': sum(image_rows[i]['governed_group'] is None for i in positions),
            'stride_counts': dict(Counter(str(r['stride']) for r in selected))}


def replay_by_image(teacher, reference, batch, config, strides, select, primary):
    """Read-only per-image repeats; every real batch must equal primary result."""
    import torch
    sums = Counter()
    per_image, identities, selected_ids = [], [], []
    distances, quality_masks = [], []
    for position in range(len(batch['im_file'])):
        rgb_mask = batch['batch_idx'].reshape(-1).long() == position
        ir_mask = batch['teacher_batch']['batch_idx'].reshape(-1).long() == position
        rgb_global = rgb_mask.nonzero().flatten().tolist()
        ir_global = ir_mask.nonzero().flatten().tolist()
        def labels(value, mask):
            return {'batch_idx': value['batch_idx'][mask] * 0, 'cls': value['cls'][mask], 'bboxes': value['bboxes'][mask]}
        one = labels(batch, rgb_mask)
        one.update(teacher_batch=labels(batch['teacher_batch'], ir_mask), im_file=[batch['im_file'][position]])
        def raw(value):
            return {'scores': value['scores'][position:position+1], 'boxes': value['boxes'][position:position+1],
                    'feats': [f[position:position+1] for f in value['feats']]}
        result = select(raw(teacher), raw(reference), one, strides=strides, config=config, mode='teacher',
                        geometry_eligible=None, geometry_verified=False, return_records=False)
        counts = {key: int(result.stats[key]) for key in COUNTS}
        distances.append(result.rgb_distances)
        quality_masks.append(result.quality_gate)
        sums.update(counts)
        per_image.append({'position': position, 'image': batch['im_file'][position], 'gate_counts': counts})
        chosen = set(result.selected_indices.tolist())
        for i, (ai, rgi, tgi) in enumerate(zip(result.anchor_indices.tolist(), result.rgb_gt_indices.tolist(), result.ir_gt_indices.tolist())):
            identity = (position, ai, rgb_global[rgi], ir_global[tgi])
            identities.append(identity)
            if i in chosen:
                selected_ids.append(identity)
    expected = list(zip(primary.batch_indices.tolist(), primary.anchor_indices.tolist(),
                        primary.rgb_gt_indices.tolist(), primary.ir_gt_indices.tolist()))
    expected_selected = [expected[i] for i in primary.selected_indices.tolist()]
    if identities != expected or selected_ids != expected_selected:
        raise AssertionError('Per-image replay changed original batch GT/anchor identities')
    if any(sums[key] != primary.stats[key] for key in COUNTS):
        raise AssertionError('Per-image replay cumulative gate counts differ from original full batch')
    if not torch.equal(torch.cat(distances), primary.rgb_distances) or not torch.equal(torch.cat(quality_masks), primary.quality_gate):
        raise AssertionError('Per-image replay DFL distances or quality mask differs from original full batch')
    return per_image


def classification_record(selection, batch, image_rows):
    chosen = set(selection.selected.nonzero().flatten().tolist())
    records = []
    for i, (bi, rgi, tgi) in enumerate(selection.base_object_ids):
        records.append({'batch_index': bi, 'image': batch['im_file'][bi], 'rgb_gt_index': rgi,
            'ir_gt_index': tgi, 'class': int(selection.labels[i]), 'selected': i in chosen,
            'eligible': bool(selection.eligible[i]), 'quality': float(selection.quality[i]),
            'valid_levels': selection.valid_levels[i].tolist(), 'governed_group': image_rows[bi]['governed_group']})
    expected = [tuple(x) for x in selection.c0_stats['selected_object_ids']]
    actual = [selection.matched_object_ids[i] for i in selection.selected_matched_indices.tolist()]
    assert expected == actual
    counts = {k: v for k, v in selection.c0_stats.items() if k.endswith('_count') or k in ('normalizer', 'nominal_dose')}
    by_class = {str(y): {'base': sum(r['class'] == y for r in records),
                       'eligible': sum(r['class'] == y and r['eligible'] for r in records),
                       'selected': sum(r['class'] == y and r['selected'] for r in records)}
                for y in sorted(set(r['class'] for r in records))}
    positions = sorted({r['batch_index'] for r in records if r['selected']})
    return {'semantics': 'C0_AND_C1_IDENTICAL_FROZEN_SELECTION', 'c0_c1_selected_exact': True,
            'selection_only_no_student_loss': True, 'counts': counts, 'class_counts': by_class,
            'selected_image_positions': positions, 'selected_unique_images': len(positions),
            'selected_source_groups': sorted({image_rows[p]['governed_group'] for p in positions if image_rows[p]['governed_group'] is not None}),
            'base_records': records}


def run(args):
    if args.batches not in (2, 64):
        raise ValueError('Only two-batch canary or fixed 64-batch diagnostic allowed')
    release = args.release.resolve()
    sys.path.insert(0, str(release))
    import torch
    import yaml
    import coverage_probe as coverage
    import runtime
    from selection_adapter import build_classification_selection
    from localization_loss import LocalizationConfig, build_localization_selection
    from diagnose_opportunities import dataset_config, split_images, source_groups
    # runtime prepends legacy/reference dirs that also contain prepare_configs.py.
    # Bind the independent-v2 generator by its actual file, not the ambiguous name.
    config_spec = importlib.util.spec_from_file_location('natural_flow_frozen_prepare_configs', release / 'prepare_configs.py')
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    configurations = config_module.configurations
    for name, relative in [('coverage_probe', 'coverage_probe.py'), ('runtime', 'runtime.py'),
                           ('selection_adapter', 'selection_adapter.py'),
                           ('localization_loss', 'task_conditional_reference/localization_loss.py')]:
        if Path(sys.modules[name].__file__).resolve() != release / relative:
            raise ValueError('Wrong frozen module loaded: ' + name)
    lease = runtime.legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1:
        raise ValueError('Exactly one existing global GPU lease required')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    prefix = {'llvip': 'llvip', 'dronevehicle': 'drone'}[cfg['dataset']]
    canonical = configurations()[prefix + '_C1']
    for key in ('teacher', 'reference', 'paths', 'imgsz', 'batch', 'workers', 'nbs', 'seed', 'evidence', 'localization', 'augmentation', 'teacher_cache_images'):
        if cfg[key] != canonical[key]:
            raise ValueError('Input differs from frozen configuration: ' + key)
    if cfg['batch'] != 32 or cfg['workers'] != 4 or cfg['imgsz'] != 640:
        raise ValueError('Original B32/workers4/640 required')
    if str(torch.__version__) != cfg['torch_version'] or runtime.legacy.ultralytics.__version__ != str(cfg['ultralytics_version']):
        raise ValueError('Pinned runtime versions differ')
    prior_receipt_path = args.coverage_dir / 'coverage_receipt.json'
    prior_trace_path = args.coverage_dir / 'natural_batches.jsonl'
    previous = json.loads(prior_receipt_path.read_text(encoding='utf-8'))
    if previous['status'] != 'COMPLETED' or previous['actual_batches'] != 64 or previous['dataset'] != cfg['dataset']:
        raise ValueError('Original completed 64-batch coverage identity required')
    old_trace = [json.loads(line) for line in prior_trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(old_trace) != 64 or [r['batch'] for r in old_trace] != list(range(64)):
        raise ValueError('Incomplete original natural trace')
    output = args.output.resolve()
    if not str(output).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('New output must be on the project data disk')
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    iterator = None
    try:
        watched = [args.config, prior_receipt_path, prior_trace_path, Path(previous['roster_path']), Path(cfg['teacher']), Path(cfg['reference'])]
        initial = {str(p): stat(p) for p in watched}
        snapshots = output / 'sources'; snapshots.mkdir()
        source_files = [Path(__file__), release / 'coverage_probe.py', release / 'runtime.py', release / 'selection_adapter.py',
            release / 'prepare_configs.py', release / 'task_conditional_reference/localization_loss.py',
            release / 'task_conditional_reference/diagnose_opportunities.py', release / 'task_conditional_reference/tracked_pair_data.py',
            release / 'task_conditional_reference/legacy_oev1/paired_rgbir_data.py',
            release / 'task_conditional_reference/legacy_oev1/object_evidence_loss.py',
            release / 'task_conditional_reference/legacy_oev1/train_object_evidence.py']
        source_manifest = []
        for i, path in enumerate(source_files):
            target = snapshots / (str(i) + '_' + path.name); shutil.copyfile(path, target)
            source_manifest.append({'source': str(path), 'snapshot': str(target), **stat(path)})
        for original, name in [(args.config, 'input_config.yaml'), (prior_receipt_path, 'original_coverage_receipt.json'),
                               (prior_trace_path, 'original_natural_batches.jsonl'), (Path(previous['roster_path']), 'original_geometry_roster.json')]:
            shutil.copyfile(original, output / name)
        write(output / 'input_manifest.json', {'inputs': initial, 'sources': source_manifest, 'hash_computed': False})
        roster, _ = coverage.load_roster(previous['roster_path'], cfg['dataset'])
        loader = coverage.build_natural_loader(cfg, seed=SEED)
        metadata_matches(loader.coverage_metadata, previous)
        names = loader.coverage_metadata['train_manifests'][0]['names']
        if names != loader.coverage_metadata['train_manifests'][1]['names'] or len(names) != cfg['expected_nc']:
            raise ValueError('Model modalities/classes differ')
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        device = torch.device('cuda:0')
        teacher = runtime.legacy.load_frozen(cfg['teacher'], cfg['paths']['privileged_data_yaml'], names).to(device)
        reference = runtime.legacy.load_frozen(cfg['reference'], cfg['paths']['student_data_yaml'], names).to(device)
        strides = tuple(int(v) for v in reference.stride)
        if tuple(int(v) for v in teacher.stride) != strides:
            raise ValueError('T/R strides differ')
        identities = {}
        for role, model in [('teacher', teacher), ('reference', reference)]:
            args_path = Path(cfg[role]).parents[1] / 'args.yaml'
            original_args = yaml.safe_load(args_path.read_text())
            if original_args['seed'] != 42 or original_args['epochs'] != 200:
                raise ValueError('Expected fixed historical E200 seed42 auxiliary')
            identities[role] = {'path': cfg[role], 'stat': stat(cfg[role]), 'args': original_args,
                                'names': model.names, 'model_eval': not model.training, 'requires_grad': any(p.requires_grad for p in model.parameters())}
        write(output / 'model_identity.json', identities)
        config = LocalizationConfig(**cfg['localization'])
        data = dataset_config(cfg['paths']['student_data_yaml'])
        group_lookup = source_groups(data, split_images(data, 'train'), cfg['dataset'], 'train')
        group_lookup.update({str(Path(k).resolve()): v for k, v in list(group_lookup.items())})
        total_gate = Counter(); gate_images = defaultdict(set); gate_groups = defaultdict(set)
        c_images = set(); c_groups = set(); c_counts = Counter(); class_counts = defaultdict(Counter)
        actual_batches = 0; selected_batches = 0; c_selected_batches = 0
        iterator = iter(loader)
        with torch.no_grad():
            for index in range(args.batches):
                raw_batch = next(iterator)
                if len(raw_batch['im_file']) != 32:
                    raise ValueError('Fixed natural window must retain full batches')
                trace_rows = coverage.summarize_batch(raw_batch, index, roster)
                compare_trace(trace_rows, old_trace[index]['images'], index)
                append(output / 'natural_batches.jsonl', {'batch': index, 'images': trace_rows})
                images = []
                for row in trace_rows:
                    group = group_lookup.get(row['image'], group_lookup.get(row['rgb_source']))
                    if group is not None and str(group).startswith('unavailable:'):
                        group = None
                    images.append({'image': row['image'], 'rgb_source': row['rgb_source'], 'governed_group': group})
                batch = runtime.to_device(raw_batch, device)
                if batch['img'].dtype != torch.float32 or batch['strong_img'].dtype != torch.float32:
                    raise ValueError('Expected one FP32 normalization')
                t = runtime.legacy.raw_prediction(teacher(batch['strong_img']))
                r = runtime.legacy.raw_prediction(reference(batch['img']))
                csel = build_classification_selection(r, t, r, batch, strides=strides, config=cfg['evidence'], selection_seed=cfg['seed'] + index + 1)
                classification = classification_record(csel, batch, images)
                loc = build_localization_selection(t, r, batch, strides=strides, config=config, mode='teacher',
                        geometry_eligible=None, geometry_verified=False, return_records=True)
                if loc.stats['geometry_verified'] or loc.stats['geometry_mask_supplied']:
                    raise AssertionError('Diagnostic must not manufacture verified geometry')
                per_image = replay_by_image(t, r, batch, config, strides, build_localization_selection, loc)
                stats = {k: v for k, v in loc.stats.items() if k != 'loss_unweighted'}
                # Frozen selector computes these as optional diagnostics; this
                # selection-only artifact deliberately does not publish them.
                for record in stats['base_records']:
                    for key in list(record):
                        if key.startswith('reference_logit_') or key.endswith('_dfl_ce') or key.endswith('_dfl_entropy'):
                            record.pop(key)
                stats.update(geometry_status=STATUS, calibration=False, training_admitted=False,
                             per_image_replay_counts_and_ids_exact=True, input_dtype='FP32',
                             base_context=selected_context(stats['base_records'], images),
                             selected_context=selected_context(stats['base_records'], images, True))
                for entry in per_image:
                    row = images[entry['position']]
                    entry.update(rgb_source=row['rgb_source'], governed_group=row['governed_group'])
                    for gate, count in entry['gate_counts'].items():
                        if count:
                            gate_images[gate].add(row['rgb_source'])
                            if row['governed_group'] is not None: gate_groups[gate].add(row['governed_group'])
                total_gate.update({key: int(stats[key]) for key in COUNTS})
                selected_batches += int(stats['selected_count'] > 0)
                c_selected_batches += int(classification['counts']['selected_count'] > 0)
                c_counts.update({k: int(v) for k, v in classification['counts'].items() if k.endswith('_count')})
                for key, counts in classification['class_counts'].items(): class_counts[key].update(counts)
                for position in classification['selected_image_positions']:
                    c_images.add(images[position]['rgb_source'])
                    if images[position]['governed_group'] is not None: c_groups.add(images[position]['governed_group'])
                append(output / 'selection_batches.jsonl', {'batch': index, 'actual_B': 32, 'trace_exact': True,
                    'geometry_status': STATUS, 'images': images, 'classification': classification,
                    'localization': stats, 'localization_per_image': per_image})
                actual_batches += 1
                print(json.dumps({'completed_batches': actual_batches, 'dataset': cfg['dataset'],
                    'C0_C1_selected': classification['counts']['selected_count'], 'L_unverified_selected': stats['selected_count']}), flush=True)
                del raw_batch, batch, t, r, csel, loc, trace_rows, per_image, classification, stats
        if actual_batches != args.batches or any(p.grad is not None for model in (teacher, reference) for p in model.parameters()):
            raise AssertionError('Incomplete diagnostic or frozen model accumulated gradient')
        for path, before in initial.items():
            if stat(path) != before: raise AssertionError('Input changed: ' + path)
        write(output / 'summary.json', {'status': 'COMPLETED', 'diagnostic_status': STATUS, 'dataset': cfg['dataset'],
            'batches': actual_batches, 'canary_only': args.batches == 2, 'seed': SEED,
            'trace_exact_all_recorded_fields': True, 'pixel_bytes_comparable': False,
            'original_coverage': str(args.coverage_dir), 'loader_metadata': loader.coverage_metadata,
            'source_group_method': 'Frozen diagnose_opportunities.source_groups: Drone train TSV required; LLVIP fit.tsv or sequence-prefix fallback; not geometry/calibration groups',
            'classification': {'counts': dict(c_counts), 'selected_batches': c_selected_batches,
                'selected_unique_images': len(c_images), 'selected_images': sorted(c_images),
                'selected_source_groups': sorted(c_groups), 'class_counts': dict(class_counts), 'C0_C1_selection_identical': True},
            'localization': {'counts': dict(total_gate), 'selected_batches': selected_batches,
                'each_gate': {key: {'objects': total_gate[key], 'unique_images': len(gate_images[key]),
                    'images': sorted(gate_images[key]), 'source_groups': sorted(gate_groups[key])} for key in COUNTS},
                'config': asdict(config), 'geometry_verified': False, 'formal_L1_admitted': False,
                'upper_bound_caveat': 'All-true geometry diagnostic; geometry can alter anchor choice, so not a guaranteed upper bound on final selected count'},
            'resources': runtime.legacy.bound_lease_resource_record_from_environment(),
            'gpu_reserved_peak_mib': torch.cuda.max_memory_reserved() / 2**20,
            'gpu_allocated_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
            'seconds': time.time() - started, 'backward_executed': False, 'training_executed': False,
            'calibration_executed': False, 'validation_or_test_accessed': False, 'new_hash_computed': False})
    except BaseException as error:
        write(output / 'failure_receipt.json', {'status': 'FAILED', 'error': repr(error), 'traceback': traceback.format_exc(),
            'seconds': time.time() - started, 'diagnostic_status': STATUS, 'training_admitted': False})
        raise
    finally:
        if iterator is not None and hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--coverage-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, choices=(2, 64), default=64)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
