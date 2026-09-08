# coding: utf-8
"""Train-GT-only proportional subset; no image reads, model loads, inference or hashes."""
import argparse
import bisect
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import yaml

NAMES = ['car', 'freight car', 'truck', 'bus', 'van']


def stat(path):
    value = path.stat()
    return {'path': str(path), 'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def parse_labels(text):
    counts = [0] * 5
    scales = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError('Expected five-column YOLO GT')
        values = [float(x) for x in parts]
        cls, x, y, w, h = values
        if not all(math.isfinite(v) for v in values) or cls != int(cls) or not 0 <= cls < 5:
            raise ValueError('Invalid GT class or nonfinite label')
        if not all(0 <= v <= 1 for v in [x, y, w, h]) or w <= 0 or h <= 0:
            raise ValueError('Invalid normalized GT box')
        counts[int(cls)] += 1
        scales.append(math.sqrt(w * h))
    return counts, statistics.median(scales) if scales else None


def cutpoints(values, fractions):
    ordered = sorted(values)
    return [ordered[max(0, math.ceil(len(ordered) * q) - 1)] for q in fractions]


def sample(records, size, seed):
    if not 0 < size <= len(records):
        raise ValueError('Invalid subset size')
    positive = [x for x in records if x['objects'] > 0]
    if not positive:
        raise ValueError('No positive training images')
    density_cuts = cutpoints([x['objects'] for x in positive], [.25, .5, .75])
    scale_cuts = cutpoints([x['median_relative_scale'] for x in positive], [1 / 3, 2 / 3])
    strata = defaultdict(list)
    for row in records:
        if row['objects'] == 0:
            density, scale = 'empty', 'empty'
        else:
            density = str(bisect.bisect_right(density_cuts, row['objects']))
            scale = str(bisect.bisect_right(scale_cuts, row['median_relative_scale']))
        key = (row['source_group'], density, scale)
        row['stratum'] = list(key)
        strata[key].append(row)
    rng = random.Random(seed)
    allocation = []
    for key in sorted(strata):
        population = len(strata[key])
        floor, remainder = divmod(population * size, len(records))
        allocation.append({'key': key, 'population': population, 'selected': floor, 'remainder': remainder, 'tie': rng.random()})
    remaining = size - sum(x['selected'] for x in allocation)
    for row in sorted(allocation, key=lambda x: (-x['remainder'], x['tie'], x['key']))[:remaining]:
        row['selected'] += 1
    selected = []
    for row in allocation:
        pool = sorted(strata[row['key']], key=lambda x: x['rgb'])
        selected += rng.sample(pool, row['selected'])
        assert abs(row['selected'] - row['population'] * size / len(records)) < 1.000000001
    selected.sort(key=lambda x: x['rgb'])
    assert len(selected) == size and len({x['rgb'] for x in selected}) == size
    return selected, allocation, {'positive_image_density_quartiles': density_cuts,
                                  'positive_image_median_scale_tertiles': scale_cuts,
                                  'scale_definition': 'Per-image median sqrt(normalized GT width * height); relative image area, not feature stride or pixel dimensions.',
                                  'boundary_rule': 'bisect_right; tied thresholds may leave bins empty; empty images have a separate stratum.'}


def distribution(rows):
    n = len(rows)
    object_counts = [sum(row['class_counts'][c] for row in rows) for c in range(5)]
    image_counts = [sum(row['class_counts'][c] > 0 for row in rows) for c in range(5)]
    return {'images': n, 'objects': sum(object_counts), 'empty_images': sum(x['objects'] == 0 for x in rows),
            'source_groups': dict(sorted(Counter(x['source_group'] for x in rows).items())),
            'class_object_counts': dict(zip(NAMES, object_counts)), 'class_image_counts': dict(zip(NAMES, image_counts)),
            'class_image_prevalence': {name: count / n for name, count in zip(NAMES, image_counts)}}


def dataset(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = data['names']
    if isinstance(names, dict):
        names = [names[k] if k in names else names[str(k)] for k in range(5)]
    if names != NAMES:
        raise ValueError('Unexpected class mapping')
    root = Path(data['path'])
    if not root.is_absolute() or not isinstance(data['train'], str) or not isinstance(data['val'], str):
        raise ValueError('Expected original absolute-root scalar train/val specification')
    train = root / data['train']
    val = root / data['val']
    if train == val or train.name != 'train' or train.parent.name != 'images':
        raise ValueError('Unexpected original train split')
    return data, root, train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-yaml', type=Path)
    parser.add_argument('--privileged-yaml', type=Path)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--expected-train', type=int, default=17990)
    parser.add_argument('--rgb-checkpoint', type=Path)
    parser.add_argument('--ir-checkpoint', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert parse_labels('')[0] == [0] * 5
        assert parse_labels('4 0.5 0.5 0.2 0.1')[0] == [0, 0, 0, 0, 1]
        for bad in ['0 .5 .5 0 .2', '5 .5 .5 .1 .1', '0 .5 nan .1 .1']:
            try:
                parse_labels(bad)
                raise AssertionError('Invalid label accepted')
            except ValueError:
                pass
        rows = [{'rgb': str(i), 'source_group': str(i % 7), 'objects': i % 9,
                 'median_relative_scale': .01 * (1 + i % 13) if i % 9 else None,
                 'class_counts': [i % 9, 0, 0, 0, 0]} for i in range(100)]
        a, alloc, _ = sample(rows, 32, 20260908)
        b, _, _ = sample(rows, 32, 20260908)
        assert [x['rgb'] for x in a] == [x['rgb'] for x in b]
        assert sum(x['selected'] for x in alloc) == 32 and all(x['selected'] <= x['population'] for x in alloc)
        print(json.dumps({'status': 'CPU_SELF_TEST_PASS', 'image_reads': 0, 'gpu_used': False, 'hashes_computed': False}))
        return
    if not all([args.student_yaml, args.privileged_yaml, args.mapping, args.groups, args.output]):
        parser.error('Input YAMLs, mapping, groups and a new output directory are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    rgb_data, rgb_root, rgb_train, rgb_val = dataset(args.student_yaml)
    ir_data, ir_root, ir_train, ir_val = dataset(args.privileged_yaml)
    source_mapping = json.loads(args.mapping.read_text(encoding='utf-8'))
    if not isinstance(source_mapping, dict) or len(source_mapping) != args.expected_train:
        raise ValueError('Original train mapping count differs')
    groups = {}
    for line in args.groups.read_text(encoding='utf-8').splitlines():
        stem, group = line.split('\t')
        if stem in groups:
            raise ValueError('Duplicate source-group stem')
        groups[stem] = group
    records = []
    label_bytes = 0
    label_mtimes = []
    for rgb, ir in sorted(source_mapping.items()):
        rgb_path, ir_path = Path(rgb), Path(ir)
        if rgb_path.parent != rgb_train or ir_path.parent != ir_train or rgb_path.stem != ir_path.stem:
            raise ValueError('Mapping leaves original train split or mismatched pair stems')
        if rgb_path.stem not in groups:
            raise ValueError('Source group missing')
        label = rgb_root / 'labels/train' / (rgb_path.stem + '.txt')
        ir_label = ir_root / 'labels/train' / (ir_path.stem + '.txt')
        if not ir_label.is_file():
            raise FileNotFoundError(ir_label)
        raw = label.read_bytes()
        counts, median_scale = parse_labels(raw.decode('utf-8-sig'))
        label_bytes += len(raw)
        label_mtimes.append(label.stat().st_mtime_ns)
        records.append({'rgb': rgb, 'infrared': ir, 'source_group': groups[rgb_path.stem],
                        'rgb_label': str(label), 'class_counts': counts, 'objects': sum(counts),
                        'median_relative_scale': median_scale})
    assert len({Path(x['rgb']).stem for x in records}) == args.expected_train
    selected, allocation, cuts = sample(records, args.size, args.seed)
    full_dist, subset_dist = distribution(records), distribution(selected)
    checkpoint_identity = {}
    for role, path in [('rgb_N42_student_warmstart_and_reference', args.rgb_checkpoint), ('infrared_T42_teacher', args.ir_checkpoint)]:
        if path is not None:
            checkpoint_identity[role] = stat(path)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.absolute()
    inputs = {name: stat(path) for name, path in [('student_yaml', args.student_yaml), ('privileged_yaml', args.privileged_yaml), ('train_mapping', args.mapping), ('source_groups', args.groups)]}
    (output / 'train_rgb.txt').write_text(''.join(x['rgb'] + '\n' for x in selected), encoding='utf-8')
    (output / 'train_infrared.txt').write_text(''.join(x['infrared'] + '\n' for x in selected), encoding='utf-8')
    def write_json(name, data):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_json('rgb_to_infrared_train.json', {x['rgb']: x['infrared'] for x in selected})
    for data, root, val, modality in [(rgb_data, rgb_root, rgb_val, 'rgb'), (ir_data, ir_root, ir_val, 'infrared')]:
        derived = dict(data)
        derived['path'] = str(root)
        derived['train'] = str(output / ('train_' + modality + '.txt'))
        derived['val'] = str(val)
        (output / ('data_' + modality + '.yaml')).write_text(yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding='utf-8')
    selected_names = {x['rgb'] for x in selected}
    with (output / 'train_metadata.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(dict(record, selected=record['rgb'] in selected_names), ensure_ascii=False) + '\n')
    manifest = {'status': 'CPU_SUBSET_CREATED_NO_TRAINING', 'method': 'NATURAL_PROPORTIONAL_STRATIFIED_V1',
                'seed': args.seed, 'subset_images': args.size, 'full_train_images': len(records), 'fraction': args.size / len(records),
                'sampling': 'Largest-remainder allocation proportional to source_group x GT-density x GT-relative-scale; uniform sampling without replacement inside each stratum. No class floors, swaps, duplicated images or second training subset.',
                'binning': cuts, 'allocation': allocation, 'full_train': full_dist, 'subset': subset_dist,
                'class_prevalence_change': {name: subset_dist['class_image_prevalence'][name] - full_dist['class_image_prevalence'][name] for name in NAMES},
                'missing_classes_in_subset': [name for name in NAMES if subset_dist['class_object_counts'][name] == 0],
                'class_sampling_used': False, 'predictions_or_dev_AP_used': False,
                'dev_unchanged': {'rgb': str(rgb_val), 'infrared': str(ir_val), 'expected_images_from_existing_contract': 1469, 'dev_files_read': 0},
                'inputs': inputs, 'label_summary': {'rgb_files_read': len(records), 'rgb_total_bytes': label_bytes, 'mtime_ns_min': min(label_mtimes), 'mtime_ns_max': max(label_mtimes), 'ir_labels_existence_checked': len(records)},
                'checkpoint_identity': checkpoint_identity, 'checkpoint_identity_scope': 'Path/size/mtime stat only; no model load, weight read or hash.',
                'group_scope': 'Existing source-folder grouping, not physical registration or acquisition-independence proof.',
                'image_files_read': 0, 'gpu_used': False, 'new_hash_computed': False, 'original_split_mapping_labels_or_yaml_modified': False,
                'interpretation': 'Approximate preservation of the observed joint strata, with rounding and random class-frequency variation. All three warmstart arms must use this same fixed subset; rare-class absence remains a limitation, not a trigger for AP-guided resampling.'}
    write_json('subset_manifest.json', manifest)
    write_json('checkpoint_identity.json', checkpoint_identity)
    print(json.dumps({'status': manifest['status'], 'output': str(output), 'images': args.size, 'strata': len(allocation),
                      'full_class_images': full_dist['class_image_counts'], 'subset_class_images': subset_dist['class_image_counts'],
                      'full_class_objects': full_dist['class_object_counts'], 'subset_class_objects': subset_dist['class_object_counts'],
                      'missing_classes': manifest['missing_classes_in_subset']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
