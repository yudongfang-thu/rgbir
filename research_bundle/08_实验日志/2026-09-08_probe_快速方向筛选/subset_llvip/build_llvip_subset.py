# coding: utf-8
"""LLVIP train-only metadata adapter for the unchanged natural proportional sampler."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixed_natural_sampler', HERE / 'natural_sampler_source.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)
P = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
PREPARED = P / 'artifacts/rgbt_p3_causal_v1/prepared/llvip'
GROUPS = P / 'data/processed/llvip/splits/grouped_v1'
EXPECTED_FIT = ['02', '03', '05', '06', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18']
EXPECTED_DEV = ['01', '04', '07', '12', '25']


def labels(raw):
    scales = []
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split()]
        if len(v) != 5 or not all(math.isfinite(x) for x in v):
            raise ValueError('Malformed person label')
        c, x, y, w, h = v
        if c != 0 or not all(0 <= a <= 1 for a in [x, y, w, h]) or w <= 0 or h <= 0:
            raise ValueError('Invalid normalized person box')
        scales.append(math.sqrt(w * h))
    return len(scales), statistics.median(scales) if scales else None


def governed(name, expected_n, expected_groups):
    path = GROUPS / (name + '.tsv')
    rows = list(csv.DictReader(path.open(encoding='utf-8'), delimiter='\t'))
    assert len(rows) == expected_n and len({x['stem'] for x in rows}) == expected_n
    assert sorted({x['sequence_prefix'] for x in rows}) == expected_groups
    for row in rows:
        assert row['stem'].startswith(row['sequence_prefix'])
        for role in ['visible', 'infrared']:
            p = Path(row[role])
            assert p.stem == row['stem'] and p.parent.name == 'train', 'Not governed original train pool'
    return path, {x['stem']: x for x in rows}


def data(role):
    path = PREPARED / (role + '.data.yaml')
    value = yaml.safe_load(path.read_text())
    assert value['names'] in [{0: 'person'}, {'0': 'person'}, ['person']]
    assert value['train'] == 'images/fit' and value['val'] == 'images/dev' and not value.get('test')
    root = Path(value['path'])
    assert root == P / 'data/processed/llvip/yolo/grouped_v1' / role
    return path, value, root


def dist(rows):
    from collections import Counter
    return {'images': len(rows), 'objects': sum(x['objects'] for x in rows),
            'empty_images': sum(x['objects'] == 0 for x in rows),
            'class_object_counts': {'person': sum(x['objects'] for x in rows)},
            'class_image_counts': {'person': sum(x['objects'] > 0 for x in rows)},
            'source_groups': dict(sorted(Counter(x['source_group'] for x in rows).items()))}


def self_test():
    assert labels(b'') == (0, None)
    assert labels(b'0 .5 .5 .2 .2') == (1, .2)
    for bad in [b'1 .5 .5 .2 .2', b'0 .5 .5 0 .2', b'0 nan .5 .2 .2']:
        try:
            labels(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid label accepted')
    rows = [{'rgb': str(i), 'source_group': str(i % 7), 'objects': i % 9,
             'median_relative_scale': .01 * (1 + i % 13) if i % 9 else None} for i in range(100)]
    first, alloc, _ = sampler.sample(rows, 32, 20260908)
    second, _, _ = sampler.sample(rows, 32, 20260908)
    assert first == second and len(first) == 32 and sum(x['selected'] for x in alloc) == 32
    assert dist(first)['objects'] == sum(x['objects'] for x in first)
    print(json.dumps({'status': 'CPU_SELF_TEST_PASS', 'gpu_used': False, 'new_hash_computed': False}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    ap.add_argument('--size', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    assert args.output and not args.output.exists() and args.size == 2048 and args.seed == 20260908
    fit_path, fit = governed('fit', 9619, EXPECTED_FIT)
    dev_path, dev = governed('dev', 2406, EXPECTED_DEV)
    assert not set(fit) & set(dev) and not set(EXPECTED_FIT) & set(EXPECTED_DEV)
    v_yaml, v_data, v_root = data('visible')
    t_yaml, t_data, t_root = data('infrared')
    mapping_path = PREPARED / 'mappings/visible_to_infrared_train.json'
    original_mapping = json.loads(mapping_path.read_text())
    assert len(original_mapping) == 9619 and len(set(original_mapping.values())) == 9619
    assert {Path(x).stem for x in original_mapping} == set(fit)
    # Directory metadata and canonical symlink targets are checked, never used as loader paths.
    for root, role in [(v_root, 'visible'), (t_root, 'infrared')]:
        for split, governed_rows in [('fit', fit), ('dev', dev)]:
            images = list((root / 'images' / split).iterdir())
            assert len(images) == len(governed_rows) and {x.stem for x in images} == set(governed_rows)
            for image in images:
                assert image.resolve(strict=True) == Path(governed_rows[image.stem][role]).resolve(strict=True)
    rows = []
    label_bytes = 0
    paired_gt_exact = True
    for v, t in sorted(original_mapping.items()):
        vp, tp = Path(v), Path(t)
        assert vp.parent == v_root / 'images/fit' and tp.parent == t_root / 'images/fit' and vp.stem == tp.stem
        vb = (v_root / 'labels/fit' / (vp.stem + '.txt')).read_bytes()
        tb = (t_root / 'labels/fit' / (vp.stem + '.txt')).read_bytes()
        vn, scale = labels(vb)
        tn, _ = labels(tb)
        assert vn == tn
        paired_gt_exact = paired_gt_exact and vb == tb
        label_bytes += len(vb) + len(tb)
        rows.append({'rgb': v, 'visible': v, 'infrared': t, 'source_group': fit[vp.stem]['sequence_prefix'],
                     'objects': vn, 'median_relative_scale': scale})
    dev_gt = [0, 0]
    for stem in sorted(dev):
        for i, root in enumerate([v_root, t_root]):
            raw = (root / 'labels/dev' / (stem + '.txt')).read_bytes()
            dev_gt[i] += labels(raw)[0]
            label_bytes += len(raw)
    assert dev_gt == [7879, 7879]
    selected, allocation, cuts = sampler.sample(rows, args.size, args.seed)
    # Deterministic replay on the exact same pre-AP train metadata.
    again, alloc_again, cuts_again = sampler.sample(rows, args.size, args.seed)
    assert selected == again and allocation == alloc_again and cuts == cuts_again
    checkpoints = {}
    originals = {}
    for role, source_yaml in [('visible', v_yaml), ('infrared', t_yaml)]:
        run = P / 'runs/rgbt_p3_causal_v1/formal_native/llvip' / (role + '_seed42_native_b32a2')
        cp, argp = run / 'weights/last.pt', run / 'args.yaml'
        argv = yaml.safe_load(argp.read_text())
        assert Path(argv['data']) == source_yaml and argv['seed'] == 42 and argv['epochs'] == 200
        assert argv['batch'] == 32 and argv['imgsz'] == 640 and argv['workers'] == 8
        assert argv['optimizer'] == 'SGD' and argv['lr0'] == .01 and argv['lrf'] == .01
        checkpoints[role] = {'checkpoint': sampler.stat(cp), 'historical_args': sampler.stat(argp),
                             'data_yaml': sampler.stat(source_yaml), 'args_data_path_exact': True,
                             'configured_terminal_lr': argv['lr0'] * argv['lrf'],
                             'identity_scope': 'Path/stat plus original args/data YAML; no checkpoint bytes read or model loaded.'}
        originals[role + '_args.yaml'] = argp.read_bytes()
        originals[role + '_data.yaml'] = source_yaml.read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.absolute()
    def write_json(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for role, value, root in [('visible', v_data, v_root), ('infrared', t_data, t_root)]:
        (out / ('train_' + role + '.txt')).write_text(''.join(x[role] + '\n' for x in selected), encoding='utf-8')
        derived = dict(value, train=str(out / ('train_' + role + '.txt')), val=str(root / 'images/dev'))
        (out / ('data_' + role + '.yaml')).write_text(yaml.safe_dump(derived, sort_keys=False), encoding='utf-8')
        (out / ('dev_' + role + '.txt')).write_text(''.join(str(root / 'images/dev' / (x + '.jpg')) + '\n' for x in sorted(dev)), encoding='utf-8')
    write_json('visible_to_infrared_train.json', {x['visible']: x['infrared'] for x in selected})
    (out / 'source_groups.tsv').write_text('stem\tsequence_prefix\n' + ''.join(x + '\t' + fit[x]['sequence_prefix'] + '\n' for x in sorted(fit)), encoding='utf-8')
    with (out / 'selected_metadata.jsonl').open('w', encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'originals').mkdir()
    for name, raw in originals.items():
        (out / 'originals' / name).write_bytes(raw)
    manifest = {'status': 'CPU_SUBSET_CREATED_NO_TRAINING', 'dataset': 'llvip',
                'method': 'NATURAL_PROPORTIONAL_STRATIFIED_V1', 'seed': args.seed, 'subset_images': args.size,
                'fraction': args.size / 9619, 'full_train': dist(rows), 'subset': dist(selected),
                'binning': cuts, 'allocation': allocation,
                'sampling': 'Unchanged previous sampler: largest remainder proportional allocation in governed prefix x train-GT density x train-GT relative scale, random ties and uniform sampling within strata. No class floors, swaps or second subset.',
                'inputs': {k: sampler.stat(p) for k, p in [('fit_tsv', fit_path), ('dev_tsv', dev_path), ('train_mapping', mapping_path), ('visible_yaml', v_yaml), ('infrared_yaml', t_yaml)]},
                'dev_unchanged': {'images': 2406, 'visible_gt': dev_gt[0], 'infrared_gt': dev_gt[1],
                                  'visible': str(v_root / 'images/dev'), 'infrared': str(t_root / 'images/dev')},
                'source_groups': {'train': EXPECTED_FIT, 'dev': EXPECTED_DEV, 'disjoint': True,
                                  'scope': 'Existing governed sequence_prefix only, not geometric registration or acquisition-independence certification.'},
                'checks': {'full_fit_mapping_and_governed_stems_exact': True, 'processed_to_governed_raw_targets_exact': True,
                           'train_visible_ir_label_bytes_exact': paired_gt_exact, 'same_seed_replay_exact': True,
                           'processed_aliases_preserved_for_label_loading': True},
                'checkpoint_identity': checkpoints, 'label_text_bytes_read': label_bytes,
                'train_gt_only_sampling': True, 'dev_used_only_for_existing_identity_counts': True,
                'test_files_read': 0, 'image_bytes_read': 0, 'checkpoint_bytes_read': 0, 'gpu_used': False,
                'new_hash_computed': False, 'original_inputs_modified': False,
                'interpretation': 'Single-class person coverage and proportional source/density/area support are descriptive. Shared RGB/IR GT is not physical geometry proof; original L1 geometry remains blocked.'}
    write_json('subset_manifest.json', manifest)
    write_json('checkpoint_identity.json', checkpoints)
    print(json.dumps({'status': manifest['status'], 'output': str(out), 'full_train': manifest['full_train'],
                      'subset': manifest['subset'], 'strata': len(allocation), 'dev': manifest['dev_unchanged']}))


if __name__ == '__main__':
    main()
