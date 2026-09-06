"""Read-only CPU inventory. Execute on 94 via: ssh 94 python3 - < this_file.

No torch/checkpoint deserialization; no image inference; no writes on 94.
"""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

B = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
RAW = Path('/mnt/dataset/yudongfang/datasets')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(p):
    return {'path': str(p), 'sha256': sha(p), 'text': p.read_text(errors='replace')}


def args(p):
    out = {}
    for line in p.read_text().splitlines():
        if line and not line[0].isspace() and ': ' in line:
            k, v = line.split(': ', 1)
            out[k] = v
    return out


def model(rel):
    p = B / 'runs' / rel
    a = p / 'args.yaml'
    c = p / 'results.csv'
    rows = list(csv.DictReader(c.open())) if c.exists() else []
    av = args(a)
    return {
        'run': str(p), 'args': record(a), 'parsed_args': av,
        'data_yaml': record(Path(av['data'])) if Path(av['data']).exists() else None,
        'results_path': str(c), 'results_sha256': sha(c) if c.exists() else None,
        'completed_csv_epochs': len(rows), 'last_csv_row': rows[-1] if rows else None,
        'weights': {q.name: {'path': str(q), 'size_bytes': q.stat().st_size}
                    for q in p.glob('weights/*.pt')},
        'identity_level': 'args + data YAML + CSV + checkpoint existence/size; no tensor metadata read',
    }


def image_dir(p):
    fs = sorted(q for q in p.iterdir() if q.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
    return {'path': str(p), 'count': len(fs), 'first_files': [str(q) for q in fs[:3]],
            'stem_list_sha256': hashlib.sha256('\n'.join(q.stem for q in fs).encode()).hexdigest()}


def labels(p):
    fs = sorted(p.glob('*.txt'))
    return {'path': str(p), 'count': len(fs)}


datasets = {}
for d, mods, split, datarel in [
    ('dronevehicle', ['rgb', 'infrared'], 'val', 'data/processed/dronevehicle/yolo/hbb_v1'),
    ('llvip', ['visible', 'infrared'], 'dev', 'data/processed/llvip/yolo/grouped_v1'),
]:
    datasets[d] = {
        'probe_split': split,
        'models': {m: model(f'rgbt_p3_causal_v1/formal_native/{d}/{m}_seed42_native_b32a2') for m in mods},
        'images': {m: image_dir(B / datarel / m / 'images' / split) for m in mods},
        'labels': {m: labels(B / datarel / m / 'labels' / split) for m in mods},
        'test_accessed': False,
    }
datasets['dronevehicle']['matched_kd_comparator_rgb'] = model('cgkd_w1/native_rgb_s42_e200')

V = B / 'cmdistill_native/data/processed'
O = V / 'VEDAI512_paper8_hbb_official_fold01'
P = V / 'VEDAI512_paper8_hbb_paper80_seed0'
datasets['vedai'] = {
    'probe_split': 'official_fold01 val (same YAML aliases test to this val; not independent test)',
    'models': {
        'rgb': model('rgbt_cmdistill_paper_reconstructed_v2/native_rgb_s42_b64_e200'),
        'ir': model('rgbt_cmdistill_paper_reconstructed_v2/teacher_ir_s42_b64_e200'),
    },
    'images': {m: image_dir(O / 'images' / m / 'val') for m in ['rgb', 'ir']},
    'labels': {m: labels(O / 'labels' / m / 'val') for m in ['rgb', 'ir']},
    'excluded_wrong_split_model': model('cmdistill_vedai_native/teacher_ir_y11_s42'),
    'prepare_manifests': [record(p / 'metadata/prepare_manifest.json') for p in [O, P]],
    'modality_caution': 'IR filename is not evidence of thermal sensing; VEDAI NIR identity must be documented from primary dataset source.',
}
sets = {n: {s: {p.stem for p in (root / 'images/rgb' / s).iterdir()} for s in ['train', 'val']}
        for n, root in [('official_fold01', O), ('paper80_seed0', P)]}
datasets['vedai']['split_counts'] = {n: {s: len(v) for s, v in splits.items()} for n, splits in sets.items()}
datasets['vedai']['cross_split_overlap'] = {
    'paper80_' + a + '_intersect_official_' + b: len(sets['paper80_seed0'][a] & sets['official_fold01'][b])
    for a in ['train', 'val'] for b in ['train', 'val']
}

F = RAW / 'FLIR_aligned/x'
datasets['flir_aligned'] = {
    'root': str(F), 'verified_trained_baseline': None, 'frozen_development_split': None,
    'model_gap_scope': 'No corresponding training args found in RGBT_campaign/runs.',
    'annotation_variants': {}, 'annotation_readme': [record(p) for p in (F / 'coco_annotations').glob('*.txt')],
}
for p in (F / 'coco_annotations').glob('*.json'):
    d = json.loads(p.read_text())
    datasets['flir_aligned']['annotation_variants'][p.name] = {
        'path': str(p), 'sha256': sha(p), 'n_images': len(d.get('images', [])),
        'n_annotations': len(d.get('annotations', [])), 'categories': d.get('categories'),
        'first_image': d.get('images', [None])[0],
    }

M = RAW / 'M3FD'
meta = {p.name: p.read_text().splitlines() for p in (M / 'meta').glob('*.txt')}
imstems = {m: {p.stem for p in (M / m).iterdir()} for m in ['vi', 'ir', 'Vis', 'Ir']}
labstems = {p.stem for p in (M / 'labels').glob('*.txt')}
datasets['m3fd'] = {
    'root': str(M), 'verified_trained_baseline': None, 'frozen_development_split': None,
    'model_gap_scope': 'No corresponding training args found in RGBT_campaign/runs.',
    'directory_counts': {q.name: len(list(q.iterdir())) for q in M.iterdir() if q.is_dir()},
    'metadata_lists': meta,
    'metadata_lists_identical': len({tuple(x) for x in meta.values()}) == 1,
    'labeled_paired_stems': len(labstems & imstems['vi'] & imstems['ir']),
    'image_stems_without_labels': {m: sorted(v - labstems) for m, v in imstems.items()},
    'metadata_files': [record(p) for p in (M / 'meta').glob('*.txt')],
}
K = RAW / 'KAIST'
datasets['kaist'] = {
    'root': str(K), 'verified_trained_baseline': None, 'frozen_development_split': None,
    'extracted_data_available': any(q.is_dir() for q in K.iterdir()),
    'files': [{'path': str(q), 'size_bytes': q.stat().st_size} for q in K.iterdir() if q.is_file()],
}

index = []
for p in sorted((B / 'runs').glob('**/args.yaml')):
    a = args(p)
    index.append({'path': str(p), **{k: a.get(k) for k in ['model', 'data', 'seed', 'epochs', 'batch']}})

out = {
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'server': '94', 'operation': 'read_only_cpu_metadata_inventory',
    'scope': str(B / 'runs'),
    'not_performed': ['training', 'GPU use', 'torch.load', 'test image inference', 'server writes'],
    'datasets': datasets, 'training_args_index': index,
    'new_dataset_arg_hits': [x for x in index if any(t in (x.get('data') or '').lower() for t in ['flir', 'm3fd', 'kaist'])],
    'governance': [record(B / 'governance' / n) for n in ['llvip_grouped_split_v1.json', 'llvip_yolo_grouped_v1.json', 'dronevehicle_yolo_hbb_v1.json']],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
