"""CPU-only paired-image registration illustrations from completed baseline probes.

No detector loading, GPU use, fitting, registration correction, or source mutation.
Selection matches feature_case_1 exactly; local ROI is a median-area RGB GT.
All outputs are new files under an exclusive registration_panels directory.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

DATASETS = ('dronevehicle', 'llvip', 'vedai')
CYAN = '#00dce7'
MAGENTA = '#ff55d2'
YELLOW = '#ffe04c'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def writej(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_source(record):
    path = Path(record['path'])
    actual = sha256(path)
    if actual != record['sha256']:
        raise ValueError(f'Source hash changed: {path}')
    return {'path': str(path), 'sha256': actual}


def gradient_magnitude(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    scale = max(float(np.quantile(mag, .98)), 1e-8)
    return gray, np.clip(mag / scale, 0, 1), scale


def gradient_overlay(rgb, ir):
    # Only a normalized-grid display resize when dimensions differ; no estimated warp.
    h, w = rgb.shape[:2]
    view_ir = ir if ir.shape[:2] == (h, w) else cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
    gr, er, sr = gradient_magnitude(rgb)
    gi, ei, si = gradient_magnitude(view_ir)
    base = .10 * (.5 * gr + .5 * gi)
    overlay = np.stack((base + .90 * ei, base + .90 * er, base + .45 * (er + ei)), axis=-1)
    return np.clip(overlay, 0, 1), {'rgb_gradient_p98': sr, 'ir_gradient_p98': si,
                                  'ir_resized_for_common_display_grid': ir.shape[:2] != (h, w),
                                  'display_grid_hw': [h, w]}


def median_area_roi(gt):
    if not len(gt):
        return [.30, .30, .70, .70], None
    area = np.clip(gt[:, 3] - gt[:, 1], 0, None) * np.clip(gt[:, 4] - gt[:, 2], 0, None)
    valid = np.flatnonzero(area > 0)
    if not len(valid):
        raise ValueError('No positive-area RGB GT')
    order = valid[np.lexsort((valid, area[valid]))]
    idx = int(order[(len(order) - 1) // 2])
    _, x1, y1, x2, y2 = gt[idx]
    cx, cy = .5 * (x1 + x2), .5 * (y1 + y2)
    rw, rh = max(3.0 * (x2 - x1), .15), max(3.0 * (y2 - y1), .15)
    roi = [float(max(0, cx - rw / 2)), float(max(0, cy - rh / 2)),
           float(min(1, cx + rw / 2)), float(min(1, cy + rh / 2))]
    return roi, {'rgb_gt_index': idx, 'class_id': int(gt[idx, 0]),
                 'gt_normalized_xyxy': gt[idx, 1:].tolist(), 'normalized_area': float(area[idx]),
                 'rank_zero_based': int((len(order) - 1) // 2), 'n_positive_area_rgb_gt': int(len(valid))}


def show(ax, im, rgb=False):
    data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if rgb else im
    ax.imshow(data, extent=(0, 1, 1, 0), interpolation='nearest')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#c7d0d8')


def boxes(ax, gt, color, style='-', linewidth=.8, selected=None):
    for idx, row in enumerate(gt):
        _, x1, y1, x2, y2 = row
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=1.7 if idx == selected else linewidth,
                               linestyle=style, alpha=.92))


def roi_box(ax, roi):
    x1, y1, x2, y2 = roi
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           edgecolor=YELLOW, linewidth=1.8, linestyle='--'))


def make_panel(root, dest, dataset):
    src = root / f'{dataset}_full'
    source_json_names = ('input_manifest.json', 'sample_files.json', 'image_metrics.json',
                         'prediction_records.json', 'completion_receipt.json')
    data = {name: readj(src / name) for name in source_json_names}
    if data['completion_receipt.json']['status'] != 'probe_completed':
        raise ValueError(f'Probe incomplete: {src}')
    rows = data['image_metrics.json']
    # Same numpy argsort / Python round as probe_rgbir.py figures() feature_case_1.
    order = np.argsort([r['rgb_luminance'] for r in rows])
    selected_index = int(order[round(.1 * (len(order) - 1))])
    row = rows[selected_index]
    sample_id = row['id']
    manifest = data['input_manifest.json']
    if manifest['sample_ids'][selected_index] != sample_id:
        raise ValueError('Image rows and input manifest order disagree')
    rec = next(r for r in data['prediction_records.json'] if r['id'] == sample_id)
    sample = next(r for r in data['sample_files.json'] if r['id'] == sample_id)
    verified = {kind: [verify_source(r) for r in sample[kind]] for kind in ('files', 'labels')}
    if rec['paths'] != [r['path'] for r in sample['files']]:
        raise ValueError('Prediction/source image path mismatch')
    ims = [cv2.imread(r['path'], cv2.IMREAD_COLOR) for r in sample['files']]
    if any(im is None for im in ims):
        raise ValueError('Image decode failed')
    gt = [np.array(rec[k], dtype=float).reshape(-1, 5) for k in ('gt_rgb', 'gt_ir')]
    roi, selection = median_area_roi(gt[0])
    edge, gradient_meta = gradient_overlay(*ims)
    identical = verified['labels'][0]['sha256'] == verified['labels'][1]['sha256']
    modality = 'NIR' if dataset == 'vedai' else 'IR'
    label_note = ('Label files are byte-identical: shared/copied boxes do not independently verify registration.'
                  if identical else 'Separate label files: differences may reflect alignment, object extent, or annotation choices.')
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    fig.patch.set_facecolor('#ffffff')
    fig.subplots_adjust(left=.025, right=.985, bottom=.19, top=.87, hspace=.14, wspace=.035)
    for r in range(2):
        show(axes[r, 0], ims[0], rgb=True)
        show(axes[r, 1], ims[1], rgb=True)
        show(axes[r, 2], edge)
        boxes(axes[r, 0], gt[0], CYAN, selected=selection['rgb_gt_index'] if selection else None)
        boxes(axes[r, 1], gt[1], MAGENTA)
        boxes(axes[r, 2], gt[0], CYAN, linewidth=.65)
        boxes(axes[r, 2], gt[1], MAGENTA, style='--', linewidth=.65)
    for ax in axes[0]:
        roi_box(ax, roi)
    x1, y1, x2, y2 = roi
    for ax in axes[1]:
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
    axes[0, 0].set_title('RGB image + RGB GT', fontsize=12, loc='left')
    axes[0, 1].set_title(f'{modality} image + {modality} GT', fontsize=12, loc='left')
    axes[0, 2].set_title('Two-color gradient overlay + both GT sets', fontsize=12, loc='left')
    axes[1, 0].set_title('Same local window: RGB', fontsize=11, loc='left')
    axes[1, 1].set_title(f'Same local window: {modality}', fontsize=11, loc='left')
    axes[1, 2].set_title('Local gradients: no estimated alignment applied', fontsize=11, loc='left')
    index_text = f'median-area RGB GT #{selection["rgb_gt_index"]}' if selection else 'fixed center window (no RGB GT)'
    fig.suptitle(f'{dataset.upper()}  |  ID {sample_id}  |  fixed RGB luminance quantile 0.10\n'
                 f'Same sample as feature_case_1; local window selected by {index_text}',
                 x=.026, ha='left', y=.975, fontsize=15, fontweight='semibold', linespacing=1.6)
    legend = [Line2D([0], [0], color=CYAN, lw=2, label='RGB gradients / RGB GT'),
              Line2D([0], [0], color=MAGENTA, lw=2, label=f'{modality} gradients / {modality} GT'),
              Line2D([0], [0], color=YELLOW, lw=2, ls='--', label='Fixed local window')]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.021, .137), ncol=3,
               frameon=False, fontsize=11)
    fig.text(.026, .114, label_note, fontsize=10.3, color='#263746')
    fig.text(.026, .088, 'Edge differences can arise from spectral response as well as geometry; this panel is not a registration-error estimator.',
             fontsize=10.3, color='#263746')
    fig.text(.026, .062, 'Gradients use per-modality 98th-percentile display scaling. Both views use normalized image coordinates; source pixels and labels are unchanged.',
             fontsize=10.1, color='#263746')
    fig.text(.026, .036, f'Original image shapes: RGB {ims[0].shape[1]}x{ims[0].shape[0]}, {modality} {ims[1].shape[1]}x{ims[1].shape[0]}. '
             'Display selection does not use prediction success, box mismatch, CKA, or gradient differences.', fontsize=9.8, color='#54616b')
    png = dest / f'{dataset}_registration.png'
    pdf = dest / f'{dataset}_registration.pdf'
    fig.savefig(png, dpi=170, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    metadata = {'dataset': dataset, 'sample_id': sample_id, 'sample_index': selected_index,
                'luminance_quantile': .1, 'selection_matches': str(src / 'feature_case_1.png'),
                'rgb_luminance': row['rgb_luminance'], 'local_roi_normalized_xyxy': roi,
                'local_selection': selection, 'label_files_byte_identical': identical,
                'gt_counts': [len(x) for x in gt], 'gradient_display': gradient_meta,
                'verified_original_files': verified,
                'source_json_sha256': {name: sha256(src / name) for name in source_json_names},
                'input_probe_script_sha256': manifest['script_sha256'],
                'output_files': [{'path': str(p), 'sha256': sha256(p)} for p in (png, pdf)],
                'interpretation': 'Illustration only. No fitted or applied registration correction, no claim of physical visibility or KD gain.'}
    writej(dest / f'{dataset}_registration.json', metadata)
    print(json.dumps({'dataset': dataset, 'id': sample_id, 'roi_gt': selection, 'png': str(png)}), flush=True)
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    dest = root / 'registration_panels'
    dest.mkdir(exist_ok=False)
    started = time.time()
    cv2.setNumThreads(2)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    shutil.copy2(__file__, dest / Path(__file__).name)
    rows = [make_panel(root, dest, name) for name in DATASETS]
    receipt = {'status': 'completed', 'time_utc': datetime.now(timezone.utc).isoformat(),
               'cpu_only': True, 'gpu_used': False, 'models_loaded': False,
               'source_images_or_labels_changed': False, 'existing_artifacts_overwritten': False,
               'script_sha256': sha256(__file__), 'source_root': str(root), 'output_root': str(dest),
               'datasets': [{'dataset': r['dataset'], 'sample_id': r['sample_id']} for r in rows],
               'duration_seconds': time.time() - started}
    writej(dest / 'completion_receipt.json', receipt)
    (dest / 'README.md').write_text('# Fixed-sample registration illustrations\n\n'
        'CPU-only generation from the completed v2 baseline probe. Each dataset uses the exact sample from feature_case_1 '
        '(RGB luminance quantile 0.10). The local window comes from the lower median RGB GT area, with index tie breaking; '
        'selection does not use registration, predictions, or feature similarity.\n\n'
        'Cyan shows RGB gradients/GT and magenta shows IR or NIR gradients/GT. Shared labels are not independent registration evidence. '
        'Gradient differences include modality response differences. No estimated alignment is applied. '
        'Original source hashes were checked, and source images/labels remain unchanged.\n\n'
        'The per-dataset JSON files contain selected IDs, ROI rules, source hashes, display details and output hashes. '
        'The source script and completion receipt are preserved in this directory.\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
