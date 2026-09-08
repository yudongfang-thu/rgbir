"""Descriptive native-box proxy only; does not execute an L selector."""
import argparse
import json
from pathlib import Path


def readout(source):
    raw = [json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    rows = []
    for r in raw:
        if r['bucket'] != 'both05_onlyT075': continue
        s, t = r['matches']['S']['0.5'], r['matches']['T']['0.5']
        rows.append(dict(stable_rgb_gt_id=r['stable_rgb_gt_id'], stable_ir_gt_id=r['stable_ir_gt_id'],
                         frame_id=r['frame_id'], S=s, T=t, S_iou_lt_070=s['iou'] < .7,
                         T_minus_S_iou=t['iou'] - s['iou'], T_lead_gt_005=t['iou'] - s['iou'] > .05,
                         C_selected=r['C_gates']['selected']))
    assert len(rows) == 11
    return dict(status='NATIVE_BOX_PROXY_READOUT_COMPLETED', scope='SAVED_FIRST32_11_BOTH05_ONLYT075_OBJECTS',
                objects=11, S_iou_lt_070_count=sum(r['S_iou_lt_070'] for r in rows),
                T_lead_gt_005_count=sum(r['T_lead_gt_005'] for r in rows),
                conjunction_count=sum(r['S_iou_lt_070'] and r['T_lead_gt_005'] for r in rows),
                source='output_attempt1/objects.jsonl',
                matching='independent native post-NMS own-GT IoU0.5 pairs; this batch unchanged at IoU0.75',
                actual_L2_selector=False, actual_L1_selector=False, GPU_used=False, new_forward=False,
                new_hash_computed=False, rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(readout(args.input), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
