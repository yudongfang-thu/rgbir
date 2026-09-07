"""Stricter paired-box support on the existing empirical exact-image contract.

No entry or correspondence is created here. Both modality GT rectangles and
their empirical-error boundary neighborhoods must lie in both observed point
hulls. Sparse point interpolation remains an empirical assumption, not dense
registration truth. The original contract's global quadrant rule is preserved.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
LEGACY_DIR = _HERE / "task_conditional_reference"
if not LEGACY_DIR.is_dir():
    LEGACY_DIR = _HERE.parent / "rgbir_task_conditional_v1"
if str(LEGACY_DIR) not in sys.path:
    sys.path.append(str(LEGACY_DIR))
from geometry_contract import GeometryContract, inside_convex
from localization_loss import LocalizationConfig, _layout, _labels, _match_objects, _iou

SUPPORT_VERSION = "paired_gt_boundary_empirical_error_v2"


def _corners(box):
    x1, y1, x2, y2 = np.asarray(box, float)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


class PairedGeometrySupport:
    """Check actual observed regions without weakening old geometry admission."""
    def __init__(self, contract):
        self.contract = GeometryContract.load(contract) if isinstance(contract, (str, Path)) else contract
        if self.contract is None:
            self.contract = GeometryContract.unverified()
        for item in self.contract.entries.values():
            summary = item["summary"]
            errors = [summary[key] for key in ("p95_error_raw_px", "max_error_raw_px", "uncertainty_raw_px")]
            if not np.isfinite(errors).all() or min(errors) < 0:
                raise ValueError("Geometry errors/annotation uncertainty must be finite and nonnegative")
            if not np.isfinite(item["rgb_points"]).all() or not np.isfinite(item["ir_points"]).all():
                raise ValueError("Independent geometry points must be finite")

    def pair_decision(self, path, rgb_box, ir_box, metadata, stride):
        """Return reason and error margin for one matched pair at one stride."""
        rgb, ir = np.asarray(rgb_box, float), np.asarray(ir_box, float)
        if rgb.shape != (4,) or ir.shape != (4,) or not np.isfinite([rgb, ir]).all():
            return "invalid_paired_box", None
        for name, box in (("rgb", rgb), ("ir", ir)):
            reason = self.contract._one(str(path), box, metadata, float(stride))
            if reason != "accepted":
                return name + ":" + reason, None
        key = str(Path(str(path)).absolute().resolve())
        item = self.contract.entries[key]
        matrix = np.asarray(metadata["rgb_matrix"], float)
        scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).max())
        summary = item["summary"]
        # The same observed maximum+annotation uncertainty used by the old
        # error discipline determines the boundary neighborhood, not a new AP-
        # tuned pixel radius. Zero observed/annotation error gives zero radius.
        margin = scale * (summary["max_error_raw_px"] + summary["uncertainty_raw_px"])
        if not np.isfinite(margin) or margin < 0:
            return "invalid_error_margin", None
        inv = np.linalg.inv(matrix)
        output = metadata.get("output_shape")
        for name, box in (("rgb", rgb), ("ir", ir)):
            expanded = box + np.asarray([-margin, -margin, margin, margin])
            if output is not None:
                h, w = map(float, output)
                if expanded[0] < 0 or expanded[1] < 0 or expanded[2] > w or expanded[3] > h:
                    return name + ":boundary_neighborhood_outside_input", margin
            corners = _corners(expanded)
            raw = np.c_[corners, np.ones(4)] @ inv.T
            if not inside_convex(raw[:, :2], item["rgb_hull"]).all() or not inside_convex(raw[:, :2], item["ir_hull"]).all():
                return name + ":uncovered_boundary_neighborhood", margin
        return "accepted", margin

    def anchor_mask(self, path, rgb_box, ir_box, metadata, stride, centers):
        reason, margin = self.pair_decision(path, rgb_box, ir_box, metadata, stride)
        xy = np.asarray(centers, float).reshape(-1, 2)
        if reason != "accepted":
            return np.zeros(len(xy), dtype=bool), reason, margin
        rgb, ir = np.asarray(rgb_box, float), np.asarray(ir_box, float)
        inside_gt = ((xy >= rgb[:2]) & (xy <= rgb[2:]) & (xy >= ir[:2]) & (xy <= ir[2:])).all(-1)
        item = self.contract.entries[str(Path(str(path)).absolute().resolve())]
        raw = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(np.asarray(metadata["rgb_matrix"], float)).T
        observed = inside_convex(raw[:, :2], item["rgb_hull"]) & inside_convex(raw[:, :2], item["ir_hull"])
        return inside_gt & observed, reason, margin


@torch.no_grad()
def build_paired_geometry_mask(reference, batch, strides=(8, 16, 32), config=None, contract=None):
    """Return bool[N_rgb_gt,A] plus diagnostic counts before T-quality gates.

    Pairing is the exact old class/cardinality-first matcher. This function
    never inspects teacher confidence, teacher DFL, or current student values.
    """
    cfg = config if isinstance(config, LocalizationConfig) else LocalizationConfig(**(config or {}))
    centers, _, ranges, size = _layout(reference, cfg, strides)
    b, nc, anchor_count = reference["scores"].shape
    ri, rc, rb = _labels(batch, b, nc, centers.device, size)
    teacher_batch = batch.get("teacher_batch", batch.get("ir_batch"))
    if teacher_batch is None:
        raise ValueError("Independent transformed teacher_batch is required")
    ti, tc, tb = _labels(teacher_batch, b, nc, centers.device, size)
    support = PairedGeometrySupport(contract)
    mask = torch.zeros((len(rb), anchor_count), dtype=torch.bool, device=centers.device)
    reasons, decisions = Counter(), []
    paths, metadata = batch.get("im_file"), batch.get("pair_info")
    if paths is None or metadata is None or len(paths) != b or len(metadata) != b:
        return mask, {"support_version": SUPPORT_VERSION, "verified": bool(support.contract.verified),
                      "reason_counts": {"missing_image_or_transform_metadata": len(rb)}, "decisions": []}
    cpu_centers = centers.detach().cpu().numpy()
    for bi in range(b):
        rglobal = torch.nonzero(ri == bi).flatten()
        tglobal = torch.nonzero(ti == bi).flatten()
        rlocal, tlocal = _match_objects(rb[rglobal], rc[rglobal], tb[tglobal], tc[tglobal], cfg.match_iou)
        for rli, tli in zip(rlocal.tolist(), tlocal.tolist()):
            rgi, tgi = int(rglobal[rli]), int(tglobal[tli])
            if float(_iou(rb[rgi:rgi + 1], tb[tgi:tgi + 1])[0, 0]) < cfg.pair_iou:
                reasons["pair_iou_below_frozen_requirement"] += 1
                continue
            rgb, ir = rb[rgi].cpu().numpy(), tb[tgi].cpu().numpy()
            for level in cfg.levels:
                lo, hi = ranges[level]
                allowed, reason, margin = support.anchor_mask(paths[bi], rgb, ir, metadata[bi], strides[level], cpu_centers[lo:hi])
                mask[rgi, lo:hi] = torch.as_tensor(allowed, device=centers.device)
                reasons[reason] += 1
                decisions.append({"batch_index": bi, "rgb_gt_index": rgi, "ir_gt_index": tgi,
                                  "level": level, "stride": int(strides[level]), "reason": reason,
                                  "boundary_margin_input_px": margin, "admitted_anchor_count": int(allowed.sum())})
    return mask, {"support_version": SUPPORT_VERSION, "verified": bool(support.contract.verified),
                  "rgb_gt_count": len(rb), "objects_with_geometry": int(mask.any(-1).sum()),
                  "geometry_anchor_count": int(mask.sum()), "reason_counts": dict(reasons), "decisions": decisions}
