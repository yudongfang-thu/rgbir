"""Empirical, exact-image geometry support for task-conditional DFL KD.

An image is not verified merely because RGB and IR labels agree. Entries need
independent physical correspondences. The supported region is the intersection
of their convex hulls; no source-folder or unobserved-image extrapolation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def convex_hull(points):
    """Deterministic CCW hull, excluding collinear interior points."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(points) < 3:
        return np.asarray(points)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def inside_convex(points, hull):
    hull = np.asarray(hull, dtype=float)
    if len(hull) < 3:
        return np.zeros(len(points), dtype=bool)
    edges = np.roll(hull, -1, axis=0)-hull
    delta = np.asarray(points)[:, None, :]-hull[None, :, :]
    cross = edges[None, :, 0]*delta[:, :, 1]-edges[None, :, 1]*delta[:, :, 0]
    return (cross >= -1e-7).all(axis=1)


def correspondence_summary(rgb_points, ir_points, original_shape, uncertainty_px=0.0):
    rgb, ir = np.asarray(rgb_points, float), np.asarray(ir_points, float)
    if rgb.shape != ir.shape or rgb.ndim != 2 or rgb.shape[1] != 2:
        raise ValueError("Correspondence arrays must both be [K,2].")
    height, width = original_shape
    quadrants = lambda p: set((p[:, 0] >= width/2).astype(int) +
                              2*(p[:, 1] >= height/2).astype(int))
    errors = np.linalg.norm(rgb-ir, axis=1)
    return dict(point_count=len(rgb), rgb_quadrants=len(quadrants(rgb)),
                ir_quadrants=len(quadrants(ir)),
                p95_error_raw_px=float(np.percentile(errors, 95)) if len(errors) else None,
                max_error_raw_px=float(errors.max()) if len(errors) else None,
                uncertainty_raw_px=float(uncertainty_px),
                point_errors_raw_px=errors.tolist())


class GeometryContract:
    """The formal mask is false outside explicitly accepted image/region evidence.

    Public inputs to object_mask / object_decisions:
      image_paths: B source paths, same order as metadata.
      rgb_xyxy: [K,4] boxes in augmented input PIXELS, not normalized xywh.
      batch_idx: [K] image indices; stride: scalar or [K] input-pixel stride.
      augmentation_metadata: B dictionaries with rgb_matrix, ir_matrix (3x3
          original -> input), original_shape [H,W], optional output_shape.
    Returns bool[K] (torch on input device when boxes are torch tensors).
    """
    def __init__(self, data):
        self.data = data
        self.verified = (data.get("mode") == "verified_identity_grid" and
                         data.get("verified") is True)
        self.entries = {}
        self.path_sources = {}
        for item in data.get("entries", []):
            if item.get("status") != "accepted":
                continue
            if item.get("source") != "independent_physical_correspondences":
                raise ValueError("Accepted geometry must use independent physical points.")
            if item.get("independent_review_required", False) and item.get("review_status") != "accepted":
                raise ValueError("An accepted entry is missing its required independent review.")
            if not item.get("annotator"):
                raise ValueError("Accepted geometry requires an identified annotator.")
            summary = correspondence_summary(item["rgb_points"], item["ir_points"],
                item["original_shape"], item.get("uncertainty_raw_px", 0.0))
            if summary["point_count"] < 6 or min(summary["rgb_quadrants"], summary["ir_quadrants"]) < 3:
                raise ValueError("Accepted entry needs >=6 physical points across >=3 quadrants.")
            for key in item.get("image_paths", []):
                canonical = str(Path(str(key)).absolute().resolve())
                self.path_sources[str(key)] = canonical
                self.entries[canonical] = dict(item, summary=summary,
                    rgb_hull=convex_hull(item["rgb_points"]), ir_hull=convex_hull(item["ir_points"]))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def unverified(cls):
        return cls(dict(mode="verified_identity_grid", verified=False, entries=[]))

    def object_decisions(self, image_paths, rgb_xyxy, batch_idx, augmentation_metadata, stride):
        boxes = _array(rgb_xyxy).astype(float).reshape(-1, 4)
        batch = _array(batch_idx).astype(int).reshape(-1)
        strides = np.broadcast_to(_array(stride).reshape(-1), (len(boxes),))
        reasons = []
        for box, bi, st in zip(boxes, batch, strides):
            reason = self._one(str(image_paths[bi]), box, augmentation_metadata[bi], float(st))
            reasons.append(reason)
        mask = np.asarray([reason == "accepted" for reason in reasons], dtype=bool)
        if hasattr(rgb_xyxy, "detach"):
            import torch
            mask = torch.as_tensor(mask, device=rgb_xyxy.device, dtype=torch.bool)
        return mask, reasons

    def object_mask(self, image_paths, rgb_xyxy, batch_idx, augmentation_metadata, stride):
        return self.object_decisions(image_paths, rgb_xyxy, batch_idx, augmentation_metadata, stride)[0]

    def _one(self, path, box, metadata, stride):
        if not self.verified:
            return "unverified_contract"
        item = self.entries.get(str(Path(path).absolute().resolve()))
        if item is None:
            return "uncovered_image"
        if not metadata or any(k not in metadata for k in ("rgb_matrix", "ir_matrix", "original_shape")):
            return "missing_transform"
        if list(metadata["original_shape"]) != list(item["original_shape"]):
            return "original_shape_mismatch"
        a, b = np.asarray(metadata["rgb_matrix"], float), np.asarray(metadata["ir_matrix"], float)
        if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
            return "invalid_transform"
        if not np.allclose(a[2], [0, 0, 1], atol=1e-7) or not np.allclose(b[2], [0, 0, 1], atol=1e-7):
            return "non_affine_transform"
        if not np.allclose(a, b, atol=1e-6, rtol=0):
            return "different_augmented_grids"
        if abs(np.linalg.det(a[:2, :2])) < 1e-12:
            return "singular_transform"
        x1, y1, x2, y2 = box
        short_edge = min(x2-x1, y2-y1)
        if short_edge <= 0 or stride <= 0:
            return "invalid_object"
        corners = np.asarray([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
        raw = (corners @ np.linalg.inv(a).T)[:, :2]
        if not inside_convex(raw, item["rgb_hull"]).all() or not inside_convex(raw, item["ir_hull"]).all():
            return "uncovered_region"
        scale = float(np.linalg.svd(a[:2, :2], compute_uv=False).max())
        summary = item["summary"]
        p95 = scale*(summary["p95_error_raw_px"] + summary["uncertainty_raw_px"])
        maximum = scale*(summary["max_error_raw_px"] + summary["uncertainty_raw_px"])
        if p95 > min(stride/4, 0.1*short_edge):
            return "p95_exceeds_tolerance"
        if maximum > stride/2:
            return "maximum_exceeds_tolerance"
        return "accepted"
