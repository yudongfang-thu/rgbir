"""CPU-testable snapshots and exact comparisons for native criterion loss items."""
from __future__ import annotations

import copy
import torch


def clone_items(value):
    """Detach tensor leaves while retaining native dict/list/tuple structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_items(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_items(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_items(item) for item in value)
    return copy.deepcopy(value)


def compare_items(left, right):
    """A recursive exact report; container, keys, dtype, shape and values matter."""
    if type(left) is not type(right):
        return dict(exact=False, left_type=type(left).__name__, right_type=type(right).__name__)
    if isinstance(left, torch.Tensor):
        same_shape = tuple(left.shape) == tuple(right.shape)
        return dict(exact=same_shape and left.dtype == right.dtype and bool(torch.equal(left, right)),
            kind="tensor", shape=list(left.shape), dtype=str(left.dtype),
            max_abs_error=(float((left.detach().float()-right.detach().float()).abs().max())
                           if same_shape and left.numel() else 0.0 if same_shape else None))
    if isinstance(left, dict):
        missing = [key for key in left if key not in right]
        extra = [key for key in right if key not in left]
        children = {key: compare_items(item, right[key]) for key, item in left.items() if key in right}
        return dict(exact=not missing and not extra and all(row["exact"] for row in children.values()),
                    kind="dict", missing_keys=missing, extra_keys=extra, children=children)
    if isinstance(left, (list, tuple)):
        children = [compare_items(a, b) for a, b in zip(left, right)]
        return dict(exact=len(left) == len(right) and all(row["exact"] for row in children),
                    kind=type(left).__name__, left_length=len(left), right_length=len(right), children=children)
    return dict(exact=bool(left == right), kind=type(left).__name__)
