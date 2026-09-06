"""Frozen engineering implementation of YOLO-OS-SSL-10K-v1.

This package deliberately contains only the reusable SSL mechanics.  Dataset
preparation and detector evaluation remain outside this package.
"""

from .manifest import PairRecord, read_pair_manifest
from .model import BYOLModel, LARS, WarmupCosineSchedule

__all__ = [
    "BYOLModel",
    "LARS",
    "PairRecord",
    "WarmupCosineSchedule",
    "read_pair_manifest",
]
