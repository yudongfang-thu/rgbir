"""Pinned, training-only integration; no edits to any live OEv1 release."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'task_conditional_reference'
LEGACY = REFERENCE / 'legacy_oev1'
if not LEGACY.is_dir():
    raise RuntimeError('Deploy the frozen task_conditional_reference with this release')
sys.path[:0] = [str(LEGACY), str(REFERENCE)]
import train_object_evidence as legacy
from tracked_pair_data import TrackedDualLabelRGBIRDataset
if Path(legacy.__file__).resolve() != (LEGACY/'train_object_evidence.py').resolve():
    raise RuntimeError('A different legacy module was already loaded')
import tracked_pair_data as _tracked_module
if Path(_tracked_module.__file__).resolve() != (REFERENCE/'tracked_pair_data.py').resolve():
    raise RuntimeError('A different tracked loader was already loaded')
for _name in ('object_evidence_loss','paired_rgbir_data'):
    _actual = sys.modules[_name]
    if Path(_actual.__file__).resolve() != (LEGACY/(_name+'.py')).resolve():
        raise RuntimeError('A different frozen helper was already loaded: '+_name)
if TrackedDualLabelRGBIRDataset.__mro__[1] is not legacy.DualLabelRGBIRDataset:
    raise RuntimeError('Tracked loader does not inherit the pinned native pair loader')
ORIGINAL_CRITERION = legacy.EvidenceCriterion
ORIGINAL_DATASET = legacy.DualLabelRGBIRDataset


def to_device(batch, device):
    batch = dict(batch)
    for key in ('img', 'strong_img'):
        batch[key] = batch[key].to(device, non_blocking=True).float() / 255
    for key in ('batch_idx', 'cls', 'bboxes'):
        batch[key] = batch[key].to(device, non_blocking=True)
    batch['teacher_batch'] = {
        key: batch['strong_' + key].to(device, non_blocking=True)
        for key in ('batch_idx', 'cls', 'bboxes')}
    return batch
