"""Load an immutable local copy of the actual OEv1 release; never edit a live run."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEGACY = HERE / 'legacy_oev1'
if not (LEGACY / 'train_object_evidence.py').is_file():
    raise RuntimeError('Missing pinned legacy_oev1 snapshot; deploy the reviewed release first')
sys.path.insert(0, str(LEGACY))
import train_object_evidence as legacy
from paired_rgbir_data import DualLabelRGBIRDataset, PairingContractError
from object_evidence_loss import object_evidence_loss, EvidenceConfig

