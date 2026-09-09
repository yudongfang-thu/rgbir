from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_rgbt_cmdistill import CMDistillCriterion, _mean_stats, load_teacher

N_ANCHORS = 16 + 4 + 1  # 4x4 + 2x2 + 1x1 grids
NC = 3
STRIDES = (8.0, 16.0, 32.0)


def _feats(batch: int, channels: int, generator: torch.Generator) -> list[torch.Tensor]:
    shapes = [(4, 4), (2, 2), (1, 1)]
    return [torch.randn(batch, channels, h, w, generator=generator) for h, w in shapes]


class _StubNative:
    """Zero native loss so any student gradient must come from the KD path."""

    def __call__(self, prediction, batch):
        return torch.zeros(()), {"box_loss": torch.zeros(()), "cls_loss": torch.zeros(())}


class _StubTeacher(nn.Module):
    """Frozen privileged-modality detector returning (y_decoded, preds mapping)."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor):
        batch = images.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        preds = {
            "boxes": torch.randn(batch, 64, N_ANCHORS, generator=g),
            "scores": torch.randn(batch, NC, N_ANCHORS, generator=g),
            "feats": _feats(batch, 8, g),
        }
        return None, preds


def _student_prediction(seed: int, requires_grad: bool = True):
    g = torch.Generator().manual_seed(seed)
    boxes = torch.randn(2, 64, N_ANCHORS, generator=g)
    scores = torch.randn(2, NC, N_ANCHORS, generator=g)
    feats = _feats(2, 8, g)
    if requires_grad:
        boxes = boxes.requires_grad_(True)
        scores = scores.requires_grad_(True)
        feats = [m.requires_grad_(True) for m in feats]
    return {"boxes": boxes, "scores": scores, "feats": feats}, (boxes, scores, feats)


def _batch() -> dict[str, torch.Tensor]:
    return {"img": torch.zeros(2, 3, 8, 8), "strong_img": torch.zeros(2, 3, 8, 8)}


def test_load_teacher_accepts_ultralytics_ema_checkpoint(tmp_path: Path) -> None:
    teacher = _StubTeacher(seed=1)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model": None, "ema": teacher}, checkpoint)
    loaded = load_teacher(checkpoint)
    assert isinstance(loaded, _StubTeacher)
    assert loaded.seed == 1


def test_mean_stats_aggregates_logged_kd_terms() -> None:
    assert _mean_stats([{"kd": 1.0, "ratio": 0.25}, {"kd": 3.0, "ratio": 0.75}]) == {
        "kd": 2.0,
        "ratio": 0.5,
    }


def test_cmdistill_criterion_gives_student_kd_gradient_and_keeps_teacher_frozen() -> None:
    teacher = _StubTeacher(seed=21)
    criterion = CMDistillCriterion(_StubNative(), teacher, strides=STRIDES)
    prediction, (boxes, scores, feats) = _student_prediction(seed=8)
    total, items = criterion(prediction, _batch())
    total.backward()
    assert torch.isfinite(total)
    assert boxes.grad is not None and boxes.grad.abs().sum() > 0
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    assert all(m.grad is not None and m.grad.abs().sum() > 0 for m in feats)
    assert torch.isfinite(boxes.grad).all() and torch.isfinite(scores.grad).all()
    assert teacher.dummy.grad is None
    for key in ("kd_pccfd", "kd_slrd", "kd_iou", "kd_cls", "kd_fg_ratio"):
        assert key in items


def test_cmdistill_criterion_total_is_native_plus_b_times_kd_sum() -> None:
    teacher = _StubTeacher(seed=13)
    criterion = CMDistillCriterion(_StubNative(), teacher, strides=STRIDES)
    prediction, _ = _student_prediction(seed=6, requires_grad=False)
    total, items = criterion(prediction, _batch())
    kd_sum = float(items["kd_pccfd"]) + float(items["kd_slrd"]) + float(items["kd_iou"]) + float(items["kd_cls"])
    assert total.item() == pytest.approx(2.0 * kd_sum, rel=1e-5)  # B=2


def test_cmdistill_criterion_falls_back_to_native_without_strong_half() -> None:
    teacher = _StubTeacher(seed=3)
    criterion = CMDistillCriterion(_StubNative(), teacher, strides=STRIDES)
    prediction, _ = _student_prediction(seed=9, requires_grad=False)
    total, items = criterion(prediction, {"img": torch.zeros(2, 3, 8, 8)})
    assert total.item() == 0.0
    assert "kd_pccfd" not in items
    assert criterion.stats_rows == []


def test_cmdistill_criterion_rejects_legacy_tensor_prediction() -> None:
    teacher = _StubTeacher(seed=17)
    criterion = CMDistillCriterion(_StubNative(), teacher, strides=STRIDES)
    legacy = torch.randn(2, 64 + NC, N_ANCHORS)
    with pytest.raises(RuntimeError, match="lacks boxes/scores/feats"):
        criterion(legacy, _batch())
