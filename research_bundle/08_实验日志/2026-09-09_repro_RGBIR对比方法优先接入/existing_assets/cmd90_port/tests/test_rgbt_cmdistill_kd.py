from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.rgbt_cmdistill_kd import (
    CMDISTILL_FG_THRESHOLD,
    CMDistillError,
    affinity_matrix,
    box_iou_xyxy,
    build_anchor_points,
    cls_logic_loss,
    cmdistill_terms,
    dfl_project,
    decode_boxes,
    iou_logic_loss,
    pcc_loss,
    pccfd_loss,
    slrd_loss,
)


def _map(value: float, shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(shape, generator=g)
    return base + value


def test_pcc_identical_maps_hit_both_variants() -> None:
    teacher = _map(0.0, (2, 3, 4, 5), seed=7)
    student = teacher.clone()
    assert pcc_loss(student, teacher, "corrected").item() == pytest.approx(0.0, abs=1e-6)
    assert pcc_loss(student, teacher, "literal").item() == pytest.approx(1.0, abs=1e-6)


def test_pcc_literal_keeps_published_wrong_direction() -> None:
    teacher = _map(0.0, (1, 2, 3, 3), seed=1)
    student = -teacher
    assert pcc_loss(student, teacher, "literal").item() == pytest.approx(-1.0, abs=1e-6)
    assert pcc_loss(student, teacher, "corrected").item() == pytest.approx(2.0, abs=1e-6)


def test_pcc_degenerate_map_has_zero_loss() -> None:
    teacher = torch.zeros(1, 2, 3, 3)
    student = torch.randn(1, 2, 3, 3)
    assert pcc_loss(student, teacher, "corrected").item() == 0.0
    assert pcc_loss(student, teacher, "literal").item() == 0.0


def test_pccfd_averages_scales_equally() -> None:
    teacher = (_map(0.0, (1, 2, 4, 4), seed=3), _map(0.0, (1, 4, 2, 2), seed=4))
    student = (teacher[0] + 1.0, teacher[1] - 1.0)
    per_scale = torch.stack(
        [pcc_loss(s, t, "corrected") for s, t in zip(student, teacher)]
    )
    combined = pccfd_loss(student, teacher, "corrected")
    assert combined.item() == pytest.approx(per_scale.mean().item(), rel=1e-5)


def test_affinity_matrix_matches_manual_cosine() -> None:
    feat = _map(0.0, (1, 3, 2, 2), seed=5)
    a = affinity_matrix(feat)[0]
    vectors = feat.reshape(1, 3, 4).transpose(1, 2)[0]
    manual = torch.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            manual[i, j] = torch.nn.functional.cosine_similarity(vectors[i], vectors[j], dim=0)
    assert torch.allclose(a, manual, atol=1e-6)


def test_slrd_scale_invariant_and_positive_off_relation() -> None:
    torch.manual_seed(11)
    teacher = torch.randn(2, 4, 5, 5)
    assert slrd_loss(teacher.clone(), teacher).item() == pytest.approx(0.0, abs=1e-6)
    assert slrd_loss(teacher * 7.0, teacher).item() == pytest.approx(0.0, abs=1e-5)
    perm = torch.randperm(5 * 5)
    shuffled = teacher.reshape(2, 4, 25)[:, :, perm].reshape(2, 4, 5, 5)
    assert slrd_loss(shuffled, teacher).item() > 0.1


def test_dfl_project_uniform_logits_give_midpoint() -> None:
    raw = torch.zeros(1, 64, 6)
    projected = dfl_project(raw)
    assert projected.shape == (1, 6, 4)
    assert torch.allclose(projected, torch.full_like(projected, 7.5))


def test_anchor_points_match_manual_grid_and_order() -> None:
    feats = (torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 1, 1))
    points, strides = build_anchor_points(feats, (8.0, 16.0))
    assert points.shape == (7, 2) and strides.shape == (7, 1)
    expected_first_level = torch.tensor(
        [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [0.5, 1.5], [1.5, 1.5], [2.5, 1.5]]
    )
    assert torch.allclose(points[:6], expected_first_level)
    assert torch.allclose(points[6], torch.tensor([[0.5, 0.5]]))
    assert strides[0].item() == 8.0 and strides[6].item() == 16.0


def test_decode_boxes_matches_hand_computed_xyxy() -> None:
    feats = (torch.zeros(1, 1, 2, 2),)
    raw = torch.zeros(1, 64, 4)
    boxes = decode_boxes(raw, feats, (8.0,))
    # uniform DFL -> 7.5 feature units; anchor (0.5,0.5)*8 = 4 px; dist 7.5*8 = 60 px
    expected_anchor_box = torch.tensor([4 - 60.0, 4 - 60.0, 4 + 60.0, 4 + 60.0])
    assert torch.allclose(boxes[0, 0], expected_anchor_box)
    anchor_last = torch.tensor([1.5 * 8, 1.5 * 8])
    expected_last = torch.cat([anchor_last - 60.0, anchor_last + 60.0])
    assert torch.allclose(boxes[0, 3], expected_last)


def test_box_iou_known_values() -> None:
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[1.0, 0.0, 3.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    iou = box_iou_xyxy(a, b)
    assert iou[0].item() == pytest.approx(1.0 / 3.0)
    assert iou[1].item() == 0.0


def test_iou_logic_loss_directions_and_empty_mask() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [10.0, 10.0, 12.0, 12.0]]])
    fg = torch.tensor([[True, False]])
    assert iou_logic_loss(boxes, boxes.clone(), fg, "corrected").item() == pytest.approx(0.0, abs=1e-6)
    assert iou_logic_loss(boxes, boxes.clone(), fg, "literal").item() == pytest.approx(1.0, abs=1e-6)
    empty = torch.zeros(1, 2, dtype=torch.bool)
    assert iou_logic_loss(boxes, boxes.clone(), empty, "corrected").item() == 0.0


def test_cls_logic_loss_minimized_at_teacher_and_bce_value() -> None:
    teacher_probs = torch.full((1, 3, 2), 0.9, dtype=torch.float64)
    teacher_logits = torch.logit(teacher_probs)
    logits = teacher_logits.clone().requires_grad_(True)
    loss = cls_logic_loss(logits, teacher_logits)
    # BCE against a soft target is minimized (stationary) at student = teacher prob,
    # where its value is the Bernoulli entropy H(0.9), not zero.
    assert loss.item() == pytest.approx(-(0.9 * math.log(0.9) + 0.1 * math.log(0.1)), rel=1e-6)
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-9
    zero_logits = torch.zeros_like(teacher_logits)
    expected = math.log(2.0)
    assert cls_logic_loss(zero_logits, teacher_logits).item() == pytest.approx(expected, rel=1e-5)


def _prediction_pair(batch_size: int = 2, nc: int = 3, seed: int = 0):
    feats = (torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 1))
    n = 16 + 4 + 1
    g = torch.Generator().manual_seed(seed)
    s_boxes = torch.randn(batch_size, 64, n, generator=g)
    s_scores = torch.randn(batch_size, nc, n, generator=g)
    s_maps = [torch.randn(batch_size, 8, f.shape[-2], f.shape[-1], generator=g) for f in feats]
    t_boxes = torch.randn(batch_size, 64, n, generator=g)
    t_scores = torch.randn(batch_size, nc, n, generator=g)
    t_maps = [torch.randn(batch_size, 8, f.shape[-2], f.shape[-1], generator=g) for f in feats]
    student = {"boxes": s_boxes, "scores": s_scores, "feats": s_maps}
    teacher = {"boxes": t_boxes, "scores": t_scores, "feats": t_maps}
    return student, teacher, feats


def test_cmdistill_terms_shapes_and_total() -> None:
    student, teacher, _ = _prediction_pair()
    terms = cmdistill_terms(student, teacher, (8.0, 16.0, 32.0), variant="corrected")
    assert terms.pccfd.ndim == 0 and terms.slrd.ndim == 0
    assert terms.iou.ndim == 0 and terms.cls.ndim == 0
    assert terms.log.item() == pytest.approx(terms.iou.item() + terms.cls.item(), rel=1e-6)
    total = terms.pccfd + terms.slrd + terms.log
    assert torch.isfinite(total)


def test_cmdistill_terms_keep_teacher_gradient_free() -> None:
    student, teacher, _ = _prediction_pair()
    teacher_boxes = teacher["boxes"].detach().requires_grad_(True)
    teacher_scores = teacher["scores"].detach().requires_grad_(True)
    teacher_maps = [m.detach().requires_grad_(True) for m in teacher["feats"]]
    teacher = {"boxes": teacher_boxes, "scores": teacher_scores, "feats": teacher_maps}
    s_boxes = student["boxes"].detach().requires_grad_(True)
    s_scores = student["scores"].detach().requires_grad_(True)
    s_maps = [m.detach().requires_grad_(True) for m in student["feats"]]
    student = {"boxes": s_boxes, "scores": s_scores, "feats": s_maps}
    terms = cmdistill_terms(student, teacher, (8.0, 16.0, 32.0))
    (terms.pccfd + terms.slrd + terms.log).backward()
    assert teacher_boxes.grad is None
    assert teacher_scores.grad is None
    assert all(m.grad is None for m in teacher_maps)
    assert s_boxes.grad is not None and s_scores.grad is not None


def test_cmdistill_terms_student_gradient_reaches_all_channels() -> None:
    student, teacher, _ = _prediction_pair()
    s_boxes = student["boxes"].detach().requires_grad_(True)
    s_scores = student["scores"].detach().requires_grad_(True)
    s_maps = [m.detach().requires_grad_(True) for m in student["feats"]]
    student = {"boxes": s_boxes, "scores": s_scores, "feats": s_maps}
    terms = cmdistill_terms(student, teacher, (8.0, 16.0, 32.0))
    (terms.pccfd + terms.slrd + terms.log).backward()
    assert s_boxes.grad is not None and torch.isfinite(s_boxes.grad).all()
    assert s_scores.grad is not None and torch.isfinite(s_scores.grad).all()
    assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in s_maps)


def test_cmdistill_terms_rejects_unknown_variant() -> None:
    student, teacher, _ = _prediction_pair()
    with pytest.raises(CMDistillError):
        cmdistill_terms(student, teacher, (8.0, 16.0, 32.0), variant="nope")
