import math
import torch
import pytest
from reference_kernels import (pool_relative_logits, classification_relative_kd,
    aligned_dfl_kd, gt_dfl_target, localization_from_target, combine_single_task)


def sample_c(classes=5):
    g = torch.Generator().manual_seed(17)
    s = torch.randn(3, 2, classes, generator=g, requires_grad=True)
    t = torch.randn(3, 2, classes, generator=g, requires_grad=True)
    v = torch.tensor([[True, True], [True, False], [True, True]])
    select = torch.tensor([True, False, True])
    labels = torch.tensor([0, 0, 0], dtype=torch.long)
    return s,t,v,select,labels


def test_relative_pool_shift_invariance():
    z=torch.arange(18,dtype=torch.float32).reshape(3,6).requires_grad_()
    f=torch.tensor([[True,True,False,False,False,False]])
    b=~f
    a,v=pool_relative_logits(z,f,b)
    shifted,_=pool_relative_logits(z+torch.tensor([[10.],[-3.],[.75]]),f,b)
    assert v.tolist()==[True]
    torch.testing.assert_close(a,shifted)
    a.sum().backward()
    assert bool((z.grad[:,:2]>0).all()) and bool((z.grad[:,2:]<0).all())


def test_relative_pool_invalid_and_empty():
    z=torch.randn(3,6,requires_grad=True)
    f=torch.zeros(1,6,dtype=torch.bool); b=~f
    out,v=pool_relative_logits(z,f,b)
    assert not v.any() and out.eq(0).all()
    out.sum().backward(); assert z.grad.eq(0).all()
    out,v=pool_relative_logits(z,f[:0],b[:0]); assert out.shape==(0,3)


def test_c_identical_zero():
    s,t,v,k,y=sample_c()
    loss=classification_relative_kd(s,s.detach(),v,k,y)
    assert abs(loss.item())<1e-6


def test_c_student_only_gradient_and_mask():
    s,t,v,k,y=sample_c()
    loss=classification_relative_kd(s,t,v,k,y); loss.backward()
    assert s.grad.abs().sum()>0 and t.grad is None
    assert s.grad[1].eq(0).all()
    assert s.grad[:,:,1:].abs().sum()>0


def test_c_y_only_no_off_target_grad():
    s,t,v,k,y=sample_c()
    classification_relative_kd(s,t,v,k,y,off_target_weight=0).backward()
    assert s.grad[:,:,1:].eq(0).all()


def test_c_single_class_finite():
    s,t,v,k,y=sample_c(1)
    loss=classification_relative_kd(s,t,v,k,y);loss.backward()
    assert torch.isfinite(loss) and s.grad.abs().sum()>0


def test_c_fixed_base_denominator():
    s,t,v,k,y=sample_c()
    first=classification_relative_kd(s,t,v,k,y)
    # Add an unselected object to base. Signal is unchanged, denominator grows.
    second=classification_relative_kd(torch.cat([s,s[:1]]),torch.cat([t,t[:1]]),
        torch.cat([v,v[:1]]),torch.cat([k,torch.tensor([False])]),torch.cat([y,y[:1]]))
    torch.testing.assert_close(second,first*3/4)


def test_c_empty_and_zero_selected():
    s,t,v,k,y=sample_c()
    zero=classification_relative_kd(s,t,v,torch.zeros_like(k),y)
    zero.backward(); assert zero.item()==0 and s.grad.eq(0).all()
    empty=classification_relative_kd(s[:0],t[:0],v[:0],k[:0],y[:0])
    assert empty.item()==0


def test_c_invalid_selected_level_rejected():
    s,t,v,k,y=sample_c(); v[0]=False
    with pytest.raises(ValueError):classification_relative_kd(s,t,v,k,y)


def test_c_student_temperature_is_used():
    s=torch.tensor([[[2.]]],requires_grad=True); t=torch.tensor([[[0.]]])
    loss=classification_relative_kd(s,t,torch.tensor([[True]]),torch.tensor([True]),torch.tensor([0]),temperature=2)
    loss.backward()
    torch.testing.assert_close(s.grad,2*(torch.sigmoid(s.detach()/2)-.5))


def test_dfl_equal_distribution_and_shift():
    g=torch.Generator().manual_seed(20)
    s=torch.randn(2,4,16,generator=g,requires_grad=True)
    t=s.detach()+3
    assert abs(aligned_dfl_kd(s,t,2).item())<1e-6


def test_dfl_teacher_detached():
    s=torch.randn(2,4,16,requires_grad=True);t=torch.randn(2,4,16,requires_grad=True)
    loss=aligned_dfl_kd(s,t,4);loss.backward()
    assert s.grad.abs().sum()>0 and t.grad is None


def test_dfl_denominator_and_empty():
    s=torch.randn(2,4,16,requires_grad=True);t=torch.randn(2,4,16)
    torch.testing.assert_close(aligned_dfl_kd(s,t,4),aligned_dfl_kd(s,t,2)/2)
    empty=aligned_dfl_kd(s[:0],t[:0],0);assert empty.item()==0


def test_same_mean_different_dfl_is_not_zero():
    p=torch.tensor([.25,.5,.25]).log().repeat(1,4,1)
    q=torch.tensor([.4,.2,.4]).log().repeat(1,4,1)
    # Both distributions have expectation 1 at T=1, but different shape.
    assert aligned_dfl_kd(p,q,1,temperature=1).item()>0.1


def test_gt_two_bins_and_tempering():
    d=torch.tensor([[1.,1.25,0.,2.5]])
    q=gt_dfl_target(d,16,temperature=2)
    torch.testing.assert_close(q.sum(-1),torch.ones(1,4))
    assert q[0,0,1]==1 and q[0,0].count_nonzero()==1
    assert q[0,1].count_nonzero()==2
    expected=math.sqrt(.75)/(math.sqrt(.75)+math.sqrt(.25))
    assert abs(q[0,1,1].item()-expected)<1e-6


def test_gt_out_of_support_rejected():
    with pytest.raises(ValueError):gt_dfl_target(torch.tensor([[-.1,1.,1.,1.]]),16)
    with pytest.raises(ValueError):gt_dfl_target(torch.tensor([[15.,1.,1.,1.]]),16)


def test_combine_single_scalar_and_no_fusion():
    native=torch.tensor([1.,2.,3.],requires_grad=True)
    kd=torch.tensor(2.,requires_grad=True)
    total=combine_single_task(native,kd,6,.1,'L1');total.backward()
    assert abs(kd.grad.item()-.6)<1e-7
    torch.testing.assert_close(native.grad,torch.ones(3))
    assert combine_single_task(native,kd,6,0,'N').item()==6
    with pytest.raises(ValueError):combine_single_task(native,kd,32,.1,'C+L')


def test_dfl_target_rejection():
    s=torch.zeros(1,4,16)
    with pytest.raises(ValueError):localization_from_target(s,torch.ones_like(s),1)
    with pytest.raises(ValueError):aligned_dfl_kd(s,torch.zeros(1,4,15),1)
