"""CPU audit of the verbatim Markdown appendix; never imports a trainer/checkpoint."""
import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch

torch.set_num_threads(1)


def audit(kernel_source):
    scope = {}
    exec(compile(kernel_source, 'verbatim_appendix_a.py', 'exec'), scope)
    kd, combine = scope['aligned_dfl_kd'], scope['combine_detection_loss']
    checks = []

    def passed(name, **details):
        checks.append(dict(name=name, status='passed', **details))

    # Independent categorical KL oracle: different distributions per side/anchor.
    s = torch.tensor([[[1., -2.], [0., 3.], [-1., 1.], [2., -1.],
                       [3., 0.], [0., 2.], [-2., 2.], [1., -1.]]], requires_grad=True)
    t = (-s.detach() * 0.7 + torch.arange(8).view(1,8,1)/9).requires_grad_()
    w = torch.tensor([[0.25, 0.75]], requires_grad=True)
    temp, norm = 2., 3.
    loss = kd(s, t, w, normalizer=norm, temperature=temp)
    oracle = 0.
    for a in range(2):
        for d in range(4):
            x = [float(s[0, 2*d+j, a])/temp for j in range(2)]
            y = [float(t[0, 2*d+j, a])/temp for j in range(2)]
            ps = [math.exp(v)/sum(math.exp(z) for z in x) for v in x]
            pt = [math.exp(v)/sum(math.exp(z) for z in y) for v in y]
            oracle += temp**2/4 * float(w[0,a])/norm * sum(q*math.log(q/p) for p,q in zip(ps,pt))
    assert abs(float(loss)-oracle) < 2e-7
    passed('independent_scalar_KL_layout_oracle', actual=float(loss), expected=oracle)
    loss.backward()
    ps = s.detach().reshape(1,4,2,2).permute(0,3,1,2).div(temp).softmax(-1)
    pt = t.detach().reshape(1,4,2,2).permute(0,3,1,2).div(temp).softmax(-1)
    expected_grad = (temp/(4*norm) * w.detach()[:,:,None,None] * (ps-pt)).permute(0,2,3,1).reshape_as(s)
    assert torch.allclose(s.grad, expected_grad, atol=2e-8, rtol=2e-6)
    assert t.grad is None and w.grad is None
    passed('analytic_student_gradient_and_teacher_gate_detach', max_error=float((s.grad-expected_grad).abs().max()))

    z = s.detach().clone().requires_grad_()
    zero = kd(z, t, torch.zeros_like(w), normalizer=7)
    zero.backward()
    assert zero.item() == 0 and torch.equal(z.grad, torch.zeros_like(z))
    passed('zero_gate_differentiable_zero')
    same = kd(s, s.detach(), w, normalizer=norm)
    assert abs(same.item()) < 2e-7
    passed('same_distribution_near_zero', actual=same.item())
    half = kd(s, t, w/2, normalizer=norm)
    assert torch.allclose(half, loss/2, atol=0, rtol=0)
    passed('fixed_denominator_half_gate_half_dose')
    repeated = kd(s.repeat_interleave(2,-1), t.repeat_interleave(2,-1),
                  w.repeat_interleave(2,-1)/2, normalizer=norm)
    assert torch.allclose(repeated, loss, atol=2e-7, rtol=1e-6)
    passed('replicated_anchor_object_balance')
    offset = torch.tensor([[[10.,-7.],[10.,-7.],[-4.,9.],[-4.,9.],
                            [6.,-3.],[6.,-3.],[15.,4.],[15.,4.]]])
    shifted = kd(s+offset, t-offset, w, normalizer=norm)
    assert torch.allclose(shifted, loss, atol=2e-7, rtol=1e-6)
    passed('per_side_logit_offset_invariance')

    native = torch.tensor([1.,2.,3.], requires_grad=True)
    c = torch.tensor(2., requires_grad=True)
    loc = torch.tensor(3., requires_grad=True)
    total = combine(native,c,loc,batch_size=5,classification_weight=.1,localization_weight=.2)
    total.backward()
    assert total.item() == 10 and torch.equal(native.grad,torch.ones_like(native))
    assert c.grad.item() == .5 and loc.grad.item() == 1.
    passed('two_KD_scalars_added_once_native_not_rescaled', total=total.item())

    half_s = s.detach().half().requires_grad_()
    half_t = t.detach().half()
    mixed = kd(half_s,half_t,w.detach().half(),normalizer=norm)
    reference = kd(half_s.float(),half_t.float(),w.detach().half().float(),normalizer=norm)
    mixed.backward()
    assert mixed.dtype == torch.float32 and torch.equal(mixed,reference)
    assert bool(torch.isfinite(half_s.grad).all())
    passed('fp16_input_fp32_kernel_finite_gradient', dtype=str(mixed.dtype))
    if hasattr(torch, 'autocast'):
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            ac = kd(s,t,w,normalizer=norm)
        assert ac.dtype == torch.float32 and torch.equal(ac,loss)
        passed('CPU_bfloat16_autocast_keeps_kernel_FP32')
    else:
        checks.append(dict(name='CPU_bfloat16_autocast_keeps_kernel_FP32',status='not_available_on_this_torch'))

    for name, ss, tt, ww in (
        ('negative_gate',s,t,-w),
        ('NaN_even_zero_gate',s*float('nan'),t,torch.zeros_like(w)),
        ('mismatched_shape',s,t[:,:,:1],w),
    ):
        try:
            kd(ss,tt,ww,normalizer=norm)
        except ValueError:
            passed('reject_'+name)
        else:
            raise AssertionError(name+' not rejected')

    # Scientific counterexample: identical decoded expectation does not identify a DFL target.
    bins = torch.arange(8,dtype=torch.float32)
    student_p = torch.tensor([.001,.001,.001,.001,.800,.194,.001,.001])
    teacher_p = torch.tensor([.001,.001,.497,.001,.001,.001,.497,.001])
    student_p /= student_p.sum(); teacher_p /= teacher_p.sum()
    student_logits = student_p.log().repeat(4).reshape(1,32,1).requires_grad_()
    teacher_logits = teacher_p.log().repeat(4).reshape(1,32,1)
    gt_bin = torch.tensor([4,4,4,4])
    native_ce = torch.nn.functional.cross_entropy(student_logits.reshape(4,8),gt_bin)
    loc_loss = kd(student_logits,teacher_logits,torch.ones(1,1),normalizer=1,temperature=2.)
    g_gt = torch.autograd.grad(native_ce,student_logits,retain_graph=True)[0]
    g_kd = torch.autograd.grad(loc_loss,student_logits)[0]
    stepped = student_logits.detach()-.01*g_kd
    ce_after = torch.nn.functional.cross_entropy(stepped.reshape(4,8),gt_bin)
    mean_s, mean_t = float((student_p*bins).sum()), float((teacher_p*bins).sum())
    assert abs(mean_t-4)<abs(mean_s-4) and float((g_gt*g_kd).sum())<0 and ce_after>native_ce
    passed('counterexample_better_expectation_not_guaranteed_GT_DFL_gradient',
           teacher_mean=mean_t,reference_mean=mean_s,GT_distance=4,
           native_gradient_dot_KD_gradient=float((g_gt*g_kd).sum()),
           native_CE_before=float(native_ce),native_CE_after_KD_step=float(ce_after),
           interpretation='Method hypothesis limitation, NOT a kernel bug or measured AP result')

    # GT control is not uniquely specified until q's temperature convention is fixed.
    q=torch.tensor([.3,.7])
    raw_optimum=q.pow(2); raw_optimum/=raw_optimum.sum()
    q_softened=q.sqrt(); q_softened/=q_softened.sum()
    recovered=q_softened.pow(2); recovered/=recovered.sum()
    assert torch.allclose(recovered,q,atol=1e-7)
    assert not torch.allclose(raw_optimum,q,atol=.01)
    passed('GT_DFL_temperature_changes_native_target_if_q_not_softened',
           GT_two_bin_probabilities=q.tolist(),
           native_optimum_when_T2_target_left_unmodified=raw_optimum.tolist(),
           native_optimum_when_target_softened_consistently=recovered.tolist(),
           interpolation_expected_distance=4.7,
           native_optimum_distance_when_T2_target_left_unmodified=4.+raw_optimum[1].item(),
           interpretation='Section12.4 needs a frozen convention; NOT an Appendix A bug')

    result=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch_threads=torch.get_num_threads(),
                scope='CPU synthetic tensors only; no model/checkpoint/data/GPU loaded',checks=checks)
    if '--native-source' in sys.argv:
        import inspect
        import ultralytics
        from ultralytics.utils.loss import v8DetectionLoss, DFLoss
        from ultralytics.utils.tal import bbox2dist
        result['native_sources'] = dict(ultralytics_version=ultralytics.__version__,
            module_file=inspect.getfile(v8DetectionLoss),
            v8DetectionLoss=inspect.getsource(v8DetectionLoss),
            DFLoss=inspect.getsource(DFLoss), bbox2dist=inspect.getsource(bbox2dist))
    return result


if __name__ == '__main__':
    here=Path(__file__).resolve().parent
    spec=here.parents[1]/'07_研究分析/RGBIR_Task_Conditional_KD_Codex_Spec_20260907.md'
    source=spec.read_text(encoding='utf-8').split('## 附录 A：')[1].split('```python\n',1)[1].split('```',1)[0]
    (here/'verbatim_appendix_a.py').write_text(source,encoding='utf-8')
    result=audit(source)
    (here/'kernel_validation_local.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
