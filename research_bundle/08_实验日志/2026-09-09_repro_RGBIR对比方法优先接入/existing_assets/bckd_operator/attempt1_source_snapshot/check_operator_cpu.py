"""Small CPU operator checks; no mmcv install, model, dataset, GPU or hashing."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import torch
import torch.nn.functional as F
from bcdl_independent import bcdl_per_anchor

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()


def original_function():
    path = ROOT / 'official/mmdet/models/losses/kd_loss.py'
    source = path.read_text(encoding='utf-8')
    module = ast.parse(source)
    fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'novel_kd_loss')
    # Slice the original def/body bytes as text; do not rewrite its arithmetic.
    extracted = ''.join(source.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    parsed = ast.parse(extracted).body[0]
    plain = copy.deepcopy(fn)
    plain.decorator_list = []
    assert ast.dump(parsed, include_attributes=False) == ast.dump(plain, include_attributes=False)
    dest = ROOT / 'extracted_novel_kd_loss.py'
    if dest.exists():
        assert dest.read_text(encoding='utf-8') == extracted
    else:
        dest.write_text(extracted, encoding='utf-8')
    namespace = {'torch': torch, 'F': F}
    exec(compile(extracted, str(dest), 'exec'), namespace)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'NovelKDLoss')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    attributes = {n.attr for n in ast.walk(forward) if isinstance(n, ast.Attribute)}
    assert 'T' not in attributes and 'threshold' not in attributes
    return namespace['novel_kd_loss'], {
        'repository_file': str(path.relative_to(ROOT)).replace('\\', '/'),
        'function_def_line': fn.lineno, 'function_end_line': fn.end_lineno,
        'decorators_removed': [ast.unparse(n) for n in fn.decorator_list],
        'function_body_AST_unchanged': True,
        'class_forward_reads_T': False, 'class_forward_reads_threshold': False,
        'mmcv_weighted_loss_and_head_execution_tested': False,
    }


def max_abs(a, b):
    return float((a.detach() - b.detach()).abs().max()) if a.numel() else 0.0


def value_grad(fn, student, teacher):
    s = student.clone().detach().requires_grad_(True)
    t = teacher.clone().detach().requires_grad_(True)
    loss = fn(s, t)
    gs, gt = torch.autograd.grad(loss.sum(), (s, t), allow_unused=True)
    return loss.detach(), gs.detach(), None if gt is None else gt.detach()


def fixed_logits(n, c, dtype):
    x = torch.arange(n * c, dtype=dtype).reshape(n, c)
    return torch.sin(x * .37) * 3.1 - .23, torch.cos(x * .23 + .17) * 2.7 + .11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing receipt overwrite; use a new attempt path')
    official, extraction = original_function()
    checks = []

    def run(name, check):
        try:
            details = check()
            checks.append({'name': name, 'status': 'PASS', 'details': details})
        except Exception as error:
            checks.append({'name': name, 'status': 'FAIL', 'error': str(error),
                           'traceback': traceback.format_exc()})

    def numeric_equivalence():
        rows = []
        for n, c in [(2, 1), (3, 5), (2, 80), (0, 5)]:
            for dtype, atol, rtol in [(torch.float64, 1e-12, 1e-12),
                                      (torch.float32, 1e-6, 1e-5)]:
                s, t = fixed_logits(n, c, dtype)
                a, ga, gta = value_grad(official, s, t)
                b, gb, gtb = value_grad(bcdl_per_anchor, s, t)
                assert a.shape == (n,) and ga.shape == s.shape
                assert torch.isfinite(a).all() and torch.isfinite(ga).all()
                assert torch.allclose(a, b, atol=atol, rtol=rtol)
                assert torch.allclose(ga, gb, atol=atol, rtol=rtol)
                assert gta is None and gtb is None
                rows.append({'n': n, 'classes': c, 'dtype': str(dtype),
                             'atol': atol, 'rtol': rtol,
                             'loss_max_abs': max_abs(a, b), 'student_gradient_max_abs': max_abs(ga, gb),
                             'teacher_gradient_is_none': True,
                             'official_loss_sum': float(a.sum()),
                             'student_gradient_l2': float(ga.norm())})
        return rows

    def beta_and_temperature():
        s, t = fixed_logits(3, 5, torch.float64)
        default = official(s, t)
        explicit = official(s, t, beta=1.0)
        beta2 = official(s, t, beta=2.0)
        invented_t10 = official(s / 10., t / 10.)
        assert torch.equal(default, explicit)
        assert max_abs(default, beta2) > 1e-3
        assert max_abs(default, invented_t10) > 1e-3
        return {'default_beta1_exact': True,
                'beta2_changes_loss_max_abs': max_abs(default, beta2),
                'invented_temperature10_changes_loss_max_abs': max_abs(default, invented_t10),
                'effective_classification_temperature': 1,
                'class_T_attribute_unused': True}

    def teacher_detach():
        s, t = fixed_logits(3, 5, torch.float64)
        _, gs, gt = value_grad(official, s, t)
        _, _, gt_enabled = value_grad(lambda a, b: official(a, b, detach_target=False), s, t)
        assert gt is None and gt_enabled is not None
        assert torch.isfinite(gs).all() and float(gs.norm()) > 0
        assert torch.isfinite(gt_enabled).all() and float(gt_enabled.norm()) > 0
        return {'default_teacher_gradient_is_none': True,
                'student_gradient_l2': float(gs.norm()),
                'explicit_detach_false_teacher_gradient_l2': float(gt_enabled.norm())}

    def student_weight_gradient():
        def wrong_detached_weight(s, t):
            q = t.detach().sigmoid()
            w = (q - s.sigmoid()).abs().detach()
            return (F.binary_cross_entropy_with_logits(s, q, reduction='none') * w).sum(1)
        s, t = fixed_logits(3, 5, torch.float64)
        a, ga, _ = value_grad(official, s, t)
        b, gb, _ = value_grad(wrong_detached_weight, s, t)
        assert torch.equal(a, b)
        delta = max_abs(ga, gb)
        assert delta > 1e-3
        return {'forward_values_bitwise_equal': True,
                'student_gradient_max_abs_if_weight_detached': delta,
                'author_gradient_l2': float(ga.norm()), 'wrong_gradient_l2': float(gb.norm())}

    def weighted_bce_is_not_weighted_kl():
        def weighted_kl(s, t):
            q = t.detach().sigmoid()
            entropy = -(q * q.log() + (1. - q) * (1. - q).log())
            ce = F.binary_cross_entropy_with_logits(s, q, reduction='none')
            return ((ce - entropy) * (q - s.sigmoid()).abs()).sum(1)
        def missing_entropy_term(s, t):
            q = t.detach().sigmoid()
            entropy = -(q * q.log() + (1. - q) * (1. - q).log())
            return (entropy * (q - s.sigmoid()).abs()).sum(1)
        s, t = fixed_logits(3, 5, torch.float64)
        a, ga, _ = value_grad(official, s, t)
        k, gk, _ = value_grad(weighted_kl, s, t)
        h, gh, _ = value_grad(missing_entropy_term, s, t)
        assert max_abs(ga, gk) > 1e-3
        assert torch.allclose(a - k, h, atol=1e-12, rtol=1e-12)
        assert torch.allclose(ga - gk, gh, atol=1e-12, rtol=1e-12)
        return {'BCE_per_anchor': a.tolist(), 'weighted_KL_per_anchor': k.tolist(),
                'weighted_teacher_entropy_per_anchor': h.tolist(),
                'student_gradient_max_abs_BCE_vs_KL': max_abs(ga, gk),
                'gradient_difference_equals_weighted_entropy_gradient': True}

    def identical_and_bad_shape():
        s, _ = fixed_logits(2, 5, torch.float64)
        loss, gs, gt = value_grad(official, s, s)
        assert torch.equal(loss, torch.zeros_like(loss))
        assert torch.equal(gs, torch.zeros_like(gs)) and gt is None
        caught = False
        try:
            official(torch.zeros(2, 5), torch.zeros(2, 4))
        except AssertionError:
            caught = True
        assert caught
        return {'identical_logits_loss_and_student_gradient_exact_zero': True,
                'shape_mismatch_rejected': True}

    run('author_vs_independent_loss_student_gradient_8_small_cases', numeric_equivalence)
    run('beta1_and_actual_temperature1', beta_and_temperature)
    run('teacher_detached_student_nonzero', teacher_detach)
    run('student_difference_weight_must_retain_gradient', student_weight_gradient)
    run('weighted_BCE_cannot_be_replaced_with_weighted_KL', weighted_bce_is_not_weighted_kl)
    run('identical_logits_and_shape_guard', identical_and_bad_shape)
    assert not torch.cuda.is_initialized()
    passed = all(row['status'] == 'PASS' for row in checks)
    source = json.loads((ROOT / 'SOURCE_RECEIPT.json').read_text(encoding='utf-8'))
    result = {'status': 'PASS_BCDL_CLASSIFICATION_OPERATOR_ONLY' if passed else 'FAIL',
              'scope': 'BCKD_BCDL_PER_ANCHOR_CPU_ONLY',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.executable, 'torch': torch.__version__, 'device': 'cpu',
              'author_repository': source['repository'], 'author_commit': source['commit'],
              'extraction': extraction, 'checks': checks,
              'passed_checks': sum(row['status'] == 'PASS' for row in checks),
              'total_checks': len(checks), 'cuda_initialized': False,
              'full_BCKD_tested': False, 'dataset_performance_tested': False,
              'weighted_decorator_reduction_tested': False, 'head_selection_tested': False,
              'new_hash_computed': False, 'global_packages_installed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'passed': result['passed_checks'],
                      'total': result['total_checks']}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
