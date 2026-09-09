"""CPU-checkable, exactly two integer-grid-bound fixes for the original CFT loss.

No source files are modified. The reference uses gain-derived integer bounds;
the runtime variant uses the same grid dimensions as Python integers, avoiding
an int(CUDA tensor) synchronization on each head. No hashes are computed.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace


def compile_variants(source_text, filename, torch):
    tree = ast.parse(textwrap.dedent(source_text))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'build_targets']
    if len(functions) != 1:
        raise RuntimeError('Expected exactly one author build_targets')
    original = functions[0]
    expected_gain = ast.parse('gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]').body[0]
    assert sum(ast.dump(n) == ast.dump(expected_gain) for n in ast.walk(original)) == 1, 'Unexpected author gain layout'

    def compile_one(mode):
        function = copy.deepcopy(original)
        changes = []
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'clamp_'):
                continue
            tensor = ast.unparse(node.func.value)
            if tensor not in ('gi', 'gj'):
                continue
            gain_index, shape_index = (3, 2) if tensor == 'gj' else (2, 3)
            expected_bound = ast.parse(f'gain[{gain_index}] - 1', mode='eval').body
            assert len(node.args) == 2 and not node.keywords and ast.unparse(node.args[0]) == '0'
            assert ast.dump(node.args[1]) == ast.dump(expected_bound), 'Unexpected author clamp bound'
            before = ast.unparse(node)
            if mode == 'grid_shape':
                node.args[1] = ast.parse(f'p[i].shape[{shape_index}] - 1', mode='eval').body
            elif mode == 'gain_integer_reference':
                node.args[1] = ast.parse(f'(gain[{gain_index}] - 1).long()', mode='eval').body
            changes.append(dict(before=before, after=ast.unparse(node)))
        assert len(changes) == 2 and {x['before'].split('.')[0] for x in changes} == {'gi', 'gj'}
        namespace = {'torch': torch}
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, filename, 'exec'), namespace)
        return namespace['build_targets'], changes

    runtime, changes = compile_one('grid_shape')
    reference, _ = compile_one('gain_integer_reference')
    original_fn, _ = compile_one('original')
    return original_fn, runtime, reference, changes


def validate_source(source_text, filename, torch):
    original, runtime, reference, changes = compile_variants(source_text, filename, torch)
    assert not torch.cuda.is_initialized(), 'CPU preflight must precede all CUDA initialization'
    owner = SimpleNamespace(na=3, nl=3, hyp={'anchor_t': 4.0}, anchors=[
        torch.tensor([[1.25, 1.625], [2., 3.75], [4.125, 2.875]]),
        torch.tensor([[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]]),
        torch.tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]])])
    # Only shape is accessed; no large prediction tensors or GPU are needed.
    predictions = [SimpleNamespace(shape=(2, 3, h, w, 6)) for h, w in ((128, 160), (64, 80), (32, 40))]
    cases = {
        'empty': torch.empty((0, 6)),
        'interior_and_edges': torch.tensor([[0., 0., .5, .5, .1, .2], [1., 0., 0., 0., .025, .03],
                                           [1., 0., 1., 1., .02, .04], [0., 0., .999, .001, .1, .2],
                                           [0., 0., .251, .749, .04, .07]]),
    }
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        else:
            assert len(a) == len(b)
            for left, right in zip(a, b): equal(left, right)
    rows = []
    original_failures = []
    for name, targets in cases.items():
        actual = runtime(owner, predictions, targets.clone())
        expected = reference(owner, predictions, targets.clone())
        equal(actual, expected)
        for level, (b, a, gj, gi) in enumerate(actual[2]):
            assert gj.dtype == gi.dtype == torch.long
            if gi.numel():
                assert 0 <= gi.min() <= gi.max() < predictions[level].shape[3]
                assert 0 <= gj.min() <= gj.max() < predictions[level].shape[2]
        try:
            original(owner, predictions, targets.clone())
        except RuntimeError as exc:
            original_failures.append(dict(case=name, error=str(exc)))
        rows.append(dict(case=name, status='PASS', matched_per_level=[len(x[0]) for x in actual[2]],
                         compared='all tcls/tbox/indices/anchors, exact tensor equality'))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS', torch_version=torch.__version__, cuda_initialized=False,
                runtime_change_count=len(changes), changes=changes, cases=rows,
                original_failures=original_failures, new_hash_computed=False), runtime


def install(loss_class, torch):
    original = loss_class.build_targets
    report, replacement = validate_source(inspect.getsource(original), inspect.getsourcefile(original), torch)
    loss_class.build_targets = replacement
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Unmodified author utils/loss.py')
    parser.add_argument('--output', type=Path, required=True, help='New exclusive CPU receipt JSON')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    report, _ = validate_source(args.source.read_text(encoding='utf-8'), str(args.source), torch)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
