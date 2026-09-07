"""Bounded CPU checks of actual alias, GT parsing, and loader admission code."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
source = HERE / 'export_full_dev.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
fixture = HERE / 'independent_attempt2_cpu_fixture'
fixture.mkdir(exist_ok=False)
image_dir = fixture / 'processed' / 'images' / 'dev'
image_dir.mkdir(parents=True)
(image_dir / '001.png').write_bytes(b'fixture_path_only_no_inference')
alias_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluation_aliases')
ns = {'Path': Path}
exec(compile(ast.Module(body=[alias_fn], type_ignores=[]), str(source), 'exec'), ns)
with patch.object(Path, 'resolve', side_effect=AssertionError('Alias resolver must not canonicalize')):
    aliases = ns['evaluation_aliases']({'path': str(fixture / 'processed'), 'val': 'images/dev'}, fixture / 'data.yaml')
assert aliases == [str(image_dir / '001.png')]
run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
label_loop = next(n for n in run.body if isinstance(n, ast.For) and ast.unparse(n.target) == '(alias, path)')
zero_guard = next(n for n in run.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'sum(label_counts) <= 0')
parser = compile(ast.Module(body=[label_loop, zero_guard], type_ignores=[]), str(source), 'exec')
cases = {'valid': '0 0.5 0.4 0.3 0.2\n', 'nan': '0 nan 0.4 0.3 0.2\n',
         'range': '0 1.1 0.4 0.3 0.2\n', 'zero_width': '0 0.5 0.4 0 0.2\n',
         'wrong_class': '1 0.5 0.4 0.3 0.2\n', 'empty': ''}
checks = {'processed_alias_preserved_no_resolve_call': True}
expected = None
for name, contents in cases.items():
    label = fixture / (name + '.txt'); label.write_text(contents)
    scope = {'Path': Path, 'np': np, 'aliases': aliases, 'all_labels': [str(label)], 'label_counts': [], 'expected_labels': {}}
    try:
        exec(parser, scope)
    except ValueError:
        assert name != 'valid'
        checks['label_' + name + '_rejected'] = True
    else:
        assert name == 'valid'
        expected = scope['expected_labels']
        checks['label_valid_parsed_float32'] = expected[aliases[0]].dtype == np.float32
on_start = next(n for n in ast.walk(run) if isinstance(n, ast.FunctionDef) and n.name == 'on_start')
context = {'np': np, 'Path': Path, 'contract': {}, 'actual': aliases, 'versions': {}, 'expected_gt': 1,
           'expected_labels': expected, 'capture_contract': lambda *args: {'roster': aliases}}
exec(compile(ast.Module(body=[on_start], type_ignores=[]), str(source), 'exec'), context)
label = dict(im_file=aliases[0], cls=expected[aliases[0]][:, :1].copy(), bboxes=expected[aliases[0]][:, 1:].copy())
v = SimpleNamespace(dataloader=SimpleNamespace(dataset=SimpleNamespace(im_files=aliases, labels=[label])))
context['on_start'](v)
assert context['contract']['loader_per_image_labels_exact'] is True
checks['actual_callback_accepts_exact_per_image_Gt'] = True
label['bboxes'][0, 0] += .01
try:
    context['on_start'](v)
except ValueError:
    checks['same_count_different_box_rejected'] = True
else:
    raise AssertionError('Same GT count with wrong coordinates was accepted')
label['cls'] = np.empty((0, 1)); label['bboxes'] = np.empty((0, 4))
try:
    context['on_start'](v)
except ValueError:
    checks['zero_loader_gt_rejected'] = True
else:
    raise AssertionError('Zero GT loader was accepted')
receipt = {'status': 'PASSED_CPU_SOURCE_LOGIC_ONLY', 'checks': checks, 'source': str(source),
           'source_stat': {'bytes': source.stat().st_size, 'mtime_ns': source.stat().st_mtime_ns},
           'scope': 'Actual AST blocks; path-only local fixtures and stubbed capture_contract; not actual Linux symlink loader/GPU',
           'ssh': False, 'gpu': False, 'hash_computed': False}
(HERE / 'independent_attempt2_cpu_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
