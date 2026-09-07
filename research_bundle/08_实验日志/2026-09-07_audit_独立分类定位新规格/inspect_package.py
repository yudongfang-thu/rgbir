"""Inspect the supplied archive without executing its Python code."""
import ast
import datetime
import json
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STEM = 'RGBIR_Independent_Class_Loc_Formal_Spec_20260907'
spec = ROOT / '07_研究分析' / (STEM + '.md')
archive = ROOT / '07_研究分析' / (STEM + '_Package.zip')
records = []
with zipfile.ZipFile(archive) as z:
    if sum(i.file_size for i in z.infolist()) > 5_000_000:
        raise ValueError('unexpected archive size')
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('duplicate archive names')
    for item in z.infolist():
        rel = PurePosixPath(item.filename)
        if rel.is_absolute() or '..' in rel.parts or '\\' in item.filename or ':' in item.filename:
            raise ValueError('unsafe archive path')
        data = z.read(item)
        entry = {'name': item.filename, 'bytes': len(data)}
        if item.filename.endswith('.py'):
            tree = ast.parse(data.decode('utf-8'))
            entry['functions'] = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            entry['tests'] = [n for n in entry['functions'] if n.startswith('test_')]
        records.append(entry)
    same = spec.read_bytes() == z.read('rgbir_independent_cl_spec/' + STEM + '.md')
    supplied_test_receipt = json.loads(z.read('rgbir_independent_cl_spec/kernel_test_results.json'))
receipt = {
    'inspected_at': datetime.datetime.now().astimezone().isoformat(),
    'spec': str(spec), 'archive': str(archive),
    'external_internal_md_bytes_equal': same,
    'files': records,
    'supplied_test_receipt': supplied_test_receipt,
    'executed_package_code': False,
    'scope': 'Byte comparison, archive inventory, AST parsing and static review only. Supplied 18-test receipt is not a local rerun.',
}
(HERE / 'package_review_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'same_md': same, 'files': len(records), 'test_functions': sum(len(x.get('tests', [])) for x in records), 'executed': False}))
