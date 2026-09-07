"""Read staged publication scope and source bytes without calculating hashes."""
from pathlib import Path
import collections
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / 'PERFORMANCE_E8_INCREMENT_20260908.json').read_text(encoding='utf-8'))
    staged = [p for p in subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT).decode('utf-8').split('\0') if p]
    forbidden = []
    secret_hits = []
    for name in staged:
        path = ROOT / name
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.onnx', '.pyc', '.pem', '.key'} or any(x in name.lower() for x in ('__pycache__', 'host_profile')):
            forbidden.append(name)
        if path.suffix.lower() in {'.json', '.jsonl', '.yaml', '.yml', '.md', '.py'}:
            text = path.read_text(encoding='utf-8-sig')
            patterns = {'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                        'github_token': r'gh[pousr]_[A-Za-z0-9]{20,}'}
            for kind, pattern in patterns.items():
                if re.search(pattern, text):
                    secret_hits.append({'file': name, 'kind': kind})
    exact = []; mismatches = []
    for record in manifest['records']:
        if record['verification'] != 'byte exact':
            continue
        source = Path(record['source']).read_bytes()
        dest = (ROOT / record['repository_path']).read_bytes()
        if source != dest or len(source) != record['source_bytes']:
            mismatches.append(record['repository_path'])
        else:
            exact.append(record['repository_path'])
    check = subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, capture_output=True)
    warnings = check.stdout.decode('utf-8', errors='replace').splitlines()
    result = dict(status='PASS' if not (forbidden or secret_hits or mismatches) else 'FAIL',
                  staged_file_count=len(staged), staged_suffixes=dict(collections.Counter(Path(p).suffix for p in staged)),
                  source_record_count=len(manifest['records']), copied_source_bytes=sum(r['bytes'] for r in manifest['records']),
                  byte_exact_non_markdown_count=len(exact), byte_mismatches=mismatches,
                  forbidden_paths=forbidden, high_confidence_secret_hits=secret_hits,
                  excluded_source_count=len(manifest['excluded']), whitespace_check_exit=check.returncode,
                  whitespace_warning_count=sum('whitespace' in x for x in warnings),
                  whitespace_scope='Upstream source bytes retained; CRLF accepted as line endings; no scientific source reformatting.',
                  new_file_hashes_computed=False, gpu_or_experiments_started=False, source_results_modified=False)
    out = ROOT / 'publication_checks/update_20260908_performance_E8/staged_review.json'
    out.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(result, ensure_ascii=True))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
