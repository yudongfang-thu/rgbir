# coding: utf-8
"""Curate the authorized completed hourly-screen increment; no training or hashing."""
from pathlib import Path
from urllib.parse import unquote
import json
import os
import re
import subprocess

WORK = Path('E:/SHARE/光sar')
REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / 'research_bundle'
ENTRY = WORK / '08_实验日志/2026-09-08_ops_小时级筛选重构'
ROOTS = [ENTRY]
EXTRAS = [WORK / '08_实验日志/README.md']
BAD_KEYS = {'cmdline', 'command_line', 'processes', 'all_processes', 'ps_output', 'process_rows', 'commands'}


def check_json(value):
    if isinstance(value, dict):
        assert not ({key.lower() for key in value} & BAD_KEYS), 'Unexpected process/command inventory'
        for child in value.values():
            check_json(child)
    elif isinstance(value, list):
        for child in value:
            check_json(child)


def main():
    manifest_path = REPO / 'HOURLY_SCREEN_INCREMENT_20260908.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'records': []}
    selected = []
    excluded = []
    for source in sorted(set([p for folder in ROOTS for p in folder.rglob('*') if p.is_file()] + EXTRAS)):
        if any(part.startswith('snapshot_') for part in source.relative_to(WORK).parts):
            excluded.append({'source': str(source), 'bytes': source.stat().st_size, 'reason': 'Repeated intermediate collection excluded; final_evidence_1022 is published, original evidence retained unchanged.'})
            continue
        if '__pycache__' in source.parts or source.suffix.lower() in {'.pyc', '.pyo', '.pt', '.pth', '.ckpt', '.gz', '.npz', '.npy', '.safetensors', '.pkl', '.pickle', '.onnx', '.zip', '.tar'}:
            excluded.append({'source': str(source), 'bytes': source.stat().st_size, 'reason': 'Cache, weight, raw tensor or compressed raw artifact excluded.'})
            continue
        if source.name == 'sample_stream.jsonl':
            excluded.append({'source': str(source), 'bytes': source.stat().st_size, 'reason': 'Repeated 30-batch sample streams retained locally/remotely; final canary_checks and postflight byte-equality receipts are published.'})
            continue
        if source.name == 'train_metadata.jsonl':
            excluded.append({'source': str(source), 'bytes': source.stat().st_size, 'reason': 'Full 17990-image label metadata omitted; frozen 2048 lists, statistics, mapping and rerunnable builder retained.'})
            continue
        if source.name == 'snapshot.json' or 'host_profile' in source.name.lower():
            excluded.append({'source': str(source), 'bytes': source.stat().st_size, 'reason': 'Whole-host heartbeat snapshot excluded; project checks and prose retained.'})
            continue
        assert source.suffix in {'.md', '.py', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.txt'}, str(source)
        assert '__pycache__' not in source.parts and 'host_profile' not in source.name.lower()
        assert source.stat().st_size < 8 * 2**20
        payload = source.read_bytes()
        text = payload.decode('utf-8-sig')
        assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}', text)
        if source.suffix == '.json':
            check_json(json.loads(text))
        elif source.suffix == '.jsonl':
            for line in text.splitlines():
                if line.strip():
                    check_json(json.loads(line))
        selected.append(source)
    mapping = {str(p.resolve()).lower(): BUNDLE / p.relative_to(WORK) for p in selected}
    desired_paths = {p.relative_to(REPO).as_posix() for p in mapping.values()}
    stale = [r['repository_path'] for r in previous['records'] if r['repository_path'] not in desired_paths]
    added = set(subprocess.check_output(['git', 'diff', '--cached', '--diff-filter=A', '--name-only', '-z'], cwd=REPO).decode('utf-8').split('\0'))
    entry_dest = (BUNDLE / ENTRY.relative_to(WORK)).resolve()
    for name in stale:
        dest = (REPO / name).resolve()
        assert entry_dest in dest.parents and name in added, 'Only this increment\'s previously staged added copies may be pruned'
        subprocess.run(['git', 'restore', '--staged', '--', name], cwd=REPO, check=True)
        dest.unlink()  # Exact derived file only; no recursive deletion or source mutation.
    records = []
    for source in selected:
        dest = mapping[str(source.resolve()).lower()]
        dest.parent.mkdir(parents=True, exist_ok=True)
        original = source.read_bytes()
        payload = original
        if source.suffix == '.md':
            def adapt(match):
                target = unquote(match.group(1).strip().strip('<>')).replace('\\', '/')
                if re.match(r'^(https?://|mailto:|#|app://|codex://)', target):
                    return match.group(0)
                name, sep, anchor = target.partition('#')
                name = re.sub(r':\d+$', '', name)
                src = (Path(name) if re.match(r'^[A-Za-z]:/', name) else source.parent / name).resolve()
                dst = mapping.get(str(src).lower())
                if dst is None:
                    try:
                        dst = BUNDLE / src.relative_to(WORK.resolve())
                    except ValueError:
                        return match.group(0)
                if dst.exists() or str(src).lower() in mapping:
                    return '](' + os.path.relpath(dst, dest.parent).replace('\\', '/') + ('#' + anchor if sep else '') + ')'
                return ']（服务器/本地保留，未包含于本阶段发布：' + target + '）'
            payload = re.sub(r'\]\(([^\n)]*)\)', adapt, original.decode('utf-8-sig')).encode('utf-8')
        dest.write_bytes(payload)
        assert dest.read_bytes() == payload and source.read_bytes() == original
        records.append({'source': str(source), 'repository_path': dest.relative_to(REPO).as_posix(), 'source_bytes': len(original),
                        'bytes': len(payload), 'verification': 'byte exact' if payload == original else 'Markdown links adapted; scientific prose retained'})
    manifest = {'status': 'FINAL_INCREMENT_COPIED_PENDING_STAGED_CHECK', 'base_commit': '7f95f08069ab1a2a93752cde9e22337b93a69e6e',
                'scope': 'Completed single-seed fixed-2048 FT3 screen: accepted analyzer/results, final three-arm train/eval/canary and initialization/resource/queue receipts, all three dev rosters required by the analyzer, baseline runtime, subset builder/lists/statistics, source/protocol/CPU checks, failed release_v1 launch and deployment identity, and current experiment index. No new experiment or analysis in curation.',
                'records': records, 'excluded': excluded, 'pruned_prepared_copy_paths': stale,
                'publication_input_limits': ['Full train_metadata.jsonl and repeated sample_stream.jsonl retained locally/remotely, not uploaded. Their already-executed checks are published; these checks cannot be re-executed solely from the curated subset.', 'Accepted analyze_hourly.py can read the published final receipts and all three development_roster.txt files; no weight or image is needed for descriptive recomputation.'],
                'new_file_hashes_computed': False, 'local_scientific_results_modified': False}
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'source_files': len(records), 'bytes': sum(r['bytes'] for r in records), 'byte_exact': sum(r['verification'] == 'byte exact' for r in records), 'excluded': len(excluded)}))


if __name__ == '__main__':
    main()
