"""Publish only the already completed N E8 endpoint and referenced status increment."""
from pathlib import Path
from urllib.parse import unquote
import json
import os
import re

WORK = Path('E:/SHARE/光sar')
REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / 'research_bundle'
E8 = WORK / '08_实验日志/2026-09-08_train_分类快速反馈E8'
OLD = WORK / '08_实验日志/2026-09-07_train_IndependentKD实施'
ROOTS = [E8 / 'endpoint_N_20260908_043812', E8 / 'heartbeat_20260908_043626', OLD / 'heartbeat_20260908_0435']
EXTRAS = [E8 / 'README.md', E8 / 'collect_arm_endpoint.py', E8 / 'GITHUB_PUBLICATION.md', OLD / 'README.md', WORK / '08_实验日志/README.md']
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
    selected = []
    excluded = []
    for source in sorted(set([p for folder in ROOTS for p in folder.rglob('*') if p.is_file()] + EXTRAS)):
        if source.name == 'snapshot.json' and source.parent.name == 'heartbeat_20260908_043626':
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
    prefix = '> **2026-09-08 首个 E8 端点**：Drone seed42 的 N 已完成固定 E8 last/EMA 与完整 dev 独立评价，mAP50–95 为 **43.37654**；17 项限定回执复核通过。这是单独的短程 N 原值，C0/C1 尚未完成匹配比较，不是蒸馏增益或 E200 等价结论。请读[原始端点与复核](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/endpoint_N_20260908_043812/README.md)。训练 CSV 的占位/末行短列不作为 AP 来源。\n\n'
    for path, note in [(REPO / 'README.md', prefix), (BUNDLE / 'README.md', prefix.replace('](research_bundle/', ']('))]:
        text = path.read_text(encoding='utf-8')
        assert not text.startswith(note)
        path.write_bytes((note + text).encode('utf-8'))
    manifest = {'status': 'REVIEWED_FOR_AUTHORIZED_INCREMENTAL_COMMIT', 'base_commit': 'c92fc6e73f5f4c1ebf02925b3df07a3e652ee8b3',
                'scope': 'Completed N E8 endpoint and its existing 17-check review; only the specifically referenced current E8 and old-run statuses. No new experiment or analysis.',
                'records': records, 'excluded': excluded, 'new_file_hashes_computed': False, 'local_scientific_results_modified': False}
    (REPO / 'N_E8_ENDPOINT_INCREMENT_20260908.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'source_files': len(records), 'bytes': sum(r['bytes'] for r in records), 'byte_exact': sum(r['verification'] == 'byte exact' for r in records), 'excluded': len(excluded)}))


if __name__ == '__main__':
    main()
