"""Curated incremental evidence export; no training or automatic Git push."""
import datetime
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

WORKSPACE=Path('E:/SHARE/光sar')
LOG=Path(__file__).resolve().parent
REPO=Path('\\\\?\\C:\\Users\\MSI-PC\\rgbir_review_worktrees\\evidence-20260906')
BUNDLE=REPO/'research_bundle'
names=['README.md','record_heartbeat_1653.py','publish_heartbeat_increment.py',
       'deploy_posthoc_adapter_cpu.py',
       'formal_live_20260907_165319.json','heartbeat_readonly_1655.txt','PUBLISH_RECEIPT_c7bbd6d.md',
       'snapshots/2026-09-07T165357.033246_0800/snapshot.json',
       'snapshots/2026-09-07T165357.033246_0800/resource_leases_raw.json']
files=[LOG/name for name in names]
for name in ('posthoc_class_adapter_v1','posthoc_class_adapter_independent_review_v1','heartbeat_20260907_1653'):
    folder=LOG/name
    if folder.is_dir():
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix.lower() in ('.py','.json','.md','.txt'))
files += [WORKSPACE/name for name in ('refine-logs/EXPERIMENT_TRACKER.md','08_实验日志/README.md',
    '99_整理回执/20260907_IndependentKD实施新增目录.md','MANIFEST.md')]
pairs=[(p,BUNDLE/p.relative_to(WORKSPACE)) for p in sorted(set(files))]
pairs += [(WORKSPACE/'README.md',BUNDLE/'workspace_context/WORKSPACE_README.md')]
mapping={str(src.resolve()).lower():dst for src,dst in pairs}
records=[];unresolved=[]
for source,target in pairs:
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix=='.md':
        def replace(match):
            raw=match.group(1);link=unquote(raw.strip().strip('<>')).replace('\\','/')
            if re.match(r'^(https?://|mailto:|#|app://|codex://)',link):return match.group(0)
            link,sep,anchor=link.partition('#');link=re.sub(r':\d+$','',link)
            src=Path(link) if re.match(r'^[A-Za-z]:/',link) else source.parent/link
            src=src.resolve();dest=mapping.get(str(src).lower())
            if dest is None:
                try:dest=BUNDLE/src.relative_to(WORKSPACE.resolve())
                except ValueError:dest=None
            if dest is not None and (dest.exists() or str(src).lower() in mapping):
                return ']('+os.path.relpath(dest,target.parent).replace('\\','/')+('#'+anchor if sep else '')+')'
            unresolved.append(dict(document=str(source),target=raw))
            return ']（未导出的工作区路径：'+raw+'）'
        content=re.sub(r'\]\(([^\n)]*)\)',replace,source.read_text(encoding='utf-8-sig'))
        target.write_bytes(content.encode('utf-8'))
    else:
        target.write_bytes(source.read_bytes())
    records.append(dict(source=source.as_posix(),path=source.relative_to(WORKSPACE).as_posix(),
                        repository_path=target.relative_to(REPO).as_posix(),bytes=target.stat().st_size))
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
inventory=dict(time=datetime.datetime.now().astimezone().isoformat(),files=records,unresolved_links=unresolved,
    scope='Increment after c7bbd6d; earlier accepted full evidence remains in the repository.',
    raw_evidence_policy='Byte-preserved raw code/numbers; derived Markdown link adaptation; no weights or dataset corpus.')
manifest=REPO/('INDEPENDENT_KD_INCREMENTAL_MANIFEST_'+stamp+'.json')
manifest.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(manifest=str(manifest),files=len(records),bytes=sum(x['bytes'] for x in records),
                     unresolved=len(unresolved))))
