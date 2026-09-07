"""Apply reviewed navigation edits and LF-only Markdown to the derived export."""
import datetime
import json
from pathlib import Path
import shutil

WORKSPACE=Path('E:/SHARE/光sar')
LOG=Path(__file__).resolve().parent
REPO=Path('\\\\?\\C:\\Users\\MSI-PC\\rgbir_review_worktrees\\evidence-20260906')
BASE=REPO/'INDEPENDENT_KD_BUNDLE_MANIFEST_20260907_154636.json'
manifest=json.loads(BASE.read_text(encoding='utf-8'))
prefix='research_bundle/'+LOG.relative_to(WORKSPACE).as_posix()+'/'
for name in ('README.md','FORMAL_LAUNCH_ACCEPTANCE_1535.md'):
    target=REPO/(prefix+name)
    content=target.read_text(encoding='utf-8').replace('均已超过24次成功更新','均已达到至少24次成功更新')
    content=content.replace('0/42/123×N/C0六轨迹全部通过，原六份端点可复用',
                            '0/42/123×N/C0六轨迹全部通过；旧六端点评价复用仍待桥接与逐类/逐对象补采')
    target.write_bytes(content.encode('utf-8'))
for name in ('review_independent_export.py','finalize_publication_stage.py','formal_live_20260907_154941.json'):
    source=LOG/name;target=REPO/(prefix+name)
    shutil.copyfile(source,target)
    if not any(row['repository_path']==prefix+name for row in manifest['files']):
        manifest['files'].append(dict(source=source.as_posix(),path=source.relative_to(WORKSPACE).as_posix(),
                                     repository_path=prefix+name,bytes=target.stat().st_size))
normalized=0
for path in REPO.rglob('*.md'):
    if '.git' in path.relative_to(REPO).parts:continue
    before=path.read_bytes();after=before.replace(b'\r\n',b'\n')
    if after!=before:path.write_bytes(after);normalized+=1
for row in manifest['files']:row['bytes']=(REPO/row['repository_path']).stat().st_size
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
manifest['time']=datetime.datetime.now().astimezone().isoformat()
manifest['supersedes_export_inventory']=BASE.name
manifest['markdown_policy']='Workspace Markdown links adapted and LF normalized; other payload bytes unchanged.'
manifest['reviewed_fixes']=['At least 24 updates, including seed123 at exactly24 in the15:32 snapshot.',
                           'Training compatibility is accepted; legacy endpoint evaluation bridge remains pending.']
name='INDEPENDENT_KD_BUNDLE_MANIFEST_'+stamp+'.json'
payload=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
(REPO/name).write_text(payload,encoding='utf-8')
(LOG/('publication_inventory_'+stamp+'.json')).write_text(payload,encoding='utf-8')
print(json.dumps(dict(manifest=name,files=len(manifest['files']),normalized_markdown_files=normalized)))
