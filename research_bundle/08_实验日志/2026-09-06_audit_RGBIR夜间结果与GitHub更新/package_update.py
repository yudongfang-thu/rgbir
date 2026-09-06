"""Append new evidence snapshots and adapt only exported Markdown navigation."""
from pathlib import Path
from urllib.parse import unquote
import json
import os
import re
import shutil

WS=Path('E:/SHARE/光sar')
REPO=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
SLUG='2026-09-06_audit_RGBIR夜间结果与GitHub更新'
PREVIOUS='2026-09-06_audit_RGBIR晚间进度与新结果'
HERE=WS/'08_实验日志'/SLUG
CHECK=REPO/'publication_checks/update_20260906_2146'
CHECK.mkdir(parents=True,exist_ok=True)

def write(path,text):
    path.write_bytes(text.encode('utf-8'))

# Update local navigation before copying it, leaving prior snapshots intact.
index=WS/'08_实验日志/README.md'
content=index.read_text(encoding='utf-8-sig')
row=f'| 2026-09-06 | [audit_RGBIR夜间结果与GitHub更新]({SLUG}/README.md) | audit | 21:46：OEv1 P123=54.637/N0=54.346新增，3/6端点但0/3配对；OS-SSL P123−shuffled123 CSV+0.669mAP/+0.495AP50；更新原GitHub分支 |'
if '[audit_RGBIR夜间结果与GitHub更新]' not in content:
    content=content.replace('|---|---|---|---|','|---|---|---|---|\n'+row,1)
write(index,content)
training_notes=[
 ('2026-09-06_train_RGBIR对象判别蒸馏三seed扩展','P42/P123/N0已完成独立评估，mAP分别54.658/54.637/54.346；N42@135、P0@28、N123@20，3/6端点、0/3完整配对。'),
 ('2026-09-06_train_RGBIR对象判别蒸馏首轮','P42=54.658独立端点保持，N42完成135轮，完整净收益仍待同代码N42。'),
 ('2026-09-06_train_OS-SSL-IR迁移','新增paired123 E200完成；与shuffled123同口径CSV差+0.669mAP/+0.495AP50。3/9完成、shuffled0@131；尚无独立last评估，初始化混杂与RGB-only控制缺口仍在。')]
for folder,note in training_notes:
    p=WS/'08_实验日志'/folder/'README.md'
    content=p.read_text(encoding='utf-8-sig')
    if '**21:46最新核验' not in content:
        title,rest=content.split('\n',1)
        write(p,title+f'\n\n> **21:46最新核验（2026-09-06）**：{note} 见[夜间结果与GitHub更新](../{SLUG}/README.md)。下文旧快照按各自时间读取。\n'+rest)
p=WS/'README.md'
lines=p.read_text(encoding='utf-8-sig').splitlines()
for i,line in enumerate(lines):
    if line.startswith('> **RGBIR最新进度'):
        lines[i]=f'> **RGBIR最新进度（2026-09-06 21:46）**：[夜间结果与GitHub更新](08_实验日志/{SLUG}/README.md)。OEv1 P42/P123/N0独立mAP54.658/54.637/54.346，3/6端点但0/3同seed配对；OS-SSL首次同seed paired−shuffled CSV差+0.669mAP/+0.495AP50，仍缺独立last评估。'
write(p,'\n'.join(lines)+'\n')

paths=[]
exclude={'PUBLISH_RECEIPT.json','github_readback_verification.json','publication_scan.json','publication_review.md','markdown_navigation.json','source_manifest.json'}
for folder in [PREVIOUS,SLUG]:
    for p in (WS/'08_实验日志'/folder).rglob('*'):
        if p.is_file() and p.name not in exclude and '__pycache__' not in p.parts:
            paths.append(p)
paths += [index]+[WS/'08_实验日志'/folder/'README.md' for folder,_ in training_notes]
source_rows=[]
for src in sorted(set(paths)):
    if src.suffix.lower() in {'.pt','.pth','.ckpt','.pyc','.pem','.key','.zip','.tar','.gz'}:
        raise ValueError('Unexpected excluded artifact: '+str(src))
    dst=REPO/'research_bundle'/src.relative_to(WS)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    source_rows.append({'source':src.as_posix(),'path':src.relative_to(WS).as_posix(),
                        'repository_path':dst.relative_to(REPO).as_posix(),'original_bytes':src.stat().st_size,
                        'category':'20260906_2146_incremental_evidence'})
src=WS/'README.md'
dst=REPO/'research_bundle/workspace_context/WORKSPACE_README.md'
shutil.copyfile(src,dst)
source_rows.append({'source':src.as_posix(),'path':'README.md','repository_path':dst.relative_to(REPO).as_posix(),
                    'original_bytes':src.stat().st_size,'category':'updated_workspace_navigation'})

manifest=json.loads((REPO/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
mapping={row['source'].replace('\\','/').lower():REPO/row['repository_path'] for row in manifest['files']}
mapping.update({row['source'].lower():REPO/row['repository_path'] for row in source_rows})
adaptation={'adapted':[],'unbundled':[]}
pattern=re.compile(r'(!?\[([^\]\n]*)\])\(([^\n)]*)\)')
for row in source_rows:
    path=REPO/row['repository_path']
    if path.suffix.lower()!='.md':
        assert path.read_bytes()==Path(row['source']).read_bytes()
        continue
    def replace(m):
        label,alt,raw=m.groups()
        link=unquote(raw.strip().strip('<>')).replace('\\','/')
        if re.match(r'^(https?://|mailto:|#|app://|codex://)',link):return m.group(0)
        target,sep,anchor=link.partition('#')
        target=re.sub(r':\d+$','',target)
        candidates=[]
        if re.match(r'^[A-Za-z]:/',target):
            if target.lower() in mapping:candidates.append(mapping[target.lower()])
            prefix=WS.as_posix()+'/'
            if target.lower().startswith(prefix.lower()):
                relative=target[len(prefix):]
                candidates.append(REPO/'research_bundle'/relative)
                if relative.startswith('09_外部审计_rgbir/'):
                    candidates.append(REPO/relative.removeprefix('09_外部审计_rgbir/'))
        elif not target.startswith('/mnt/'):
            candidates += [path.parent/target,REPO/target,REPO/'research_bundle'/target]
            original=(Path(row['source']).parent/target).resolve().as_posix().lower()
            if original in mapping:candidates.append(mapping[original])
        found=next((c for c in candidates if c.exists()),None)
        if found:
            url=os.path.relpath(found,path.parent).replace('\\','/')+('#'+anchor if sep else '')
            if url!=raw:adaptation['adapted'].append({'document':row['repository_path'],'from':raw,'to':url})
            return label+'('+url+')'
        adaptation['unbundled'].append({'document':row['repository_path'],'original':raw})
        return alt+'（未收录的来源路径：`'+raw+'`）'
    text=pattern.sub(replace,path.read_text(encoding='utf-8-sig'))
    write(path,text)

for row in source_rows:
    row['exported_bytes']=(REPO/row['repository_path']).stat().st_size
    row['documentation_links_rewritten']=sum(x['document']==row['repository_path'] for group in adaptation.values() for x in group)
updated={row['repository_path']:row for row in manifest['files']}
updated.update({row['repository_path']:row for row in source_rows})
manifest['files']=list(updated.values())
manifest['file_count']=len(manifest['files'])
manifest['bytes']=sum(row['exported_bytes'] for row in manifest['files'])
manifest.setdefault('incremental_manifests',[])
name='publication_checks/update_20260906_2146/source_manifest.json'
if name not in manifest['incremental_manifests']:manifest['incremental_manifests'].append(name)
write(REPO/'BUNDLE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
result={'captured_snapshot':'2026-09-06T21:46+08:00','source_snapshot_policy':'Original raw bytes retained; Markdown links adapted in export only. Earlier published raw snapshots unchanged.','files':source_rows}
write(CHECK/'source_manifest.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
write(CHECK/'link_adaptation.json',json.dumps(adaptation,ensure_ascii=False,indent=2)+'\n')
write(HERE/'source_manifest.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'copied_files':len(source_rows),'source_bytes':sum(r['original_bytes'] for r in source_rows),'adapted_links':len(adaptation['adapted']),'unbundled_links':len(adaptation['unbundled'])}))
