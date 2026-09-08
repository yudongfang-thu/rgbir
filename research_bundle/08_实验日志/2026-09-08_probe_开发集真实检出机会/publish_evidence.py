"""Copy accepted CPU cache analysis and its finalized navigation to the evidence branch."""
from pathlib import Path
import os,re,json,shutil,subprocess
src=Path(__file__).parent;workspace=src.parent.parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/full-evidence-20260906':raise ValueError('Wrong branch')
if not (src/'FINAL_REPORT.md').is_file():raise ValueError('Final readout required')
if 'ACCEPTED' not in (src/'REVIEW.md').read_text(encoding='utf-8'):raise ValueError('Independent acceptance required')
dest=repo/'research_bundle/08_实验日志'/src.name
rows=[]
for p in sorted(src.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(src)
    if '__pycache__' in rel.parts or p.suffix.lower() not in {'.py','.md','.json','.jsonl','.csv','.tsv','.yaml','.txt','.gz'}:continue
    b=p.read_bytes()
    if len(b)>8000000:raise ValueError('Unexpected large publication artifact '+str(rel))
    if any(line.strip().startswith(b'-----BEGIN ') and b'PRIVATE KEY' in line for line in b.splitlines()):raise ValueError('Unexpected credential')
    t=dest/rel
    if dest.resolve() not in t.resolve().parents:raise ValueError('Invalid destination')
    t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    if t.read_bytes()!=b:raise ValueError('Copy differs')
    rows.append(dict(path=rel.as_posix(),bytes=len(b)))

index=repo/'research_bundle/08_实验日志/README.md'
data=index.read_bytes();nl=b'\r\n' if b'\r\n' in data else b'\n';lines=data.decode('utf-8').splitlines()
local_lines=(workspace/'08_实验日志/README.md').read_text(encoding='utf-8').splitlines()
row=next(l for l in local_lines if '[probe_开发集真实检出机会](' in l)
if any('[probe_开发集真实检出机会](' in l for l in lines):lines=[row if '[probe_开发集真实检出机会](' in l else l for l in lines]
else:
    i=next(i for i,l in enumerate(lines) if l.startswith('| 2026-09-08 |'));lines.insert(i,row)
index.write_bytes(nl.join(l.encode('utf-8') for l in lines)+nl)
local=(workspace/'README.md').read_text(encoding='utf-8').splitlines()
banner=next(l for l in local if l.startswith('> **完整dev检出机会复核完成'))
banner=banner.replace('](08_实验日志/','](research_bundle/08_实验日志/')
p=repo/'README.md';ls=p.read_text(encoding='utf-8').splitlines()
if any(l.startswith('> **完整dev检出机会复核完成') for l in ls):ls=[banner if l.startswith('> **完整dev检出机会复核完成') else l for l in ls]
else:ls=[banner,'']+ls
p.write_bytes(('\n'.join(ls)+'\n').encode('utf-8'))
checks=repo/'publication_checks/update_20260908_dev_opportunities';checks.mkdir(parents=True,exist_ok=True)
(checks/'manifest.json').write_text(json.dumps(dict(files=rows,total_bytes=sum(r['bytes'] for r in rows),
 new_hash_computed=False,weights_uploaded=False,credentials_uploaded=False,dataset_images_uploaded=False),ensure_ascii=False,indent=2),encoding='utf-8')
(checks/'README.md').write_text('# 发布范围\n\nLLVIP既有完整dev缓存的CPU检出/定位互补分析：冻结输入与协议、原生匹配源码、小样例、实际逐对象统计、完成回执及独立审阅。没有新增模型推理或训练；不上传权重、凭据、原图或主机快照。Drone完整IR缓存缺口单列。\n',encoding='utf-8')
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows)),ensure_ascii=False))
