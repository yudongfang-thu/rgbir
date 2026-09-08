"""Publish finalized small evidence into the existing review branch, without raw images/weights."""
from pathlib import Path
import json,shutil,subprocess,re,os

src=Path(__file__).parent
workspace=src.parent.parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/full-evidence-20260906':
    raise ValueError('Wrong branch')
dest=repo/'research_bundle/08_实验日志'/src.name
allowed={'.py','.md','.json','.jsonl','.yaml','.csv','.tsv','.txt'}
skip_roots={'evidence_1252','analysis_1252'}
rows=[];skipped=[]
for p in sorted(src.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(src)
    if ('__pycache__' in rel.parts or rel.parts[0] in skip_roots or
        p.suffix.lower() not in allowed or p.name.endswith('_resource_profile.json')):
        skipped.append(rel.as_posix());continue
    if p.stat().st_size>5000000:raise ValueError('Unexpected large artifact '+str(rel))
    data=p.read_bytes()
    if any(line.strip().startswith(marker) for line in data.splitlines()
           for marker in (b'-----BEGIN OPENSSH PRIVATE KEY-----',b'-----BEGIN RSA PRIVATE KEY-----')):
        raise ValueError('Unexpected credential marker')
    target=dest/rel
    if dest.resolve() not in target.resolve().parents:raise ValueError('Invalid destination')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,target)
    if target.read_bytes()!=data:raise ValueError('Copy differs')
    rows.append(dict(path=rel.as_posix(),bytes=len(data)))

# Preserve the existing index line endings and add only this completed entry.
index=repo/'research_bundle/08_实验日志/README.md'
data=index.read_bytes();nl=b'\r\n' if b'\r\n' in data else b'\n'
lines=data.decode('utf-8').splitlines()
row='| 2026-09-08 | [probe_同帧选择覆盖](2026-09-08_probe_同帧选择覆盖/README.md) | probe | 两次零更新推理完成；32图80GT旧候选机会9→原生检出机会1，selected31/32双方已检出；修正低置信解释，下一项复用完整dev缓存 |'
if any('[probe_同帧选择覆盖](' in line for line in lines):
    lines=[row if '[probe_同帧选择覆盖](' in line else line for line in lines]
else:
    i=next(i for i,line in enumerate(lines) if line.startswith('| 2026-09-08 |'))
    lines.insert(i,row)
index.write_bytes(nl.join(line.encode('utf-8') for line in lines)+nl)

banner='> **2026-09-08 同帧原生检测核对完成**：LLVIP固定32张训练图，旧assigned定义的教师机会9个，原生NMS后一对一匹配仅1个且已选中；selected中31/32双方已检出。旧低置信候选不能直接解释为实际漏检，完整dev AP不受此定义修正影响。见[结果、源码及原始小回执](research_bundle/08_实验日志/2026-09-08_probe_同帧选择覆盖/FINAL_REPORT.md)。下一项优先CPU复用完整dev预测，不新增E200。'
p=repo/'README.md';ls=p.read_text(encoding='utf-8').splitlines()
if any(l.startswith('> **2026-09-08 同帧原生检测核对完成') for l in ls):
    ls=[banner if l.startswith('> **2026-09-08 同帧原生检测核对完成') else l for l in ls]
else:ls=[banner,'']+ls
p.write_bytes(('\n'.join(ls)+'\n').encode('utf-8'))

corrections=['07_研究分析/RGBIR数据分析综合报告_20260908.md',
 '08_实验日志/2026-09-08_probe_快速方向筛选/DIRECTIONS_AND_PROXY_LIMITS.md']
for name in corrections:
    p=workspace/name;t=repo/'research_bundle'/name
    if not t.is_file():raise ValueError('Expected existing narrative '+str(t))
    shutil.copyfile(p,t)
    data=t.read_bytes();nl='\r\n' if b'\r\n' in data else '\n'
    content=data.decode('utf-8')
    content=re.sub(r'\]\(E:/SHARE/光sar/([^\)\n]+)\)',
      lambda m:']('+os.path.relpath(repo/'research_bundle'/m.group(1),t.parent).replace('\\','/')+')',content)
    t.write_bytes((nl.join(content.splitlines())+nl).encode('utf-8'))

checks=repo/'publication_checks/update_20260908_selection_witness'
checks.mkdir(parents=True,exist_ok=True)
(checks/'manifest.json').write_text(json.dumps(dict(files=rows,skipped=skipped,
    total_bytes=sum(r['bytes'] for r in rows),narrative_scope_corrections=corrections,
    new_hash_computed=False,weights_uploaded=False,credentials_uploaded=False,
    dataset_images_uploaded=False),ensure_ascii=False,indent=2),encoding='utf-8')
(checks/'README.md').write_text('# 发布范围\n\n两次已完成同批零更新推理、原生匹配见证、分析器及独立复核、不可变执行源码、小回执和完整逐对象表。只加入两个历史叙述的日期化解释修正，原始结果不变。完整资源采样原件留本地/94，发布安全资源摘录与不含进程信息的准入回执；不上传权重、凭据、数据集原图或主机快照。\n',encoding='utf-8')
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),destination=str(dest)),ensure_ascii=False))
