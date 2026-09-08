"""Mirror and publish completed training-side CPU evidence, preserving executed outputs."""
from pathlib import Path
import base64,json,subprocess,shutil,sys
root=Path(__file__).parent;workspace=root.parent.parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
mode=sys.argv[1]
if 'ACCEPTED' not in (root/'independent_review/README.md').read_text(encoding='utf-8'):raise ValueError('Core review required')
if 'ACCEPTED' not in (root/'independent_review/BRIDGE_REVIEW.md').read_text(encoding='utf-8'):raise ValueError('Bridge review required')
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if '__pycache__' in rel.parts or p.suffix.lower() not in {'.py','.md','.json','.jsonl','.yaml','.csv','.tsv','.txt'}:continue
    b=p.read_bytes()
    if len(b)>6000000:raise ValueError('Unexpected large file')
    files.append((rel,b))
if mode=='mirror':
    dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_train_localization_coverage_20260908/review_v1'
    rows=[dict(path=p.as_posix(),data=base64.b64encode(b).decode()) for p,b in files]
    code="""from pathlib import Path
import json,base64
root=Path(%r);root.mkdir(parents=True,exist_ok=False);rows=[]
for r in json.loads(%r):
 p=root/r['path'];b=base64.b64decode(r['data'])
 if root not in p.resolve().parents:raise ValueError('Invalid path')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 if p.read_bytes()!=b:raise ValueError('Mirror differs')
 rows.append(dict(path=str(p),bytes=len(b),byte_exact=True))
r=dict(status='TRAIN_LOC_CPU_EVIDENCE_MIRRORED',destination=str(root),local_origin='E:/SHARE/光sar/08_实验日志/2026-09-08_probe_训练侧定位覆盖',files=rows,new_hash_computed=False)
(root/'mirror_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
"""%(dest,json.dumps(rows))
    py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
    p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    r=json.loads(p.stdout)
    with (root/'REMOTE_MIRROR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(r['status'],len(r['files']))
elif mode=='publish':
    if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/full-evidence-20260906':raise ValueError('Wrong branch')
    dest=repo/'research_bundle/08_实验日志'/root.name;rows=[]
    for rel,b in files:
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,p)
        if p.read_bytes()!=b:raise ValueError('Copy differs')
        rows.append(dict(path=rel.as_posix(),bytes=len(b)))
    index=repo/'research_bundle/08_实验日志/README.md';data=index.read_bytes();nl=b'\r\n' if b'\r\n' in data else b'\n'
    lines=data.decode('utf-8').splitlines()
    row=next(l for l in (workspace/'08_实验日志/README.md').read_text(encoding='utf-8').splitlines() if '[probe_训练侧定位覆盖](' in l)
    if any('[probe_训练侧定位覆盖](' in l for l in lines):lines=[row if '[probe_训练侧定位覆盖](' in l else l for l in lines]
    else:lines.insert(next(i for i,l in enumerate(lines) if l.startswith('| 2026-09-08 |')),row)
    index.write_bytes(nl.join(l.encode('utf-8') for l in lines)+nl)
    p=repo/'README.md';ls=p.read_text(encoding='utf-8').splitlines()
    banner='> **2026-09-08 训练侧定位门覆盖核对完成**：复用首32图缓存与历史同批L2记录，11个双方粗检出而仅IR达IoU0.75的对象中，L2实际选4个，另7个因参考IoU≥0.70被前置门排除。全程CPU、无新增前向；门的覆盖与学习效果分别判断。见[原始记录及独立验收](research_bundle/08_实验日志/2026-09-08_probe_训练侧定位覆盖/FINAL_REPORT.md)。'
    if not any(l.startswith('> **2026-09-08 训练侧定位门覆盖核对完成') for l in ls):ls=[banner,'']+ls
    p.write_bytes(('\n'.join(ls)+'\n').encode('utf-8'))
    checks=repo/'publication_checks/update_20260908_train_loc_coverage';checks.mkdir(parents=True,exist_ok=True)
    (checks/'manifest.json').write_text(json.dumps(dict(files=rows,total_bytes=sum(r['bytes'] for r in rows),new_hash_computed=False,weights_uploaded=False,dataset_images_uploaded=False),ensure_ascii=False,indent=2),encoding='utf-8')
    (checks/'README.md').write_text('# 发布范围\n\n首32缓存定位读出、历史L2记录桥接、逐对象表及独立复核，包含真实旧base记录副本。不含新模型前向/训练、凭据、权重或原图。\n',encoding='utf-8')
    print('TRAIN_LOC_CPU_EVIDENCE_PUBLISHED',len(rows))
else:raise ValueError('Expected mirror or publish')
