"""Publish/mirror only small project evidence after independent acceptance."""
from pathlib import Path
import base64, io, json, shutil, subprocess, sys, tarfile

root=Path(__file__).parent
workspace=root.parent.parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
mode=sys.argv[1]
for name in ('ACTUAL_CAPTURE_REVIEW.md','independent_review/DRONE_ANALYZER_REVIEW.md','independent_review/CROSS_GT_REVIEW.md'):
    if 'ACCEPTED' not in (root/name).read_text(encoding='utf-8'):
        raise ValueError('Independent acceptance missing: '+name)
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if '__pycache__' in rel.parts or p.name.endswith('_resource_profile.json'):continue
    if p.name.startswith(('GITHUB_PUBLICATION_RECEIPT','REMOTE_MIRROR_RECEIPT')):continue
    if p.suffix.lower() not in {'.py','.md','.json','.jsonl','.yaml','.csv','.tsv','.txt','.gz'}:continue
    if p.stat().st_size>16000000:raise ValueError('Unexpected large artifact '+str(rel))
    files.append((rel,p.read_bytes()))
if mode=='mirror':
    dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_drone_teacher_capture_20260908/review_v1'
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz') as tf:
        for rel,b in files:
            ti=tarfile.TarInfo(rel.as_posix());ti.size=len(b);tf.addfile(ti,io.BytesIO(b))
    code="""from pathlib import Path
import sys,tarfile,json
root=Path(%r);root.mkdir(parents=True,exist_ok=False);rows=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tf:
 for m in tf:
  p=root/m.name
  if not m.isfile() or root not in p.resolve().parents:raise ValueError('Invalid artifact path')
  b=tf.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
  if p.read_bytes()!=b:raise ValueError('Mirror mismatch')
  rows.append(dict(path=m.name,bytes=len(b),byte_exact=True))
r=dict(status='DRONE_CAPTURE_AND_CPU_EVIDENCE_MIRRORED',destination=str(root),files=rows,new_hash_computed=False)
(root/'mirror_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
"""%dest
    encoded=base64.b64encode(code.encode()).decode()
    command="import base64;exec(base64.b64decode('"+encoded+"'))"
    py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
    run=subprocess.run(['ssh','94',py,'-c',"\""+command+"\""],input=archive.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode:raise RuntimeError(run.stderr.decode(errors='replace'))
    result=json.loads(run.stdout)
    with (root/'REMOTE_MIRROR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(result['status'],len(result['files']))
elif mode=='publish':
    if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/full-evidence-20260906':raise ValueError('Wrong branch')
    mirror=root/'REMOTE_MIRROR_RECEIPT.json'
    if not mirror.is_file():raise ValueError('Mirror receipt missing')
    files.append((Path(mirror.name),mirror.read_bytes()))
    dest=repo/'research_bundle/08_实验日志'/root.name
    rows=[]
    for rel,b in files:
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,p)
        if p.read_bytes()!=b:raise ValueError('Copy mismatch')
        rows.append(dict(path=rel.as_posix(),bytes=len(b)))
    index=repo/'research_bundle/08_实验日志/README.md';old=index.read_bytes();nl=b'\r\n' if b'\r\n' in old else b'\n'
    lines=old.decode('utf-8').splitlines()
    row=next(l for l in (workspace/'08_实验日志/README.md').read_text(encoding='utf-8').splitlines() if '[probe_Drone教师完整推理](' in l)
    if any('[probe_Drone教师完整推理](' in l for l in lines):lines=[row if '[probe_Drone教师完整推理](' in l else l for l in lines]
    else:lines.insert(next(i for i,l in enumerate(lines) if l.startswith('| 2026-09-08 |')),row)
    index.write_bytes(nl.join(l.encode('utf-8') for l in lines)+nl)
    p=repo/'README.md';ls=p.read_text(encoding='utf-8').splitlines()
    banner='> **2026-09-08 双数据集完整推理分析完成**：Drone教师1469图28.69秒，完整队列3分33秒；CPU分析19.67秒。结合LLVIP训练侧覆盖，明确定位/置信度/特征的证据与反证。见[方向判断及复核入口](research_bundle/08_实验日志/2026-09-08_probe_Drone教师完整推理/FINAL_REPORT.md)。'
    if not any(l.startswith('> **2026-09-08 双数据集完整推理分析完成') for l in ls):ls=[banner,'']+ls
    p.write_bytes(('\n'.join(ls)+'\n').encode('utf-8'))
    checks=repo/'publication_checks/update_20260908_drone_full_opportunity';checks.mkdir(parents=True,exist_ok=True)
    (checks/'manifest.json').write_text(json.dumps(dict(files=rows,total_bytes=sum(x['bytes'] for x in rows),excluded='raw host resource profiles, caches, weights and images',new_hash_computed=False),ensure_ascii=False,indent=2),encoding='utf-8')
    (checks/'README.md').write_text('# 发布范围\n\n完整教师小体积预测缓存、独立GT配对表、失败attempt及复核源码。资源只发布项目范围的独立验收摘要，原始host profiles保留服务器和本地。无凭据、权重或原图。所有执行输出保留原字节及原PENDING状态，以外部验收文件解释接受范围。\n',encoding='utf-8')
    print('DRONE_EVIDENCE_PUBLISHED',len(rows),sum(x['bytes'] for x in rows))
else:raise ValueError('Expected mirror or publish')
