"""Mirror and publish completed CPU evidence only; never alter original outputs."""
from pathlib import Path
import base64,io,json,re,shutil,subprocess,sys,tarfile

root=Path(__file__).parent;workspace=root.parent.parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
mode=sys.argv[1]
review=json.loads((root/'independent_review/EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'))
if review['integrity_status'] not in ('pass','warn'):raise ValueError('Independent scoped review required')
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if '__pycache__' in rel.parts or p.name.endswith('_resource_profile.json'):continue
    if p.name in ('REMOTE_MIRROR_RECEIPT.json','GITHUB_PUBLICATION_RECEIPT.json'):continue
    if p.suffix.lower() not in {'.py','.md','.json','.jsonl','.yaml','.csv','.tsv','.txt','.gz'}:continue
    b=p.read_bytes()
    if len(b)>16000000:raise ValueError('Large artifact '+str(rel))
    if p.suffix.lower()!='.gz' and re.search(rb'BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}',b):raise ValueError('Credential marker '+str(rel))
    files.append((rel,b))
if mode=='mirror':
    dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_localization_learning_target_20260908/review_v1'
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz') as tf:
        for rel,b in files:
            m=tarfile.TarInfo(rel.as_posix());m.size=len(b);tf.addfile(m,io.BytesIO(b))
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
r=dict(status='LOCALIZATION_LEARNING_TARGET_EVIDENCE_MIRRORED',destination=str(root),files=rows,new_hash_computed=False)
(root/'mirror_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
"""%dest
    encoded=base64.b64encode(code.encode()).decode()
    command="import base64;exec(base64.b64decode('"+encoded+"'))"
    py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
    p=subprocess.run(['ssh','94',py,'-c','"'+command+'"'],input=archive.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    r=json.loads(p.stdout)
    with (root/'REMOTE_MIRROR_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(r['status'],len(files))
elif mode=='publish':
    assert subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()=='research/full-evidence-20260906'
    mirror=root/'REMOTE_MIRROR_RECEIPT.json';files.append((Path(mirror.name),mirror.read_bytes()))
    dest=repo/'research_bundle/08_实验日志'/root.name;rows=[]
    for rel,b in files:
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,p)
        if p.read_bytes()!=b:raise ValueError('Published bytes differ '+str(rel))
        rows.append(dict(path=rel.as_posix(),bytes=len(b),byte_exact=True))
    index=repo/'research_bundle/08_实验日志/README.md';old=index.read_bytes();nl=b'\r\n' if b'\r\n' in old else b'\n'
    lines=old.decode('utf-8').splitlines();marker='[probe_定位学习位置与目标]('
    row=next(l for l in (workspace/'08_实验日志/README.md').read_text(encoding='utf-8').splitlines() if marker in l)
    if any(marker in l for l in lines):lines=[row if marker in l else l for l in lines]
    else:lines.insert(next(i for i,l in enumerate(lines) if l.startswith('| 2026-09-08 |')),row)
    index.write_bytes(nl.join(x.encode('utf-8') for x in lines)+nl)
    p=repo/'README.md';lines=p.read_text(encoding='utf-8').splitlines()
    banner='> **2026-09-08 定位学习位置与目标诊断完成**：复用首32图与8批L2记录，核对实际学习anchor、教师坐标目标和GT控制，给出本版L2去留决定。见[完整结果与边界](research_bundle/08_实验日志/2026-09-08_probe_定位学习位置与目标/FINAL_REPORT.md)。'
    if not any(x.startswith('> **2026-09-08 定位学习位置与目标诊断完成') for x in lines):lines=[banner,'']+lines
    p.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    checks=repo/'publication_checks/update_20260908_localization_learning_target';checks.mkdir(parents=True,exist_ok=True)
    (checks/'manifest.json').write_text(json.dumps(dict(files=rows,total_bytes=sum(x['bytes'] for x in rows),new_hash_computed=False,raw_host_profiles_excluded=True,weights_uploaded=False,images_uploaded=False,credential_marker_scan_passed=True),ensure_ascii=False,indent=2),encoding='utf-8')
    (checks/'README.md').write_text('# 发布范围\n\n仅新诊断源码、历史输入小副本、CPU原值、失败及独立复核回执；字节一致。原始PENDING和失败attempt保留，外部验收解释范围。无模型前向/新训练、权重、原图或凭据。\n',encoding='utf-8')
    print('LOCALIZATION_LEARNING_TARGET_PUBLISHED',len(files),sum(x['bytes'] for x in rows))
else:raise ValueError('Expected mirror or publish')
