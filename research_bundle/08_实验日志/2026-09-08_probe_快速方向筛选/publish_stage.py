"""Copy only reviewable small evidence; no credentials, weights, source images or host snapshots."""
import json,shutil,subprocess
from pathlib import Path
src=Path(__file__).parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
dest=repo/'research_bundle/08_实验日志'/src.name
if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/full-evidence-20260906':
    raise ValueError('Wrong publication branch')
dest.mkdir(parents=True,exist_ok=True)
rows=[];skipped=[]
for p in sorted(src.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(src)
    if '__pycache__' in rel.parts or p.suffix.lower() not in ('.py','.md','.json','.jsonl','.yaml','.csv','.tsv','.txt'):
        skipped.append(str(rel));continue
    if rel.parts[0].startswith('results_') and rel.parts[0]!='results_1203_snapshot':continue
    if rel.parts[0].startswith('analysis_') and rel.parts[0]!='analysis_1203_snapshot':continue
    if p.name.endswith('_admission.json') or p.name.endswith('_resource_profile.json'):continue
    if p.stat().st_size>5000000:raise ValueError('Unexpected large publication artifact: '+str(rel))
    content=p.read_bytes()
    if any(line.strip().startswith(marker) for line in content.splitlines()
           for marker in (b'-----BEGIN OPENSSH PRIVATE KEY-----',b'-----BEGIN RSA PRIVATE KEY-----')):
        raise ValueError('Private-key marker in publication')
    target=dest/rel
    if dest.resolve() not in target.resolve().parents:raise ValueError('Destination escaped publication root')
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    if target.read_bytes()!=content:raise AssertionError('Publication copy differs')
    rows.append(dict(path=rel.as_posix(),bytes=len(content)))
index=repo/'research_bundle/08_实验日志/README.md'
text=index.read_text(encoding='utf-8')
link='2026-09-08_probe_快速方向筛选/README.md'
row='| 2026-09-08 | [probe_快速方向筛选]('+link+') | probe | 六个已准入短训/full dev共32m；Drone C2未扩大C1收益、LLVIP L2未超过GT，原F因系数约束blocked；另冻结LLVIP C0与F-GM独立短探针 |\n'
if link not in text:
    lines=text.splitlines(keepends=True)
    i=next(i for i,line in enumerate(lines) if line.startswith('| 2026-09-08 |'))
    lines.insert(i,row);index.write_bytes(''.join(lines).encode('utf-8'))
readme=repo/'README.md';text=readme.read_text(encoding='utf-8')
banner='> **2026-09-08 双数据集快速方向筛选**：首批六个准入短训及完整dev评价共32分2.5秒。Drone N/C1/C2 mAP=54.518217/54.543548/54.540724；LLVIP N/L2-box/L2-GT=32.171224/32.147232/32.162699，当前新增方法未显示强信号，不自动E200。原F-rel有梯度但超事先数值系数上限而blocked、没有AP；原记录保留，另在首次F AP前公开修订为实际梯度匹配的独立F-rel-GM探针。LLVIP原C0置信度N/C0也另行冻结。请读[首批结果](research_bundle/08_实验日志/2026-09-08_probe_快速方向筛选/STAGE_REPORT.md)、[全部入口](research_bundle/08_实验日志/2026-09-08_probe_快速方向筛选/README.md)、[F-GM协议修订](research_bundle/08_实验日志/2026-09-08_probe_快速方向筛选/FEATURE_GM_FOLLOWUP_PLAN.md)。单seed成熟模型短训不替代正式归因。\n\n'
if '2026-09-08 双数据集快速方向筛选' not in text:readme.write_bytes((banner+text).encode('utf-8'))
checks=repo/'publication_checks/update_20260908_direction_screen';checks.mkdir(parents=True,exist_ok=True)
(checks/'manifest.json').write_text(json.dumps(dict(files=rows,total_bytes=sum(x['bytes'] for x in rows),new_hash_computed=False,weights_uploaded=False,credentials_uploaded=False,dataset_images_uploaded=False),indent=2,ensure_ascii=False),encoding='utf-8')
(checks/'README.md').write_text('# 发布范围\n\n新方向源码、冻结配置、算子及分析器小样例、完整dev小回执、校准对象表、BN交换与初始复验。首批原F保留blocked，无F评价；两个后续有独立scope及公开协议修订。未上传权重、凭据、原图或完整主机快照。\n',encoding='utf-8')
print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),destination=str(dest))))
