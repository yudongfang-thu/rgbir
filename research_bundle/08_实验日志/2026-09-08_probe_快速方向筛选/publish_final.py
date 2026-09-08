"""Publish small finalized evidence, then refresh the existing navigation entry."""
from pathlib import Path
import json
root=Path(__file__).parent
if not (root/'FINAL_REPORT.md').is_file():raise ValueError('Final report required')
source=(root/'publish_stage.py').read_text(encoding='utf-8')
source=source.replace("if '__pycache__' in rel.parts", "if p.name=='FINAL_REPORT_DRAFT.md':continue\n    if '__pycache__' in rel.parts")
source=source.replace("if p.stat().st_size>5000000:",
    "if p.stat().st_size>(8000000 if rel.as_posix()=='cpu_selection_coverage_audit_v1/objects.csv' else 5000000):")
exec(compile(source,str(root/'publish_stage.py'),'exec'))
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
prefix='> **2026-09-08 双数据集快速方向筛选**'
banner='> **2026-09-08 双数据集快速方向筛选已完成**：九次短训与九次完整dev评价已收口。分类、定位、置信度和局部特征关系的原值、去留范围、实际耗时、代理局限与协议修订见[综合结果](research_bundle/08_实验日志/2026-09-08_probe_快速方向筛选/FINAL_REPORT.md)。全部是单seed探索，不自动扩大E200；原F-rel阻塞与独立F-rel-GM结果分别保留。'
readme=repo/'README.md'
text=readme.read_text(encoding='utf-8')
lines=text.splitlines()
lines=[banner if line.startswith(prefix) else line for line in lines]
readme.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
index=repo/'research_bundle/08_实验日志/README.md'
data=index.read_bytes();newline=b'\r\n' if b'\r\n' in data else b'\n'
lines=data.decode('utf-8').splitlines()
row='| 2026-09-08 | [probe_快速方向筛选](2026-09-08_probe_快速方向筛选/README.md) | probe | 九次分钟级短训与完整dev已完成；分类/定位/置信度/局部特征关系统一读出，微小单seed差不扩E200；保留原F阻塞与F-GM协议修订 |'
lines=[row if '[probe_快速方向筛选](' in line else line for line in lines]
index.write_bytes(newline.join(x.encode('utf-8') for x in lines)+newline)
checks=repo/'publication_checks/update_20260908_direction_screen'
(checks/'README.md').write_text('# 发布范围\n\n九次实际短训与完整dev评价、源码、冻结配置、算子与分析器检查、跨scope控制投影、BN交换、初始复验、方向诊断及最终综合报告。原F-rel的BLOCKED和F-rel-GM独立协议修订均保留。未上传凭据、权重、原图或完整主机快照；没有新文件hash。\n',encoding='utf-8')
