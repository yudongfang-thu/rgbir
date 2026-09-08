"""Update only human-readable navigation after all nine short trainings finish."""
from pathlib import Path
root=Path(__file__).parent
workspace=root.parent.parent
def replace_line(path,needle,replacement):
    data=path.read_bytes();newline=b'\r\n' if b'\r\n' in data else b'\n'
    rows=data.decode('utf-8').splitlines()
    found=[i for i,r in enumerate(rows) if needle in r]
    if len(found)!=1:raise ValueError('Expected exactly one existing navigation line: '+str(path))
    rows[found[0]]=replacement
    path.write_bytes(newline.join(x.encode('utf-8') for x in rows)+newline)
replace_line(workspace/'README.md','新方向筛选实施中（2026-09-08）',
    '> **快速方向筛选已收口（2026-09-08）**：[九次短训的综合结果](08_实验日志/2026-09-08_probe_快速方向筛选/FINAL_REPORT.md)。两数据集分类/定位/置信度/局部特征关系均有实际反馈；F-rel-GM 比 N/C1 高0.037257/0.011926 pp，其余当前新版本也无明显扩训信号，不新增E200。原F阻塞与后续修订分列；短训尚不能预测E200排名。')
replace_line(workspace/'08_实验日志/README.md','[probe_快速方向筛选](',
    '| 2026-09-08 | [probe_快速方向筛选](2026-09-08_probe_快速方向筛选/README.md) | probe | 九次分钟级短训/full dev已完成；分类/定位/置信度/局部特征关系统一读出，F−N/C1=+0.037257/+0.011926 pp；微小单seed差不扩E200 |')
