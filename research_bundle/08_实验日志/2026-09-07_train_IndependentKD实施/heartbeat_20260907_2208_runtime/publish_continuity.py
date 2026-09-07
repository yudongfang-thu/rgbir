"""Publish a bounded continuity fix, preserving the routine snapshot bytes."""
from pathlib import Path
import json
w=Path('E:/SHARE/光sar');r=Path(__file__).resolve().parent
repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
(r/'INDEPENDENT_CONTINUITY_REVIEW.md').write_text('# 独立文档连续性复核\n\n审阅者/root/loc_stress只读核验后返回PASS：修后tracker共39行、ID唯一；B02定位压力、B03完整AP、B04配置准备未准入，B05自然64批。各实际独立接受回执存在且范围一致，B04的9份草案/6CPU检查不等于训练/profile/geometry准入。\n\n实施README指向22:08；双数据集README保留21:55时间戳。22:08五任务更新增量全正、六项资源/lease检查通过、无新端点。无科学决策或训练变化，整个冲刺未完成。本记录由root根据独立代理实际回复落盘；审阅者没有修改文件。\n',encoding='utf-8')
p=w/'99_整理回执/20260907_IndependentKD实施新增目录.md'
with p.open('a',encoding='utf-8') as f:f.write('\n22:08心跳：本地新增实施日志heartbeat_20260907_2208_runtime，保存只读状态、原文before副本和文档连续性修正回执；无服务器目录或原实验产物变更。跟踪表B02–B04恢复原ID语义，新增B05；不改结果、门槛或训练。\n')
files=[w/'refine-logs/EXPERIMENT_TRACKER.md',w/'08_实验日志/README.md',w/'08_实验日志/2026-09-07_train_IndependentKD实施/README.md',w/'99_整理回执/20260907_IndependentKD实施新增目录.md',w/'08_实验日志/2026-09-07_train_IndependentKD实施/running_state_20260907_220857.json']+[p for p in r.rglob('*') if p.is_file()]
for p in files:
    dest=repo/'research_bundle'/p.relative_to(w);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(p.read_bytes());assert dest.read_bytes()==p.read_bytes()
print(json.dumps(dict(copied_files=len(files),scope='documentation continuity repair and attributable routine snapshot; no new scientific result')))
