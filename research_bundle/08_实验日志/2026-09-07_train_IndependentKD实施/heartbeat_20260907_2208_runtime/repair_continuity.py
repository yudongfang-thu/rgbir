"""Repair current navigation statuses only; keep all executed evidence immutable."""
from pathlib import Path
import json,re
W=Path('E:/SHARE/光sar');R=Path(__file__).resolve().parent
paths=[W/'refine-logs/EXPERIMENT_TRACKER.md',W/'08_实验日志/2026-09-07_train_IndependentKD实施/README.md',W/'08_实验日志/README.md']
before=R/'before';before.mkdir(exist_ok=False)
for i,p in enumerate(paths):(before/(str(i)+'_'+p.name)).write_bytes(p.read_bytes())
p=paths[0];s=p.read_text(encoding='utf-8-sig');lines=s.splitlines()
lines=[x for x in lines if not re.match(r'^\|B0[2-5]\|',x)]
new_rows=[
'|B02|诊断|LLVIP优先、Drone对照定位25条件压力|固定train/dev缓存|USER_REQUEST|ACCEPTED_DESCRIPTIVE|两数据集attempt2独立全量复算通过；保留旧汇总计数错误，非物理配准或L1准入|',
'|B03|诊断|完整dev AP与少数类桥接|Drone旧N/C0三seed+LLVIP旧42|USER_REQUEST|ACCEPTED_DESCRIPTIVE|Drone1469dev六端点、LLVIP2406dev两模型；官方TIDE/独立COCO及真值通过，非KD增益|',
'|B04|准备|LLVIP新协议baseline/定位配置迁移核对|42/0/123|PREPARE|PREPARED_NOT_ADMITTED|9份草案、单类退化/数据流与评价profile核对完成；新N兼容、profile适配及L几何/校准仍未准入|',
'|B05|诊断|真实自然64批C/L选择|20260907|USER_REQUEST|ACCEPTED_UNVERIFIED_GEOMETRY|各2048图、原流exact、L207/602；有候选批62/63不等于非零梯度批；未启动新定位训练|']
idx=next(i for i,x in enumerate(lines) if x.startswith('|B01|'));lines[idx+1:idx+1]=new_rows
ids=[x.split('|')[1] for x in lines if re.match(r'^\|[A-Z]\d{2}\|',x)]
assert len(ids)==len(set(ids))
lines[1:1]=['','> **例行核对（2026-09-07 22:08）**：五任务持续更新、六项资源/lease检查通过；C1三seed第22轮，旧shuffled/same-modal为69/60轮。无新E200或故障。本次仅修正文档重复B02–B04编号及过期状态，新增B05表示已完成自然流诊断；不变更科学协议或训练任务。']
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=paths[1];s=p.read_text(encoding='utf-8-sig');first,rest=s.split('\n',1)
note='最新运行核对（2026-09-07 22:08）：[五任务持续更新、资源合规](heartbeat_20260907_2208_runtime/README.md)。C1三seed第22轮，旧shuffled/same-modal为69/60轮，无新E200端点或故障。下方早期时间戳是历史快照。\n\n新的[双数据集证据阶段](../2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)已完成并独立接受：完整dev AP、定位压力与两组真实64批；LLVIP定位候选207、Drone602，均未几何认证。当前证据优先LLVIP定位目标核验与Drone少数类混淆；旧“低置信占多数”仅为200dev对象计数，不能推广为完整macro AP瓶颈。已发布证据提交00bfb59及回执a819fe0。\n'
rest=rest.replace('最新查询（2026-09-07 20:09）','历史查询（2026-09-07 20:09）',1)
p.write_text(first+'\n\n'+note+rest,encoding='utf-8')
p=paths[2];s=p.read_text(encoding='utf-8-sig');lines=s.splitlines()
for i,x in enumerate(lines):
    if '| [train_IndependentKD实施]' in x:
        lines[i]='| 2026-09-07 | [train_IndependentKD实施](2026-09-07_train_IndependentKD实施/README.md) | train | 22:08：C1三seed第22轮、旧控制69/60轮，五任务持续更新，3卡资源/lease合规；无新完整端点或故障，修正跟踪表重复编号，原科学条件不变 |'
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
receipt=dict(status='DOCUMENTATION_REPAIRED_ONLY',removed_duplicate_ids=['B02','B03','B04'],
 canonical_ids=dict(B02='localization_25_condition_stress',B03='full_dev_AP_and_class_bridge',B04='LLVIP_configuration_preparation_not_admitted',B05='fixed_64_batch_selection_diagnostic'),
 all_table_ids_unique=True,scientific_decisions_changed=False,training_changed=False,raw_results_changed=False,
 before_snapshots=str(before),files=[str(p) for p in paths])
(R/'continuity_repair_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
(R/'CONTINUITY_REPAIR.md').write_text('# 文档连续性修正\n\n仅修复当前跟踪表重复B02–B04及过期状态；原编号含义保留：B02定位压力、B03完整AP、B04配置准备，新增B05自然64批。配置准备完成不表示训练准入。实施README新增最新证据入口，旧时间戳保留历史身份。\n\n此前原文副本保存在before目录，原始模型、结果、失败attempt及训练代码未修改。22:08监控为NO_ACTIONABLE_CHANGE；此次文档纠正不是新实验增益或新阶段科学决策。\n',encoding='utf-8')
print(json.dumps(receipt))
