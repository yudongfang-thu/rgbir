"""Close this evidence stage in human-readable indexes; raw evidence is untouched."""
from pathlib import Path
W=Path('E:/SHARE/光sar');R=Path(__file__).resolve().parent
p=R/'README.md';s=p.read_text(encoding='utf-8').replace('自然64批选择诊断正在推进','两组自然64批选择诊断也已完成并获独立限定接受')
s=s.replace('已有训练21:29：C1三个seed第20轮，旧shuffled/same-modal第66/58轮','已有训练21:55：C1按42/0/123为第22/21/22轮，旧shuffled/same-modal第69/59轮')
s=s.replace('固定流、真实confidence-first anchor、逐gate图/组与分类剂量分布','LLVIP207/62批，Drone602/63批；真实anchor、逐gate图/组与分类对象分布')
p.write_text(s,encoding='utf-8')
p=R/'natural_flow_diagnostic/README.md';s=p.read_text(encoding='utf-8')
old='**源码与 6 项 CPU 合约检查通过，尚未执行真实 GPU canary/64 批诊断。结果身份固定为 `UNVERIFIED_GEOMETRY_DIAGNOSTIC`，不提供 L1/L_GT 或校准准入。**'
new='**有效attempt2完成：LLVIP/Drone各2批技术短测和64批真实自然流，原记录字段逐批exact；L待认证候选207/602，有候选批62/63。真实结果已获[独立限定接受](independent_review/REAL_RESULTS_EXPERIMENT_AUDIT.json)。结果身份为 `UNVERIFIED_GEOMETRY_DIAGNOSTIC`，不提供L1/L_GT或校准准入。**\n\n[完整结果](completed_readout_attempt2/README.md) · [限定解读](completed_readout_attempt2/FINDINGS.md) · [129文件逐字节采集回执](collection_receipt.json) · [启动失败及最小修复](ATTEMPT1_STARTUP_FAILURE.md)。SCP嵌套目录超Windows路径长度导致采集部分失败，使用tar流和extended路径补齐缺失，已有文件字节核对未覆盖；远端诊断结果未改变。'
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
p=W/'08_实验日志/README.md';s=p.read_text(encoding='utf-8');rows=s.splitlines()
for i,line in enumerate(rows):
    if '| [probe_双数据集证据优先推进]' in line:
        rows[i]='| 2026-09-07 | [probe_双数据集证据优先推进](2026-09-07_probe_双数据集证据优先推进/README.md) | probe | 已完成并独立接受：完整dev AP/逐类/定位压力+两组自然64批；LLVIP定位候选207/62批、Drone602/63批；优先LLVIP定位与Drone少数类混淆，非KD收益或L1准入 |'
    elif '| [train_IndependentKD实施]' in line:
        rows[i]='| 2026-09-07 | [train_IndependentKD实施](2026-09-07_train_IndependentKD实施/README.md) | train | 21:55：C1按42/0/123为22/21/22轮、旧控制69/59，3卡5训练资源合规；无新完整端点，C0专项复核和原L几何阻塞保持 |'
p.write_text('\n'.join(rows)+'\n',encoding='utf-8')
p=W/'refine-logs/EXPERIMENT_TRACKER.md';s=p.read_text(encoding='utf-8')
new='> **本阶段完成（2026-09-07 21:55）**：两组真实64批及全量独立复算通过，LLVIP定位待认证候选207（62批/187图/14来源组），Drone602（63批/268图/31组）。LLVIP优先定位目标认证/学习验证，Drone优先少数类与来源归因；不是L1准入。原C1按42/0/123为22/21/22轮、旧控制69/59轮、3卡5训练、RSS141.51GiB，无新E200端点。'
first,rest=s.split('\n',1);s=first+'\n\n'+new+'\n'+rest
marker='|B01|诊断|'
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith(marker):
        lines[i+1:i+1]=[
        '|B02|诊断|两数据集完整dev AP与少数类桥接|旧N/C0三seed+LLVIP旧42|USER_REQUEST|ACCEPTED_DESCRIPTIVE|Drone1469dev六端点、LLVIP2406dev两模型；官方TIDE/独立COCO及真值通过，非KD增益|',
        '|B03|诊断|定位固定25条件压力|固定train/dev缓存|USER_REQUEST|ACCEPTED_DESCRIPTIVE|两数据集attempt2独立全量复算通过；旧汇总计数错误保留，非物理配准证明|',
        '|B04|诊断|真实自然64批C/L选择|20260907|USER_REQUEST|ACCEPTED_UNVERIFIED_GEOMETRY|各2048图、原流exact、L207/602；有候选批62/63非非零梯度批；未启动新定位训练|']
        break
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=W/'99_整理回执/20260907_IndependentKD实施新增目录.md'
with p.open('a',encoding='utf-8') as f:f.write('\n双数据集阶段完成：94新增有效 llvip_full_eval_attempt2 与 natural_flow_attempt2，均在原rgbir_evidence_priority_20260907根。两个旧attempt1分别因标签路径/配置导入失败，原始产物保留；本地保存完整有效预测、自然流/选择记录、CPU原始统计与独立审阅。仅补充导航/解释，不移动、覆盖、删除旧结果或权重。自然流初次SCP长路径失败通过tar流补齐缺失文件，已有文件逐字节一致；新增governance_group_sources为原TSV只读副本。\n')
