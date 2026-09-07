# -*- coding: utf-8 -*-
"""Summarize frozen TIDE results without re-evaluation, hashing, or threshold tuning."""
from pathlib import Path
import collections,csv,json
import numpy as np
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'drone_results_v1'
d=json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))['results']
def stats(v):return dict(values=v,mean=float(np.mean(v)),sample_SD=float(np.std(v,ddof=1)),positive_seeds=sum(x>0 for x in v))
def fmt(v):return f'{np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}'
summary={}
lines=['# Drone 完整检测 AP 错误贡献（CPU，2026-09-07）','',
'> 官方 TIDE 在六类main error中，AP50下类别错误的独立贡献最大，AP75下定位最大；special FP oracle另列且贡献可更大。旧C0的AP50并未三seed全升，固定.25背景误检增加也不等于背景dAP三seed增加。本结果不预测可达KD收益。','',
'## 目的与设置','',
'解释完整检测排序/AP瓶颈，区别上一阶段对象子集阈值计数。六份 N/C0 × seed0/42/123，固定last/EMA、全部1469 dev图/22462 GT；每份缓存已包含全部该协议post-NMS输出（conf=.001，NMS IoU=.7，multi-label，max_det300，FP32，native rect544×672）。CPU官方TIDE commit49a5d2a4aeb56795e93a3ed7cc7e6d25757bb4c1，正IoU=.50/.75、背景IoU=.10、max_dets300，无ignore/crowd。','',
'每份包含空预测图并完整计入GT；六份GT与完整roster一致。AP、dAP单位均为百分点；SD为三seed样本SD。TIDE用分数优先匹配/101点均值；原pinned用IoU候选优先去重/插值梯形积分，二者不混称同口径。','',
'## 原值','', '|端点|原pinned AP50|原pinned AP75|原pinned mAP50–95|TIDE AP50|TIDE AP75|', '|---|---:|---:|---:|---:|---:|']
for name,r in d.items():
    p=r['pinned_metrics'];lines.append('|'+name+'|'+ '|'.join(f'{v:.5f}' for v in [p['AP50']*100,p['AP75']*100,p['mAP50_95']*100,r['thresholds']['0.5']['AP'],r['thresholds']['0.75']['AP']])+'|')
lines+=['','## 独立oracle贡献','', '六类dAP是逐项单独纠正后的AP变化，不可相加；Miss oracle还涉及调整漏检分母，不等于合成一批可学出的新检测。','', '|IoU|臂|Cls|Loc|Both|Dupe|Bkg|Miss|FP oracle|FN oracle|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in ('0.5','0.75'):
    summary[t]={}
    for arm in ('N','C0'):
        measures={key:[d[f'{arm}_s{s}']['thresholds'][t][category][key] for s in (0,42,123)] for category,keys in [('main_dAP',['Cls','Loc','Both','Dupe','Bkg','Miss']),('special_dAP',['FalsePos','FalseNeg'])] for key in keys}
        summary[t][arm]={key:stats(v) for key,v in measures.items()}
        lines.append('|'+t+'|'+arm+'|'+'|'.join(fmt(v) for v in measures.values())+'|')
lines+=['','## C0−N配对变化','', '|量|seed0|seed42|seed123|mean±SD|正方向seed|','|---|---:|---:|---:|---:|---:|']
for label,t,category,key in [('TIDE AP50','0.5',None,'AP'),('TIDE AP75','0.75',None,'AP'),('AP50 Cls dAP','0.5','main_dAP','Cls'),('AP50 Bkg dAP','0.5','main_dAP','Bkg'),('AP75 Loc dAP','0.75','main_dAP','Loc'),('AP75 Bkg dAP','0.75','main_dAP','Bkg')]:
    def get(arm,s):
        r=d[f'{arm}_s{s}']['thresholds'][t]
        return (r[category] if category else r)[key]
    v=[get('C0',s)-get('N',s) for s in (0,42,123)];summary[label]=stats(v)
    lines.append('|'+label+'|'+'|'.join(f'{x:+.5f}' for x in v)+'|'+fmt(v)+'|'+str(sum(x>0 for x in v))+'/3|')
lines+=['','dAP残余增加意味着该oracle相对当前模型的贡献变大，不直接等价于训练导致该错误机制恶化；AP变化与多个错误、排序、召回共同有关。原正式mAP50–95的C0−N仍是+.266655±.144373pp，本实验不重定义它。','',
'## 排序与类别分母','', '|端点|所有低阈值预测数|其中空预测图|达到max_det图|score≥.25 TP|score≥.25 FP|其中Bkg|score≥.5 Bkg|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in d.items():
    bins=r['thresholds']['0.5']['score_diagnostic']['pooled']['bins'];a=r['input_audit']
    numbers=[a['predictions'],len(a['empty_prediction_images']),a['images_at_max_det'],sum(b['tp'] for b in bins[3:]),sum(b['fp'] for b in bins[3:]),sum(b['error_counts'].get('Bkg',0) for b in bins[3:]),bins[4]['error_counts'].get('Bkg',0)]
    lines.append('|'+name+'|'+'|'.join(map(str,numbers))+'|')
lines+=['', '三seed C0−N的score≥.25 Bkg为+3/+5/+46，复现旧专项复核触发；但AP50 Bkg dAP为两降一升，AP75为三降。seed0的score≥.5 Bkg还减少41个，而[.25,.5)增加44个。**固定阈值背景数增加不能单独推断背景AP损伤。**','',
'完整GT中car=18965（84.43%），freight car=710、truck=1336、bus=751、van=700；宏平均AP给每类相同权重。上一阶段top1对象机会中的少量纯类别错误不能排除完整多标签预测中的少数类混淆/排序瓶颈。','',
'|类别|GT|N seed0 score≥.25 FP|其中Cls|其中Bkg|', '|---|---:|---:|---:|---:|']
for c,n in enumerate(['car','freight car','truck','bus','van']):
    bins=d['N_s0']['thresholds']['0.5']['score_diagnostic'][str(c)]['bins'][3:]
    lines.append(f"|{n}|{d['N_s0']['input_audit']['gt_by_class'][str(c)]}|{sum(b['fp'] for b in bins)}|{sum(b['error_counts'].get('Cls',0) for b in bins)}|{sum(b['error_counts'].get('Bkg',0) for b in bins)}|")
lines+=['','上述逐类表只用seed0展示分母结构，完整六端点逐类分数区间、TP/FP分位数保存在JSON；不据单seed挑方法。','',
'## 验证、限制与下一步','',
'1. 官方TIDE与独立NumPy score-greedy/AP101实现对全部六端点×两个IoU逐类AP、GT/TP/FN交叉一致（AP容差1e-9pp），已知真值七状态、空图、score ties/无GT类别、单调分数变换检查通过。','2. 属完整dev的描述性分析，未证明教师能修复这些错误，更未证明RGB学生可学出；下一步应将真实高分混淆/低分TP与统一IR/N0知识比较，保留当前C1训练。','3. AP75下定位oracle大说明高IoU对定位敏感，不验证任何IR目标几何合同，不自动准入L1。','4. 标签未标注对象可能被归为背景；原低阈值.001与max_det300已限定可观测空间，不声称无限召回。','5. 需独立审阅接受后才可升级分析器证据等级；不完成三seed四臂跨模态归因。','',
'## 产物与来源','',
'- `drone_results_v1/{N,C0}_s{0,42,123}.json`：六端点完整统计；`summary.json`、`tide_endpoints.csv`。','- `drone_results_v1/synthetic_truth.json`、`synthetic_validation.json`：真值与测试。','- `run_tide_audit.py`：后续不新增hash的CPU入口；原次源码 `drone_results_v1/runner_source.py` 保留。','- `PROTOCOL.md` 首次冻结；`PROTOCOL_ADDENDUM.md` 来源记录勘误。','- 输入本地根：`../../2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/`。服务器原始根：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/legacy_diagnostics_v1/`。本次未连接服务器。','- 官方作者：[TIDE仓库](https://github.com/dbolya/tide)、[论文](https://arxiv.org/abs/2008.08115)。运行未修改官方源码。','']
(HERE/'README_DRONE.md').write_text('\n'.join(lines),encoding='utf-8')
(HERE/'drone_aggregate.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Report and aggregates written')
