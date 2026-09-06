# Task-Conditional 首轮实施（2026-09-07）

> **03:22 阶段结论：独立模块已部署94，N/C与两种C内容对照均完成24次成功更新，新旧C/N精确等价通过；D1/D2发现定位机会，但已接纳几何覆盖内有效对象为0，CL42/CGT42不准入。按预设分支继续C归因：C-shuffled42正式训练已启动，C-same-modal42等待资源。**

## 目的
保持 OEv1 C 不变，检验条件定位教师内容是否同时超过 C 与同掩码 GT 重监督。正式冻结见 EXPERIMENT_PLAN.md。

## 设置
主数据集 DroneVehicle；几何与 D2 不通过而 LLVIP 通过时转 LLVIP并补匹配C。第一批仅 CL42/CGT42 E200；已有 N/C/random 三 seed 继续。资源按根 AGENTS 与现行全项目 lease 执行，正式训练双开须实测。

## 结果
详见 [阶段执行验收](EXECUTION_REVIEW.md)、[D1/D2独立分析](../2026-09-07_probe_TaskConditional机会诊断/INDEPENDENT_RESULTS.md) 和 [C内容归因](../2026-09-07_train_OEv1内容归因/README.md)。截至03:20旧N/C/random仍为4/9独立端点；只有seed42配对，C−N mAP +0.144554pp，AP75 −0.327243pp。新正式长训尚无AP。

## 结论
未产生新的定位训练结论；几何、校准、canary未通过前不启动L长训。未通过只限制本版L，C的可解释实验继续。

## 产物路径
- 本地代码：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/`。
- 94目标：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/`。
- 94 runs：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_task_conditional_v1_20260907/`。
- 几何证据：`../2026-09-07_probe_TaskConditional几何审计/`。

## 局限与下一步
λL尚未校准，CL/CGT/L真实训练canary、E200及后续16臂均未执行。当前没有几何合格D2，不能用无几何诊断选中对象绕过门槛，也不能宣称L无效。待独立配准覆盖得到补充后再按原阈值执行校准及准入；没有静默改门、box fallback或重定义GT控制。阶段证据同步既有GitHub分支。
