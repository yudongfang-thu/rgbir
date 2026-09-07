# 双数据集证据优先推进（2026-09-07）

**完整dev AP、逐类对象桥接与定位压力诊断已完成且独立接受：LLVIP优先定位，Drone优先少数类混淆；两组自然64批选择诊断也已完成并获独立限定接受。不把baseline差距或oracle当成KD增益，原L1尚未准入。**

## 目的
落实用户最新指示“llvip和drone都需要做，但是优先做证据空间大的”。两个数据集都保留；LLVIP优先定位，Drone的完整AP证据将新增分析重点收敛至少数类混淆。现有C1三seed及C0内容控制继续按冻结协议完成。

## 冻结设置与工作分工
- LLVIP为首要定位诊断，Drone做同设置对照。复用上一阶段raw DFL/框/GT/anchor缓存，固定对象和基础集合，检查0与1/2/4输入像素扰动的教师定位质量、支持域、DFL分布及头部梯度；具体坐标变换和读出规则由执行前PROTOCOL明确。压力测试不证明真实配准误差，也不替代原L1几何证据。
- Drone优先实际post-NMS错误/AP分析，并清点LLVIP可用预测。复用已完成N/C0六端点，不重做模型推理；原生评价口径与独立错误分析分别标注，不能把固定阈值对象计数称作AP上限。
- 所有新分析先记录协议、检查已知真值及独立审阅，再形成结论。仅CPU分析不申请GPU；必要GPU任务由根代理统一使用已有lease、canary实测及screen管理，不建立第二资源池。
- 不改C1的lambda/样本流或既定结果门槛，不放宽原L1的0.70门，不自动启动联合/feature新方法。需要另立定位版本时保留原版本身份并完整冻结新合同。
- 仅既有train/dev，test封存。原始结果、失败attempt、回执与权重不覆盖、不移动、不删除。

## 当前结果
详细结果和下一步依据见[阶段报告](STAGE_REPORT.md)。

|工作|主要结果|复核入口|
|---|---|---|
|Drone六端点完整1469dev AP|三少数类占N的macro AP50分类oracle贡献94.26%|[主分析](ap_error/README_DRONE.md)、[独立逐类审阅](ap_error/independent_review/SUPPLEMENT_AUDIT.md)|
|LLVIP旧RGB/IR完整2406dev|mAP32.8784/48.8529，定位是六类main error主要瓶颈|[实际评估](llvip_full_eval/README.md)、[独立审阅](ap_error/llvip_independent_review/EXPERIMENT_AUDIT.md)|
|两数据集25条件定位压力|固定dev门内四像素八方向门控存活LLVIP34/61、Drone9/47|[结果](localization_stress/README.md)、[独立审阅](localization_stress/independent_review/EXPERIMENT_AUDIT.md)|
|LLVIP工程准备|9份NOT_ADMITTED配置，单类C1=C1_y、评价profile仍待适配|[准备报告](llvip_preparation/README.md)|
|真实自然64批选择|LLVIP207/62批，Drone602/63批；真实anchor、逐gate图/组与分类对象分布|[执行入口](natural_flow_diagnostic/README.md)|

已有训练21:55：C1按42/0/123为第22/21/22轮，旧shuffled/same-modal第69/59轮，无新完整端点。独立评价优先、GPU统一lease、按实测峰值预约，未抢停旧任务。

## 路径
本目录保存冻结协议、脚本、小产物和独立审阅。94若需要新增诊断，使用 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/`，实际任务与资源回执另登记。未使用目录不冒称已部署。

## 局限与下一步
LLVIP当前visible42是历史模型，尚无新协议N三seed；新定位训练仍需解决目标可用性/自然训练流覆盖和完整梯度校准。最终方法收益仍需同协议三seed与四臂归因。
