# RGBIR 对象判别蒸馏首轮（2026-09-06）

> **21:46最新核验（2026-09-06）**：P42=54.658独立端点保持，N42完成135轮，完整净收益仍待同代码N42。 见[夜间结果与GitHub更新](../2026-09-06_audit_RGBIR夜间结果与GitHub更新/README.md)。下文旧快照按各自时间读取。

> **17:30最新核验（2026-09-06）**：首个P42训练与独立评估已完成，mAP50–95=54.658；N42已接续运行，尚无同代码净收益。 见[晚间进度与新结果](../2026-09-06_audit_RGBIR晚间进度与新结果/README.md)。下文保留较早状态。

> 结论：代码及真实训练检查通过；94 的物理 GPU 4 已运行 DroneVehicle paired seed42/E200，weight0 同配置排在后面串行执行。本轮只用这一张 GPU，性能结果尚未产生。

> 后续（2026-09-06 07:42）：用户授权按项目AGENTS扩展资源，已补seed0/123至GPU5/6；本条目原seed42队列保持原样，当前约89/200。后续状态见[三seed扩展记录](../2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md)。

## 目的
根据521对baseline诊断，验证对象相对邻近背景的类别判别证据迁移能否改善RGB学生。首轮比较净收益可行性，避免把普遍定位优势当作前提。

## 设置
模型YOLO11n，IR教师与RGB参考均为已核实seed42原生baseline，冻结；学生从通用yolo11n初始化。RGB原生GT检测监督，IR独立GT只用于训练期对象对应/教师质量。每个对象提取P3/P4正确类别logit的前景−局部背景证据，质量选择后SmoothL1蒸馏；不增加部署结构。具体定义以本目录EXPERIMENT_PLAN.md与配置副本为准。

首轮 N/P 各一个seed42，E200、batch32、nbs64、workers4。同卡串行，优先paired，weight0排队；不会因早期AP改变阈值或停止另一臂。既有其他研究任务保持不变。

## 结果
19 项 pinned CPU 检查全部通过；独立审查发现的 canary 梯度检查、eval roster 和队列退出码问题均已修复。

| 检查 | paired | weight0 |
|---|---:|---:|
| 实际 batch / 参数更新 / AMP skip | 30 / 24 / 6 | 30 / 24 / 6 |
| 入选对象实例数（跨 batch 累计） | 3767 | 3767 |
| 首批 KD 对 scores 梯度 L2 | 0.003656818 | 0.003656818（最终剂量为0） |
| NVML 显存峰值 MiB | 6304 | 6304 |
| 主进程与子进程 RSS 峰值 MiB | 28895 | 28690 |

两臂初始 student state 与首批 RGB/IR 输入、学生标签直接逐张量相等。canary 核验 zero-weight 总损失及 scores 梯度与 native 精确相等；paired 单次加权的梯度组合最大误差3.81e-6。教师/reference无梯度且不进optimizer/EMA参数。初始AMP动态缩放各跳过6次更新，记录与真实更新分别保存；不把固定epoch数等同于长期成功更新数必然相等。

screen：`rgbir_oev1_queue_s42`。队列：paired E200 → last/EMA full val → weight0 E200 → last/EMA full val；统一开发集1469张，不使用test。正式预约显存10000MiB、RSS49152MiB，比实测峰值留出余量。所有阶段由resource guard约束，锁定GPU4。

## 结论
当前已确认代码能在真实数据上执行、蒸馏有非零梯度、同代码weight0对照路径正确；尚不能判断方法是否改善检测性能。canary不用于按AP挑选超参，完整训练不读取中间AP调参。

## 产物路径
- 本地代码：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1/`。
- 94脚本/小产物：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/`。
- 94训练结果：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/`。
- 框架审查：framework_notes.md；独立review：EXPERIMENT_CODE_REVIEW*.md。
- 参数冻结：EXPERIMENT_PLAN.md、code/config_drone.yaml；代码副本：code/。
- 工程证据：canary_summary.json、canary_comparison.json、cpu_tests_attempt1.log、canary_*_receipt.json。
- 正式代码：94 artifacts/rgbir_object_evidence_v1_20260906/release_v2/；canary使用release_v1，两版trainer/loss/loader/config通过直接文件比较一致，仅队列异常状态和eval单位标记补充。
- 训练run：`.../runs/rgbir_object_evidence_v1_20260906/full_paired_s42_attempt1/`；下一run：`full_weight0_s42_attempt1/`（排队，尚未开始）。
- 总日志：94 artifacts/rgbir_object_evidence_v1_20260906/full_attempt1.log；状态：full_queue_status.json；本地启动快照：launch_status.json。

## 局限与下一步
本次两个seed42 run是探索性首轮；三seed和paired/shuffled/same-modal/weight0、同剂量随机等对照仍是后续claim要求。旧baseline不可直接替代同代码N；新KD不是已验证的论文创新。
