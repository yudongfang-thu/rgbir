# 单批 DFL 生产源码独立验收

**结论：`READY_FOR_SINGLE_BATCH`，仅准入已冻结首 32 图的一次 S/R/T 零更新数据前向及其原始分布导出。真实输出尚未产生，本结论不是实际数据或 DFL 效用验收。**

审阅者 `/root/dfl_readout_review`；可观察模型身份 `unavailable`。审阅覆盖 release_v1 的七份源/配置，以及原 witness driver、raw_shapes、原 OEv1 布局/解码、原 DFLoss、pinned setup/check_amp 源、历史 80 GT/native anchor 记录和实际 CPU 回执。未启动 GPU、模型主干前向、训练、官方 test 或新 hash。

## 实际执行证据

- `pinned_cpu_v1.json`：两个命令均 exit 0，用固定 Torch 2.10.0+cu128 / Ultralytics 8.4.115，`CUDA_VISIBLE_DEVICES=''`，4.1164 秒。包括 6 项 portable CPU 单元测试，以及一次实际安装 Detect/DFL 的 synthetic-raw exporter 集成检查（回执计 9 项检查）；合并字段 tests=15，不应误写为 15 次独立实验。
- 集成脚本实际调用 `export_dfl`，使用合成 raw 而非 checkpoint/backbone：确认 native conv 输入概率 hook、DFL 输出、FP32 和 native 解码闭合、9 个请求去重分布、T 同 S/R native 索引和本模态 GT 语义、state 不变/无梯度/CUDA 未初始化。
- 独立 reviewer 已执行 `LOCAL_CPU_TRUTH_attempt2.json` 的 8 项语义/身份检查，以及分析器 5 项核心与 2 项 wrapper 小真值。第一次独立人口检查误将非空 GT 帧数设为 32，失败源和回执保留；修正为完整 32 帧/31 个非空帧后通过。

## 源码检查结论

|检查|结论与证据|
|---|---|
|固定人口/身份|`run_dfl_probe.py` 在数据前向前核 checkpoint path/stat、80 个稳定 GT ID、历史 native anchor 及完整 first_batch_stream/identity exact；身份失败先退出。完整 32 帧包含一张空 GT 帧。|
|请求完整性|S/R 在历史 R 候选及各自 native anchor；T 覆盖这些相同索引、自身 native 及真实历史 selected T。缺项为 null，分布按 model/image/anchor 去重。独立原始名单预期 317 份：S83/R83/T151，详见 `EXPECTED_REQUESTS.json`。|
|真实 DFL 内容|`dfl_export.py` 直接从 raw boxes[B,64,A] reshape 为 [B,4,16,A] 并沿 bin softmax；未从 NMS 框/均值构造分布；完整 16-bin logits、FP32/native 概率、dtype、中心/stride/level、解码框及每角色未 clamp GT 距离均保留。|
|一次数据前向/零更新|本批 S/R/T 各一次 model hook 计数；其后只解码已有 raw，无新 NMS/匹配/loss/backward。三模型 state_dict 前后逐 tensor exact，无梯度，optimizer 空状态且 optimizer/EMA 更新为 0。普通 head 缓存可变已明示。|
|BN/AMP 与 setup|S train+BN frozen，R/T eval，实际 autocast 继承旧 actual_amp。新版仅在 setup 临时替换通用 check_amp，值须等于原已验收回执，finally 恢复，并有成功/异常恢复 CPU 真值；避免额外 YOLO26n 通用校验前向。该 setup 改动明确记录，不宣称 setup 与旧执行完全等价。|
|两种解码|FP32 期望对原 OEv1 `_decode_boxes` exact；actual native DFL 输出对原 `decode_bboxes` exact。两条路径分别闭合，不要求彼此逐位相等；AMP 舍入和 native 概率质量和不重新归一。|
|GT 读出与含义|分析器只在有限 `0≤d<15` 上计算未 clamp 相邻 bin CE，非法边为 null；这是诊断，未还原训练中的 14.99 clamp、分配/权重。熵 nat、方差 bin²，缺失/非法分母分开。不同 anchor/bin 不作直接 KL。|
|新旧执行边界|所有新记录使用同一个新 current_forward_id；旧 frame/GT ID 仅作稳定身份。未比较历史完整像素或 DFL，不宣称是历史同一次 forward 或 bitwise 重演。|
|资源与入口|七份 reviewed_source 快照用于 launcher 本地/远端逐字节比较；唯一原 global lease + screen、数据盘写入、单批自身实测，无额外 full/自动重试。动态资源准入仍由原 dispatcher 执行。|

已修正的关键源码缺口包括：T 同 native 索引漏项、全模型 state 检查和真实计数、额外通用 AMP 前向。早期发现按 `SOURCE_FINDINGS.json` 保留；本文件是对冻结版本的当前判断。

## 范围与后续验收

真实一次前向仍须通过全部运行时 guards，并记录实际 NVML/RSS 峰值。输出到齐后独立核对 80 GT / 317 分布、全部角色/缺项、FP64 对已保存 FP32 数值闭合、真实 native 输出回执及 CPU 派生结果。CPU 集成成功不证明 GPU AMP 的实际数值或资源结果，任何失败都保留，不能先写通过。

本探针最多证明分布内容及同一模型/anchor 的 GT 读出确已保存。它不能证明空间物理配准、分布可直接迁移、教师内容更可靠或 KD 可学效用；不能据小样本选阈值/λ 或授权训练。

`audited_input_hashes = not computed`：按任务明确边界未计算 hash。路径/大小/实际源内容/逐字节快照构成本次版本约束，不宣称 SHA-256 证据。
