# RGBIR 对象判别蒸馏首轮：独立审查初检

> **状态：设计预检完成，尚未收到新实现，不构成训练放行。** 审查日期 2026-09-06；reviewer 为独立 Codex agent `/root/rgbir_code_review`，按 `experiment-bridge` 的代码复审要求执行。本审查不运行 GPU、不修改训练实现。

## 范围与证据边界

依据 `07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md`、上一轮 `minimal_kd_validation.md`、本目录 `framework_notes.md` 及现有 paired loader / CMDistill trainer / evaluator。

本轮用户授权范围是 DroneVehicle 上 N/P 两个 seed42 探索实验，至多占用一张 GPU。该范围不等于上一轮建议的 N/P × 3 seeds 净收益门，也不允许据单 seed 作论文增益结论。新前景—环带相对证据表示、选择公式与超参应在训练前形成新协议及源码/配置副本回执；现有 probe 提供设计动机，不提供方法收益。工程级 AGENTS 禁止新写 hash/SHA，以源码、输入、环境副本和 tensor 直接等价核验为准。

## 训练前必须核查

1. **独立标注与学生不变性。** 现有 `PairedDetectionDataset._image_record` 复制学生 record 后换教师图像，`transform_paired_record` 又要求两侧 targets 相等，不能用于声称独立 IR GT 下的正确性。新 adapter 必须保留 RGB 原生 img / cls / bboxes / batch_idx，另存真实 IR 标注，并在同一几何参数下变换。随机裁剪可使两侧不同对象消失，不能靠数组位置一一对应。学生 tensor / target 与变换后 RNG 必须有实际等价测试。
2. **原生监督与单次缩放。** 当前 pinned 原生 detection loss 返回三项向量，trainer 求和。新实现应先取得原生标量，再加一次 `B * lambda * kd_mean`；向三项向量广播 KD 会放大三倍。weight0 的损失和学生梯度应与同 batch native 一致。
3. **相对证据区域。** 各模态在自身变换后的 GT 坐标中提取 ROI 与环带；环带排除该模态所有已标注对象，包括未配对及其它类别。空 ROI、空环带、图边裁剪与小目标格点不足要有冻结政策，不能让无效对象产生 NaN 或虚构知识。单对象先归一化，再按固定 RGB 候选数归一化 KD。
4. **质量选择的实际含义。** 教师与冻结 RGB reference 的比较应是训练 GT 支持下的任务质量，不能把未经校准的置信度差称为信息量。教师/reference、选择权重与目标全部 detach。reference 见过训练图，故它是训练代理；只能在独立开发对象上分析一般化。
5. **学生图与状态隔离。** teacher/reference 为 eval、requires_grad=False、no_grad forward；不进入学生 optimizer、EMA 与 checkpoint。绑定 criterion 的时机要避免 ModelEMA deepcopy 出教师副本。训练/评估数据角色固定，last/EMA 端点一致，不用 test 或 best epoch 选择方法。
6. **资源与运行证据。** 两任务从候选卡动态选择同一张物理 GPU 后锁定，串行执行；同时遵守工程级每卡最多 2 个本项目 CUDA PID、项目显存严格低于整卡 70%、实测峰值后至少 2 GiB 空闲，以及 aggregate RSS 上限。有限 IR 缓存、计入 workers RSS；长训练 screen/tmux；输出只写项目 runs/dataset 盘。技术失败保留 attempt，禁止框架静默 OOM 降 batch 改变 N/P 契约。每个实际 run 使用 `write_jstars_run_receipt.py`，身份为 PROTOCOL-ADAPTED，并标明原创 exploratory pilot 名称。

## 非阻塞研究局限

- N/P 首轮只检验本轮配对干预对匹配 native 的单 seed 方向，不证明配对特异性、选择价值或优于同模态蒸馏；后续 S/M/R 与多 seed 门保持待执行。
- 环带可能含漏标目标；logmeanexp 相对证据与冻结 reference 选择本身不保证可学性或跨模态校准。
- 老 evaluator CLI 未识别新 arm 时会写成 HNEWA-INSPIRED；应由新 wrapper 写准确方法身份。

## 后续

新协议、loader、loss 与 trainer 到齐后逐行复审，并在 `EXPERIMENT_CODE_REVIEW.md` 记录阻塞项、修复与实际 canary 证据。此初检文件保留，不覆盖。
