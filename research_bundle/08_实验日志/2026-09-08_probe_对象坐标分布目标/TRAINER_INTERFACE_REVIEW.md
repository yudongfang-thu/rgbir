# L3 短筛训练接口有界核对

**结论：可复用已执行 direction 短筛的训练骨架，以新 criterion 单分支接入 N / L3_DFL / L3_GT；不能只修改旧 YAML 的 arm，也不能把旧 L2 最终教师 anchor 与温度设置当成新 DFL 协议。当前为静态接口审阅，尚未准入或启动 L3 训练。**

日期：2026-09-08。审阅者：`/root/baseline_feature_analysis`。本次只读本地既有源码、配置及来源回执，没有 GPU、前向、新 AP、SSH 或新 hash；不接管另一路 C1 提速部署。未重新审计整个历史训练实现，也未将旧结果当作新接口验收。

## 来源与实际执行身份

读取原条目 `2026-09-08_probe_快速方向筛选/newentry/release/`，该条目的 `deployment_attempt1.json` 指向 94：

```
/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/release_v1
```

`localization_box_v2.py`、`direction_criterion.py`、`calibrate_direction.py` 已有前阶段对实际 94 release_v1 的逐字节回执；本次将当前本地文件再次与那三份远端来源副本逐字节比较，全部相同。其余 direction 文件此次作本地静态接口检查，不声称重新验证当前远端全部文件。13 份小源码/配置副本及逐字节复制记录在 [runtime_inputs/COPY_RECEIPT.json](runtime_inputs/COPY_RECEIPT.json)，原三份身份回执保留为 [PRIOR_SOURCE_IDENTITY.json](runtime_inputs/PRIOR_SOURCE_IDENTITY.json)。下文行号均针对这些原样副本。

训练命令的 `--reference-dir` 为原已执行的：

```
/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5
```

其 `runtime.py:5–24` 绑定本目录 `task_conditional_reference/legacy_oev1`、原 `train_object_evidence` 和 tracked 双标签 loader，并检查模块来源。direction 并非在磁盘上改写旧 IndependentCriterion：`train_direction.py:12–31` 克隆原 `legacy.build_trainer` 的 globals，局部绑定新 criterion、tracked dataset、冻结模型加载器。原 selected-only 库仍来自早前已接受路径且复制前比较 approved 字节（18–23 行）；它对 LLVIP L2 只是构建时依赖，不使 L2 走 C1 的池化损失，更不自动采用当前另一路待切换的 C1 快版。

## 最小新分支接入

| 位置 | 可保留的部分 | 新 L3 必须显式替换的部分 |
|---|---|---|
| [direction_criterion.py](runtime_inputs/direction_criterion.py):7–28 | `make_type(api)` 继承原 criterion 的初始化；传入 student/teacher/reference raw、batch、strides | 添加唯一 `compute_l3(..., variant)`；返回未乘剂量的标量与 JSON stats。原代码仅识别 L2 前缀，改 cfg 名字不会调用 L3 |
| 同文件:30–63 | 完整 native loss、S 原训练前向；T/R eval + no_grad；`native.sum() + B * kd_coefficient * kd`；共享参数梯度检查 | 新 N 的辅助 request 应与 L3 分支一致、权重为字面量 0；L3_GT 和 L3_DFL 的 mask/学习 anchor/分母共用，只有目标切换。空 selected 返回连接学生 DFL logits 的零标量 |
| [train_direction.py](runtime_inputs/train_direction.py):16–31、50–166 | 私有 builder、完整状态 warm-start、fresh optimizer/EMA、BN 冻结、真实更新计数、输入权重 stat 不变 | 新 wrapper 的 config/scope/endpoint/status/config 文件名必须一致；加入新算子源码及实际目标/温度/支撑拒绝计数，不能仍输出 L2 科学身份 |
| [calibrate_direction.py](runtime_inputs/calibrate_direction.py):18、53–88 | 固定 8 批，逐批恢复完整初始状态，真实共享参数梯度读数，没有 optimizer/EMA 更新 | arm 列表、目标剂量、系数约束及 GT 共用规则由本次协议固定。旧 L2 λ 和 min(1,median) 不能被视为 DFL 的已校准剂量 |
| [direction_common.py](runtime_inputs/direction_common.py):25–36；[run_direction_queue.py](runtime_inputs/run_direction_queue.py):18–29、96–143 | 输出目录不得覆盖、原 global resource_dispatch、配置冻结和逐臂 canary | arm/scope 白名单、系数识别（原来仅 `startswith('L2')`）、新校准与终态字段须统一改到独立新目录 |
| [evaluate_direction.py](runtime_inputs/evaluate_direction.py):14–18、49–90、244–257 | 同一原生 evaluator、完整 dev roster、训练配置 bytes/权重 stat 绑定 | 允许新三臂及新 scope/endpoint，读取新训练 completion/config 名称；原旧三臂身份不能直接冒充新结果 |

现有入口足以局部绑定，不需要迁移新快版训练循环。建议新文件使用唯一模块名或显式路径 import，保留 `train_direction.py:65–66` 的 pinned 来源断言；`runtime.py` 会前插 legacy/reference 搜索路径，不依赖含糊的同名 import。仅复用旧 helper 的接口不等于新的损失、选择与梯度已验收。

## 旧 L2 的真实选择、目标与分母

[localization_box_v2.py](runtime_inputs/localization_box_v2.py):15–19 的 `FIXED` 是真实执行参数；并未读取 YAML 的 `localization` 字典来覆盖它。

1. 按双模态各自增强 GT 配对：初次同类 match IoU≥.5，再要求 pair IoU≥.8。R/T 候选限 P3/P4；所有本图 GT 都参与 native unique-owner 排除（98–151 行）。
2. R anchor 先满足 own RGB GT 四边距离 `0≤d≤14.99`、native owner、有效框、confidence≥.05、IoU≥.1；按 **confidence→IoU→最小 anchor ID** 选择（143–158 行）。到此便计入 base，早于后续可靠性/教师质量门。
3. R 同类且 confidence≥.25，R IoU<.70 才进入教师搜索。T 在自己的 IR GT 上同类、confidence≥.25、IoU≥.5，按 **own GT IoU→confidence→最小 ID** 选教师 anchor；随后将教师框按双 GT 对象坐标仿射，要求 RGB mapped IoU≥.60 且严格领先 R>.05（159–188 行）。这不是教师在 R 同索引的原分布选择。
4. 返回 `ids[:, :] = (batch_index, reference_anchor, teacher_anchor, rgb_global_gt_row, ir_global_gt_row)`，以及 RGB GT、mapped teacher box、stats、centers、stride_values（194–201 行）。`normalizer=max(1,base_count)`，不改成 selected 数。
5. 学生在 **R anchor** 读原 `raw['boxes']`，FP32 softmax 后取 DFL 期望再解码 xyxy；L2-box 将学生/教师框除以对应 RGB GT 宽高归一为对象坐标，L2-GT 目标为 `[0,0,1,1]`。每对象四边 SmoothL1 β=.1 取 mean，再对 selected 求 sum / base（204–230 行）。两臂同选择、同分母，教师/GT目标 detached；native loss 未替换。

**温度区别：**旧 L2 的 `decode_selected` 直接 `softmax(-1)`，实际温度为 1；YAML `localization.temperature: 2.0`、`evidence.temperature: 2.0` 没有进入这项 L2 坐标损失。新完整 DFL 目标须显式定义温度在哪里应用、运输前后顺序、是否带温度平方因子，不能继承一个未生效的 YAML 数字。

**对新接口的确定差异：**本条目的预列主输入是 T 在历史 R 同索引上的分布，旧 L2 `_select` 则返回 teacher-own 最优 anchor，且只检查 R 的 GT 距离支撑，没有检查教师完整正概率质量运输到学生 bins 的支撑。因此可以复用 `_layout`、双 GT 读取、匹配和 coarse-R 构造等 helper；不能原封不动把旧 `_select` 最终结果称为本次同索引分布协议。新的质量越界拒绝发生于哪个阶段、是否保留原 coarse base 分母，须按 root 冻结协议写入并让 GT/DFL 两臂同掩码，不能在看结果后改变。

## 校准与 canary 的真实含义

旧 LLVIP 校准对每批计算 `∇(native.sum())` 和 `∇(B*L2)`；参数集合是 [pinned_gradient_observation.py](runtime_inputs/pinned_gradient_observation.py):16–26 中检测头 `head.f[:2]` 对应的两个 P3/P4 源模块可训练参数，**不是全模型参数，也不是局部输出坐标梯度**。8 批逐批重置学生全部参数/缓冲区、冻结 BN running stats；至少 4/8 finite nonzero 比值才有效。旧 L2-box 用 `.1*native_norm / unit_B_KD_norm` 的 median 并截到 1，L2-GT共用它（calibrator 57–85 行）。这些是旧 L2 的实际规则，不预先承诺新 DFL 沿用。

训练 N 仍执行同样 T/R 前向和辅助 L2-box 计算，外层 λ=0，并检查 `total == native.sum()`（criterion 38–55 行）；不能把旧 N 描述为无教师计算的吞吐基线。新 N 同样可作为零权重学习控制，不复用旧 N 的 AP 来充当新 scope 匹配臂。

旧 canary 用新解释器/fresh run，在真实 optimizer 成功更新数达到 24 后请求停止；结束检查至少 24 更新、非 N 有 selected 且共享 KD 梯度 finite/nonzero，BN 缓冲区未变、完整初始化相同、fresh EMA/optimizer、输入权重 stat 不变（trainer 90–150 行）。它仍配置 3 epoch，因此有限最多 192 次 batch 访问后失败，不会无限等待 AMP 跳步。它不是仅运行 24 batch；停止计数与样本数必须读实际 receipt。

原 queue 比较三臂 canary **首 30 批**源文件、增强后 RGB/IR 标签及 batch ID，并将正式训练首 30 批与各自 canary exact 比较（queue 70–93、291–314 行）；这份流回执不等于全部像素 tensor 逐字节比较。旧 `max_steps=None` 路径没有触发 legacy 的 `first_batch.pt` 分支（[pinned_train_object_evidence.py](runtime_inputs/pinned_train_object_evidence.py):172–175）。不要把其他 performance canary 的像素证明移植成这次自动已有的证明。

原 global lease 可复用，资源应以新算子 canary 实测决定。旧 queue 的初始上限 8192MiB/32768MiB，training reservation 使用实测 NVML/allocated/reserved 最大值加 max(256MiB,5%)、RSS 加 max(2048MiB,5%) 并取整（49–68 行）；这是代码上限和预约公式，不是新 DFL 的实测峰值。旧 canary 或另一路 C1 的资源不能当新方法已实测。

## 数据与完整评估入口

可保持 seed42、自然分层 subset seed20260908、2048 图/B32/640/4 workers/AMP、SGD lr=1e−4、lrf=1、warmup=0、BN running stats 冻结但 affine 可训练、fresh optimizer/EMA、3 epoch。**每轮是 64 个 batch，合计 192 个 batch；nbs=64 和 AMP 下不等于 64/192 次成功 optimizer update。**原 trainer 从完整 `ema or model` 权重逐 tensor 验证 warm-start，原生训练内部评估被禁用，固定 last/EMA 独立评估。

LLVIP 模型与数据原路径来自 [llvip_N_s42_FT3.yaml](runtime_inputs/llvip_N_s42_FT3.yaml)：

```
S/R: .../runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt
T:   .../runs/rgbt_p3_causal_v1/formal_native/llvip/infrared_seed42_native_b32a2/weights/last.pt
subset: .../artifacts/rgbir_direction_screen_20260908/subset_llvip_v1/
        data_visible.yaml / data_infrared.yaml / visible_to_infrared_train.json
full visible: .../artifacts/rgbt_p3_causal_v1/prepared/llvip/visible.data.yaml
```

省略前缀均为 `/mnt/dataset/yudongfang/projects/RGBT_campaign`。auxiliary_data_identity 保留旧完整 RGB/IR YAML 用于冻结模型身份，训练 paths 指向 subset；不能混换这两层。

可沿用新 scope 适配后的命令形状：

```
python <new_calibrator> --reference-dir <release_gpu5> --config <new_N_template> --output <new_calibration>
python <new_trainer> --reference-dir <release_gpu5> --config <frozen_effective_cfg> --output <new_canary> --canary
python <new_trainer> --reference-dir <release_gpu5> --config <same_cfg> --output <new_run> --canary-receipt <new_canary/canary.json>
python <new_evaluator> --reference-dir <release_gpu5> --config <same_cfg> --checkpoint <new_run/weights/last.pt> --output <new_eval> --native-config <llvip_native_evaluation.yaml>
```

LLVIP `--native-config` 是[完整 projection cfg](runtime_inputs/llvip_native_evaluation.yaml)，不是直接传 data YAML；不传 Drone-only `--native-profile-binding`。evaluator 将 subset YAML 的 val roster 逐项与原完整 roster 相等，要求 LLVIP 2406 图/7879 GT/1 类；Drone 若保留通用接口则为 1469/22462/5 类并需要既有 binding。原生 FP32 `quantize=None`、conf=.001、NMS IoU=.7、max_det300、rect、无增强，按 `metric_units=fraction_0_to_1` 输出；配置/完成回执/last.pt stat 不闭合即拒绝。新三臂结果需等各自新 completion 后才能汇总。

本次未发现必须弃用 direction runtime 的接口问题。实际需要实现的是新目标/选择分支及其一组一致的配置、校准、canary、queue 和 evaluator 身份适配；不需借本任务接管或迁移未在这条短筛链路验收的提速版本。
