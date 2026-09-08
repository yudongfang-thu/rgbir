# 同一真实 forward 的全 GT 与实际选择门链

核心源码冻结后本地 CPU 6/6 通过。这里只准备导出，未运行 GPU、模型推理或读取新 AP；真实训练首批结果由 root 的独立 driver 产生。

`export_selection(batch, student_raw, teacher_raw, reference_raw, *, api, evidence_config, strides, selection_seed, identity_contract, batch_index=1)` 返回 `records`、`ir_records`、`frames`、`selector_counts` 及合同元数据。driver 把 `records` 写为 `objects.jsonl`，没有重复的 `objects` 别名。三个 raw 必须来自同一实际 batch 的一次 S/T/R forward；本函数只有无梯度的原函数计算，不再推理模型。

## 身份与分母

- `records` 遍历全部增强后仍存活的 RGB GT；`ir_records` 遍历全部增强后 IR GT，包含未配对对象；`frames` 保留全部真实 batch 图像，包括空 GT 图。
- `identity_contract` 要求 `STABLE_GT_IDENTITY_VERIFIED`、`frame_id`、按实际 global GT row 排列的 `rgb_rows` 与 `ir_rows`。每行保留 image index、canonical source image path、原 native verified-label 表行号、stable ID；原行号不是 XML/text 行号猜测。原始未增强标签及真实过滤事件由 mapping trace 合同另行保留。
- `frame_id` 是 forward 标识加 image index。每个 RGB record 含自己的增强后 pixel `rgb_gt_xyxy`、global/stable RGB ID；配对对象另含自己的 `ir_gt_xyxy`、global/stable IR ID 及原 selector 配对 IoU。
- 全 GT、matched、base、eligible、selected 分母分开。没有用旧 200 图 dev 表连接这批 train GT；也没有把这个首批外推全训练集或 AP oracle。

## 对象状态

S/R 对 RGB own GT、T 对 IR own GT，先复用 baseline 的 GT 辅助一对一空间分配：最高类别置信度至少 .05、空间 IoU 至少 .1、分配不使用预测类别。最终正确要求置信度至少 .25、类别正确、own-GT IoU 至少 .5。互斥优先级固定为 `no_candidate → low_confidence → class_and_localization → class_only → localization_only → correct`。

`states.T` 是 IR own-GT 主状态；`states.T_to_RGB` 用同一已分配 teacher box 直接对 RGB GT 测量，明确没有几何映射。二者不能互换。T 另含 `iou_to_rgb_gt`。未配对 RGB 的 T/T_to_RGB 为 null，不把未知计为错误。

这是一对一 raw-anchor 对象诊断，不是 post-NMS 检测 AP。原 C 选择的 reference/teacher 门是 dense any-candidate 判定，可以与该一对一状态不同；代码与聚合均不强迫二者相等。

## 原选择门

`gates` 保留 `matched`、P3/P4 `valid_levels`、`region_valid`、`reference_candidate`、`teacher_correct_own`、`quality_q`、`quality_positive`、`base`、`eligible`、`selected`，及每层两模态 fg/bg 数、各 raw valid、common valid、S/T/R 标量 evidence。matched/base row 从 0 起；selected/eligible rank 从 1 起。

原并列条件是 `base = region_valid & reference_candidate`，`eligible = base & teacher_correct_own & (q > 0)`。原 `q = max(softplus(-R_evidence) - softplus(-T_evidence), 0)`。配对选择在真实整批上进行，rho=.5、取 ceil(rho × eligible)，q 降序，精确 tie 保持原 matched 顺序；没有逐图重选。未 matched 对象除 `matched=false` 外，所有后继门字段为 null。

首先调用 selected-only API 的 `full_diagnostics=True` 得到真实完整 selector；薄路径与完整诊断的 IDs、mask、q、选择顺序及原 C0 stats 必须逐项 exact。再用原 scalar helpers 公开 adapter 丢弃的 nonbase matched 字段，要求重建 matched/base 身份、valid mask、base q、eligible、selected、选择顺序及全部原计数 exact。重建值不代替真实 selector 决策。

`selector_counts` 固定包括 rgb/teacher GT、common、valid-region、reference-candidate、base、teacher-correct-base、eligible、selected、normalizer。reference-candidate 原计数覆盖全部 matched，不能被误读为先通过 region 后的累计数量。normalizer 为 max(1, base_count)。按门先后汇总的首次流失仅是记账顺序，不是门的因果贡献。

## CPU 复核

入口可移植，无机器路径常量：

```text
<python> test_export_cpu.py --reference-dir <pinned-release> --candidate-source <selected_only_v1.py> --output <new-receipt.json>
```

本地 `EXPORT_CPU_attempt1.json`：torch 1.8 CPU，6/6 PASS。覆盖全部状态顺序、own/cross GT 区别、原 P3/P4 selector、多图/空图/交错 global GT 行、未配对 GT、FP16 薄路径和 FP32 fallback、nonbase/q=0、any-candidate 与一对一差异、错身份拒绝和 raw 不变。小 raw 使用真实 P3/P4/P5 网格 8/4/2 与 64px 输入；不构造模型、不加载权重。实际 pinned 环境复跑及真实首批完成合同仍由 root 另行验收。
