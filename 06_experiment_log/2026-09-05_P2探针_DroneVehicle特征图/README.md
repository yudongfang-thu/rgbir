> **⚠️ 勘误（2026-09-05 晚，依据全项目复盘审计 §5.1/§5.2）**：
> ① S_native 实为 **VEDAI 模型**（checkpoint 身份错误），"蒸馏拉力 +0.32"混杂数据集/配方/任务身份，**结论作废**；
> ② `linear_cka` 未中心化，非标准 CKA，"P3 架构饱和"推断不成立；
> ③ shuffled 为单一固定反转 null；same-input 是教师 OOD 输入；FFT 比值归一化不可比——四项指标均需按审计建议重设计后方可复用。
> 本条目保留为过程记录；重跑需换用正确 native checkpoint 并修 CKA/FFT 实现。

> 实验日志条目 #001。目录规范与模板见 [08_实验日志/README.md](../README.md)。
> 服务器产物：`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/p2_feature_probe_20260905/`（本目录存脚本与 summary.json 副本）。

# 特征图探针报告 · DroneVehicle（P2，2026-09-05）

> 94/GPU1 只读推理，294 对 val 图像，640px。教师 = `infrared_seed42_native_b32a2`（IR，y11）；
> 学生对照 = `native_rgb_s42_b64_e200`（RGB，未蒸馏）与 `adapted_v2 dronevehicle_seed42`（RGB，已蒸馏）。
> 条件：paired `T(IR_i)↔S(RGB_i)` / shuffled `T(IR_i)↔S(RGB_perm(i))` / same-input `T(RGB_i)↔S(RGB_i)`。
> 产物：`94:RGBT_campaign/artifacts/p2_feature_probe_20260905/{summary.json,per_image.csv,probe.log}`；脚本 `p2_feature_probe_20260905.py`。

## 主表（294 图均值）

| 层 | 学生 | CKA paired | CKA shuffled | CKA same-in | 显著性corr paired | 显著性corr shuffled | FFT 高/低散度比 |
|---|---|---|---|---|---|---|---|
| P3 | native | 0.9892 | 0.9878 | 0.9906 | 0.262 | 0.133 | 0.645 |
| P3 | **dist** | 0.9974 | 0.9948 | 0.9982 | **0.716** | 0.338 | 0.530 |
| P4 | native | 0.9153 | 0.9037 | 0.9308 | 0.359 | 0.207 | 0.870 |
| P4 | **dist** | 0.9764 | 0.9548 | 0.9844 | **0.704** | 0.252 | 0.936 |
| P5 | native | 0.5601 | 0.5534 | 0.5813 | 0.297 | 0.286 | 0.784 |
| P5 | **dist** | 0.8833 | 0.7288 | 0.8685 | **0.721** | 0.343 | 1.025 |

**蒸馏拉力**（CKA_dist − CKA_native，paired）：P3 +0.008（饱和）｜P4 +0.061｜**P5 +0.323**。

## 五条解读

1. **模态差距集中在 P4/P5，P3 被架构相似性饱和**（native CKA 0.99）。在 P3 做特征蒸馏没有空间——未来的定位蒸馏应瞄准 P4/P5 或直接蒸馏头内分布（DFL），这与 YOLOv11-RGBT"P3 中层融合即可、深层仍有油水"的观察互补。
2. **蒸馏确实把学生拉向教师**：P5 CKA 0.56→0.88。且蒸馏学生的教师相似度变得**依赖配对**（P5 paired−shuffled：native +0.007 → dist +0.155）——KD 注入的是"配对特异"信息，与 HNEWA 线行为学差距（h1 vs h2）互证。
3. **显著性图相关是最灵敏的配对信号度量**：所有层 paired/shuffled 都是 ~2 倍关系，蒸馏把它整体翻倍（P3 0.26→0.72）。CKA 在浅层饱和，**今后 probe 以显著性相关为主指标、CKA 为辅助**。
4. **FFT 散度比与 FreqKD 相反**（0.53–1.03 vs 其 2.4×）：航拍 RGB-IR 的师生差异更多在**低频带**（场景布局/尺度语境），不同于行人 RGB 预训练设定的高频主导。**频段解耦策略不能跨数据集照搬**——先测再定权重。
5. **same-input 与 paired 的差很小**（native P5：0.581 vs 0.560）——native 深层大差距主要来自**架构/训练轨迹**而非输入模态本身；说明"换更强教师架构"与"跨模态蒸馏"是两个独立的增益来源，论文里要分开归因。

## 局限与下一步

- 单 seed checkpoint（s42）；LLVIP/VEDAI 探针缺 y11 对照模型（VEDAI 仅有 IR y11 教师，v5 模型与 8.4.115 不兼容）——待补 y11 基线后同款跑。
- 显著性相关按"配对 vs 打乱"检验了空间对应，但未按类别/昼夜分桶——下一步并入。
- 本探针只回答"特征是否被拉向教师"，不回答"拉向教师是否划算"（任务增益）——后者仍需四臂训练归因。

## 对方法预注册的直接输入

- 蒸馏层位：**P4/P5（深层）为主，P3 免做特征项**；定位分布（DFL）蒸馏不受此限（头内分布另算）。
- 教师路由信号候选：显著性图 paired 相关（本探针指标）可在训练中在线计算，作为"教师对该样本可信度"的门控量。
- 报告口径：CKA 与显著性相关并列，注明 CKA 浅层饱和问题。
