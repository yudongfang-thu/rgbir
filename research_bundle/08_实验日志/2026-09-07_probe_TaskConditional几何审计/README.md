# Task-Conditional 独立几何审计（2026-09-07）

> 已实现可执行几何合同并冻结 300 对 train-only roster；45 对实际无框初筛，LLVIP050001 的 6 个静态点获独立复核，仅接纳单图局部区域；第 7 点扩域未接纳，未扩展整个 prefix。另精查 3 帧但未增加 accepted 图像；14 项 CPU 测试通过。

## 目的

独立检查 RGB–IR 近似恒等网格的适用图像、区域和误差范围，为定位 DFL 蒸馏提供拒绝未覆盖区域的几何合同。标签框一致性不作为独立对应真值。

## 设置

- 94 服务器，CPU 数据读取与审计，不使用 GPU。
- DroneVehicle：按 governance/rgb_train_source_groups.tsv 的训练来源 folder，首、中、末三帧，预期 86 组、258 对。
- LLVIP：既有 fit split 的 14 个 prefix，首、中、末三帧，预期 42 对。
- 每对至少 6 个可辨认物理结构对应点，覆盖至少三个象限；固定 20% 独立复核。
- P95 误差阈值 min(stride/4, 0.1×目标短边)，最大误差 stride/2；增强后按尺度传播。未测量、未覆盖及无可信对应均拒绝。

## 结果

| 项目 | 实际完成量 / 状态 |
|---|---|
| Drone roster | 86 组、258 对；实际 TSV 位于 `data/processed/dronevehicle/yolo/hbb_v1/`，不是 governance 根目录 |
| LLVIP roster | 14 个 fit prefix、42 对；固定独立复核 9 对 |
| 独立复核冻结抽样 | Drone 52 / 258，LLVIP 9 / 42；均为 ceil(20%)、均匀分散固定索引 |
| 真实图像初筛 | Drone 3 对 + LLVIP 42 对 = 45 对；其余 Drone 255 对未观察 |
| 第一批主标 | Drone17930 / LLVIP020001 各 7 个近似点，root 独立复核未接纳为精确对应；约 10px 粗标残差不作为真值 |
| 第二批主标 | LLVIP050001：原始 1280×1024 图中的 3 个井盖中心 + 3 个路桩顶中心，root 已独立复核接纳近似对应，见 `INDEPENDENT_REVIEW_BATCH2.md` |
| 第三批新增点 | 050001 最右路桩顶部存在模态边界歧义，root 未接纳第 7 点，不生成扩域 accepted v2；旧点/容差不变 |
| 第四批有界精查 | 030237 / 030474 各仅 2 个可信度较高井盖中心；060258 给出 6 个静态候选，但墙柱/栅栏/杆根存在轮廓歧义，未接纳 |
| 几何 CPU 验证 | 14 项，本地与 94 实际 Python 环境均通过；新项验证 canonical path 别名 |
| 正式几何合同 | `geometry_contract_accepted_exact_v1.json` 仅有 050001 一图条目，区域/尺度/短边限制仍逐对象执行；旧 unverified 文件保留历史身份 |

第二批静态结构的观察和第一批形成了有意义的区别：不是所有图都像车窗/近景枝叶那样难以核定物理点。该区别仅用于优先安排审计，未形成数据集配准率或定位增益结论。已接纳 6 点的原图 P95=3.4947px、max=3.6056px，加 3px 人工不确定度后 640 输入 P95 约 3.2474px：P3 无对象可通过；P4 仍需短边≥32.474px、全框落入凸包以及增强尺度条件。

## 结论

当前只有单图区域的 ACCEPTED_EMPIRICAL_GEOMETRY，不能视为整个 LLVIP 的几何验证。D2 可输出未验证机会诊断，正式定位训练必须读取覆盖合同。数据配对、复制 GT 框以及相似图像布局都不能独立支持直接 DFL 对齐。

已确认旧 paired loader 仅重放 RNG，未保存原图到输入矩阵；root 在新模块实现旁路矩阵追踪，不修改原 OEv1 loader。

## 接口与验收

`GeometryContract.load(path).object_mask(image_paths, rgb_xyxy, batch_idx, augmentation_metadata, stride)` 返回每对象 bool；`rgb_xyxy` 必须是增强输入像素 xyxy。`stride` 可为标量或逐对象向量，torch 输入保持 device；`object_decisions` 同时返回拒绝原因。

`augmentation_metadata` 为逐图 `pair_info`：至少有 `rgb_matrix`、`ir_matrix`（3×3 原图→输入）和 `original_shape=[H,W]`。模态增强矩阵必须相同且为非奇异仿射。合同仅接受实际源图像 keys；对象四角逆变换后需同时位于两模态对应点凸包内，误差与人工不确定度按矩阵最大奇异值传播，再检查 stride/短边阈值。

已测试：零证据拒绝、未见图像拒绝、全框区域覆盖、缺失/不同增强、缩放平移翻转、误差尺度传播、小对象阈值、标注不确定度、独立复核缺失、少点拒绝、真实残差统计、空对象。

## 产物路径

- 本目录：`frozen_roster.json`、`roster_receipt.json`、两批真实主标与 summary、无框/点标图、`llvip_overviews/` 七页与 42 对独立图、`TRIAGE_45_PAIRS.md`、脚本副本。
- 后续批次：batch3/4 主标及 summary、`INDEPENDENT_REVIEW_BATCH3.md`；各批 summary 的 viewed_pairs 是该批主标范围，不是累计初筛量，累计仍为 45 对。
- D2 单帧名单：`d2_accepted_exact_v1_images.txt` 和 `d2_accepted_exact_v1_roster.json`，050001 的 RGB/IR 与 label 路径存在性已核实；标签内容未用于几何标记。
- 源码：03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/geometry_contract.py、geometry_audit_tools.py。
- 94 预定产物：/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/geometry/。

## 局限与下一步

本次几何首轮按 root 调度收口，不继续挑更多 frame。050001 只覆盖一个训练图的局部凸包，不能直接为全部 9,619 个 fit 图授权。050 中/末原图已进一步精查，但道路点被人车遮挡，无法完成组范围六点条件，详见 `LLVIP05_TEMPORAL_SCOPE.md`。额外 030237 / 030474 的路面白线在 IR 不可辨认，060258 的多个固定物体轮廓未达到可自动接纳的精度；详见 batch4 的逐点可信度备注。其余尚未标记的样本是未验证，不是配准失败。
