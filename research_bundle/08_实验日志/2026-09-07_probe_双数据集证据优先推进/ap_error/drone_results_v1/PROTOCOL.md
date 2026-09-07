# 完整检测 AP 错误诊断协议（2026-09-07，结果运行前冻结）

本实验只用已有完整 dev post-NMS 预测做 CPU 描述性分析，不设计或启动新损失，不改变 C1/L1 准入，也不读取 test。

## 输入与范围

- DroneVehicle：`../../2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/{N,C0}_s{0,42,123}_attempt1/predictions/objects.jsonl.gz`，六份每份 1469 图、22462 GT，原补评对历史五汇总指标 exact。
- 使用逐图保存的全部 GT，显式登记空预测、空 GT 图；逐 image 路径比对六份 roster 与 GT 完全一致，保留 checkpoint、原评估 receipt 和输入 SHA256。
- 保存的是同一实际 rect/letterbox 画布中的绝对 xyxy，实际画布 544×672；GT 与预测直接同坐标变换成 TIDE xywh，禁止把名义 imgsz640 当成坐标范围。不另缩放或裁剪。
- 类别 0 car / 1 freight car / 2 truck / 3 bus / 4 van。低阈值 conf=.001，max_det=300；完整是该固定后处理协议的全部输出，并非无阈值或 pre-NMS 全候选。
- LLVIP 现有 `2026-09-06_probe_RGBIR数据特性与可迁移知识/llvip_full/prediction_records.json` 为 200 图抽样，不能当完整 dev。上一阶段 1024train+200dev 是对象关联 pre-NMS 子集，不能计算正式 AP。若根任务取得完整低阈值逐图输出，可按同样协议接入并明确模型 recipe 与单 seed 限制。

## 冻结的估计量

1. 原 pinned evaluator 的 AP50/AP75/mAP50–95只作原始参考。官方 TIDE 分别在正匹配 IoU=.50 和 .75、背景 IoU=.10 上运行 main error 六类（Cls/Loc/Both/Dupe/Bkg/Miss）和 special FP/FN oracle，保持其官方分数优先匹配、101点均值积分及独立 oracle 语义。采用 max_det=300，防止官方默认100额外截断。TIDE AP不冒充 pinned AP；实现匹配、坐标clip/scale和积分可能不同。
2. 报六端点逐值、C0−N逐seed差和 mean±sample SD（ddof=1）；只进行预设方向描述，无显著性或新方法增益结论。独立 oracle dAP 不可相加成总可达收益，不能表述为 KD 能达到的收益。
3. 排序辅助：按 TIDE 对每类别及整体 pooled FP/TP 匹配输出固定分数区间（[.001,.01),[.01,.05),[.05,.25),[.25,.5),[.5,1]）计数；高分 FP 的错误类型；各类 TP/FP 分数分位数。不调阈值，不把 pooled precision 当 macro AP。记录同分数 ties。
4. 标注范围限制：GT 为保存的 RGB 标签，无 crowd/ignore 字段，不虚构 ignore 区；未标注对象可能进入背景错误。低分检测数量本身不衡量 AP 损伤。

## 验证与复核入口

- 官方作者源码固定 commit，保留源与 license/hash；隔离 CPU 依赖不安装到正式训练环境。
- 已知真值小例覆盖正确、类别、定位、同时错、背景、重复、漏检（含空预测图与纯背景图），无GT类别/分数ties和xyxy→xywh检查；预期标签先冻结。
- 独立 NumPy score-greedy AP101 实现核对 TIDE AP50/AP75，保留逐类 GT 分母、TP/FP/FN及类别AP；额外验证单调分数变换保持 AP。
- 保留脚本、配置、机器可读报告与原始来源，可由根或独立审阅者重跑。未有 accepted analyzer 前不升级论文 claim。

## 一手来源

- 作者仓库：https://github.com/dbolya/tide
- 论文：https://arxiv.org/abs/2008.08115
- Data xywh 与 max_dets：https://github.com/dbolya/tide/blob/master/tidecv/data.py
- 匹配和错误/独立 oracle：https://github.com/dbolya/tide/blob/master/tidecv/quantify.py
- AP积分：https://github.com/dbolya/tide/blob/master/tidecv/ap.py
