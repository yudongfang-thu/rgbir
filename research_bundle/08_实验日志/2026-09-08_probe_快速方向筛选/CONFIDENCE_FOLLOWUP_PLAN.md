# LLVIP C0 置信度：独立的两臂短训（2026-09-08 11:49 冻结）

**在本数据集 C0 新 AP 尚未产生时，新增 N/C0 两臂；不修改已结束 L2 或在跑 Drone 矩阵。**

## 为什么值得这一个小对照

LLVIP 初始 visible/IR dev mAP 为 32.878/48.853；单类使“跨类别混淆”不适用，但前景相对背景置信度仍可传递。刚完成的对象定位短训覆盖稀疏、梯度较弱且与 GT 目标高度相似，只约束该定位版本，未检验分类置信度。

本次直接复用原 OEv1 C0；不把 C0 改称新算子，不改选择门控、区域或损失。每对象前景相对背景类别证据、原质量选择、P3/P4、温度2、原 SmoothL1、λ=0.1。完整 native 损失保留，N 沿相同辅助路径但 λ=0。

## 固定协议

- scope `LLVIP_CONFIDENCE_FT3`，endpoint `LLVIP_CONFIDENCE_FT3_LAST_EMA`。
- 复用已冻结 LLVIP 2048 自然分层 train 子集；完整 dev2406/7879，test不使用。
- 两臂相同原 visible42 last/EMA 初始化，原 IR42 教师、visible42参考冻结。
- seed42，E3/192batch，640/B32/nbs64/workers4，SGD lr0=0.0001/lrf=1，warmup0，BN running statistics冻结、affine可学，fresh optimizer/EMA，原增强和loader随机流。
- C0 λ固定0.1，不扫系数、不依据本轮 AP 改步数。这里不套用新C1的梯度校准，因为算子和系数均为原C0定义。
- N/C0各做24有效更新canary，完整初始化和BN实际检查，前30batch顺序、双标签、增强结果配对；按实测峰值经原全局lease运行。
- 两臂固定last/EMA做原生FP32完整dev独立评估，不用best；本轮新N作为直接对照，先前L2矩阵N只作额外兼容参照。

## 排程和解释

源 `newentry/confidence_release/`，94发布独立 `confidence_release_v1` 与 `confidence_attempt1`。协调器等待主 `screen_attempt1` 完成后再执行，所有GPU继续共用原dispatcher，不增加物理卡或第二资源池。

仅新增两次分钟级短训练及必需canary/评估。结果报告 C0−N、AP50/AP75/召回，并与初始模型做有身份绑定的绝对参照。无论正负，均不凭一个seed/短训自动新增E200；不把成熟模型短训排序当作从头长训排序的已验证代理。
