# OS-SSL 红外域迁移（DroneVehicle，2026-09-06）

> **21:46最新核验（2026-09-06）**：新增paired123 E200完成；与shuffled123同口径CSV差+0.669mAP/+0.495AP50。3/9完成、shuffled0@131；尚无独立last评估，初始化混杂与RGB-only控制缺口仍在。 见[夜间结果与GitHub更新](../2026-09-06_audit_RGBIR夜间结果与GitHub更新/README.md)。下文旧快照按各自时间读取。

> **17:30最新核验（2026-09-06）**：2/9微调完成，paired123当前103/200；已证实复用W1 native存在检测头初始化混杂，旧“同init”表述不成立；冻结迁移门为+1.0 AP50，旧文+0.1误写。尚缺独立last评估与RGB-only SSL控制。 见[晚间进度与新结果](../2026-09-06_audit_RGBIR晚间进度与新结果/README.md)。下文保留较早状态。

> **一句话**：OS-SSL（全项目唯一 3/3 强正例）首次移植红外域——SSL 三臂 10k 步已完成，9 个微调在守卫队列中自动等待并行会话臂完成后接续（预计明天出齐 3-seed 四臂对照）。
> 预注册：`07_研究分析/方法预注册_OS-SSL-IR_20260906.md`（迁移门 +0.1 AP50/归因门/失败判读，读出前冻结）。

## 设置
- 数据：DroneVehicle 17,990 训练对（paired-tree 模式，identity 分组；"sar"槽=IR，"eo"槽=RGB）
- SSL：BYOL on yolo11n 骨干（train_yolo_osssl 从 pyc 恢复运行），10,000 步/b32，三臂 manifest（paired / sar_only=IR-only / shuffled，shuffle seed=42）
- 注入：build_yolo11n_template(nc=5) 模板 + strict_inject（240 骨干张量，non-backbone SHA 记录）；inject 工具的 stable_state_sha256 缺失已用本地确定性实现修补（副本 inject_patched.py）；旧版 trainer 的 final.pt 缺 checkpoint_kind 标记已如实补注（偏差见下）
- 微调：协议 yaml（e200/b32/640）+ clean 重建 checkpoint（标准 DetectionModel，nc=5）；native 臂复用 cgkd_w1 三 seed（同 init SHA + 同 recipe）
- 执行：GPU4/5/6（SSL）→ GPU0/1/2 队列（微调，经 resource guard 重试 worker——与并行会话的 rgbir_oev1 三臂共享 3-GPU 配额）

## 结果
- SSL 终态 loss：paired 0.124 / ir-only 0.083 / shuffled 0.049（**不可跨臂比较**：shuffled 用固定 permutation，模型可记忆固定错配；判读只看下游检测）
- 微调进度（2026-09-06 15:00）：shuffled s123 **完成 200/200** ✅；sar_only s42 @155/200（GPU1）；其余 7 run 在守卫队列（与并行会话 rgbir_oev1 三臂共享 3-GPU 配额，轮询自动接续）。注意：**并行会话臂完成释放租约后，我们的任务是逐个获取**，全程无需人工干预
- 队列状态健康：w0/w1/w2 三 worker + 放宽条款第 4 卡已用完（shuffled s123 即第 4 卡跑完的）

## 结论
（待微调完成后填四臂 3-seed 对照表）

## 偏差记录
- D1：旧版 pyc trainer 保存的 final.pt 缺 checkpoint_kind/checkpoint_format 字段，注入器拒绝；已按官方 SpaceNet6 格式补注标记（不改变任何张量内容），登记于此。
- D2：为兼容注入器版本漂移，stable_state_sha256 用本地确定性实现替代（仅影响记录哈希，不影响模型内容）。

## 产物路径
- 94: `RGBT_campaign/artifacts/osssl_ir_20260906/`（ssl/{arm}/final.pt、manifest、injected_*.pt、clean_*.pt、protocol_clean_*.yaml、worker 日志）
- 94: `RGBT_campaign/runs/osssl_ir_20260906/{arm}_rgb_s{seed}_e200/`（9 个微调输出）
