# 自然流选择诊断首次独立审阅

首次机器审阅 `EXPERIMENT_AUDIT.json` 为 **PASS_FOR_REAL_CANARY**：源码审阅与 14 项独立 CPU 真值/负例通过。该结论仅表示可进行已授权的只读真实 canary，未宣称真实运行完成。随后真实 attempt 1 在 0 batch 因同名 `prepare_configs` 导入冲突失败；原文件保留，本轮后续放行入口为 `ATTEMPT2_IMPORT_REVIEW.json/md`。

审阅者 `/root/ap_error`，实现作者 `/root/baseline_feature_analysis`，campaign/部署作者 `/root`。使用 experiment-audit 技能；用户禁止新增 hash，采用源文件路径/size/mtime、源码副本及直接字节比较。所有审阅测试均在本地 CPU，未调用 GPU/SSH。

核心检查：C 调用保留整 batch 的 rho/ceil 配额；多图反例显示全局 eligible=3 时选 2，逐图会选 3。L 以原整批 teacher same-anchor 调用为主，每图重放只分配 gate 的图/组归属，真实每批必须恢复并精确比较所有 COUNTS、全局 RGB/IR GT 索引、base/selected anchor 身份、DFL 距离和质量 mask；不累加 normalizer。R 只复用分类 student 槽以调用现有选择，不建立新学生，不输出学生 loss。T/R 均 frozen eval FP32，每模型每批一次前向，无 backward/校准。

`cpu_test_result_v2.json` 为原 14 项已执行结果，包括 C 配额反例、L ragged/空图及篡改负例、原 trace/loader 元数据微改拒绝、2/64 CLI 约束、summary scope 与 NaN/Inf/非正资源拒绝，以及模拟 full=63 不得写 completion。torch 1.8.0+cu111，CUDA 未初始化；测试不代替 94 pinned 环境的真实 canary。

`run_campaign.py` 分别读取两数据集实测 2-batch 峰值，之后预约固定 64 batch；现有全局 lease 保留 GPU/进程数、2 GiB 显存余量与总内存限制。canary 和 full 都必须通过结果语义与有限正资源检查。部署 helper 的入口只核验审阅文件存在，根执行者明确承担读取实际 PASS 内容后才调用的职责；审阅者未运行 helper。

None/false geometry 是 **UNVERIFIED_GEOMETRY_DIAGNOSTIC**。忽略 geometry 可能改变最优候选 anchor，因此最终 selected_count 不能称为严格上界；计数不支持几何、梯度、可学性或 KD 收益结论。旧覆盖日志没有像素字节，因此 exact 只限于它记录的 source/augmentation/双标签/shape/dtype 等字段。

已知覆盖缺口由真实 attempt 1 暴露并在追加导入审阅中处理。后续结果是否执行、是否与旧自然流一致，须另读真实运行回执；不得将源码/CPU PASS 升级为已完成 64 batch。
