# 实现与首轮运行协议：Object Evidence v1

> 2026-09-06，训练前冻结。来源为上一轮RGBIR实测诊断与用户本轮“实现代码，94启动1–2个实验，最多一个GPU”的明确授权。先CPU算子检查/独立审查/真实batch短程canary，再启动两个E200探索性run；不提前声称跨模态收益。

## 1. 主张、范围与对照
假设：教师更可靠、学生有候选线索且可对应的对象，其前景相对背景的类别判别证据具有可迁移价值。N/P首轮只检验这套干预的净结果，不能分解选择、软内容或证明超过同模态KD。

首轮正式：paired、weight0，各seed42一次；学生YOLO11n，从原通用pretrain初始化，不从已训RGB模型微调。完整E200，last/EMA独立val，主指标mAP50–95，次指标AP50/AP75/precision/recall。train17990、val1469；不打开test。

IR教师：formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt；冻结RGB参考：同目录rgb_seed42_native_b32a2/weights/last.pt。教师和参考均在train见过图，质量/候选是训练代理，不是独立可学性估计。两臂相同student输入/标签/初始化/更新预算。

## 2. 精确知识定义
输入同几何增强的RGB和IR各自真实标注。学生管线先执行一次原生transform，IR真实图/GT重放同几何随机状态，最后恢复学生执行后的RNG。禁用mosaic/mixup/cutmix/HSV/Albumentations；保留translate0.1、scale0.5、fliplr0.5。两侧目标不按索引强配。

同图同类GT用IoU≥0.5最大有效匹配数优先、IoU破平局的一对一对应。用P3/P4原生anchor中心，前景为各自GT内部，背景为同心2倍框内排除该模态全部GT的区域（包括其他类和未配对GT）；每层至少1前景、4背景点。只平均两模态共同有效的层。

`e_i^m = mean_valid_levels[(logmeanexp(z_fg,c) - logmeanexp(z_bg,c))/2]`。

不使用独立objectness分支，不蒸馏DFL/位置/全图特征。T=2；教师target截断到[-8,8]；SmoothL1 beta=1。

基础集合E：有真实跨模态GT对应、两侧区域有效，并且冻结RGB参考有任意类别pre-NMS候选conf≥0.05、对RGB GT IoU≥0.1。它依赖对应/区域有效性，**不是纯RGB-only候选集合**。

教师正确性代理：IR teacher存在argmax类别等于GT、该类conf≥0.25、对自身IR GT IoU≥0.5的pre-NMS候选。没有NMS/一对一TP含义，不使用验证集成功标签。

`q_i = max(softplus(-e_i^ref)-softplus(-e_i^T),0)`。

该q衡量局部判别margin的排序代理，不是校准置信差或物理信息量。在E中取teacher-correct且q>0的eligible集合，按q降序（稳定索引破平局）选择 `K=ceil(0.5*|eligible|)`，整个batch统一选择；不是按类/尺度分层固定K。

`KD = sum_selected SmoothL1(e_i^S, stopgrad(clip(e_i^T))) / max(1,|E|)`。

`total = native_total.sum() + 0.1 * batch_size * KD`；native_total已经包含原生B尺度，KD只加一次，禁止向三项native向量广播。weight0用相同数据/选择/辅助模型计算，最终乘0，便于核验学生路径与剂量；教师和参考eval/no_grad，不在optimizer或部署checkpoint中。

## 3. 相对于前一轮建议的明确细化
上一轮给出对象类别响应与相对背景证据两种候选，本版本固定后者；不是临时因训练效果替换知识。
rho作用在teacher-correct且q>0集合，候选分母仍为E，因此有效剂量会随教师质量变化。本版本不声称固定保留所有E的50%。未来paired_random必须取与P同K、同E、同λ和同分母，才能分析选择价值。模块提供random/uniform/same_modal算子，但本轮只启动N/P；shuffled与GT-only内容策略未冻结时必须显式拒绝，不能假装已有完整归因结果。
IR独立标签是新增训练期辅助信息；旧prepare receipt的teacher_labels_used_by_kd=false属于旧方法，不能照抄到本方法。

## 4. 训练与执行
E200、batch32/nbs64、workers4，SGD lr0/lrf=.01、momentum=.937、wd=.0005、warmup3/.8/.1，AMP开启、deterministic、patience0、不自动batch减半、不自动NaN恢复。与历史workers8差异已登记，首轮N/P统一workers4。

首轮seed42结果无论正负都完整记录；不因中间指标低提前停另一个可解释实验。只有技术故障、数据/梯度错误、资源越界才技术停止。候选定义、λ=.1、ρ=.5、T=2等不按AP调参。后续是否扩seed0/123另作研究判断，达到三seed之前不应用上一版三seed增益门为正式claim。

启动时查看94已有任务，从已有项目使用的物理卡选择一张可容纳的卡并写入launch allocation；整个canary/训练/评估/两臂排队都锁定该卡，不在方法配置硬编码GPU。现行同卡已有一个其他研究CUDA任务时，本次最多同时一个训练（全项目同卡最多两个CUDA PID），N/P串行screen；每次通过resource_guard，实测峰值+余量、实际空闲≥任务预估+2GiB，项目显存<70%，总RSS<300GiB。

## 5. 工程门和记录
CPU：实际Ultralytics几何transform上的独立标签与学生tensor/RNG一致；loss梯度方向、teacher/reference无梯度、对整体logit平移不变、随机同K、空候选和最大有效匹配。
真实batch canary：两臂各24 optimizer updates，检查student初始化直接tensor等价、weight0损失和梯度与native一致、KD仅加一次、选择非空/梯度非零、教师无梯度、teacher/reference不进EMA/保存图；测NVML/allocated/reserved/RSS及吞吐。canary不用AP选择方法。
部署前由独立Codex reviewer审查。使用write_jstars_run_receipt.py保存源码/配置/命令/环境/输入清单/资源/训练及评估指标/test exposure副本；遵守工程AGENTS，不写hash。
