# RGBIR 对象判别蒸馏首轮：独立实现复审

> **最终工程结论：静态复审与两臂真实 canary 独立复核全部通过，无剩余工程阻塞项；可按已冻结协议、同一 GPU 串行启动已授权的 N/P 首轮。此结论只证明当前实现与启动条件通过工程核验，不证明检测增益。** 日期 2026-09-06；独立 reviewer `/root/rgbir_code_review`。只读实现、证据与 CPU checkpoint，不运行 GPU、不改变代码。

## 适用范围

本次是用户已授权的 DroneVehicle IR→RGB、N/P、seed42、200 epochs 探索性首轮，最多一张 GPU，N/P 同卡串行。缺少其它 seeds 或后续归因臂不阻止这两次可解释的授权实验；同时禁止把首轮结果升级为多 seed 增益、配对特异性或论文机制结论。

依据 workspace AGENTS 与工程内 AGENTS；源码/配置/输入/环境用副本回执保留，不新写 hash/SHA。历史最小验证方案提供研究背景，本轮准确方法以训练前冻结的新协议为准。

代码根：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1/`。下列行号来自 2026-09-06 本轮源码快照。

## 已审查实现

| 检查项 | 证据 | 判断 |
|---|---|---|
| 学生原生输入与 GT | `paired_rgbir_data.py:175`：base 获取 record、base.transforms 执行一次；输出来自 weak，仅追加 strong 字段 | 静态通过 |
| 独立 IR GT 与 RNG | `paired_rgbir_data.py:168` 读取 teacher_base 真标注；`:180` 保存 before，`:182` 保存 after，`:192` 重放，`:195` finally 恢复；`:200` 保留独立 strong cls/boxes | 静态通过 |
| teacher 标签 collate | `paired_rgbir_data.py:207` 分离两侧目标数，`:215` 重新构造每个 IR 目标的 image batch index | 静态通过 |
| 缓存与学生状态 | `paired_rgbir_data.py:114` 限缓存 1–64；`:138` 仅 dedicated teacher dataset 关闭 native buffer；`:170` LRU 淘汰 | 静态通过 |
| 对象对应 | `object_evidence_loss.py:92`：同类、IoU≥0.5；最大有效配对数优先、再最大 IoU；不按两侧标注数组位置对应 | 静态通过 |
| foreground—ring 相对证据 | `object_evidence_loss.py:177`：各自 GT ROI；`:187` 环带排除该模态 ALL GT；`:189` 有效点数；`:192` logmeanexp；`:194` 除温度 | 静态通过 |
| 原生监督旁路 | 模块不改变标签或 assigner；只读取 student raw class logits，解码 DFL 仅使用 teacher/ref detach 输出 | loss 模块通过，trainer 待核查 |
| 教师和参考 detach | `object_evidence_loss.py:255`、`:269` no_grad，`:290`、`:293` detach；smoothL1 仅 student evidence 可导 | 静态通过 |
| 归一化与随机控制 | `object_evidence_loss.py:209` 选择；`:296` 固定当前 base 数为分母；`:217` private CPU Generator 不推进训练 RNG | 按当前实现通过；与旧草案差异需冻结登记 |
| 空目标与空区域 | `object_evidence_loss.py:191` 有限 sentinel、valid mask；`:249` 可导零；`:288` 无共同对象返回零；`:298` 无选中对象返回零 | 静态通过 |

## 需在协议明确登记的差异

已复核新增 `EXPERIMENT_PLAN.md` 与 `config_drone.yaml`：下列差异均已在新协议第 2–3 节准确登记，默认值与 loss 配置一致；不构成首轮 N/P 的阻塞项。

1. 当前 E/base 已依赖 RGB–IR GT 对应与两侧有效环带，再要求冻结 RGB reference 的类无关候选；它不是旧草案里的纯 RGB 公共 E。
2. `K=ceil(rho * eligible_count)`，eligible 还要求教师候选通过且 q>0；并非 `ceil(rho * E)`。选择在全 batch pooled 对象上进行，未做类别/大小分层。实际名义剂量 K/E 随 batch 变化，不能称为跨教师固定剂量。首轮 N/P 仍可直接解释为整套已定义干预对 native 的比较。
3. 教师候选由 pre-NMS decoded boxes 中“存在正确 argmax 类、conf≥0.25、IoU≥0.5 的候选”定义；同一候选可能支持邻近多个 GT。它是候选支持代理，不是上一轮 probe 的 NMS 后一对一真实命中统计。RGB reference 则用类无关 conf≥0.05、IoU≥0.1 候选。
4. q 为相对证据的 softplus surrogate 改善，无法保证跨模态概率已校准。`sigmoid(evidence)` 的熵描述的是二元边际证据目标，不是完整 nc 类别向量的熵。冻结 reference 见过训练图，其选择只是训练代理。
5. 模块的 paired_random 使用同一批次同样 K 从整个 base 采样，检验完整正确性/质量选择；不是在 teacher-correct 子集里仅打乱 q 排名。same_modal/shuffled 的公平候选与预算政策需在实际授权运行前另行冻结，本轮不使用这些臂推论。

## 已收到测试证据

loader agent 报告在 94 的 pinned Ultralytics 8.4.115 环境通过 6/6 CPU 测试；reviewer 已逐项阅读 `test_paired_rgbir_data.py`。测试用合成图像/独立标签与真实 RandomPerspective、RandomFlip、Format：24 seeds 学生 tensor 和 Python/NumPy/Torch RNG 等价；IR 标签数不同；真实裁剪后两侧独立保留标签；collate IDs；same-modal 直接一致；LRU 边界；教师读取异常后恢复学生 RNG。该证据不冒充真实 DroneVehicle GPU canary。

loss agent 另提交 `loss_cpu_validation.md` 与 `loss_cpu_tests.log`，本地 Torch 1.8 CPU 通过 13 项测试。reviewer 已读测试源码，覆盖学生前景/背景梯度方向、教师/reference/学生 DFL 无 KD 梯度、相对证据平移不变、独立教师 GT、环带排除未匹配 GT、Hungarian 反例、固定 base 分母、空对象/空区域、private RNG、同模态 IR 标签隔离。完整 pinned 环境仍由 root 重跑。

## 完整 trainer 复审（第二轮）

已审查 `train_object_evidence.py` 与 pinned BaseTrainer 源码快照：

- `:47` / `:98`：`native_total.sum() + B * weight * kd` 标量一次相加；保留 native 自身的 B 缩放。
- `:156` / `:165`：仅 train 包装；真实 IR dataset、独立 strong GT 传入 `teacher_batch`，学生 batch 先完成 native preprocess。
- `:177` / `:187`：先完成 `super()._setup_train()` 的 optimizer/EMA，之后装 ordinary Python criterion 与两 frozen 模型；不注册为学生子模块。`:190` / `:193` 检查状态键及辅助参数不进 optimizer。
- `:151`：OOM 自动降 batch 在 pipeline 重建前被拒绝；`:222` / `:227` 禁止损失 NaN 自动恢复和非有限 live/EMA checkpoint 修复。
- `:205`：记录 update_attempts、真实成功更新与 AMP skips；`:213` canary 按成功更新数停止。`:282` 正式 E200 终态预算检查。
- `:300`：使用工程要求的统一 run receipt；方法身份读取 PROTOCOL-ADAPTED，记录原创 method_id、独立 IR label 使用及不访问 test。

**B1，canary 检查 bug，已修复：** 原 `:103` 条件为首批或“有选择且 gradient_checks 为空”。首批无 eligible 对象时仍会写入零 KD 梯度记录，导致后续出现有效选择也不再做梯度检查，终态因无非零证据而误报失败。root 已将第二项改为“有选择且尚无任何 kd_score_gradient_l2>0 的记录”；reviewer 已逐行核对新 `:103–104`，修复通过。

**非阻塞预算口径：** native `optimizer_step` 在 GradScaler skip 后仍会更新 EMA。相同 E200 固定 batch/尝试 schedule，真实成功 optimizer 更新数可能因 N/P 的 AMP skips 不同而有差异，EMA.updates 则记录尝试次数。当前 receipt 三者均有保存，准确；在终态复核前不能宣称成功更新数已精确相同。两个 24-successful-update canary 也可能经历不同 batch 数。

**非阻塞回执补强：** root 已把 native DetectionTrainer 实际源码加入 trainers 副本，reviewer 已核对；BaseTrainer 的 AMP/EMA/schedule 源码也可一并保存。运行环境 pinned 版本已在 launch manifest 记录。

## 此前待集成证据（由下文最终 canary 核验闭合）

- trainer 将 native 三项先求和、KD 标量仅加入一次；同 batch weight0 loss/gradient 等价。
- teacher/ref 在 optimizer/EMA 建立后绑定；checkpoint/EMA 仅含部署学生，参数集合和初始化一致。
- 实际 N/P 学生 batch/目标一致，KD 非零且梯度生效，teacher/ref 无梯度；完整真实数据读取和独立 IR 标注接入。
- 相同 last/EMA 固定端点、相同实际 updates、独立 GT evaluator、无 test 暴露、准确 PROTOCOL-ADAPTED 与原创 pilot 名称回执。
- 资源 canary 与动态同卡锁定：合并当前 legacy 项目 CUDA PID 核算；用户本轮一张卡限制、每卡项目最多两 CUDA PID、显存和 RSS 余量；不可自动 OOM 改 batch。

## evaluator 与 launcher 复审（第三轮）

**B2，评估回执必然报错，已修复：** 初版 `evaluate_object_evidence.py` 把空 `split_rosters` 传入统一 receipt；`write_jstars_run_receipt.py:304` 对任何 run_kind 都要求至少一个实际 roster，故评估完成后必然抛异常。root 已在新版 evaluator `:32–37` 解析 frozen val 目录，冻结 1469 张图并保存完整路径 roster；`:51` 传入该文件。reviewer 已核对修复。

评估 `:30` 固定 `weights/last.pt`，`:39` 使用 RGB data YAML、split=val、640、batch32 和相同 YOLO.val 口径；`:42–46` AP50/AP75/mAP50–95/precision/recall 与既有 evaluator 字段相同，数值为 0–1。只有 training_completed 才可评估，拒绝覆盖 eval_val；输出固定项目 runs 子目录。没有 teacher 预测充当 GT，也没有 test 读取。

**B3，exit 2 混淆资源排队与子程序失败，已修复：** 初版 queue 只要 guard 返回 2 就永远重试，而 guard 会原样返回子程序 exit 2，包括 argparse 错误。root 已改为流式读取输出：只有 exit 2 且解析到 guard `QUEUED`、没有 `LAUNCHED` 才同卡等待。reviewer 对照 guard `:730` 的 admission JSON 与 `:778` 的子程序返回码核查，修复通过；实际子程序错误不再当排队重试。

串行 queue 使用 allocation.json 锁定一个物理 GPU，flock 防止同 campaign 并行，每次 guard 只给一个 candidate、一个 CUDA PID；canary P/N 各 24 个成功 optimizer updates，随后 CPU 直接比较 initial_student 和 first_batch tensors。full stage 要求 canary_comparison passed，顺序为 P→eval→N→eval。已有 training attempt 目录直接拒绝覆盖，训练失败产物保留。

首次审查曾提醒旧未登记 SSL 任务可能不计入 lease guard；随后 root 的新 `allocation.json` 明确记录旧 SSL 已完成、动态选择 GPU4 时无 compute PID、空闲 24082 MiB。这一实际状态消除了该次启动的未登记 legacy 占用问题，后续每个 stage 仍需正常 guard admission。

非阻塞运维建议：queue 顶层异常可补写 failed/current_arm/error，避免 screen 退出后状态 JSON 仍停留 running；当前训练 failure_receipt 与日志会保留，故不会覆盖或伪造成功。

## 复审结论更新

所有本轮源码均完成静态复审，B1/B2/B3 已修复。首次设计预检保留在 `EXPERIMENT_CODE_REVIEW_INITIAL.md`。

## 最终真实 canary 独立核验（2026-09-06）

reviewer 经 `ssh 94` 只读取得两次完成回执、统一 run receipt、runtime_ready、allocation 和 canary comparison，并在 pinned 环境以 `CUDA_VISIBLE_DEVICES=''`、`torch.load(map_location='cpu')` 读取两个 `last.pt`。审查前后 `torch.cuda.is_initialized()` 均为 False；没有新增 GPU 任务。

| 实测项 | paired | weight0 | 判断 |
|---|---:|---:|---|
| canary 完成状态 | canary_completed | canary_completed | 通过 |
| 实际 batch / optimizer 尝试 | 30 / 30 | 30 / 30 | 当前短程预算相同 |
| 成功 optimizer updates / AMP skips | 24 / 6 | 24 / 6 | 如实区分成功更新与跳步 |
| EMA updates | 30 | 30 | 与原生每次尝试更新 EMA 的行为一致 |
| 选中对象累计 | 3767 | 3767 | 选择信号非空 |
| 首批 KD score 梯度 L2 | 0.0036568181 | 0.0036568181 | 算子非零梯度生效；N 最终乘零 |
| 总梯度单次缩放最大误差 | 3.8147e-6 | 0 | 在声明 AMP 容差内 |
| weight0 与 native loss / score gradient | 精确一致 | 精确一致 | 通过 |
| 教师 / reference 是否有梯度 | 否 / 否 | 否 / 否 | 通过 |
| CUDA PID / 物理 GPU | 1 / GPU4 | 1 / GPU4 | 同卡串行 |
| NVML 进程峰值显存 MiB | 6304 | 6304 | 低于正式预约 10000 MiB |
| PyTorch reserved / allocated 峰值 MiB | 5770 / 5087.686 | 5770 / 5087.686 | 与 NVML 口径分别保留 |
| 进程树 RSS 峰值 MiB | 28895 | 28690 | 正式预约调整为 49152 MiB |

`canary_comparison.json` 为 passed；两臂 `initial_student.pt` 与 `first_batch.pt` 字段集合及每个 tensor 均在 CPU 直接精确比较通过。该证据是初始模型与首批输入/目标等价，不冒充对全部长程 batch 的逐元素比较。

两个 `last.pt` 的独立 CPU 检查均为：EMA 内普通 `DetectionModel`，`model` 字段为 None，`criterion` 为 None；无 teacher/reference/trainer 辅助属性或对应 state key。499 个 state tensor 的键与形状均与 initial_student 一致，参数总数 2,590,815；全部 checkpoint tensor 位于 CPU，所有浮点张量有限。部署 checkpoint 确认不包含训练期教师/reference。

统一 run receipt 均为 COMPLETED / PROTOCOL-ADAPTED / development_train，明确记录 method_id=RGBIR-OBJECT-EVIDENCE-v1、canary=true、student_native_gt_only=true、teacher_labels_used_by_kd=true。所有声明的源码/配置/split 副本实际存在。暴露账本仍是 UNVERIFIED_SEALED / confirmatory=false，不能把本开发实验称为正式封存测试验证；本轮完成回执没有 test 访问。

资源解释：canary 原预约 RSS 为 16384 MiB，实测超过这一预约但远低于主机上限；root 已据此把正式任务预约改为 49152 MiB，显存仍按 10000 MiB、每次至少 2048 MiB 空闲余量准入。正式释放仍必须通过同一 GPU4 的 guard 重新检查当时空闲资源。这是实测后的资源预算修正，没有更改 trainer/loss/方法配置或根据 AP 选阈值。

**启动结论：工程核验通过，允许进入用户授权的 N/P、seed42、E200 同卡串行训练。无需等待完整 AP 结果或追加其它 seeds 才启动。** 完整训练终态仍需报告实际更新数、AMP skips、固定 last/EMA 的 RGB GT val 结果；单 seed 方向不升级为多 seed 增益或跨模态机制证明。

独立核验产物：`reviewer_canary_readonly.py`（审查脚本）、`reviewer_canary_verified.json`（完整读取回执与 CPU checkpoint 核验）。远端原始根为 `94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/` 和 `94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/canary_{paired,weight0}_s42_attempt1/`，原始产物未改动。
