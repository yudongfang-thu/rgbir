# OGSOD400 本地 / L20 训练协议与方法实现审计

**审计日期：** 2026-07-12（Asia/Shanghai）  
**审计范围：** 本地 `ogsod400_clean_protocol`、L20 `/private/projects/ogsod400_clean_protocol`、已完成与正在运行的 baseline / CCLKD / FGD / LD / CMDistill / legacy LADD / LADD identity / LCSR / RIF  
**证据切点：** 不可变快照分别为本地 12:37、L20 12:35；L20 活任务在 12:53 再次只读核验  
**远端操作：** 只读；未停止、重启、修改或新启动任何实验

## 总结论：FAIL（“全部完全符合且 claim 正确”不成立）

这不是指标造假或数据泄漏导致的 FAIL。经典实验完整性结论为 **WARN**：检测 AP 使用真实 OGSOD 测试标签，未发现自归一化或虚构 AP；但 registry、哈希和状态记录不完整，CCLKD 的验证损失列全程为 NaN。

整体判为 FAIL 的原因是方法与 claim 层面存在会改变科学含义的问题：

1. 正在运行的 `legacy_ladd` 直接从通用 `yolo11n.pt` 进入 phase B，未加载前置分解阶段 checkpoint；phase B 又冻结随机初始化的分解模块。因此它不是有效的历史 LADD 复现。
2. 当前 `FGD` 只实现 focal foreground/background 与 attention mask，显式移除了 global relation；不能以完整 FGD 名义进入主表。
3. `LADD identity` 的 corrected feature 代数上恒等于原始 feature，且配置声称 foreground-only，代码却对全空间做 z loss。
4. `LCSR` 的可执行路径与 zero-init 是正确的，但 “公共子空间”“SAR 可预测比例 rho” 等机制 claim 尚未被代码或 gate 证据识别；正式 400 epoch 在预声明 gate 前启动。
5. `RIF` 正式实验没有启动，这是正确决定；但当前 formal pipeline 的 cache/reuse 绑定仍不足，不能安全放行。

## 1. 审计基线与协议

主协议文件：`configs/protocols/ogsod_yolo11n_nomosaic_direct400_sgd_b64_v1.yaml`，SHA256：

`4471603c315925b44b7cca5cfa53d2920dd2834f787b3309695e04bc8e2b6d6d`

固定口径：

| 项目 | 固定值 |
|---|---|
| 任务 / 模型 | OGSOD HBB / YOLO11n |
| 初始化 | `yolo11n.pt`，source epochs = 0 |
| 训练 | exact 400 epochs，patience 400 |
| 输入 / batch | 256 / 64，strict batch |
| 优化器 | SGD，lr0/lrf=0.01，momentum=0.937，weight decay=0.0005 |
| warmup / schedule | 3 epochs，0.8，0.1；cos_lr=false |
| 主增强 | mosaic/close_mosaic/mixup/cutmix=0 |
| 其他增强 | translate=0.1，scale=0.5，fliplr=0.5；degrees/perspective/HSV/erasing=0 |
| 随机性 | deterministic=true；seeds 0/42/123 |
| 正式评估 | SAR-only AP50、AP50-95；best 与 epoch400 |

协议 YAML 内部状态仍是 `canonical_candidate`，而 README 将其称为固定/唯一 canonical。数值口径明确，但治理状态需要统一。

## 2. 数据与 baseline 完整性

### 2.1 数据

L20 实际数据审计结果：

| split | RGB 图像/标签 | SAR 图像/标签 | boxes | 类别计数 |
|---|---:|---:|---:|---|
| train | 14,664 / 14,664 | 14,664 / 14,664 | 38,975 | 0:25,533；1:3,306；2:10,136 |
| test | 3,667 / 3,667 | 3,667 / 3,667 | 9,614 | 0:6,389；1:803；2:2,422 |

- RGB/SAR 文件 stem 完全相同；
- train/test 无 stem 交叠；
- RGB/SAR 标签树哈希逐 split 完全相同；
- 未发现缺图、缺标签或 background 样本；
- paired loader 按相对文件名取 RGB teacher 图像，并同步几何增强。

本地与 L20 dataset YAML 的语义差异只有 `path`：本地是占位路径，L20 是 `/private/data/...`。但协议记录的是本地占位 YAML 哈希，L20 实际 YAML 哈希不同，因此**数据内容匹配，证据哈希不闭环**。

### 2.2 已完成 baseline

| baseline | 完成状态 | AP50-95 | 判定 |
|---|---|---:|---|
| SAR s0 retry1 | 400/400 | 0.49003 | exact matched |
| SAR s42 retry1 | 400/400 | 0.48840 | exact matched |
| SAR s123 retry1 | 400/400 | 0.48810 | exact matched |
| RGB s42 retry1 | 400/400 | 0.58743 | exact matched |
| RGB s123 retry1 | 400/400 | final 0.58768；best 0.58780@399 | exact matched；文档把 final 值写在 best.pt 口径下 |
| RGB s0 retry1 | 约 299/400 后中断 | 不可用于正式 teacher | 已归档排除 |
| RGB s0 restart2 | 正确从 `yolo11n.pt` 重跑 | 运行中 | 配置正确，完成前不可批准 |

三个 SAR seed、RGB s42/s123 的初始化权重、数据、优化器、增强与 400 epoch 均匹配协议。RGB s0 restart2 也从干净 `yolo11n.pt`、`resume=false` 启动，没有续接被中断的 retry1。

## 3. L20 已启动实验逐项判定

以下进度是 2026-07-12 12:53 的瞬时切点；`epoch index` 是 CSV 第一列的零基索引。

| run family | 当前状态 | 固定训练参数 | baseline / teacher | 科学可用性 |
|---|---|---|---|---|
| SAR s0/s42/s123 | 完成 | exact matched | 直接 `yolo11n.pt` | 正式 baseline 可用 |
| RGB s42/s123 | 完成 | exact matched | 直接 `yolo11n.pt` | 正式 RGB teacher 可用 |
| RGB s0 restart2 | index 25，持续写入 | exact matched | 直接 `yolo11n.pt` | 完成前不可用作正式 teacher |
| CCLKD s0 | index 234，持续写入 | 除 `strict_batch_size=false` 外匹配 | **在线、可训练 RGB teacher**；不是冻结 RGB baseline | 仅可称 clean-protocol YOLO11 CCLKD adaptation |
| FGD s42 | index 2，持续写入 | exact matched | 正确的同 seed 冻结 RGB s42 teacher | loss 不完整；不得称完整 FGD |
| LD s42 | index 14，持续写入 | exact matched | 正确的同 seed 冻结 RGB s42 teacher | 可保留为 LD-YOLO11 cross-modal adaptation |
| CMDistill s42 | index 21，持续写入 | exact matched | 正确的同 seed 冻结 RGB s42 teacher | 可保留为 paper-aligned adaptation，非精确复现 |
| legacy LADD s42 | index 15，持续写入 | 外层超参 matched | RGB teacher 正确；分解初始化错误 | 仅随机分解诊断，不是 legacy LADD |
| LADD identity s42 | index 16，持续写入 | exact matched | 正确的同 seed冻结 RGB s42 teacher | 只能作为 projected-feature KD identity control |
| LCSR s42 | index 2，持续写入 | exact matched | 正确的同 seed 冻结 RGB s42 teacher | 探索性；机制 gate 未通过 |
| RIF formal | 未启动 | N/A | 缺冻结 fusion-oracle cache | 正确阻塞 |
| RIF smoke | 2 样本、1 epoch | intentional smoke deviation | zero-target cache | `synthetic_proxy / engineering_smoke`，无 AP 含义 |

12:53 时八条活动结果文件均继续更新，GPU 为 28,678/46,068 MiB、99% 利用率；此前和本次日志扫描未发现 Traceback、CUDA OOM、OutOfMemoryError、RuntimeError 或 AssertionError。**运行健康不等于方法科学有效。**

FGD/LD/CMDistill launcher 会检查并记录 SAR baseline 路径，但实际模型初始化由 `MODEL=yolo11n.pt` 决定，未设置 `B_DETECTOR_SOURCE` / `B_DECOMP_SOURCE`；因此 SAR baseline 没有被加载。这一点符合 direct-init source-epochs=0 的公共协议，不是 400+400 continuation，但也意味着 “使用 SAR baseline 初始化” 的说法不成立。

## 4. 方法实现与 claim 审计

### 4.1 legacy LADD — FAIL

证据链：

- `methods/legacy_ladd/tools/run_legacy_ladd.sh:39-54` 直接以 `--phase b --model "$MODEL"` 启动，默认 MODEL 是通用 `yolo11n.pt`；
- 历史 launcher 自己声明 later phases 必须使用上一阶段 `best.pt`：`comparison/runtime/current_hbb/scripts/ogsod_public/run_ladd_phase.sh:19-21`；
- phase B 在 `ladd_b_a2_core=false` 时冻结 teacher decomposition/reachability：`comparison/runtime/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py:1125-1150`；
- 活动日志显示只迁移 448/499 个参数，新增辅助模块未从前置阶段恢复。

因此当前训练实际是 “raw YOLO detector + 对随机初始化且冻结的 decomposition target 做 KD”。其 AP 即使最终提高，也不能作为历史 LADD 的证据。可选处置只有两种：完整执行并绑定前置 phase checkpoint，或把 run 明确重命名为 `random-decomposition diagnostic`。

### 4.2 LADD identity — claim 不匹配

`methods/ladd_identity/src/ladd_identity/modules.py:36-46`：

`shared = W^T W x`，`private = x - shared`，`corrected = private + shared = x`。

检测头确实读取 `corrected`（`model.py:49-53`），所以不存在隐藏 raw bypass；但该 corrected 在代数上严格等于 raw feature。真正产生训练作用的是同一个可学习 projector 上的 RGB/SAR z KD（`loss.py:37-44`），不是 detector feature correction。

此外 `configs/nomosaic400.yaml:7` 声明 `foreground_only: true`，实际 `loss.py:37-44` 对所有空间 token 做 SmoothL1，没有任何 foreground mask。当前 run 应定义为 “全空间 projected-feature KD identity control”，不能声称 foreground-only 或 deployable correction。

### 4.3 LCSR — 路径正确，机制 claim 未验证

确认正确的部分：

- 检测头实际读取 corrected feature：`method/src/lcsr/model.py:66-70`；
- predictor 最后一层零初始化：`modules.py:54-61`；
- 初始 corrected 等于原 detector feature，训练后由 SAR-only predictor 产生增量：`modules.py:102-106`；
- teacher 冻结，gap target detach，部署不需要 RGB。

尚不成立的部分：

- `rho` 在 `modules.py:120-129` 是成对 SAR/RGB code 的逐通道对角相关性的 EMA，不是 “从 SAR 可预测的比例” 或 predictive R²；
- 两套可训练投影器加 redundancy-reduction loss 不保证识别唯一、语义化的公共子空间；
- `oracle_replace` 只定义、未进入训练或正式 eval；
- 预声明计划要求 G0 配对 > shuffled、G1 paired oracle > native、G2/G3 干预与可预测性通过后才跑 400 epoch（`refine-logs/EXPERIMENT_PLAN_20260712_LCSR.md:9-35`），但目前直接启动了 s42 formal；
- `foreground_mask` 在 `method/src/lcsr/loss.py:21-27` 对每个 GPU bbox 多次调用 `.item()`，引入 CPU/GPU 同步，解释了 LCSR 明显较慢。

因此当前 LCSR run 只能标为 exploratory。只有 paired/shuffled/oracle/adapter gates 通过后，才能使用 “公共子空间”“可恢复公共差值”“因果替换” 等 claim。

### 4.4 RIF — formal 未就绪

RIF smoke 的 target 在 `build_smoke_cache.py:54-63` 明确为全零，只证明工程链路，不能产生 detector AP claim。

新 pipeline 正确拒绝包含 `zero_target` 的 formal cache，但仍有以下放行漏洞：

- `rif_pipeline.py:59-95` 只检查 samples 非空、描述不含 zero_target、至少两个长度为 64 的字符串；不验证十六进制、源 checkpoint 实体/哈希、sample 文件/哈希、target keys/shape/非零性、split/seed/protocol；
- `rif_pipeline.py:247-249` 只要旧 `summary.json` 存在就复用 projector，不绑定 cache hash、target kind、epochs、seed 或代码 hash；
- `rif_pipeline.py:288-305` 只要旧 `metrics.json` 存在就复用 evaluation，不绑定 projector hash；
- `load_state` 只校验 schema 与 seed，不校验 target kind、epochs、batch、协议；
- `train_rif_interaction.py:34-48` 使用 `torch.load(weights_only=False)`，且未校验 SAR/target shape 一致性；
- projector 是 feature-level proxy 训练，不等价于 direct400 detector，不能直接与 AP baseline 并表。

在补齐 provenance/schema/hash binding、真实 Stage-A/cache 生成和 detector evaluation 前，继续保持 blocked 是正确的。

## 5. 对比方法的命名与机制边界

### FGD — FAIL as “FGD”

`_fgd_style_loss`（`comparison/runtime/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py:1133-1167`）只有 focal fg/bg feature loss 与 attention mask loss；manifest 明确写 `fgd_relation=removed`（launcher :351-357）。FGD 原论文的核心由 focal distillation 和 global distillation 两部分组成，后者用于建模像素关系。当前实现必须改名为 `FGD-focal-only`，或补齐 global component 后重跑。[FGD 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)，[官方代码](https://github.com/yzd-v/FGD)。

### LD — 可用但必须加 adaptation 限定

代码在 YOLO11 DFL logits 上做 temperature=10 的 KL，并包含 main 与 VLR-style 区域（`loss.py:1169-1264`），与 LD 的 localization-distribution / valuable-localization-region 核心一致。但这是同容量、跨模态、YOLO11 的适配，不是论文原设置的精确复现。主表名称应为 `LD-YOLO11 cross-modal adaptation`。[LD 论文](https://arxiv.org/abs/2102.12252)，[官方代码](https://github.com/HikariTJU/LD)。

### CMDistill — 可用但不是精确复现

代码自己已声明 `paper-aligned adaptation`（`loss.py:1281-1287`）。PCC 使用 PKD 风格通道标准化 MSE，SLRD 使用采样 token cosine affinity，IBCLD 使用 BCE + aligned IoU（`loss.py:1336-1426`）。组件方向对应论文 PCCFD/SLRD/IBCLD，但具体张量约化是项目自定义近似；必须保留 adaptation 限定。[CMDistill 论文条目/DOI](https://doaj.org/article/0f0694405cf54dc19e948babe3409841)。

### CCLKD — clean-protocol adaptation，不是 paper reproduction

L20 独立 reproduction checker 要求 batch=32、mosaic=1、positive MixUp（snapshot `cclkd_reproduction/code/check_cclkd_repro_protocol.py:10-17,117-132`），其 launcher 也明确说明 YOLO11 只是 extension/adaptation（`launch_cclkd_paper_repro_job.sh:4-19`）。当前活动 run 则是 batch64、mosaic0、mixup0、`strict_batch_size=false`。它可以作为统一 clean protocol 下的在线 CCLKD 适配，但不能称原论文 reproduction；`formulation=paper` 只说明 loss 分支，不改变这一点。

## 6. 经典实验完整性检查（独立 reviewer + 本审计复核）

| 检查 | 状态 | 结论 |
|---|---|---|
| A. Ground-truth provenance | PASS | detector AP 均由 OGSOD dataset labels 与 Ultralytics DetectionValidator 计算 |
| B. Score normalization | PASS | 未发现用模型自身 max/min/mean 重标 AP；原始 CSV 数值与文档数字一致 |
| C. Result existence / traceability | WARN | registry 覆盖不全且 hash 大面积空缺；RGB s0 restart2 未完成却被 ACTIVE_BASELINES 放在 approved 列表 |
| D. Dead code / actual execution | WARN | claimed loss 主路径大多实际执行；LCSR oracle 未进入 evaluation；方法标签与实际路径有上述错位 |
| E. Scope | WARN | 只有 5 个完成的 baseline；方法结果全部未完成，不能写稳定/正式收益 |
| F. Evaluation type | PASS with qualifier | detector runs 为 `real_gt`；RIF zero-target smoke 为 `synthetic_proxy / engineering_smoke` |

独立 reviewer 的经典完整性 verdict 是 WARN；它确认无伪 GT、无 AP 自归一化、已报告 baseline 数字可追溯，同时发现 CCLKD 所有已写 epoch 的 validation loss 组件均为 NaN。独立 reviewer 没有识别出 legacy LADD 初始化、FGD global component、Identity foreground 和 LCSR gate 等机制问题，因此本报告的整体 verdict 更严格。

Reviewer 通过本机只读 CLI 运行，调用时请求 Claude Opus；返回元数据实际报告模型为 `deepseek-v4-pro`，因此报告如实记录为跨家族独立 reviewer，而不冒充 GPT-5.4/Opus。

## 7. 证据登记与运维缺陷

1. `registry/runs.csv` 只有少量 smoke/四方法记录，缺全部六个 baseline、RGB restart2、CCLKD、FGD/LD/CMDistill；现有行的 init/data/teacher hash 也大量为空。这违反项目自己的证据规则。
2. 协议 YAML 记录的 dataset YAML hash 是本地占位文件 hash，不是 L20 实际路径 YAML hash。
3. 远端 `README_CN.md` 与实际方法目录/运行状态不同步；`ACTIVE_BASELINES.md` 把未完成的 RGB s0 restart2 放在 approved outputs。
4. `scripts/queue_frozen_rgb_comparisons_l20.sh:47-57` 中 `run_tag` 是函数局部变量，监控循环却在 :99 直接引用；在 `set -u` 下首个任务退出时会触发 unbound variable，seed123 队列很可能不会启动。
5. CCLKD AP 有效，但 validation loss 列全程 NaN，应修 validator 诊断输出，不能用这些 loss 曲线判断收敛。

## 8. 测试结果与证明边界

- 本地 `tools/check_workspace.py`、`tools/check_four_methods.py` 通过；
- 本地 LCSR 5 tests 通过、2 个因本机缺 Pillow 跳过；Identity 1 test、RIF 3 tests 通过；
- L20 在禁用 CUDA 的独立 CPU 测试中，LCSR 7 tests、Identity 1 test、RIF 3 tests 全部通过；
- 主协议与上述关键方法源码在本地/L20 的 SHA256 一致。

这些测试证明目录、代数恒等式、forward/backward 与最小 pipeline 可执行，不证明方法的语义可识别性、因果机制或最终 AP 优势。

## 9. 优先处置

### P0：阻止错误结果进入主表

1. 将当前 FGD 标为 `FGD-focal-only`，或补 global relation 后从头重跑。
2. 将当前 legacy LADD 标为 `random-decomposition diagnostic`；正式 LADD 必须从正确前置 phase checkpoint 启动。
3. 将 Identity 当前 run 标为 `all-token projected-feature KD identity control`；不要写 correction/foreground-only claim。
4. 将 LCSR s42 标为 exploratory，补 G0-G3、shuffled/oracle/adapter control 后再决定是否进入正式矩阵。
5. RIF 保持 blocked，不能用 zero-target smoke 或未绑定 provenance 的 cache 放行。

### P1：修复可复现性与队列

1. 修复 queue 监控循环的 `run_tag` 作用域，再验证 seed42→seed123 wave transition。
2. 为每个实际 run 补齐 registry 行、初始化/数据/teacher/code SHA256、effective config 和 canonical result。
3. 明确 protocol 状态为 canonical，并记录 L20 实际 dataset YAML/hash。
4. 修复 CCLKD validation loss NaN 与 `strict_batch_size`。

### P2：性能与命名

1. 向量化 LCSR foreground rasterization，消除 bbox 级 GPU `.item()` 同步。
2. 主表统一使用 `adaptation` / `focal-only` / `exploratory` 等准确方法名。

## 10. Claim impact

| claim | 判定 |
|---|---|
| SAR s0/s42/s123 与 RGB s42/s123 baseline 数字真实、协议匹配 | supported |
| RGB s0 restart2 是正确的干净重跑 | supported，但完成前不是 approved baseline |
| 已运行 detector AP 使用真实 GT，无自归一化 | supported |
| FGD/LD/CMDistill 均是原方法精确复现 | unsupported；FGD 尤其不完整，LD/CMDistill 只能称 adaptation |
| 当前 legacy run 代表历史 LADD | unsupported |
| Identity 对 detector feature 做了有效 correction 且只在前景蒸馏 | unsupported |
| LCSR 已识别语义公共子空间与 SAR 可恢复比例 | unsupported / 尚未验证 |
| RIF 已有可比较的正式实验 | unsupported；目前只有 zero-target proxy smoke |
| 所有启动任务完全遵循证据登记与预声明 gate | unsupported |

## 审计限制

这是代码、配置、日志、结果文件与实时进程层面的审计，不等价于对论文新颖性或最终统计显著性的审稿。活动 run 尚未完成；本报告不根据早期 AP 判断方法优劣。
