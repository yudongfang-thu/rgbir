# 复现实验线独立复核（2026-09-09）

**复现线已取得三次作者发布权重的完整LLVIP官方test复评，并有CMDistill、CCLKD partial等历史YOLO11n适配端点；尚未取得本轮CFT/LLVIP作者模型从通用预训练初始化重训的完整端点。90在22:33没有本项目训练进程、tmux或资源租约，不能写成正在训练。**

本文件只读复核已保存的来源、执行回执、日志、指标及部分代码，服务于“三线进度总整理”。不启动SSH、GPU、新实验或重算hash，不改原始实验文件、索引或Git。以下AP为百分数，差值为百分点（pp），三seed SD为样本标准差（ddof=1）。

| 审阅元数据 | 记录 |
|---|---|
| date | 2026-09-09；最新服务器状态证据为22:33只读快照 |
| auditor | `/root/audit_reproduction_line`，与总整理执行者分开的审阅代理 |
| model_identity | 本次工具未暴露可独立核验的具体模型身份；不声称跨模型独立 |
| overall_verdict / integrity_status | `warn` / `warn` |
| reason_code | `EXECUTED_RESULTS_EXIST_WITH_PROTOCOL_AND_HISTORICAL_SOURCE_LIMITS` |
| 来源版本 | 以现存路径、run/attempt、record字段及日期识别；遵任务边界未计算新摘要，未保证检查后内容永久不变 |

## 1. 方法×数据×实际完成范围

“可比”分为同任务描述性参照和严格同条件净效应；前者不能自动升级成后者。

| 方法/身份 | 数据与模型 | 实际完成内容 | 未完成项 | 与当前YOLO11n RGB-only HBB主线直接比较 |
|---|---|---|---|---|
| **CFT作者权重复评，PAPER-RECONSTRUCTED** | LLVIP official test3463对、previous标注7931GT；RGB+IR两流融合，约206.2M参数；1024、eval B64、FP16 | 原作者test.py/AP路径完整复评，97.376907 AP50 /72.891348 AP75 /63.537478 mAP；90.363秒，COMPLETED。原始指标数组、预测及roster已保存 | 原训练源码/标签历史同一性未完全证明；24个融合block有显式类兼容重绑；无完整重训、多seed训练或整篇论文主张复现 | **否**：双输入、模型规模、官方test及历史标注不同。不能称RGB-only蒸馏对手 |
| **LLVIP作者RGB/IR单模态权重复评，PAPER-RECONSTRUCTED** | 两个作者YOLOv5l发布模型，各约46.63M参数；official test3463、previous7931GT；1280、eval B32、FP32、作者val.py | RGB AP50/AP75/mAP=90.788183/56.342893/52.664239；IR=96.378278/76.420674/67.014908；完整回执COMPLETED，90.696/88.061秒 | 未恢复作者历史训练全过程；未完成本地重训；论文版本差异及旧环境兼容需保留 | **否**：RGB虽单模态，仍不是YOLO11n/640/grouped dev；只能作为作者模型复评与模态能力背景 |
| **CFT原训练重建/资源canary** | 原两流模型、COCO通用初始化、B32/1024/E200目标 | 单卡两attempt、双卡一attempt，随后三卡一attempt，均首批前向OOM、0次成功更新；三卡11/11/10分批与AMP生效有诊断 | 未通过资源准入，未启动正式E200；无训练效果与稳定吞吐 | **否**：技术失败不是方法负结果；也未证明完整24GB或其他硬件无法运行 |
| **LLVIP作者单模态原训练重建/资源canary** | YOLOv5l、COCO通用初始化、B8/1280、目标E200 | CPU初始化/loss兼容预检；attempt1是Path序列化错误0更新，attempt2实际10次SGD更新、0 AMP skip，第11批backward OOM | 24更新准入目标未达到，E200未启动；历史初始化版本与完整配方仍有缺口 | **否**；早期6批机械外推80.27h不构成完整训练实测 |
| **CMDistill-corrected历史协议适配** | YOLO11n RGB学生+冻结IR教师；Drone train17990/val1469、LLVIP grouped fit9619/dev2406；640/B32/nbs64/E200 | 两数据集各3seed完整训练和独立last指标；训练receipt标PROTOCOL-ADAPTED。Drone每seed56722更新、LLVIP30326更新 | 原作者YOLOv5s链未逐项重现；LLVIP缺匹配三seed专用native；当前N/C0执行流与历史trainer不完全等价；四臂归因不齐 | **有条件描述性可比**：同RGB-only HBB任务，必须标适配版与各自native；不能直接拿历史workers8结果减当前workers4的N/C0作净收益 |
| **CMDistill新90接线** | 独立来源目录，拟接已治理Drone/LLVIP；真实IR教师仍TO_BIND | CLI help成功、既有22个CPU单测通过；回执明确synthetic tests only、gpu_used=false | 真实教师绑定、真实batch KD/冻结检查、资源canary、完整训练 | **否**：没有新检测AP；不能把历史94六端点写成90新结果 |
| **CCLKD adapted partial，LLD+CCL** | Drone RGB-only YOLO11n，实际640/B32/nbs64/E200/workers8；固定IR教师 | 三seed历史训练各56722更新；9月6日补完整1469图固定last/EMA独立评估，54.297299±0.215820 mAP | FLD/RLD未启用；历史v2完整执行源码未绑定；缺本版weight0和三seedshuffled/same-modal | **有条件描述性可比**；不是完整CCLKD。当前OEv1还直接使用IR GT，须注明额外监督差异 |
| **HNEWA-CMKD-MSE-inspired** | Drone/LLVIP；六通道RGB+IR融合教师→RGB YOLO11n学生，P3–P5全图MSE；E200/B32/640 | paired/shuffled/same-modal均有0/42/123独立last；自身native只有0/123来源闭合 | 原Hnewa作者条件未原样复现；native42端点不清，不能拼他campaign补齐；强自模态优势不稳 | **有条件描述性可比**，须保留inspired身份及融合教师；不是作者完整方法复现 |
| **OS-SSL-IR迁移探索** | Drone；RGB/IR图像自监督预训练→RGB YOLO11n检测；每臂SSL仅seed42一份，10000steps/B32，再E200微调 | 三个SSL预训练完成；9个微调中4个完成（P123/S0/S123/IR-only42）；9月6日因优先级暂停，IR-only0保留第6轮。P123/S123有独立last评估 | 其余微调未完成；同模板零SSL native、RGB-only SSL缺失；旧pyc/预训练源和评估执行回执不完整 | **不能作严格净效应比较**：增加SSL阶段、非骨干起点混杂、非完整三seed；IR-only不是RGB自模态增强 |
| **BCKD的BCDL分类算子** | 作者分类函数摘取；CPU小张量，不是检测数据实验 | 6/6检查通过，含loss/学生梯度、教师隔离、差值权重不可detach；回执明确full_BCKD_tested=false | 完整head、定位分支、MMCV weighted reduction、检测训练与AP | **否**：算子核验不是BCKD论文复现或数据集基线 |
| **M²D-LIF / C2Former** | 原方法是RGB+IR融合；Drone原OBB协议，不是当前HBB学生 | M²D作者完整源码包取得，原入口/参数/类别/OBB路径核查；C2有源码/config核查 | M²D作者融合权重及标签未取得；C2 Drone最终权重与matched标签清单未闭合；无本项目模型运行/AP | **否**：仅资产/协议准备。M²D发布权重复评无需教师，重训才需双教师；不能将dv_ir.pt同时填两位教师的入口直接视为正确重训 |

AMFD、ICAFusion、CrossFusionKD等在本轮限定证据内属于文献/资产审查对象，不计为已完成数据集复现。AMFD的“single-stream”仍读取RGB+IR，不能因此放进RGB-only对比表。FGD、LD旧实现仅FGD-like/LD-style，不把缺Global或VLR的实现改名为完整作者方法。

## 2. 抽查得到的关键事实

### 作者复评：有实际执行证据，但不是训练过程复现

CFT的`official_test_attempt1/receipt.json`记录`ap_computed=true`、`training_started=false`、`split=official_test`和`status=COMPLETED`；实际wrapper第140–149行捕获作者AP输入后调用原`author_test.test`。本次本地再次计数，保存的roster为3463项且全部唯一，预测txt为3461份；无预测图不应从GT分母移除。既有同执行环境`pinned_cpu_recompute.json`记录7931GT、18275预测，三项AP与执行值差0；本审阅没有重新执行这一AP重算或重做IoU匹配。

LLVIP可见光/红外两份原receipt均记录3463图、7931GT和COMPLETED；指标小数乘100与上表一致。执行wrapper捕获原AP函数和数据迭代后调用`author_val.run`，不是从论文表手填指标。已有`LLVIP_EVALUATION_REVIEW.json`保存44项数值/覆盖核查并保留WARN：不重新匹配box/GT、源文件大小不等于源码同一性证明、执行者复用上下文及device选择副作用。不能把原复核写成盲审或跨模型审阅。

两份LLVIP原冻结receipt仍保留初版v1参照（RGB mAP50.0、IR61.9）；现README另列修订版52.7/67.0。这里应写“接近修订版公开结果”，不是事后把v1的差值解释为复现增益。CFT的1024/FP16/NMS .5及匹配器与LLVIP单模态1280/FP32/NMS .6不同，不能用63.54对67.01推断融合优劣。

### 从头重训：执行失败的类型已经可分清

CFT三卡原`failure.json`为`RESOURCE_FAILURE_OOM`、`batches_entered=1`、`successful_optimizer_steps=0`；并非因AP低而停止。LLVIP attempt2的`failure.json`保存10次非跳过更新及第11批OOM，失败时仍有6.34GiB物理空闲，受0.68 allocator预算限制。因此只能判断当前受限执行路径未准入，不能宣称原模型在3090上绝对无法训练。

最新`status_20260909_2233/status.json`原件：`project_processes=[]`、`tmux_sessions=[]`、`project_lease_count=0`、`full_retraining_started=false`。该快照支持“90当前空队列，原协议重训待解决”，不支持“90仍在后台重训”或“等待任务自动完成”。

### 适配版检测端点：原值与口径核对

本次从原始指标JSON/来源容器重新提取数值，并独立按同seed作差及ddof=1聚合，得到：

| Drone独立last mAP | seed0 | seed42 | seed123 | mean±SD |
|---|---:|---:|---:|---:|
| 历史native | 54.357648 | 53.815556 | 53.687433 | 53.953546±0.355778 |
| CMD corrected | 53.900286 | 53.244840 | 53.669168 | 53.604765±0.332435 |
| CMD−历史native pp | −0.457363 | −0.570716 | −0.018264 | −0.348781±0.291793，0/3正 |
| CCLKD partial | 54.066006 | 54.493295 | 54.332596 | 54.297299±0.215820 |
| CCLKD−历史native pp | −0.291643 | +0.677738 | +0.645163 | +0.343753±0.550510，2/3正 |

CMD的双数据集六份completion receipt均是`completed`，含实际optimizer更新和非零KD记录；六份指标均指向各自`weights/last.pt`。LLVIP三个指标为34.060274/33.443988/33.316849，均值33.607037±0.397629；`split=val`仅是YAML槽名，`evaluation_role=dev`，不可混作官方test。缺少其匹配native三seed，不能补造净差。

CCLKD三份`evaluation_val.json`均为完整1469图、非canary、fixed_budget_last_ema；本次逐份roster计数与唯一计数都是1469。抽查seed42新评估receipt，明确`historical_training_source_not_fully_bound=true`，来源快照绑定此次eval而不是旧训练。旧训练receipt只列LLD/CCL，不能被新评估的完整回执“补成”完整CCLKD。

**特别容易误读的配置：** 同资产包`raw/SpaceNet6_OTD_official_reproduction/configs/research/rgbt_cclkd_protocol_drone.yaml`写512/B16/mosaic1；已执行`raw/log_excerpts/cclkd_dr_s42_v2_head.txt`第2行明确640/B32/nbs64/mosaic0/workers8。应按日志描述历史实际设置，不能从邻近YAML推定执行身份。

HNEWA-inspired原始CSV抽查能对应paired的三seedlast端点，另有`_best`及`/tmp/best_probe/last.pt`标为provenance_check_needed；不能填入native42空格。已有逐seed分析给出Drone paired−shuffled +0.3360±0.2894pp（3/3正），paired−same-modal +0.2068±0.3597pp（2/3正）；这不是旧“+12.7pp”。本审阅未全量重算该方法的全部指标/预测。

OS-SSL P123/S123原`metrics_record.json`分别为53.932152/53.223747 mAP，差+0.708405pp；对应AP50差+0.533024pp，仅一组微调同seed比较。训练completion receipt里的`arm=native_weight0`指检测阶段不加KD，实际SSL臂须由`protocol_clean_paired/shuffled.yaml`与初始化一起识别，不能据字段把它们当成零SSL native。后续停止回执记录IR-only0第6轮、目标残留PID为空、没有正式完成回执；4/9完成是停止时最新状态，较22:28的3/9快照更新。

## 3. 审计判定与应修正的进度表述

| check / claim | 判定 | 影响 |
|---|---|---|
| gt_provenance | warn | 作者复评使用previous人工GT；适配检测使用真实开发GT。未独立重放原图→标注→匹配全部链路 |
| score_normalization | pass（本次范围） | 指标0–1转百分数、同seed差值和样本SD算术通过；作者AP独立重算仅查已有回执，不冒称本轮重算 |
| result_existence | pass（列出的端点） | 完成回执、原指标、部分roster及实际代码存在；失败attempt、停止记录与正式端点分列 |
| dead_code | warn | CFT/LLVIP有实际原评估器调用与捕获；BCDL及新90 CMD仅CPU作用域；历史CCLKD/SSL源码链不能补全 |
| scope | warn | 原论文权重复评≠从头训练、多seed完整原模型复现≠迁移版收益；90最新没有训练运行 |
| eval_type | pass | 检测AP=`real_gt`；BCKD/CMD CPU小检查=`synthetic_proxy`工程核验；OS-SSL预训练=`self_supervised_proxy`，下游检测AP仍为real_gt |
| R1：CFT/LLVIP作者权重性能已基本核对 | supported，需限定为公开协议重建与修订论文版本 | 可报告本次复评分数，保留PAPER-RECONSTRUCTED/WARN |
| R2：本轮作者原模型已完整重训复现 | unsupported | 当前只有资源canary失败；正式E200未启动 |
| R3：CMD适配在Drone相对自身历史native的mAP三seed均负 | supported于限定比较 | 不支持否定作者论文、所有CMD实现或所有选择性KD |
| R4：CCLKD完整论文已复现且稳定增益 | unsupported | 只有partial，2/3正且缺对照；新评估不能修复历史源码缺口 |
| R5：OS-SSL已通过三seed归因或仍在运行 | unsupported | 矩阵4/9完成后暂停；初始化与RGB自模态对照未补 |
| R6：新90已复现四种方法并形成新AP榜 | unsupported | 三个作者模型复评分开计；CPU算子、接口与资产准备不能计为检测方法复现 |

总整理必须覆盖旧状态：CFT不再仅“3图前向无AP”；CCLKD不再“缺独立端点”；CMD双数据集六端点已完成；OS-SSL不再“3/9、正在继续”；90现在无重训运行。证据总表开头已有日期覆盖声明，正文“CMD seed123 e138/200”“FreqMix+9”等属于保留的历史记录，不应再按当前事实引用。

## 4. 关键原件路径与范围限制

以下为本地审阅入口，服务器路径只记录，不代表本次已SSH验证：

1. `08_实验日志/2026-09-09_repro_CFT原文协议/server_execution/official_test_attempt1/{receipt.json,executed_wrapper.py,evaluated_roster.json,author_metric_inputs.npz}`；同级父目录`pinned_cpu_recompute.json`；`server_training/training_feasibility.json`。
2. `08_实验日志/2026-09-09_repro_90原协议继续/llvip_execution/llvip_{visible,infrared}_full_attempt1/{receipt.json,executed_wrapper.py}`；`LLVIP_EVALUATION_REVIEW.json`；`llvip_training_execution/llvip_train_canary_attempt2/failure.json`；`cft_dp3_execution/cft_dp3_canary_attempt1/failure.json`；`status_20260909_2233/status.json`。
3. `08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_sources.json`内`files[*].text`：CMD六份完成/指标及native原记录；`08_实验日志/2026-09-05_audit_跨模态蒸馏全项目复盘/rgbt_eval_and_code_snapshot.json`、`rgbt_eval_rows.csv`。
4. `08_实验日志/2026-09-06_ops_OEv1优先级与对比实验/cclkd/eval_raw/runs/oev1_comparators_20260906/cclkd_partial_s{0,42,123}_full_attempt1/{evaluation_val.json,evaluation_roster.txt,eval_evidence/run_receipt.json}`；同`cclkd/`下`raw/log_excerpts/`、历史完成receipt、`historical_native/`。
5. `08_实验日志/2026-09-06_audit_实验全景与设置对照/osssl_new_eval/raw/runs/osssl_ir_20260906/{paired_rgb_s123_e200,shuffled_rgb_s123_e200}/{metrics_record.json,completion_receipt.json}`；9月6日ops目录`stop_receipt.json`、`final_status.json`。
6. `08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/existing_assets/{cmd90_port/cmd_cpu_receipt_attempt1.json,bckd_operator/CPU_RECEIPT_90_attempt1.json}`；`drone_research/`与`08_实验日志/2026-09-09_repro_CFT原文协议/DRONE_ORIGINAL_PROTOCOL_QUEUE.md`。
7. 90本轮根：`/mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/{cft_author_protocol_20260909_attempt1,original_repro_continue_20260909_attempt1,reproduction_20260909_attempt1}/`；94历史模型根：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/{rgbt_cmdistill_adapted_v2,rgbt_cclkd_adapted_v1,rgbt_hnewa_cmkd_mse_inspired_v1,osssl_ir_20260906}/`。

旧`rgbt_cmdistill_paper_reconstructed_v2`目录名也不能当原文学生训练完成证据：抽查9月4日data_first_repro_v2 tracker实际讨论VEDAI official_fold01教师/native与20步canary（1089/121），不同于证据总表旧行的Drone标签。该时间快照不能推定今天仍在训，也不补出不存在于本次证据集的完整CMD学生端点。

OGSOD/SiXiang、SpaceNet6 OTD/OS-SSL属于历史任务层；其中SN6 OS-SSL归入外部方法的适配/复现实验，不属于本项目原创方法贡献。本次读取总表及历史说明辨别其边界，没有重新审计这些远端完整结果或原论文协议，因此其“作者原模型完整复现”状态在本审阅中为`unavailable`，不作通过认证。历史SpaceNet6的OS-SSL正例不能代替Drone上的迁移归因，SAR部署任务也不能直接与RGB-only HBB主线排榜。

本次对文件存在、执行字段、部分实际代码与日志、三份CCLKD roster及关键算术做了独立检查；没有加载checkpoint、运行作者模型、重新生成预测、重算作者框匹配/AP或全面比较每份训练源码。来源审阅的独立性不等于跨模型独立，也不等于无任何未发现缺陷。此文件新增结论仅用于进度与证据分级，不升级论文claim。

## 5. 主报告第二部分限定回审

审阅对象：`07_研究分析/项目三线进度与证据总整理_20260909.md`，本次读到的第二部分为第109–155行。仅与上文已经核对的本地证据对照，没有扩大检索或改主报告。

**实质数值与当前状态没有发现必改错误。** CFT、LLVIP RGB/IR的mAP与CFT AP50/AP75取舍正确；约90秒是评价耗时而非训练耗时；PAPER-RECONSTRUCTED身份、CFT 0更新及LLVIP 10更新失败、allocator边界、90在22:33没有项目训练/排队状态均准确。历史CMD六端点、CCLKD partial模块/实际640/B32配置、HNEWA两个native端点限制、OS-SSL 4/9后暂停均与已查证据一致。第三小节正确拒绝用历史workers8 CMD/CCLKD直接减当前workers4 N/C0作算法净效应，并注明C0直接用IR GT。

交给主报告作者的两项必须收窄/修正之处：

1. **第155行限定主语为“本轮RGBIR外部方法”。** 原句“目前还没有……完整两阶段证据链”在同一节已列SN6原公开实现R1与YOLO OS-SSL v2之后，容易被读为整个项目均未完成过原实现→迁移这条线；本复现审阅没有核验这个全项目否定。建议改为：“本轮RGBIR外部方法尚未形成‘按原论文完整重训，再迁移到当前YOLO11n RGB-only HBB协议并与匹配N比较’的完整证据链。”这样与当前范围和证据一致。
2. **第143行按审阅责任分配证据链接。** 当前把整张历史表的“完整证据路径、逐seed表和复现身份”统一指向本文件，但本文件明确未独立重审SN6原实现和YOLO OS-SSL原始结果。建议改为：“SN6两行的原值及范围见RGB–SAR独立审阅；其余RGBIR复现身份、逐seed表和执行原件见复现线独立审阅。”SN6数值是否最终接受由实际读取其原结果的审阅者确认，本次不补作认证。

这两项是范围与证据归属修订，不要求改变已核对的RGBIR数值，也不要求新实验。具体模型身份仍不可独立核验，不声称跨模型审阅。主报告修订由父任务完成，本代理只在此留回审记录。
