# CFT／LLVIP 原论文协议复现（2026-09-09）

**已完成完整权重复评及有限范围独立复核：CFT 作者 checkpoint、官方旧标注、原评估器在 3463 对 LLVIP test 上得到 AP50/AP75/mAP=97.376907/72.891348/63.537478%，与论文三项差均不超过0.13pp。90 单GPU完整进程90.36秒，退出码0；同环境 CPU 重算三个 AP 完全一致。这是作者权重复评，还不是从头训练复现。原B32/1024的三次训练资源检查均在首批前向 OOM、0次成功更新，完整重训未启动；新增迁移版实验暂停。**

## 已执行结果（11:05）

| 指标 | 论文百分数 | 本次百分数 | 差值 pp |
|---|---:|---:|---:|
| AP50 | 97.5 | 97.376907 | −0.123093 |
| AP75 | 72.9 | 72.891348 | −0.008652 |
| AP50–95 | 63.6 | 63.537478 | −0.062522 |

冻结的设置为原CFT两流模型、1024、作者CLI默认eval B64、FP16、conf=.001、NMS IoU=.5、原双流loader的rect/pad=.5及原test.py匹配/AP函数。没有按结果调阈值或换模型；实际重建旧标注全test为7931对象，新版为8302对象，版本不能混用。实际输入为864×1056，与原矩形padding一致。

原B64资源canary完成3批192图：NVML峰14308MiB、整卡最低空闲10033MiB；正式按实测+余量预约15360MiB。正式评估峰11412MiB、进程树RSS约18.07GiB，实际只占一张动态分配GPU3，两个任务顺序执行。运行使用tmux和统一lease，完整官方评价90.363秒、退出0，GPU已释放。

[原始完成回执](server_execution/official_test_attempt1/receipt.json)、[原指标输入](server_execution/official_test_attempt1/author_metric_inputs.npz)、[实际图像名单](server_execution/official_test_attempt1/evaluated_roster.json)、[比较表](server_execution/comparison.json)、[新旧标注审计](server_execution/annotation_audit.json)。作者输出3461份预测txt；240113、240165无预测，但其GT计入分母。独立复核确认3463图、7931GT、55批全部计入，未发现NMS超时截断。

[独立复核](EVALUATION_REVIEW.md)核对了覆盖、预测与指标数组，以及作者AP聚合；没有独立重做框与GT的IoU匹配。本地不同NumPy版本复算最大差0.002965pp，保留warning；[同执行环境CPU复算](server_execution/pinned_cpu_recompute.json)三个AP与执行结果完全相同。原协议与历史代码限制仍保留，不能称三项精确复现或AUTHOR-EXACT。

原始图像不复制到本地或GitHub；小原始结果、失败/成功记录和实际脚本副本已留存。此作者checkpoint存在运行时类名兼容适配，准确历史训练代码/私有YOLO标签未取得，保留PAPER-RECONSTRUCTED身份与[协议审阅](PROTOCOL_REVIEW.md)。公开权重的性能基本对齐，不意味着原训练过程、多seed稳定性或全部论文主张已验证。

论文指标来源：[CFT Table 3](https://arxiv.org/html/2111.00273v2#S4.T3)；执行源码来源：[作者仓库](https://github.com/DocF/multispectral-object-detection)。CFT使用RGB+IR双输入、约2.06亿参数的融合模型，不能直接列作本项目RGB-only部署的同条件蒸馏对手。

## 从头训练资源检查：未准入，未启动E200

已取得作者提供的通用COCO预训练yolov5l.pt（94,562,742字节），使用作者train_rgb_ir、原双流模型及损失/增强/优化器路径。B32、1024、E200来自作者checkpoint命名等公开证据重建，不声称全部超参均有论文明确记载。只补运行环境兼容与诊断包装，未把已训练LLVIP权重用作重新训练的初始化。

| 独立attempt | 执行方式 | 实测结果 | 成功optimizer更新 |
|---|---|---|---:|
| train_canary32_attempt1 | 单卡，总B32；Torch分配上限65% | 首批前向OOM，进程NVML峰16038MiB | 0 |
| train_canary32_attempt2 | 单卡，总B32；按实测框架外占用将Torch上限设68% | 首批前向OOM，进程NVML峰16806MiB | 0 |
| train_canary32_dp_attempt1 | 作者DataParallel双卡，总B32；每卡Torch上限68% | 首批前向OOM；两卡Torch保留峰16570/16572MiB | 0 |

这三次是技术准入失败，**不是方法负结果，也不证明占满24GB或采用其他硬件仍无法训练**。项目显存必须低于单卡70%，所以没有为启动训练而取消既有资源上限。双卡NVML轮询漏采了GPU3短时峰值，不能把该卡采样值6772MiB当实际峰；保留框架峰值和完整OOM日志。各次进程树RSS峰约37.6/39.7/38.9GiB。

完整训练没有启动，尚无可用的每步训练耗时，**不能由90秒评价时间推算E200，也不能给出已经测得的E200时长**。三次失败原件均留存：[汇总](server_training/training_feasibility.json)、[训练目录](server_training/)、[启动包装静态审阅](TRAIN_CANARY_REVIEW.md)。本轮资源lease已全部释放，见[结束快照](resource_closeout.json)。

## 当前排程与下一步边界

1. 本阶段CFT作者权重复评已经完成，表明作者权重在公开协议重建条件下得到接近论文的结果；从头训练仍单列未复现。
2. 暂停新增CMD等迁移到我方YOLO11n/划分的实验。原协议确认前，不用迁移负结果判断原论文或作者实现。
3. Drone优先核对M²D-LIF原模型权重、OBB标签和原评价入口；[原协议队列](DRONE_ORIGINAL_PROTOCOL_QUEUE.md)已完成核查，但作者权重/标签尚未取得，因此尚未启动Drone复评或训练。C2Former同样缺完整资产。此处是待办，不是已排入GPU运行的任务。
4. 后续重训需先有符合资源规则且保持原训练语义的实际可运行路径；若改变硬件并行、微批或模型实现，逐项说明等价证据，不能静默降输入尺寸/模型规模后标原文复现。

## 目的

先验证论文结果能否在作者条件下基本复现，再讨论迁移到 YOLO11n/RGB-only 后的方法效果。迁移负结果不能区分原论文可复现性、实现错误与适用条件变化。原条件未对齐也不能仅凭差异认定造假。

## 执行身份和授权

用户本轮明确授权原文协议/模型复现实验，包含原协议官方 test 复评；覆盖此前本轮只读 fit 的限制。官方 test 暴露单列为作者方法复现，不用于本项目方法选择、阈值调整或与 grouped dev 混比。既有 train/dev/test 划分原件保持不变，另建 author_protocol 数据目录。94 不连接、不恢复。

原模型、官方 train12025/test3463、作者预处理和评价路径优先；数据版本及最小运行时兼容差异逐项记录，不能标 AUTHOR-EXACT。前一轮 checkpoint 类兼容补丁原样沿用并留证，不能把有差异的执行包装写成原环境精确复现。

## 顺序

1. 核对作者论文、评估入口与原始标签转换；冻结模型、1024输入及评价参数。
2. 恢复官方测试集；原 test 图像/标注和当前 grouped dev 分离登记。
3. 通过统一 lease 做原评价 batch 的资源 canary，测量显存和吞吐。
4. 启动完整作者 checkpoint 复评，保存原预测、作者指标及完成回执。
5. 依据原训练配置的实测成本选择重训启动路径，不以改模型/日程冒充原文复现。

资源仍遵 AGENTS，全部新产物写 90 dataX 项目根，长任务 screen/tmux，所有 GPU 调用共用既有 resource guard/lease，不新建资源池、不干扰其他任务。未通过的技术 attempt 保留，结果低不作为早停理由。

本地目录与服务器 `/mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/` 互指。论文/作者协议核查由并行审阅完成，不能用 CPU 小样本冒称完整训练或论文表复现。
