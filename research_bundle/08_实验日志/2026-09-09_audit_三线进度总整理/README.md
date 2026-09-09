# 数据、复现、自研方法三线进度总整理

**当前有系统数据诊断、作者权重复评和 C0 三 seed 小幅信号，但没有完成跨模态四臂归因的自研方法。按用户要求暂停新增实验与自动扩展，先统一梳理三条工作线。**

完整报告：[项目三线进度与证据总整理](../../07_研究分析/项目三线进度与证据总整理_20260909.md)。

## 目的

纠正把数据机会、论文权重复评、训练复现、适配实验、短训和完整方法效果混在一起的进度表述。涵盖 RGB–SAR 与 RGB–IR，不追加实验矩阵。

## 设置与实际操作

本轮按 experiment-audit/analyze-results 工作流，主线程核 RGB–IR 数据与当前方法，两名独立协作者分别核 RGB–SAR 历史和复现线。没有新模型前向、AP 评价、训练探针或输入内容摘要计算；历史大权重及全量数据来源链未重新核验。

94 的有限 CPU 排队器于 22:41 停止，仅停止新任务调度，不中断已运行 trainer。应用 automation 工具返回原任务不存在，本地自动任务目录亦为空；没有重建。首个暂停匹配检查因包含父 shell 而断言失败、未执行停止；修正为精确 Python 进程后只终止 queue，保留训练 tmux。

用户对两项现有训练是否一并停止的澄清尚未回复，因此暂时保留。新队列停止是本次暂停安排，不把已经被取代的旧 campaign RUNNING 字段当作现状。

## 结果

| 项目 | 截止状态 |
|---|---|
| 94，22:46 | C1 seed42 已完成162/200、seed0恢复attempt2完成139/200；新seed123未启动且已暂停。两项尚无新完整AP |
| 90，22:33 | 没有项目进程、tmux或lease；CFT/LLVIP作者权重复评已完成，原协议E200重训未启动 |
| 当前完整自研证据 | Drone N/C0/random各三seed，共九端点；C0−N +0.266655±0.144373pp，C0−random +0.174939±0.038853pp，均3/3正 |
| 分类内容归因 | 旧shuffled/same-modal缺完整端点；C1_y无正式结果，不能声称四臂齐备 |
| 定位与特征 | L1准入阻塞；L2/L3及F仅短训，没有可靠完整效用结论 |
| RGB–SAR | 保留SpaceNet6 OS-SSL外部适配正例；不能概括为无空间。LADD/SX/MM/P3各按具体终态保留负面或受限结果 |

## 结论与局限

主要缺口是将教师条件优势变成学生可学且超过GT/同模态控制的完整效用。复现线仍缺90原协议训练闭环。整体审计WARN；本轮只重新组织、核对既有证据，没有重跑全部历史评价，不认证权重或原图内容未变。

不自动启动、恢复或扩展任何待运行实验。现有两项C1仅保留已有训练与末轮评价责任。

## 产物与互指

- [RGB–SAR独立审阅](independent_rgbsar_review.md)
- [复现线独立审阅](independent_reproduction_review.md)
- [94状态快照](status94_20260909_review.json)
- [停止新增队列回执](pause_new_work_20260909_review.json)
- [审计判定](EXPERIMENT_AUDIT.md)及对应JSON
- [三seed数值摘录与复核](paired_statistics.json)，来源由同名脚本记录；只复核现存端点算术，不生成新模型指标

94暂停原件：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/c1_budget10h_20260909/pause_new_work_20260909_review.json`。本次审阅镜像安排在`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/project_three_track_review_20260909/`，是否同步完成以同步回执为准。

本地：`E:/SHARE/光sar/08_实验日志/2026-09-09_audit_三线进度总整理/`。公开资料沿用GitHub分支`research/full-evidence-20260906`，只同步审阅、小结果和索引，不包括大模型、数据集原图或凭据。
