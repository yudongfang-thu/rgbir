# RGBIR 实施起点审计（2026-09-07）

> **C内容对照接线审查**：[shuffled / same-modal 独立复核](content_control_wiring/README.md)。发现的回执身份与正式canary准入缺口已修正；两臂实际TaskCriterion的CPU接口合成检查通过，未发现阻断24更新canary的接线问题。此为实施复核，不代表真实模型/GPU canary已通过。

> 分析器v2更新：实际receipt绑定recipe比较与launch协议交叉核对完成，20项CPU测试通过；旧快照重算结论不变，详见[复核回应](ANALYZER_V2_REVIEW_RESPONSE.md)。下文18项是首版验证记录，原报告与源码保留。

> **02:56真实Loader验收通过**：[CPU等价与矩阵追踪](loader_equivalence/README.md)。release_v4在Drone/LLVIP各3个train样本的原字段、标签、collate、CPU RNG完全一致；80个输出框投影最大误差7.0572e-5px。未用GPU，不替代训练canary或物理配准证据。

> **02:23快照：OEv1 C42−同代码N42为+0.144554 mAP pp、−0.327243 AP75 pp，仅1/3完整配对。九个N/C/random端点中4个独立评估完成，另5个正常训练。约+1pp的历史参照不能充当严格净增益。**

## 目的
冻结实施起点，核对三 seed 结果、真实配对增益、已有对比方法的可比性，并验证结果分析器。

## 设置
同代码 OEv1 paired=C / weight0=N / paired_random=R，seeds0/42/123，固定last/EMA开发val独立评估。Drone train17990/val1469，YOLO11n、E200、640、batch32/nbs64/workers4。原始指标0–1，下表乘100，差值单位pp。先创建本README，再只读采集94，没有启动/停止/修改任务，没有调用会刷新lease的guard inspect，没有访问test或计算哈希。

## 结果
| 臂 | seed | mAP50–95 | AP50 | AP75 | 独立回执 |
|---|---:|---:|---:|---:|---|
| N | 0 | 54.346185 | 76.992515 | 63.932917 | 完整 |
| N | 42 | 54.513608 | 77.179654 | 64.166566 | 完整 |
| C | 42 | 54.658162 | 77.069620 | 63.839323 | 完整 |
| C | 123 | 54.636847 | 77.304557 | 64.214774 | 完整 |
| C−N | 42 | **+0.144554** | **−0.110034** | **−0.327243** | 单seed描述 |

四个端点的完整训练/eval回执、绑定metric snapshot、seed/arm/checkpoint与1469张有序roster通过新分析器检查。C42的precision/recall对N42差为−1.215073/−0.493206pp。不能从mAP略升直接宣称所有错误类型改善。

### 运行状态与ETA（02:23:37 +08:00）

| 任务 | 已完成轮次 | 当前轮次 | 最近每轮秒 | 剩余训练估计 |
|---|---:|---:|---:|---:|
| C0 | 104 | 105 | 216.15 | 5.76小时 |
| N123 | 98 | 99 | 212.25 | 6.01小时 |
| R42 | 44 | 45 | 251.96 | 10.92小时 |
| R0 | 34 | 35 | 251.16 | 11.58小时 |
| R123 | 16 | 17 | 200.08 | 10.23小时 |

progress.epoch是当前进入的轮次；CSV是已完成轮次。ETA仅按最近约10轮中位耗时估计，不含资源等待和独立评估。原snapshot中完成任务旧ETA约一轮，是collector跳过val=False最终缺列CSV行所致；**以derived/progress_eta.json的completion优先修正为准**，原件未覆盖。CSV不充当独立结果。

### 资源

5个绑定lease、GPU2/4/5/6共4卡；GPU2=R42+R0双开，GPU4=R123，GPU5=C0，GPU6=N123。GPU1/3/7完全空卡，GPU0属于其他使用者。项目进程树RSS140.33GiB（RSS可能重复计共享页），预约180224MiB=176GiB。GPU2已用15303MiB、余8781MiB；其他项目单任务卡约7654MiB、余16430MiB。单任务guard峰值约7632MiB、RSS约28.2GiB。四卡符合AGENTS放宽条款，仍有三张完全空卡；任何新增任务必须由root重新核验和实测。

旧random队列仍负责R42训练及独立评估；未发生的退出不预写为正常退役。

### 分析器与历史对比

新模块`analyze_results.py/test_analyze_results.py`完成18项CPU真值fixture：固定三seed顺序、ddof1、负seed保留、单位换算、缺失/FAILED receipt、metric/roster/ckpt不一致、异协议不合并、禁止多attempt择优及pilot阈值。不自动升级主张，不停止已运行实验；待root独立审阅接受。

pilot仅在完整CL/C/CGT seed42、实施校验通过、CL−C≥.3pp且CL−CGT>0时允许扩展。已有历史比较复核见[COMPARATORS.md](COMPARATORS.md)，旧定位probe勘误见[PROBE_ERRATA.md](PROBE_ERRATA.md)。

## 产物路径
94：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/`。

- `snapshots/2026-09-07T022337.461405_0800/`：114个原始小产物、来源manifest、进程/lease/GPU快照；`derived/`为端点、资源、ETA重算。
- `comparator_snapshot/`：02:30重新采集54个历史对比小产物；`comparator_analysis.json`为本轮CPU重算。
- 采集脚本：`collect_readonly.py/collect_comparators.py`；重算脚本：`build_audit.py/compare_existing.py`。build拒绝覆盖已有derived；再次复算用分析器CLI输出新文件。

## 局限与下一步
保留5个在跑任务直至完整端点评估，按批准计划实现L/GT并先完成几何、D2、canary。当前没有新增定位结果，不预写L成功；分析器待独立review。历史native/CMD不具备新合同全部回执，保持历史适配参照身份。

