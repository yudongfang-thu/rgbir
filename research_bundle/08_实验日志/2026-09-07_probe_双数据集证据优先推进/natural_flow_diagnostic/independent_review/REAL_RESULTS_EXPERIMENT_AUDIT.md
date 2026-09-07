# 真实自然流结果独立验收

**PASS_FOR_OBSERVED_FIXED_FLOW_DIAGNOSTIC：LLVIP 与 Drone 各 2-batch canary 和 64-batch full 均有真实完成记录，独立 CPU 复核通过。** 每组 full 为固定 seed 20260907、64×32=2048 个无替换自然源图。该验收只支持本次冻结选择诊断，未准入几何、校准、L 训练或 KD 收益。

审阅者 `/root/ap_error`，实现/汇总作者 `/root/baseline_feature_analysis`，94 执行者 `/root`。本审阅使用 experiment-audit 技能，未访问 GPU/SSH、未重推理、未计算新 hash；用户禁止 hash 的指令优先，使用 path/size/mtime 与源码字节副本。机器入口为 `REAL_RESULTS_EXPERIMENT_AUDIT.json`，表格独立验收追加于 `real_results_v1/readout_tables_review.json`。

## 已核实的实际结果

| 数据集 | RGB GT | C base / eligible / selected | C非空批 / 图 / 来源组 | L base / selected | L非空批 / 图 / 来源组 |
|---|---:|---:|---:|---:|---:|
| LLVIP | 5201 | 5128 / 4406 / 2217 | 64 / 1317 / 14 | 5033 / 207 | 62 / 187 / 14 |
| Drone | 30197 | 28297 / 15616 / 7821 | 64 / 1317 / 64 | 21488 / 602 | 63 / 268 / 31 |

C 的选中数为各真实 batch 独立 `ceil(rho×eligible)` 后相加；因此不必等于所有 eligible 总数的一半。全 batch 配额和稳定 quality 排序已逐批重算，未使用逐图 C 选择。C0/C1 在本诊断共享同一原选择集合，R 复用 student 槽没有新学生参数。

L selected/base 为 LLVIP 4.11%、Drone 2.80%。最大累计对象流失发生在 R 的定位缺口条件：both reliable 的 4731/18614 对象，在要求 R IoU<0.70 后分别余 278/744；再经原 teacher IoU 与 margin 条件选中 207/602。R 是固定旧 E200 seed42 RGB 辅助模型，不能把这条分布归给本阶段新训练的 N42。

Drone 各 GT 类别计数如下，规则对五类完全相同。

| 类别 | C base | C eligible | C selected | L base | L selected |
|---|---:|---:|---:|---:|---:|
| car | 24226 | 13530 | 6935 | 18328 | 561 |
| freight car | 905 | 370 | 179 | 702 | 9 |
| truck | 1270 | 593 | 320 | 956 | 21 |
| bus | 1221 | 822 | 247 | 964 | 6 |
| van | 675 | 301 | 140 | 538 | 5 |

L 的选中对象中 car 占 93.19%；Drone 原来源组标签 `B` 占 C selected 的 49.83%、L selected 的 60.47%。这些 `B/image/modalA` 等原治理标签粒度不等，不能视为同等粒度的几何标定组或独立场景。非空 batch 数表示有对象被选择，不表示非零真实共享梯度，也不证明少数类 AP 瓶颈被覆盖或可学。定位选中 anchor 的 P3/P4 为 LLVIP 61/146、Drone 549/53；分类 P3/P4 是共同有效层，可同时有效，不是互斥尺寸档。

## 独立复核覆盖

`real_results_v1/result.json` 记录了 132 个真实 batch（两组各 2+64），耗时 109.29 秒 CPU，torch 1.8.0+cu111、CUDA 未初始化。逐项核验包括：

- 四个 run 的 11 份真实源码副本分别与已审源码/原 release 直接 byte identity；根 129 文件完整收集回执为 `COLLECTED_BYTE_EXACT`，本地逐文件 size 再核。真实运行环境版本与初始导入 preflight 另见根记录（torch 2.10.0+cu128、Ultralytics 8.4.115）。
- 每批 source、增强、RGB/IR 标签、位置、shape/dtype 等字段与上一阶段原 64 流完全一致；full 的两组 loader 元数据与原覆盖回执逐字段一致。canary 前 2 批的完整 selection 记录与 full 前 2 批也完全相等。旧覆盖没有像素字节，因此不声称像素 byte identity。
- 原 LLVIP `fit.tsv` 与 Drone `rgb_train_source_groups.tsv` 对照全部 4096 full 图，group 真值完全一致，无 fallback、无未知组；不只检验内部汇总闭合。来源路径、size、mtime 见 `../governance_group_sources/receipt.json`。
- C 逐批 base/eligible/selected、global GT 与 class、原 stable quality 排序、ceil 配额、whole-batch normalizer、图/组/类别均重算。L 所有 per-image gate 到 batch/summary 全量重积，base 记录的可靠性/IoU/margin 后段 gate 与 selected 身份逐条重算；原整 batch normalizer 保留。
- L 的 GT 决定前段（RGB/IR GT、common、pair IoU、geometry、inside、support、unique owner）用真实双标签与固定 640 格点在 CPU 原函数逐批重算。模型 scores 置无候选只用于隔离这部分，不产生预测、结果或新阈值。
- 作者 7 份 CSV 共 15060 行、90780 个单元从原 full JSONL 独立重建并核对，未调用作者汇总函数。覆盖 overview、全部 gate/class/level/stride/concentration，以及 14974 行完整 image/group histogram。

资源四个回执均 COMPLETED、exit0、monitor 无错误。四卡例外各有原 admission，占用后均至少留 2 张完全空卡；cuda 进程数回执为每 GPU 1。所有采样中最小 GPU 空闲为 13269 MiB，最大 project RSS 为 151679 MiB，分别满足 2048 MiB 余量与 300 GiB 上限。此为原监测采样证据，不声称外部采样覆盖每个瞬时峰值。full 预约来自各自 canary 实测并加余量，未用理论估算代替实测。

## 证据边界与失败保留

所有 anchor 的真实 T/R logits 未保存，因此离线审阅无法独立重新计算 reference-candidate 是否存在、最高置信 anchor，或分类 quality 的原始模型值。这部分依赖原 frozen 模型前向、已审定入口和真实逐图/整 batch exact 断言；审阅没有伪称独立重新前向通过。`geometry=None,false` 可改变候选 anchor，不保证最终 selected_count 严格上界。

首轮远端 attempt1 的导入冲突在 0 batch 失败，完整保留，修复与独立 19 项 CPU 回归见 `ATTEMPT2_IMPORT_REVIEW.json/md`。最终审阅归档时普通 Windows 路径无法 stat 一个超过 MAX_PATH 的已收集嵌套源码；首次脚本/错误另存 `real_results_v1/finalize_initial_source.py` 和 `finalize_initial_failure.json`，只改用 Windows extended path 读取。真实统计、原始源码/输出和判断标准未改。

机器产物：`real_results_v1/result.json`、`governance_and_metadata_review.json`、`readout_tables_review.json`。复核入口源码：`review_real_natural.py`、`finalize_real_review.py`、`review_readout_tables.py`。执行输入为 `../remote_completed_attempt2/`，作者读出为 `../completed_readout_attempt2/`。各命令采用新输出/独占写入，保留已有 receipt。
