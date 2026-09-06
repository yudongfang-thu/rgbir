# CCLKD partial：三 seed 独立 last 评估（2026-09-06 23:20–23:21）

> 三个历史 CCLKD partial（仅 LLD+CCL）端点已补完统一独立验证，mAP=54.297±0.216；相对历史同seed native的描述性差为+0.344±0.551个百分点，逐seed为负/正/正，尚非稳定三seed收益。评估回执已完整保存，不能据此追认历史训练源码完全绑定。

## 设置与方法身份

复用9月3日已完成的三个 `cclkd_drone_seed{0,42,123}_b32_e200/weights/last.pt`，不重新训练，不选best，不修改checkpoint。实际历史训练为YOLO11n、RGB学生/冻结IR教师、640、E200、batch32/nbs64、SGD、弱几何增强；直接KD项仅LLD+CCL，FLD/RLD未启用。更完整历史设置与局限见同目录README。

当次独立评估为 `split=val`、640、batch32、workers4，同一1469张Drone RGB验证图像。环境固定Ultralytics8.4.115、PyTorch2.10.0+cu128、Python3.10.20。指标单位原始JSON为0到1，下表乘100；SD采用三seed样本标准差（ddof=1）。

## 原始独立端点

| seed | mAP50–95 | AP50 | AP75 | precision | recall |
|---|---:|---:|---:|---:|---:|
| 0 | 54.066006 | 76.485618 | 63.281834 | 77.236983 | 72.932142 |
| 42 | 54.493295 | 77.003218 | 63.549107 | 76.629545 | 72.777405 |
| 123 | 54.332596 | 76.312105 | 63.793740 | 76.551846 | 73.087270 |
| mean±SD | **54.297299±0.215820** | **76.600314±0.359549** | **63.541561±0.256036** | — | — |

## 相对历史native的逐seed描述性比较

历史native原始独立JSON来自 `runs/cgkd_w1/native_rgb_s{seed}_e200/metrics_record.json`，固定last、同Drone val；小文件副本已保存在 `historical_native/`。该native与CCLKD主要训练预算、通用预训练与451/499张量加载路径一致，但不是此次新运行的CCLKD同代码weight0。

| seed | CCLKD partial mAP | 历史native mAP | 差值pp | AP50差值pp | AP75差值pp |
|---|---:|---:|---:|---:|---:|
| 0 | 54.066006 | 54.357648 | **−0.291643** | −0.262784 | −0.559558 |
| 42 | 54.493295 | 53.815556 | **+0.677738** | +1.002080 | +0.982645 |
| 123 | 54.332596 | 53.687433 | **+0.645163** | +0.581962 | +0.487416 |
| mean±SD | 54.297299±0.215820 | 53.953546±0.355778 | **+0.343753±0.550510** | **+0.440420±0.644202** | **+0.303501±0.787380** |
| 正向seed数 | — | — | **2/3** | **2/3** | **2/3** |

因此，此前“缺独立检测结果、不能判断”的状态现在可以更新为：**这三个历史partial模型有可核验的完整开发集端点，均值略高于主要recipe匹配的历史native，但seed0下降，稳定性和跨模态独特贡献仍未验证。** 不能写为完整CCLKD已复现，也不能宣称当前partial配方有稳定正收益。

## 独立审计结果

`analyze_eval.py` 从本地原始下载文件检查以下条件，三个seed全部通过：

- `evaluation_completed`、seed正确、`cclkd_partial`、固定历史`weights/last.pt`。
- `fixed_budget_last_ema`、`split=val`、`canary=false`、`official_test_accessed=false`。
- 三个roster均1469个唯一条目，三个有序roster逐项相同，receipt中的roster副本与run原件一致。
- 每个receipt为`run_kind=eval`、`terminal_status=COMPLETED`、`method_identity=PROTOCOL-ADAPTED`。
- 每个receipt含独立绑定lease ID，GPU2上单CUDA PID；全部source/config/roster/metric引用存在，指标快照与独立JSON完全一致。
- receipt明确 `historical_training_source_not_fully_bound=true`、`teacher_labels_used_by_kd=false`。
- exposure ledger标注开发用途、`confirmatory=false`；此轮没有访问test，不能将开发集结果升级为确认性测试结果。

### 资源与canary

主任务先在释放的空GPU2运行按GT目标数选取的64张高密度图像canary，然后在同卡依次完成三个完整评估。该canary仅测试执行和资源，64张指标没有混入方法结果。

| 项目 | 实测或预约 |
|---|---:|
| canary PyTorch reserved峰值 | 884 MiB |
| full每任务显存预约 | 2560 MiB |
| full守护器观测GPU峰值 | 1370 MiB |
| full每任务CUDA PID | 1 |
| full最高RSS | 3941 MiB |
| full PyTorch allocated峰值 | 约712 MiB |

分配回执保存起任务前5张空卡（0/1/2/3/7），评估期间与OEv1现有3张卡合计4张；引用用户AGENTS §2.1放宽条款，仍留下4张完全空卡。三次eval串行运行，没有新增训练。子任务本身只负责后续CPU证据拉取与审计。

## 公平比较与证据边界

1. **方法范围：** 仅LLD+CCL的历史适配版，不是原论文全部模块。旧训练源码未逐字冻结，当前损失源码含后续v3修订，不能用当前文件改写历史执行身份。
2. **这次源码回执绑定评估：** `source_snapshot/trainer/` 中是新评估脚本、Ultralytics `DetectionValidator` 与 `YOLO` 源码，说明“这些指标如何算得”；不是9月3日历史trainer/loss来源。`historical_training_receipt`是指针，历史receipt副本另存 `raw/`，不能把二者混写。
3. **历史native：** recipe主要匹配，差值可作描述性参考；没有本次CCLKD同代码weight0以及三个seed的shuffled/same-modal归因。+0.344均值不能独立证明配对IR的因果价值。
4. **相对OEv1：** OEv1在KD阶段直接使用独立IR GT，而本CCLKD partial使用RGB GT、IR图像和已训练教师；不同辅助标注使用应明列，OEv1仍需自身same-modal、dose-matched random与标签匹配控制。
5. **种子范围：** 三个seed为学生训练重复，教师固定seed42；不覆盖教师训练的不确定性。

## 完整证据位置

94公共输出根：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/oev1_comparators_20260906/`。

- `cclkd_partial_s0_full_attempt1/`
- `cclkd_partial_s42_full_attempt1/`
- `cclkd_partial_s123_full_attempt1/`
- `cclkd_partial_s42_canary_attempt1/`（仅资源canary）

本地 `eval_raw/` 保留上述每个run的独立JSON、protocol、roster、eval_evidence完整receipt与源码/配置/指标副本，以及artifact汇总/分配/脚本。共下载62个小文件、1,929,766字节，没有checkpoint、图片或cache。

可重算入口：`analyze_eval.py`；机器可读结果与审计表：`summary.json`；下载来源与时间：`eval_source_manifest.json`；拉取脚本：`collect_eval_remote.py`、`fetch_eval_evidence.py`。
