# 旧 N/C0 与新 C1 的评价桥接候选

结论：九份旧端点原始 receipt 均通过现有验证，N/C0 的六份已完成桥接候选。六份使用完全相同的旧评价入口源码，配置、1469 图名单及数据 YAML 与新成功探针一致；旧 N42 的五个总指标也与新 native/evidence 两路逐位一致。当前仅生成 **DRAFT_AWAITING_INDEPENDENT_REVIEW**，未写 ACCEPTED，不阻塞分类正式训练。

## 输入与实际证据

- 旧输入：`old_endpoint_manifest_1407.json` 指定的 `snapshots/2026-09-07T140740.823638_0800/raw/`。N0/N42/N123、C0/C42/C123、R0/R42/R123 均为完整原始训练+评价端点，九份 receipt 检查均为 complete。本轮仅前六份进入候选，旧目录中的 C0 表示 **旧 C 的 seed0**，新分析器的 C0 表示 **旧 C 方法**，候选保留原 `source_arm=paired` 及真实 seed 以免混淆。
- 实际 probe：94 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/`，对应冻结 `release_gpu5`。关键 JSON、实际绑定源码、数据 YAML、当前安装 `default.yaml`、实际 receipt writer 的只读副本位于 `evaluation_bridge_candidates_v1/probe/`。
- 评价类型为 `real_gt`：RGB Drone dev1469 独立标签，固定 last/EMA；未访问 sealed test。probe 使用原 N42 权重，不产生新的 E200 训练回执。

实际 probe 的 native 与 evidence 均 seen1469，loader 顺序一致，五个总指标及逐类 AP 完全相等。五个总指标同时与旧 N42 原始 `evaluation_val.json` 相等：

| 指标 | 三处相同的原始 fraction |
|---|---:|
| AP50 | 0.7717965366938525 |
| AP75 | 0.6416656614678693 |
| mAP50–95 | 0.5451360845035178 |
| precision | 0.7864732611282166 |
| recall | 0.7292365679591741 |

资源数据来自根调度的真实 probe：NVML 项目峰值 1370 MiB、进程树 RSS 6260 MiB。它支持评价路径预约，不表示训练峰值。

## 静态核对

六份旧评价源码独立原字节比较完全相同，均为 3505 bytes。其实际 `model.val` 调用显式设置 data/val/640/batch32/workers4/device0/plots=False/save_json=False/verbose=False，并使用固定 last 权重。全部旧 bound data YAML 与 probe 的实际绑定 YAML 原字节一致，全部 bound roster 与新 canonical roster 相等，Torch/Ultralytics 版本均为 2.10.0+cu128 / 8.4.115。

当前真实安装源码说明未显式设置的参数如何得到：`Model.val` 将 rect 设为 True；BaseValidator 对 detect 将空 conf 设为 0.001；默认 quantize=None、iou=0.7、max_det=300、agnostic_nms=False、augment=False，推理为 FP32。这与成功 probe 记录的实际 effective kwargs 一致。

`Model._reset_ckpt_args` 仅保留 imgsz/data/task/single_cls；所有六份历史 args sidecar 为 task=detect、single_cls=False。本工具明确标注这些 sidecar **不是 receipt-bound 配置**，不能把它们重新包装为历史执行时已经被绑定的事实。旧 receipt 仅保存项目评价入口，未保存当时所有原生库的源码字节。因此，把六份历史端点映射到新合同仍包含基于共同入口、同版本与 N42 重复验证的语义推断，须独立审阅接受这个范围。

## canonical 源集合与顺序

候选没有照搬 probe 的 12 个来源。它解析实际 `evaluate_independent.py` 中 `legacy.implementation_files(...)` 参数次序，并核对实际 receipt writer 的去重和编号规则，构造未来正式评价的七份 trainer snapshot：

1. `01_evaluate_independent.py`
2. `02_val.py`（DetectionValidator）
3. `03_validator.py`（BaseValidator）
4. `04_model.py`（YOLO.val 实際定义于 Model）
5. `05_utils.py`（check_det_dataset）
6. `06_metrics.py`（DetMetrics/Metric/ap_per_class 去重）
7. `07_nms.py`

未来 C1 的实际 eval receipt 到达后，`verify-formal-sources` 必须检查真实七份 source set、编号顺序及原字节。该功能只核源码，不签 bridge。若正式评价源码或收集顺序改变，需要新的候选审阅；本轮没有修改任何冻结 NEW 源码。

## 工具与产物

- `prepare_evaluation_bridge.py`：LOG 目录工具，prepare 子命令仅生成 draft，没有接受开关；verify-formal-sources 子命令只检查未来实际来源。
- `evaluation_bridge_candidates_v1/prepared_a1/evaluation_compatibility_candidate.json`：六个 checkpoint/seed 对应的实际 bound train/eval 配置、旧评价源码独立副本、待审语义合同、canonical 源副本。
- `prepared_a1/manifest_candidate.json`：保留 `oev1_frozen_drone_e200` 原协议名，并显式保留 source_arm；不以更换 protocol_id 强行造成可比。
- `prepared_a1/candidate_summary.json`：九份 inventory、六份 candidate、零静态 mismatch、probe/历史 N42 等价，以及 **六份 draft 均被新 analyzer 拒绝升级** 的实际检查。
- `test_prepare_evaluation_bridge.py`：六项 CPU/实际源码文件检查通过，包括 probe 数值篡改、旧 kwargs 变化、formal 收集顺序变化和未来源码字节变化的拒绝。

重现准备命令（在项目根，输出必须为新目录）：

```powershell
& 'D:/Anaconda/envs/KGJ_proj/python.exe' '08_实验日志/2026-09-07_train_IndependentKD实施/prepare_evaluation_bridge.py' prepare --old-manifest '08_实验日志/2026-09-07_train_IndependentKD实施/old_endpoint_manifest_1407.json' --analyzer-root '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2' --accepted-analyzer '08_实验日志/2026-09-07_train_IndependentKD实施/analyzer_acceptance_v2_20260907/analyzer_acceptance.json' --probe '08_实验日志/2026-09-07_train_IndependentKD实施/evaluation_bridge_candidates_v1/probe' --output '<新的候选目录>'
```

## 待决范围与使用边界

1. 独立审阅者须明确接受或收窄“旧共同源码、同版本、N42 全 dev 复算”向六份历史端点的评价语义桥接。当前 candidate 不允许 analyzer 升级。
2. 首份未来正式 eval receipt 必须匹配上述 canonical 七源、实际 kwargs 和完整名单；训练 recipe 与 KD intervention 的兼容仍由现有 analyzer 和训练兼容验收各自检查。
3. 原六份旧评价没有保存逐类 AP、实际 seen/loader 顺序和 objects。这些字段未填入旧 entry 的执行证据，合同中不复制 probe 的 observed_images/actual_loader_roster。新 N42 probe 的逐类结果作为独立真实产物保留，不能假称其他五个模型已有这些诊断。
4. 桥接若接受，范围仅是已有五个总指标的开发集比较。缺少旧逐类/对象记录时，负迁移 harm review 仍可能为 INCOMPLETE；不能用该桥接签出 CLEAR、四臂归因或论文主张。

本轮没有重新评估六模型，没有启动 GPU、读取测试集或改动旧结果；只读获取源码/成功 probe 的小产物，准备后续结果分析入口。分类正式训练继续按既有准入推进。

## 独立审阅补充

`/root/review_matrix_spec` 已独立阅读候选工具和真实产物，并重跑六项 CPU 测试全部通过，接受“候选生成与未来实际源码匹配检查”的工具代码范围。审阅者明确保留 DRAFT，不接受仅从 N42 外推到六模型的更强运行级逐位等价，也不填补历史 library 源、seen/loader、逐类/对象诊断缺失。工具代码接受与六端点 bridge 正式接受分开记录；本目录首份 audit 的 PENDING 表示发起独立审阅时点，不能据此推断 bridge 已获得接受。

## C1 启动后的必做最小补采

旧 N/C0 的逐类与对象证据必须补齐，不能因历史没有保存而省略。安排在新 C1 正式训练启动后，作为独立短评价队列；不重训旧模型，不等待这部分完成才启动 C1。

**覆盖六个固定 checkpoint：**N0/N42/N123、C0/C42/C123，均只读原 run 的 `weights/last.pt`，训练来源仍是原 E200 completion 与原 train receipt。冻结一个新的诊断评价 manifest，逐条记录原 run、原 checkpoint、原 completion/train receipt 路径、seed、实际原 arm、方法规范名，以及新的诊断输出目录。目录统一使用当前 BASE 下 `legacy_diagnostics_v1/<原臂>_s<seed>_attempt1/`；不得写回任何原 run。

**每个 checkpoint 只需一次完整 evidence 路径的 dev1469 前向。**直接复用冻结的 `make_evidence_validator`、native DetectionValidator 和 metric/capture 函数，不再做一次 native/evidence 双重前向，也不另造 AP、NMS 或匹配实现。若旧身份读取需要独立入口，后续只写一个 LOG 目录的薄适配器：验证旧训练来源并把输出指向新目录，仍调用冻结的评价组件；本轮暂不实现，不改 NEW release。

N42 新 probe 已有真实逐类 AP 与 objects，可保留作为已有诊断证据。但其 probe receipt 不是 `object_error_analysis.load_evaluation` 当前要求的完整 eval receipt（后者还直接绑定 objects 原字节和配置）。默认预算包含 N42 的一次规范补采，共 **6 次单路完整评价**；若独立审阅事先确认现有真实 probe 证据可通过一个明确标为“后续诊断绑定”的入口完整复用，则可免掉 N42，减为 **5 次**。不迟造一份看起来是在旧训练时代执行的 eval receipt，也不将 probe 里的对象记录复制成旧 run 的历史文件。

**调度与资源：**复用已通过的 gpu5 evaluation profile 所证明的 native/evidence 单模型评价路径、B32/W4/640、FP32、rect=True、conf=.001、IoU=.7、max_det300、single_cls=False、agnostic_nms=False、augment=False。所有任务仍走统一 lease，GPU 动态选择，预约采用实测 profile 与现行 guard 的余量；不因 probe 峰值1370MiB而绕过70%、2GiB、任务数或RSS规则。薄适配器必须独立确认仍调用同一冻结推理/指标路径并绑定其实际源，不能把训练或额外教师前向借用为同一 evaluation profile。

**每次必须新存的完整证据：**五个总指标及全部五类 AP50/AP75/mAP50–95、objects.jsonl.gz、实际 evaluation_contract（完整 roster、loader 顺序、seen1469、实际 kwargs）、原图/输入 canvas 尺寸和原生预测；以及新 eval receipt、实际评价源码副本、配置、数据名单、原字节对象与指标快照。新 receipt 明确标注“对历史训练 checkpoint 于本次新做的诊断评价”。原训练 completion/receipt 只作为只读输入或有原路径的原字节引用副本；不新建 E200 训练成功回执。

**新旧端点不混写：**每个新评价的五个总指标与同 checkpoint 的旧原始评价逐项比较并存差值；一致才能直接使用旧 AP 作为同端点历史比较说明。若不一致，保留两次评价和全部差值，先调查源码/参数/数据/数值执行差异，不覆盖旧 AP，也不悄悄把新对象结果搭配成旧执行证据。新的诊断 manifest 显式指定本次统一评价 attempt 和原训练来源；若现有 analyzer 的固定目录接口需要引用旧 train evidence，使用原字节来源副本加 origin manifest，避免伪装成发生了第二次训练。

**CPU 诊断与验收：**使用已冻结的对象分析规则，并先确认其独立接受回执仍有效；至少形成三 seed 的 C0−N，以及新 C1 完成后 C1−N 的逐类 AP 和 background FP/image；C1 与 C0 的差异另外报告。对象检查固定 confidence≥.25、同类一对一 IoU≥.50、粗候选IoU≥.10、背景为对所有GT IoU<.10，输出 repaired/damaged 的分子与各自基线正确/错误分母，并按既定类/尺度/来源/亮度代理分组。缺失 metadata 保持 UNKNOWN。比较前严格验证同 seed、完整图集、逐图 GT 原数组/顺序、canvas、实际评价合同一致。

只有以上旧证据补齐，且 accepted analyzer 的逐类、背景误检等检查实际通过，才允许把相应 harm 状态从 INCOMPLETE 升级；旧缺失不能成为跳过 harm 检查的理由。这些补采不扩大方法矩阵，不停止已批准的 E200 分类训练。
