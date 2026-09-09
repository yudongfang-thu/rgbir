# 90 原生 RGB / IR baseline 重建准备

**可以在没有 94 教师权重时独立重建四个 baseline：原 `formal_native` 的真实入口是 `train_rgbt_kd_detector.py --arm native`，该分支不加载或前向 T/R。当前交付仅为本地准备，四份配置均为 `PREPARED_NOT_ADMITTED`，未执行远端命令、加载权重、训练或计算新 hash。每个完整任务单独受 10 小时上限约束。**

## 原入口与最小接入

- 原队列 [`run_rgbt_campaign_queue.py`](../../03_现行工程/SpaceNet6_OTD_official_reproduction/tools/run_rgbt_campaign_queue.py) 的 `_add_train_job`（约 168 行）对 `formal_native` 实际调用 [`train_rgbt_kd_detector.py`](../../03_现行工程/SpaceNet6_OTD_official_reproduction/tools/train_rgbt_kd_detector.py)；保留 `--arm native --seed 42 --batch 32`，不传两个 teacher 参数。
- `build_trainer`（约 305 行）在 native 时保留原 `DetectionTrainer` 单模态 dataset 和 preprocessing，不包 `PairedDetectionDataset`；`set_model_attributes` 只在非 native 指定臂加载教师。`_ordinary_models` 也只加载通用 student 预训练。另一模态 YAML 和 paired/shuffled mapping **仍被预检**，但不意味着 native 训练读取其像素或使用其教师权重。
- 原 loss 是 [`FPSelectiveLoss`](../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/fp_selective_distillation.py) → `selective_fp_criterion` 的 native 分支 → [`b0_distillation.native_student_loss`](../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/b0_distillation.py:131)。它调用原 criterion 的 parse / `get_assigned_targets_and_loss`，校验捕获的 assignment，保留 `loss_parts.sum() * B`；无辅助教师前向、无 KD。**不是把当前 IndependentKD 的 N_fast 填假 T/R，也不擅自改为另一种 loss 调用。**
- 名称看似更直接的 `tools/train_rgbt_native_detector.py` 确实无教师，但其 `val=True` 与原四个 `args.yaml` 的 `val=False` 不同；本轮不选它作为原 recipe 入口。
- 需新建小的 90 准入/计时适配器，加载原 native 函数并只加资源、成功更新、限时及身份回执。不能直接调用历史完整 campaign 队列，它会自动规划 KD / 扩 seed。旧源不改；新适配器及实际安装 runtime 在首个性能短测前另存来源副本并核对接口。

## 已执行参数与四份纯配置

来源是旧四臂实际 `args.yaml`，不是按当前默认值猜测：

| 数据/模态 | 配置模板 | 原 effective args | 完整 train / dev / dev GT |
|---|---|---|---|
| Drone RGB | [模板](native_rebuild_templates/dronevehicle_rgb_seed42_native_E200.template.yaml) | [RGB args](../2026-09-08_ops_小时级筛选重构/baseline_runtime/original_rgb42/args.yaml) | 17990 / 1469 / 22462 |
| Drone IR | [模板](native_rebuild_templates/dronevehicle_infrared_seed42_native_E200.template.yaml) | [IR args](../2026-09-08_ops_小时级筛选重构/baseline_runtime/original_ir42/args.yaml) | 17990 / 1469 / 24490 |
| LLVIP visible | [模板](native_rebuild_templates/llvip_visible_seed42_native_E200.template.yaml) | [visible args](../2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/N42_full_attempt1/original_args.yaml) | 9619 / 2406 / 7879 |
| LLVIP IR | [模板](native_rebuild_templates/llvip_infrared_seed42_native_E200.template.yaml) | [IR args](../2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/T42_full_attempt1/original_args.yaml) | 9619 / 2406 / 7879 |

四臂均为 YOLO11n 通用预训练、E200、B32 / nbs64、640、workers8、seed42、正常 BN、fresh optimizer/EMA、SGD lr0=.01 / lrf=.01、momentum=.937、weight_decay=.0005、warmup3（momentum .8 / bias lr .1）、线性调度、AMP true、deterministic true、patience0、fraction1、rect false、resume false。正常阶段 accumulate2；warmup 内原生动态 accumulate，不能全程硬写 2。

增强保持 translate .1、scale .5、fliplr .5；mosaic/mixup/cutmix/degrees/perspective/flipud/HSV/erasing 均 0，close_mosaic0。原 effective args 的 box7.5 / cls.5 / dfl1.5、cache false、compile false、channels_last false、single_cls false、multi_scale0 等也逐项保存在各模板 `rebuild90.expected_effective_args` 中。它们是**运行后待核对的完整预期**；原 CLI 只把 `_overrides` 中明确的字段送到 native，其余依赖绑定的实际默认配置，不能把这个 metadata 表冒充已执行参数。

原 `val=False` 并不保证从不评价：已捕获的 native trainer 在末轮/stop 时调用 validate，`final_eval` 会 strip 自己生成的 checkpoint 并可能验证 best。保持原行为和每轮 save / save_period−1，并对 **last endpoint** 做同口径独立 full-dev 评价；不要拿 CSV 的占位 AP、best checkpoint 或最后一次验证含混替代它。

四模板从原 `configs/research/rgbt_*.yaml` 生成，只改 model/data 路径并添加独立 90 元数据。旧冻结 `method_id=RGBT-P3-CAUSAL-v1` 留给原 validator，**新权重身份**是 `90_native_rebuild_<dataset>_<modality>_s42_E200_v1`。模板保留原允许 seeds 列表是为兼容 validator，本任务仅 seed42。[本地 YAML / 参数核对回执](native_rebuild_templates/TEMPLATE_CHECKS.json) 不代表远端准入；旧 CLI 不识别 `rebuild90.status`，因此必须由新外层检查该状态，不能直接执行模板。

## 数据、初始化与随机性边界

本条目 [环境初查](environment90_metadata_attempt2.json) 找到普通预训练候选；根代理随后确认新环境 `/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python` 已 CPU 导入通过，普通预训练已复制到 `/mnt/dataX/ydf/projects/RGBT_campaign_90/weights/pretrained/yolo11n.pt` 并通过 80 类 YOLO11n CPU 加载。四模板已更新为这两个实际路径。此项是根代理的执行回报，本子任务未加载权重；仍未证明与旧 94 通用预训练张量相同，不填造逐位同源。[新环境原回执](environment_ready90.txt) 保留实际版本。

模板数据路径采用本条目已有 `prepare_data90.py` 的目标 `/mnt/dataX/ydf/projects/RGBT_campaign_90/data_attempt1/prepared/{dronevehicle,llvip}`；这不声称 preparation 已完成。Drone 原始 `/mnt/dataset/DroneVehicle` 与 M2D 的文件数/配对存在性不能代替当前 HBB、class order、split/label 合同。LLVIP 本轮接入脚本指向 `/mnt/dataY/ydf/dataset/LLVIP.zip`，需要正式 preparation 成功回执；早前“未找到 LLVIP”是当时盘点范围的历史状态，不用于否认现已发现的 ZIP。LLVIP 必须保留治理的 train9619 / dev2406 及原 native label 行序，禁止用 official test 补训练。数据 YAML 不含 test。保留 processed image 路径让 YOLO 正确推导 labels，不能把它替成 raw canonical 路径造成零 GT。

原 `build_trainer` 不设置自定义数据生成器。已捕获原生 [engine_trainer.py](../2026-09-08_ops_训练吞吐诊断/runtime_sources/engine_trainer.py:139) 使用 `init_seeds(args.seed+1+RANK, deterministic=...)`；[data_build.py](../2026-09-08_ops_训练吞吐诊断/runtime_sources/data_build.py:229) 使用 worker 的 torch seed 初始化 NumPy/Python，loader generator 固定 `6148914691236517205+RANK`。保持单进程单 GPU、同 train 序列、原 shuffle/最后不足批/InfiniteDataLoader 生命周期、workers8 和增强调用顺序。seed42 不等于跨 workers/安装源码/设备逐位同流保证；worker8→4 是另一个 recipe 身份，不能因速度或内存而静默改变。配置名中的 b32a2 也不是裁掉不足 32 的末批。

## 是否计算 hash、仍需完成的技术接入

本次只读了本地源码/小回执、写纯配置和本文，**没有计算文件 hash**。所查 `train_rgbt_kd_detector.py`、`fp_selective_distillation.py`、`b0_distillation.py`、`write_jstars_run_receipt.py` 及直接 project guard/contract 文件未见 hash 调用；`emit_bound_run_receipt` 主要复制来源文件和记录资源。不过本地没有在本轮穷尽实际 90 安装包的 dataset cache / checkpoint 下载等传递路径，**不能因此宣称运行全过程已保证无新 hash**。`cache=False` 仅是图像缓存配置，不足以证明标签缓存校验不会算摘要。

执行前剩余的具体检查是：在已部署新目录绑定实际 native dataset/cache/checks 源，若标签缓存有 digest 路径，使用仅新副本的显式无摘要标签读取/原值比较适配，保留原 labels 与排序，独立核对；不返回伪 digest、不改共享安装包。`check_amp` 也可能触发额外模型下载/推理：不能把 94 的“已验证 AMP=true”无条件移植到 90。90 首次本模型技术短测须绑定 AMP / finite 梯度 / skip 证据，准备好本地依赖后再使用有来源的局部 AMP setup。无这项就留 blocked，不能静默下载或伪造通过。

原 `--max-steps` 只数 `optimizer_step()` 调用，包含可能的 AMP skip；旧 completion 即使短测也写 completed。因此新适配器必须区分 `PERFORMANCE_CANARY_COMPLETED` 与 E200 completion，记录 attempts / successful_updates / AMP skips / EMA updates / 实际 epoch / batch 数。独立记录 native optimizer 的真实成功调用，而不是从旧计数猜测。不要只给原 CLI 加 `--max-steps` 就宣称“24 成功更新”或完整训练。

## 单任务 ≤10h 的短测与执行门（未运行）

短测只回答吞吐、资源和运行正确性，不读取/比较 AP，不筛选模型或日程。建议每数据集/模态各使用真实全量 loader，保持 **E200 调度**，新独立 canary 最多 10 分钟：先至少 24 次真实成功更新，然后继续至 120 个完整 batch；覆盖真实双/单模态路径是指分别测各 native 模态，每个任务只前向自身模型。全量数据本身的 setup/标签读取计入任务总成本，不用 2048 子集代替。

记录 setup、逐 batch 同步墙钟（含 loader/preprocess/S forward/native loss/backward/optimizer）、首 20 批后稳定区间、audit 时间、保存和完整 dev 评价秒数；记录整进程树 RSS、NVML 峰值、torch allocated/reserved、整卡最低余量。计时记录不能额外大量保存每批图像或全 state；必要的小 CPU 真值只验新计数/限时接线。若 120 批无法在 10 分钟完成、资源不合规或出现非有限/缺失输出，该 canary 技术失败，不改日程求通过。

单卡 B32 时预计每轮 batch 数 Drone `ceil(17990/32)=563`、LLVIP `ceil(9619/32)=301`，**以真实 len(loader) 为准**。预留完整任务上限 36000 秒，采用事先固定 `setup + 1.25 × (200 × 实测 batches/epoch × 稳定批均值 + 200 × 保存开销 + 已知末轮/终端评价开销)`，并单列尚未实测的项；缺完整评价/保存实测则先完成相应技术测量，不能填零。通过预算后才另开正式 fresh E200；该上限是每一个 baseline job，不是四臂合计。

**不要设原 native `args.time=10` 作为上限**：已捕获 trainer 会按已耗时重估 epochs、重建 warmup/LR，破坏 E200 配方。使用外层本任务专属 watchdog + worker wall 检查，到上限前保留已写产物并明确 `INCOMPLETE_TIME_LIMIT`，不改 scheduler、不写 E200 completed；只终止本 job 所有进程。首完整 epoch 后再次用实测含保存开销核预算，预计超 10h 时明确未准入/停止并回报工程瓶颈，不把不足 E200 的结果当正式 baseline。所有 GPU 调度继续由根代理统一 lease 执行，此计划不自行启动资源池。

94 原 RGB/IR E200 的 [CSV 时间证据](../2026-09-08_ops_小时级筛选重构/baseline_runtime/README.md) 分别 3.473139 / 3.696167 小时；这是训练 CSV 时间，不含独立评价/排队的完整任务墙钟。它支持先测原 native 入口，但不能按 4090/3090 名义算力直接承诺 90 时长。共享卡负载、CPU/存储和新 data view 均须实测；短测的快慢不裁决方法输赢。

## 新身份的使用边界

重建完成后，90 RGB/visible seed42 可成为**新90**的 baseline/同模态 reference，90 IR seed42 可成为**新90**的跨模态 teacher。先绑定 new90 last checkpoint、数据/label 合同、初始化和 full-dev evaluator。旧 94 baseline AP 与新 90 teacher 的组合不是原实验恢复；不能把旧 94 N 分母直接用于新90 KD 增益比较。若后续在 90 比方法，使用同一新 pretrained、同一数据与 recipe、同一 90 N 对照重新建立比较；seed42 只能提供单 seed 观察，不能升级多 seed 增益。

附加交付是 [payload_native](payload_native/README.md)：12 份 Python 源、1 份历史 exposure 配置、4 份本轮模板，共约 202 KB，原代码逐字节复制且 AST 解析通过。没有修改 `payload_code` 或实现新调度器。Linux CPU import/help 留给根代理在上述新环境执行；本地 Windows 不用伪造 `fcntl` 绕过 Linux guard。原 guard 必须通过环境绑定根代理已部署的 `ROOT/resources` 共用 lease，不创建独立资源池。

数据 preparation、实际 native runtime/cache/AMP 与性能准入尚须有真实回执；CPU 普通模型加载已经由根代理完成，但不等于 E200 已准入。本次没有改旧训练源、父 README 或索引。
