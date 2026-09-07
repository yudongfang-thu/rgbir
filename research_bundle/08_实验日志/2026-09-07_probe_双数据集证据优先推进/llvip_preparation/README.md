# LLVIP 独立 N / L1 / L_GT 配置准备

**NOT_ADMITTED：9 份三 seed 配置已生成，6 项 CPU 真值检查通过；LLVIP 的真实兼容性/训练 canary、L1 几何与校准，以及正式评价 profile 尚未准入。本次未启动训练。**

## 目的与执行范围

依据工作区 AGENTS、`refine-logs/EXPERIMENT_PLAN.md` 和实际 `refine-logs/EXPERIMENT_TRACKER.md`，为双数据集证据优先推进准备 LLVIP 定位分支。当前工程能表达 LLVIP N/C1/L1，并不意味着 Drone 已有 receipt 可以迁移。本次只读本地工程、已执行 source snapshot、旧模型身份与 split 产物，写本目录并运行 CPU 检查；未 SSH、使用 GPU、计算哈希、改公共模块或修改运行队列。

源模块是 `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2`。18 个相关文件与 `2026-09-07_train_IndependentKD实施/remote_admission_1532/compat_C0_s0_attempt1/implementation_snapshot` 逐字节相等。该已执行快照绑定 94 上 `artifacts/rgbir_independent_kd_v2_20260907/release_gpu5`。这里只验证这些本地文件，未宣称现场远端代码未经变化。

## 已准备的配置与真实支持

`prepared_v1/llvip_{N,L1,L_GT}_seed{42,0,123}_NOT_ADMITTED.yaml` 共 9 份，来自现有 `prepare_configs.configurations()`，保留本数据集 IR42 teacher、visible42 reference 和现行新 N recipe。全部 draft config 检查通过。N 的分类/定位系数均为 0；L1/L_GT 的分类系数为 0，定位系数为 `null`。所有 readiness、calibration、canary、geometry、D2 字段保持空值，`formal_training_authorized=false` 和 `protocol_status=NOT_ADMITTED`，未捏造 receipt 或 lambda。

| 路径 | 已验证的实际行为 | 仍需满足的条件 |
|---|---|---|
| N | `build_trainer` 将 N 映射到旧 `weight0`，使用 IndependentCriterion 和 TrackedDualLabelRGBIRDataset；原合成函数的总 loss 与 score 梯度和 native exact | 本数据集实际兼容性、canary、readiness、资源 lease；不能以 CPU stub 代替真实数据轨迹 |
| C1 / C1_y | Bernoulli 分类 KD 在单类时二者 loss、梯度 exact，且非零 | LLVIP 没有非目标类别项，因此不能声称多类软目标增量；本次不准备此长训分支 |
| L1 / L_GT | 同严格定位 mask 和 anchor；分别使用教师 DFL 与 GT 两 bin 目标。空 geometry 会立即拒绝构造 | 独立接受的 exact-image 几何、train D2、真实 train-mode 64 batch 校准和 canary；相同口径对照共享实际 LLVIP lambda |

N 虽然是零辅助梯度的 native 优化目标，仍通过同一 paired loader，并执行冻结 teacher/reference 前向与辅助诊断。不能称其完全绕过教师或等价于单模型成本。teacher/reference 在 optimizer/EMA 建好后附加，不进入学生优化器；paired loader 先跑学生变换，再重放配对图像/标签 RNG，并恢复学生后 RNG 状态。它保留原始 RGB 训练路径及 no-mix/no-HSV 配置，要求原始配对尺寸相等；共享标签和同一次随机变换都不能证明跨传感器几何准确。

本地 CPU 使用 `D:/Anaconda/envs/KGJ_proj/python.exe`、PyTorch `1.8.0+cu111`，CUDA 未初始化；这不是正式 pinned 2.10 训练环境。6 项检查覆盖单类 C1=C1_y 非零梯度、原 weight0 loss/梯度 exact、真实 build_trainer 的映射逻辑（替身依赖）、空几何拒绝、LLVIP evaluator profile 拒绝，以及 9 份 draft 不伪造准入证据。单类真值 loss=0.4291123152，梯度 L2=0.2845637798；它们只用于算子检查。

## 数据与旧模型身份

LLVIP 单类为 `{0: person}`。旧 visible/infrared data YAML 分别指向 `data/processed/llvip/yolo/grouped_v1/{visible,infrared}`，`train=images/fit`、`val=images/dev`，没有 test 字段。已保存 population 为 train 9619 / dev 2406；本地诊断 roster 分别为 2048 / 200 图，样本 stem 和 source group 均无交集。已见 train groups 为 02/03/05/06/08/09/10/11/13/14/15/16/17/18，dev 为 01/04/07/12/25。这是旧全量计数与诊断子集的复核，不是重新遍历远端完整数据集。

模型身份来源为 `2026-09-07_probe_Baseline蒸馏机会重诊断/remote_exports/llvip_full_attempt1/model_identity.json`。旧 N42 为 `runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt`，T42 为对应 `infrared_seed42_native_b32a2/weights/last.pt`。它们是 E200、seed42、YOLO11n640、B32/nbs64、SGD；比较到的优化器、学习率、warmup、AMP/deterministic、全套 augmentation 值均与新 family 一致，**明确差异为旧 workers8、新 workers4**。旧 visible42 只作固定 reference 和独立诊断 baseline，不能冒充新 family N42；新 N0/N123 仍须真实训练。

## 具体阻塞与必要差异

1. **LLVIP 自己的兼容性/readiness 缺失。** 现有 `admission.check_readiness` 要求 N/C0 × seeds 0/42/123 六条真实兼容性轨迹，各至少 30 loader batches、24 successful updates，并校验 dataset/model/T/R/recipe。Drone receipts 不匹配。C0 是技术兼容性 fixture，此处没有因此追加 C0 长训练任务。N 还需本 arm 的实际新路径 canary 和资源预约。CPU 数学相等只证明测试输入下的 loss 组合。
2. **原 L1 几何仍未准入。** LLVIP 共用标注、ROI/anchor oracle 读出，以及 IR 定位信息更强均不替代 exact-image 点/对象空间支持合同。此前 image roster 或单张 050001 的点集证据不能覆盖真实 natural loader 中所选对象。L1/L_GT 需要同一已接受 geometry 下的 train D2 与 64 batch 校准；校准至少 16 个非零 batch、16 张 unique image、2 个 source。当前 lambda 保持空值，不用 Drone 数值补齐，也不放宽冻结阈值。
3. **正式 evaluation profile 有实质 Drone 硬编码。** `evaluate_independent.py` 已列 LLVIP=2406，通用原生指标/对象捕获可复用；但 `evaluator_profile.configuration_identity` 明确只接受 Drone/1469，`resource_dispatch.py` 的 evaluation_profile 完成计数也要求 1469。必须新建或正式泛化 LLVIP 的模型/2406 roster/运行参数绑定并取得真实同路径资源与 native/capture 一致性证据。仅改 dataset 字符串、沿用 Drone profile 或 receipt 会被拒绝。本次未改公共实现。
4. **旧权重重评与新 run 评价入口不同。** 现行 `evaluate_independent.py` 正式入口要求新训练 config、completed receipt、E200/EMA last 等。不可给旧 visible42 伪造新 N completed receipt。相邻 `llvip_full_eval` 已另设旧 N42/T42 全 dev 重评，复用通用 helper，绕开 Drone-only profile 身份函数；它只补旧模型机会证据，不解除新训练 evaluator profile 阻塞。
5. **归因控制尚未全实现。** v2 当前执行只接受 paired，虽有 shuffled/same-modal 配置字段，执行校验会拒绝它们。不能把 C0 改名成这两个控制。未来若主张跨模态训练增益，仍需真实实现并固定所需控制与三 seed 归因；当前准备仅支持已定 N/L1/L_GT 可表达部分。

## 可运行入口与下一步

本次已执行 `prepare_llvip.py` 和 `check_llvip_cpu.py`。前者只生成配置/本地来源记录，目标目录必须不存在；需要重做时使用新 attempt 路径，保留 `prepared_v1`：

```powershell
python 'E:/SHARE/光sar/08_实验日志/2026-09-07_probe_双数据集证据优先推进/llvip_preparation/prepare_llvip.py' --out 'E:/SHARE/光sar/08_实验日志/2026-09-07_probe_双数据集证据优先推进/llvip_preparation/prepared_v2'
```

正式 trainer 的已有入口为 release_gpu5 的 `train_independent.py --config <真实准入配置> --output <新数据盘run目录> --arm N|L1|L_GT --source paired --seed 42|0|123`；诊断 canary 另带 `--max-steps 24`。本段仅定位真实 CLI，当前 NOT_ADMITTED 配置不应作为正式启动输入。下一步先准备 LLVIP 兼容性 fixture 与新 N canary，并独立补齐定位几何覆盖；只有实际校准/准入后另生成冻结配置，才进入所需长训顺序。不能靠改状态字符串解除这些条件。

## 产物

- `prepare_llvip.py`、`check_llvip_cpu.py`：本地准备与 CPU 检查脚本。
- `cpu_checks.json`：6 项检查结果及环境边界。
- `prepared_v1/preparation_manifest.json`：9 份配置与 18 文件字节比较记录。
- `prepared_v1/recipe_and_model_comparison.json`：旧 T/R 和新 N recipe 逐字段比较。
- `prepared_v1/split_identity.json`、`data_yaml_source_copies/`：已保存 split 身份与 YAML 来源。
- `prepared_v1/source_copies/`：本次实际审阅的实现副本。
- 相邻 `llvip_full_eval/INDEPENDENT_CODE_REVIEW.md` 和 `independent_code_review_receipt.json`：独立代码复核接受其执行 canary；不等于实际 canary/full 完成。

以上是配置和已执行 CPU 证据，不是新模型精度结果，也未产生任何 KD 增益结论。工作区索引与 campaign 状态由根任务统一维护。
