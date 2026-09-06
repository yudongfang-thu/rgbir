# OS-SSL-IR 夜间结果更新（2026-09-06 21:46）

> 新增 paired seed123 的完整 E200 训练：CSV mAP50–95=53.942、AP50=75.849；同 seed paired−shuffled 为 +0.669 mAP / +0.495 AP50 pp。首次出现可比较的同 seed SSL 内部正差信号，但尚无独立 last 评估或三 seed 配对归因结论。

## 目的

在 17:29 晚间审计的基础上，检查新增完整结果，核对其口径与控制组身份，为 GitHub 外部模型审计保存原始小证据和可复算的分析。本轮不启动检测推理、不加载 checkpoint、不修改 GPU 任务、队列、方法或门限。

## 设置与时间

- 快照：94 服务器，2026-09-06 **21:46:07.027–21:46:07.114 +08:00**，非原子采集；73 个源文件在各自读取期间均无尺寸/mtime 变化。
- 实验不变：DroneVehicle，三种 SSL 初始化 `paired / shuffled / sar_only`（后者是 IR-only），每臂 10,000 步、一次 seed42 预训练；下游 RGB 检测 E200，微调 seeds0/42/123。当前“三 seed”只覆盖微调随机性。
- 评估范围仍为开发 val；本轮未访问 test。全部表格来自同一口径的 **E200 `results.csv` 末行**，不将 stdout 的 best 验证或历史 native 的独立 last 指标混入表中。
- 初始化结论引用 **17:31:43** 已完成的 CPU 实测，并附原文件副本；未重新加载权重进行重复检查。旧原始日志与失败尝试继续保留。

## 进度：3/9 完成，1 个运行，5 个排队

| SSL 初始化臂 | seed0 | seed42 | seed123 |
|---|---|---|---|
| paired | 排队 | 排队 | **200/200，新增完成** |
| IR-only（sar_only） | 排队 | **200/200，已完成** | 排队 |
| shuffled | **131/200，运行中** | 排队 | **200/200，已完成** |

`worker3_gpu1.log` 记录 paired123 于 **19:19:46** 正常退出，随后接续 shuffled0。paired123 的 200 行 CSV 轮次连续为 1–200，完成回执的 seed、config、output 均匹配实际 run。该回执继承通用训练器的 `arm=native_weight0 / method_identity=CGA-KD-W1`，属于已知身份字段问题，不能只据这两个字段认定它是 native。本审计依照配置路径、权重路径、运行目录与 worker 命令识别实际 SSL 臂，不改写原回执。

## 完整结果与新增比较

指标为百分数，差值为百分点（pp）。

| SSL 初始化臂 | seed | E200 CSV AP50 | E200 CSV mAP50–95 | 对同 seed 历史 native 的 ΔAP50 / ΔmAP |
|---|---:|---:|---:|---:|
| **paired** | **123** | **75.849** | **53.942** | **+0.091 / +0.244** |
| shuffled | 123 | 75.354 | 53.273 | −0.404 / −0.425 |
| IR-only | 42 | 76.822 | 54.623 | +0.828 / +0.825 |

历史 native123 的对应 CSV 为 AP50=75.758、mAP=53.698；native42 为 75.994、53.798。这些对照存在前次审计已识别的全检测器初始化差异，表中的相对 native 差值仅为背景观察，不是干净的 SSL 效应估计。

### 新增的同 seed 内部比较

| 固定比较 | ΔAP50 | ΔmAP50–95 | 当前证据范围 |
|---|---:|---:|---|
| paired123 − shuffled123 | **+0.495** | **+0.669** | 一个微调 seed，两臂 E200 CSV，同非骨干模板；尚缺独立 last 评估 |

两臂 `args.yaml` 仅 `model/name/save_dir` 不同；相同 seed、数据、训练 recipe 与非骨干初始化让这个比较比“SSL−旧 native”更接近要检查的配对信息问题。该方向与“正确配对比固定错配更有利”的假说一致，值得继续核对另两个 seed。

但它还不能证明配对知识提供了稳定净收益：一方面，**paired 比 shuffled 好可以同时包含 shuffled 受损这一因素**；另一方面，paired 相对有初始化混杂的旧 native 只有 +0.091 AP50 / +0.244 mAP，不能靠内部差值替代干净 native 和 RGB-only SSL 对照。也不能用不同 seed 的 IR-only42 和 paired123 数值排名三种方法。

冻结归因门是 **+0.5 AP50 pp**，保存 CSV 算出的单 seed 差值是 **+0.495**；不将其四舍五入为“已过门”。更根本地，门限要求的三 seed 汇总和统一独立评估尚未齐备，因此本轮既不宣布归因成功，也不以这一个近门限差值宣布整条路线失败。冻结的迁移门仍为 **+1.0 AP50 pp 且 3/3 正向**，没有根据当前数字调整。

## 独立评估及初始化问题仍未解决

1. 本轮遍历已有 OS-SSL run 的小产物及评估相关子目录，**仍为 0 个独立 `metrics_record.json`**。当前 worker 脚本只执行训练，未调用预注册的 `eval_rgbt_detector.py`；已有完成日志末尾验证的是 `best.pt`。因此本表不称为独立 last/EMA 终态，也不与 OEv1 独立评估数字直接排序。
2. 17:31 CPU 检查确认三种 SSL checkpoint 的 **259 个非骨干张量逐项相等**；W1 native 的 COCO80→nc5 构造、3/5 分类行映射、451/499 张量加载，与 clean nc5 模板的 499/499 加载不等价。新结果没有解除这个混杂，也不能据此断言初始化差异解释了全部性能变化。
3. 下游目标模态是 **RGB**，IR-only 是辅助红外单模态预训练，不能替代 **RGB-only SSL**。要检验跨模态知识的独有价值，仍需按独立补充协议加入同模板零 SSL native 和 RGB-only SSL。

## 结论与下一步

1. 新进展是首次完整的同 seed paired/shuffled 比较出现正方向，证据强度从“各臂孤立单次结果”推进到“一个可解释的内部对照”。仍未支持三 seed 增益、配对归因或论文主结论。
2. 保持当前冻结训练矩阵与门限，收齐剩余微调；按预注册统一 last/EMA、同 evaluator、val 评估后，再计算三 seed mean±SD 和逐 seed 方向。本次只做读取，没有新增推理任务。
3. 同模板零 SSL 与 RGB-only SSL 是后续补充实验的明确缺口，需要单独登记设计，保留原始 W1 和原预注册；不要以单 seed 正差为依据改步数、改阈值或追加机制解释。

## 产物路径与复算

- 94 原始 run：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/osssl_ir_20260906/`。
- 94 原始配置及队列：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/`。
- [snapshot.json](snapshot.json)：时间、73 个源路径/尺寸/mtime/读取一致性、run 与权重文件清单、队列进程；权重仅记录路径与元数据，未复制。
- [summary.json](summary.json)：原始 CSV 的十进制复算、同 seed 比较、配置差异、完成证据检查及结论边界。
- [raw](raw)：73 个原始小文件副本；其中 6 个大日志只保留最后 64 KiB，文件名带 `partial_tail`，清单记录偏移量，不冒充完整日志。
- [初始化原始证据副本](initialization_inspection_173143_reused.json)与[其原采集脚本](inspect_initialization_173143_source.py)：来自 17:31 检查，本轮未重新运行。
- 本轮脚本：[采集](collect_snapshot.py)、[分析](analyze_snapshot.py)。本地安装 PyYAML 后运行 `python analyze_snapshot.py` 可仅用本包小产物复算；采集脚本需要配置好的 SSH 94，读取远端当前状态。
- [前次完整初始化分析](../../2026-09-06_audit_RGBIR晚间进度与新结果/osssl/README.md)和[冻结预注册](../../../07_研究分析/方法预注册_OS-SSL-IR_20260906.md)继续作为背景依据。本次未生成新哈希，原始文件未移动、覆盖或重命名。
