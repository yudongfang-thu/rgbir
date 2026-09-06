# OS-SSL-IR 晚间只读审计（2026-09-06）

> **已有 2/9 个检测微调完成，但尚无 paired 终点或独立 last 评估；更重要的新发现是复用的 W1 native 与 SSL 三臂检测头初始化不等价，当前 +0.825 pp 的单次差值不能归因于 SSL。**

## 目的

核对 94 服务器 OS-SSL-IR 的实际训练进度、已完成输出和控制组身份，区分值得跟踪的数值与可以成立的方法结论。本轮只读日志、配置、小产物和 CPU checkpoint 元数据/张量，不运行检测推理，不启动或修改 GPU 任务，不改变既有队列。

## 设置与时间

- 小产物快照：2026-09-06 **17:29:25 +08:00**；初始化实测：**17:31:43 +08:00**。非原子快照；70 个小文件采集中无文件尺寸/mtime 变化。
- SSL：DroneVehicle 17,990 对训练图，`paired / shuffled / sar_only` 三臂，每臂预训练一次 seed42，10,000 步；`sar_only` 在这里表示 **IR-only**。
- 下游检测：输入 **RGB**，同一 `rgb.data.yaml`，5 类，E200、batch32、640、SGD；微调 seeds0/42/123。val 为 1,469 张、22,462 个目标，test 未读。
- 真正冻结的预注册是 `07_研究分析/方法预注册_OS-SSL-IR_20260906.md`：迁移门为 **paired−native ≥ +1.0 AP50 pp 且 3/3 seeds 正向**；配对与 IR-only 归因门分别为 +0.5 AP50 pp。旧训练 README 括号中的“+0.1 AP50”不是有效阈值。

## 进度

| SSL 初始化臂 | seed0 | seed42 | seed123 |
|---|---|---|---|
| paired | 排队 | 排队 | **103/200，运行中** |
| IR-only（sar_only） | 排队 | **200/200，完成** | 排队 |
| shuffled | 排队 | 排队 | **200/200，完成** |

`worker3_gpu1.log` 记录 IR-only seed42 在 **15:32:39** 正常退出并接续 paired seed123；worker0/2 仍按资源守卫重试。只有 2 个完整微调，不是三 seed 归因矩阵已完成。旧失败尝试日志仍保留；当前两个完成臂均有完成回执和 best/last 权重，不能把历史格式兼容错误当作当前训练失败。

## 已有数字：同 seed、同 CSV 口径的探索观察

下表全部来自 E200 的 `results.csv` 末行，数值单位为百分数，差值单位为百分点。这里只与 **native 同口径 CSV** 对照，不借用它的独立 final 指标拼表。

| 臂 | seed | AP50 | mAP50–95 | native 同 seed AP50 / mAP | 相对 native ΔAP50 / ΔmAP |
|---|---:|---:|---:|---:|---:|
| IR-only SSL | 42 | 76.822 | 54.623 | 75.994 / 53.798 | **+0.828 / +0.825** |
| shuffled SSL | 123 | 75.354 | 53.273 | 75.758 / 53.698 | **−0.404 / −0.425** |

它们说明两个已完成配置没有呈现一致方向，不能把两个不同 seed 的臂直接相减作 paired/IR-only/shuffled 归因；paired 当前还没有 E200 终点。IR-only 的单次正差值得跟踪，**但不能称为“IR-only SSL 已有效”**，原因包括初始化混杂、单 seed 和独立评估缺失。

### 评估口径与回执身份

1. 所有 OS-SSL run 均没有 `metrics_record.json`，worker 仅执行训练，未接入预注册的 `eval_rgbt_detector.py --checkpoint .../weights/last.pt`。W1 native 三个 seed 已有该独立 val 指标。
2. 已完成训练 stdout 的最后一段明确为 `Validating .../weights/best.pt`；例如 IR-only 日志显示约 77.2 / 54.9，**这是 best 评估，不是冻结 last 终点**，不能拿来替代上表或与 native last 直接比较。
3. 当前 completion receipt 从通用 native trainer 继承了 `arm=native_weight0 / method_identity=CGA-KD-W1`。它证明脚本跑完，却没有准确编码 OS-SSL 臂；必须通过 receipt 中实际 `config/output`、args 中 `clean_{arm}.pt` 和 worker 命令还原身份。本轮没有改写原回执。
4. OEv1 已完成的 paired42 独立 last/EMA 评估与本表 OS-SSL 的 CSV 末行属于不同实验线和不同评估产物。**不能凭 OEv1 54.658 与 IR-only 54.623 的接近就排序方法，更不能把另一条线未完成的 weight0 当作现成对照。**

## 关键新结论：native 初始化存在混杂

单看 args，两个完成 SSL 臂与同 seed W1 native 只有 `model/name/project/save_dir` 不同；训练数据、训练 recipe 与其余保存字段一致。但 `model` 的差别包含了骨干以外的变化。

**实际证据如下：**

| 检查项 | W1 native | OS-SSL 三臂 |
|---|---|---|
| 输入检测 checkpoint | 原始 COCO `yolo11n.pt`，80 类 | 已构造的 `clean_{arm}.pt`，5 类 |
| 日志中的类别转换 | `nc=80 with nc=5` | 已为 nc5 |
| 日志中的类别行映射 | **`Remapped 3/5 cls head rows ... by class name`** | 未发生该映射，checkpoint 类名为 `0,1,2,3,4` 占位名 |
| 预训练张量加载日志 | **451/499** | **499/499** |
| 非骨干模板共享 | 不能由“同源权重”推成相同 | CPU 实测 **259 个非骨干张量三 SSL 臂逐项相等** |

框架检测模型加载代码会按真实类别名映射 COCO 分类输出，遇到占位名称则跳过。OS-SSL 模板构造只复制形状兼容的张量，对类别数改变而形状不兼容的检测头保留模板初始化；下游再加载全部 499 个张量。W1 native 在微调时才构造 nc5 模型，并发生 3/5 类的输出映射。两种流程不能凭同一个原始 `yolo11n.pt` 就声明全初始化一致。

因此：

- **SSL 内部对照仍可继续解释**：paired、shuffled、IR-only 的 259 个非骨干模板张量一致，没有发现三 SSL 臂彼此更换检测头的情况。
- **相对 W1 native 的变化混合了 SSL 骨干、检测头构造和类别偏置映射差异**；本轮并未测定这些因素各自能产生多少 AP。因此既不能把 +0.825 全归于 SSL，也不能断言这个正差完全由初始化造成。
- 旧记录“同 init + 同 recipe、native 可直接复用”的表述需要降级为“recipe 一致，同源初始化，但全检测器初始化不等价”。

## 另一个归因缺口：缺 RGB-only SSL

此路线的目标模型只看 RGB，而已有 `sar_only` 是 **IR-only 预训练→RGB 微调**。它可以检验“只用辅助 IR 数据预训练”这个替代解释，但不是目标 RGB 自模态 SSL。若要主张跨模态信息带来独有收益，需要增加 **RGB-only SSL**，与 paired 使用相同步数、样本预算、模板和微调协议。

三臂 SSL 各只有一次 seed42 预训练；0/42/123 目前仅指检测微调阶段。最终报告不能把它说成覆盖了三次独立 SSL 预训练的随机性。

## 局限与下一步

1. 保留并完成当前冻结的三 SSL 臂；这些发现不要求中止可解释的内部对照。
2. 按冻结协议，对所有终点统一运行 **last/EMA、val、同 evaluator**，产出独立小结果后再计算三 seed 差值；不混用 stdout best。
3. 在独立补充协议中增加 **同 nc5 模板、同非骨干张量、零 SSL 步的 native**，用于隔离 SSL 骨干贡献；保留旧 W1 native 作为原始监督训练参考，不能覆盖旧结果或把补充试验伪装成原先已预注册。
4. 增加 RGB-only SSL 机制控制。阈值保持原先 +1.0 / +0.5 AP50，不根据当前数值下调。
5. 本轮不升级“RGBIR 蒸馏有效”“配对信息有贡献”或“IR-only 优于 paired”等主张，尚未有这些结论所需的证据。

## 产物路径

- 94 原始训练：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/osssl_ir_20260906/`。
- 94 原始配置/队列/SSL：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/`。
- 94 native：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/cgkd_w1/`。
- 本地 `raw/`：70 个配置、脚本、回执、CSV、native metrics 和日志副本；大日志仅复制最后 64 KiB，并在文件名/manifest 中明确为 partial tail。
- [snapshot.json](snapshot.json)：时间、每个源文件路径/尺寸/mtime、run 清单和队列进程；未生成新哈希。
- [initialization_inspection.json](initialization_inspection.json)：CPU checkpoint 元数据、原日志前 32 KiB 的相关原句、三臂 259 个非骨干张量相等性。
- [summary.json](summary.json)：复算数值、args 差异、进度和结论边界。
- 脚本：[collect_snapshot.py](collect_snapshot.py)、[inspect_initialization.py](inspect_initialization.py)、[analyze_snapshot.py](analyze_snapshot.py)。
