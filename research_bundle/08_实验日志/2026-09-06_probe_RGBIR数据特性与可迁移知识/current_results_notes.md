# 当前 RGB–IR 结果复核（2026-09-06）

> **现有 RGB–IR 结果支持“直接模仿强模态并不稳定转成弱模态净收益”，但还不能证明“RGB–IR 没有蒸馏空间”或“门控没有价值”。DroneVehicle CMDistill 相对新 native 的开发集差值为 −0.349±0.292 mAP 百分点；HNEWA 对强同模态对照的收益小且种子方向不一致。**

## 目的、范围与证据等级

核验本项目最新 W1 native 与 CMDistill、HNEWA、CCLKD 和 P3 已有结果，避免把方法设计建立在错误指标、错误端点或过度机制推断上。

- 本轮只读取 94 的既有小型 metrics、配置、训练/评估脚本、完成日志，以及三个约 5.5 MB checkpoint 的文件 SHA256；没有加载模型 tensor、启动 GPU、训练或评估，没有读取封存 test，也没有写入服务器。
- 应用 `analyze-results` 的原始数据→配对统计→观察/解释/下一步流程。
- 所有 AP 先乘 100，以**百分点**报告；SD 为 sample SD（ddof=1）；两臂比较先按同一 seed 求差，再计算差值 mean±SD。
- 本轮是描述性复核，**不替代 accepted analyzer，不将开发集结果升级为最终论文 claim**。
- 原始小文件、路径、SHA256、采集代码和复算结果均保存在同目录 `current_results_sources.json`。此前审计的原始结果表也按本地路径+哈希收录。

## 1. W1 native 与 CMDistill：确有小幅开发负结果

N 为 `runs/cgkd_w1/native_rgb_s{seed}_e200`；L 为 `runs/rgbt_cmdistill_adapted_v2/dronevehicle_seed{seed}_b32_e200`。六份独立 `metrics_record.json` 均明确指向各自 `weights/last.pt`，均使用同一 `prepared/dronevehicle/rgb.data.yaml`、`split=val`、`evaluation_role=val`。

| 指标 | 臂/差值 | seed0 | seed42 | seed123 | mean±sample SD |
|---|---|---:|---:|---:|---:|
| mAP50–95 | N native | 54.3576 | 53.8156 | 53.6874 | 53.9535±0.3558 |
| mAP50–95 | L CMDistill-corrected | 53.9003 | 53.2448 | 53.6692 | 53.6048±0.3324 |
| mAP50–95 | **L−N** | **−0.4574** | **−0.5707** | **−0.0183** | **−0.3488±0.2918** |
| AP50 | N native | 76.7484 | 76.0011 | 75.7301 | 76.1599±0.5274 |
| AP50 | L CMDistill-corrected | 76.2351 | 75.1644 | 76.2499 | 75.8831±0.6225 |
| AP50 | **L−N** | **−0.5133** | **−0.8367** | **+0.5197** | **−0.2768±0.7085** |
| AP75 | L−N | −0.6820 | −0.0729 | +0.0246 | −0.2434±0.3829 |

**观察：** mAP50–95 三 seed 全负，但 seed123 几乎持平；AP50/AP75 只有两 seed 为负，不能写成“所有指标均 3/3 稳定负迁移”。

**解释：** 在这个学生、教师、数据与训练设置下，PCCFD/SLRD/IBCLD bundle 未产生净收益。它能说明当前干预不划算，不能自动定位到对齐误差、模态不可观测、特征项、定位项或门控缺失中的任何一个原因。

**对设计的含义：** 新方法不能只打赢 L；应以 N、same-modal 与 paired/shuffled 为核心对照，防止把“比已有负方法少掉点”误写成跨模态增益。

### 1.1 recipe 核对：主要参数一致，精确 weight0 契约尚未全部闭合

当前原始 args 确认三组 seed 都共享：同一初始化路径 `.../int8_cross_modal_stage2_v1/weights/yolo11n.pt`、同一学生 data YAML、e200、batch32、nbs64、imgsz640、SGD、lr0/lrf0.01、momentum0.937、weight_decay0.0005、warmup3、同 augmentation、deterministic=True、patience0。

旧 `formal_native/...native_b32a2` 的当前 args 也显示 **batch32、nbs64**。**不能仅凭目录名 b32a2 与 b32_e200 判定训练预算不匹配**；前者可以描述自动累计的有效 batch64。过去文档仅凭名称推断“不同 recipe”应以实际 args 修正。

实际代码差别：

1. N 使用标准 `YOLO(...).train`，val=True、plots=True；L 使用 `CMDistillTrainer`，val=False、plots=False，重写 validate/final_eval 并包裹 paired loader。
2. `RGBTSharedGeometryDataset` 使用 `PairedDetectionDataset`。后者先对学生执行原 transforms，然后恢复 Python/NumPy/torch RNG 为教师回放同一变换，最后恢复学生变换后的 RNG。静态代码**没有显示学生主动使用了另一套增强**，不能把“自定义 loader”本身当作已证实的混杂。
3. L 完成 receipt 记录 `optimizer_steps=56722`、`kd_batches_logged=112600`。N 完成 receipt 未记录实际 optimizer updates，只有共享超参和完整训练。N/L 的训练时初始化字节、运行时依赖哈希、逐 batch 学生输入/target 等价和相同更新数尚未逐项动态证明。
4. 六份独立 evaluator 记录均为 last.pt；seed42 分桶是另一评估流程，batch 默认16而全量 evaluator 使用32，二者约0.01–0.02 pp的小差别不能混成端点变化或蒸馏效应。

因此合适表述是：**“主要 recipe 一致、采用独立 final 评估的三 seed 开发负结果；精确同代码 weight0 契约仍可补强。”** 比“负迁移已被毫无混杂地证实”更准确，也比“不能比较”更符合当前证据。

## 2. W1 昼夜分桶：新重评存在，机制推断仍过强

新文件 `artifacts/cgkd_20260905/native_s42_buckets.json` 的数字与旧 P3 summary 的 native 数字完全相同。核查后发现：

- 新 `bucket_native.py` 明确加载 **`runs/cgkd_w1/native_rgb_s42_e200/weights/last.pt`**。
- 同目录 `bucket_native.log` 包含 day/night/full 的实际评估输出和 `BUCKETS_DONE`，对应735/734/1469张图。
- 所以**不能仅根据结果相同就指控复制旧表**；存在重评的正面证据。新 JSON 未将 checkpoint 哈希绑定到结果，旧 JSON 也缺这类身份字段，因此 tensor 身份/等价性不在本轮证明范围。
- 三个 checkpoint 文件哈希不同，但不同序列化文件可包含相同权重，文件哈希差异本身同样不能证明模型 tensor 不同。

| seed42 亮度代理桶 | N mAP50–95 | L mAP50–95 | L−N | N AP50 | L AP50 | L−N |
|---|---:|---:|---:|---:|---:|---:|
| 高亮度/day | 62.796 | 62.173 | −0.623 | 80.307 | 79.710 | −0.597 |
| 低亮度/night | 45.101 | 44.322 | −0.779 | 71.177 | 69.424 | −1.753 |
| full，分桶评估流程 | 53.823 | 53.260 | −0.563 | 76.012 | 75.167 | −0.845 |

桶按RGB平均亮度中位数78.283划分，不是官方昼夜标签。只有seed42；本轮未重评其它seed。

**可以说：** 在这个 seed 和亮度代理划分下，KD 在两桶均有负差值，当前结果没有呈现简单的“只在高亮度桶受损、低亮度桶获益”。

**不可以说：** “负迁移不随条件集中”“门控 H3 失去实证基础”“问题不在 When、只在 What”。理由：

- 两个均值桶不能排除桶内目标大小、局部可见性、配准、遮挡等条件作用；同时为负也不意味着损失同质。
- mAP50–95 两桶负差值只相差 **0.156 pp**；没有多 seed 或子组差值不确定性评估。不能因AP50差异较大就替代主指标的条件交互检验。
- IR 与 RGB 模型分桶AP是对各模态标签的独立总体成绩；未在同一目标集合逐目标核对教师对错，不能把 IR 的 +11.248 AP50 当作 RGB 必然可学的收益上限。
- 没有实际有效的条件门控干预及同平均剂量随机门控对照，无法检验门控机制是否有效。

若预注册门1仅规定 day 桶下降≥0.3，当前seed42点估计越过了数值线（−0.623），但“数值门通过”不等于已完成机制因果解释或 accepted analyzer 升级。

## 3. HNEWA：配对信号存在，但净增益证据弱

本轮重新读取 `runs/rgbt_hnewa_cmkd_mse_inspired_v1/eval_records/*.json`，未发现较9月5日审计更新的完整 final 对照矩阵。下表沿用逐seed独立final JSON，排除带 `_best` 以及 `/tmp/best_probe/last.pt` 的端点不清记录。

| 数据集 | 对比，mAP pp | seed0 | seed42 | seed123 | mean±sample SD | 正向 |
|---|---|---:|---:|---:|---:|---|
| DroneVehicle | paired−shuffled | +0.5662 | +0.0111 | +0.4307 | **+0.3360±0.2894** | 3/3 |
| DroneVehicle | paired−same-modal | +0.5331 | +0.2660 | −0.1788 | **+0.2068±0.3597** | 2/3 |
| LLVIP | paired−shuffled | +0.2955 | −0.8525 | +1.6067 | **+0.3499±1.2305** | 2/3 |
| LLVIP | paired−same-modal | +0.2495 | −0.5643 | +0.5874 | **+0.0909±0.5920** | 2/3 |

| 数据集 | 臂 mAP mean±SD（seeds0/42/123） |
|---|---|
| DroneVehicle | paired54.1359±0.2317；shuffled53.7999±0.1074；same-modal53.9291±0.3337 |
| LLVIP | paired34.6786±0.8004；shuffled34.3287±0.7854；same-modal34.5877±0.3451 |

HNEWA 自身 b0 native 只有seed0/123 final来源明确。共同两seed的 paired−native：DroneVehicle **+0.2256±0.3012**（+0.0126/+0.4386），LLVIP **+0.2802±0.2502**（+0.4571/+0.1033）。不能把新CGKD N seed42随意拼入HNEWA凑满三seed。

**观察→解释→设计：** DroneVehicle paired−shuffled 更稳定，但seed42几乎无差；paired−same-modal两数据集均有负seed，说明强同模态教师已经解释了大部分收益。优先问“特定目标/知识是否提供同模态教师没有的可学信息”，比继续提高全图模仿权重更有依据。paired>shuffled本身仍可能部分来自shuffled的伤害。

注意本线名称明确是 `cmkd_mse_inspired`，不应在论文里表述为已逐项复现原始Hnewa方法。

## 4. CCLKD 与 P3：不要把完成日志混为有效性证据

### CCLKD

`runs/rgbt_cclkd_adapted_v1/cclkd_drone_seed{0,42,123}_b32_e200/completion_receipt.json` 三份均显示完整e200、56722 optimizer steps。末几步 `kd_ccl≈0.69316`、`kd_lld≈10^-5–10^-4`。

在该run目录和 `rgbt_cclkd_native_v1` 下，未找到独立 `metrics_record.json` 或 final eval JSON。工程 smoke/confirm 完成日志不是检测效能评估。因此**不能将CCLKD列作已证明有效或无效的RGB–IR方法**。这些loss值值得检查剂量与梯度是否有效，但loss趋近log2或数值小本身不能证明梯度失效。

该实现receipt明确 `CCLKD-adapted ... not an exact reproduction`。`rgbt_cclkd_native_v1` 中混有名为native_sar/teacher_rgb的旧训练，不能仅凭rgbt目录名归为RGB–IR证据。

### P3 causal v1

此前审计保存了以下**单seed42、E200训练CSV终点**（不是本轮新增独立final评估，不能升为正式收益结论）：

| 数据集 | native | paired P3 | shuffled | same-modal | random-dose |
|---|---:|---:|---:|---:|---:|
| DroneVehicle | 53.798 | 53.509 | 53.936 | 54.095 | 54.367 |
| LLVIP | 32.867 | 33.532 | 35.398 | 35.209 | 33.308 |

作为历史诊断，它提示：DroneVehicle paired低于所有对照；LLVIP paired虽高于native，却低于shuffled和same-modal。因而“相对native升一点”不足以说明学到了有用的跨模态信息。这条证据与HNEWA不同协议，不拼表算均值。

## 5. 当前最可取的研究判断

1. **转向RGB–IR有理由，但优势来自可检验的数据条件，不来自当前已有大增益。** RGB–IR可提供更直接的共同目标、局部配准和条件差异分析；现有实验没有证明只换数据就容易成功。
2. **“蒸馏哪些信息”是合适主问题，但可迁移性至少有三道条件：教师在该目标上更正确、目标在学生模态中有足够观测、被蒸馏表示在两模态间有可靠对应。** 教师整体AP高、特征相似或亮度低都不能单独替代这三个判断。
3. **先分清共同目标上的错误类型再选最小知识。** 如果教师优势在分类但定位/配准差，先考虑语义/目标关系；若目标框可对齐且教师定位更准，再考虑位置或框分布；若教师优势主要来自RGB看不到的目标，降低强制模仿优先级。
4. **控制剂量仍有价值。** 应将可见性/对应质量的选择与同平均剂量随机选择比较；目前结果既不支持盲目按图像亮度门控，也没有否定对象级可靠性门控。
5. **为了快速形成可投稿结果，停止把新名称和多个组件绑定。** 在一个主数据集先验证一类可学知识、一个有效干预，再补四臂和三seed；特征图是机制筛查证据，不能替代检测净增益。

## 产物与可追溯路径

- 本文件：当前结果判断、统计与限制。
- `current_results_sources.json`：远端原始文本+SHA256、本地参考文本+SHA256、收集代码、N/L复算JSON、checkpoint文件哈希。
- 94根路径：`/mnt/dataset/yudongfang/projects/RGBT_campaign/`；训练/eval/loader源码根：`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/`。
- 保留所有历史原件，本轮未更改旧W1/P2/P3结论头；需要主任务将本记录作为新的解释链入口。
