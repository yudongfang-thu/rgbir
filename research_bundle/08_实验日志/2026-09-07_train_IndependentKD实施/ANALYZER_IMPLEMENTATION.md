# 独立分类/定位分析器与协议逻辑实施

**状态：四个纯逻辑文件已实现，37项本地CPU/unittest通过；未参与编写分析器的L实现agent独立复核36项时接受了当前三个被绑定源码。接受回执见`analyzer_acceptance_v2_20260907/analyzer_acceptance.json`。CLI仍必须显式读取有效回执，且实现检查/实际实验证据未齐时不能自动扩展或升级主张。没有SSH、GPU、训练或新AP评价。**

本次依据用户已授权的执行补充：C1_y 可按明确规则作为简单候选回退；LLVIP 可以替代 Drone 的 L 矩阵，需增加其自身 N×3，核心上限 15、两条新方法四臂上限 27。它不是同时复制 Drone/LLVIP 两套 L，也不是一次提交 27 个作业。

## 产物与接口

实现目录：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/`。

| 文件 | 功能 |
|---|---|
| `protocol.py` | 冻结 arm/source/seed、系数与单任务拒绝、配置逻辑、分支预算；无 Torch |
| `test_protocol.py` | 8 项协议真值测试 |
| `analyze_independent.py` | 文件回执核验适配、百分数/pp、逐seed配对与样本SD、效用/内容/损伤/四臂判读、接受门 |
| `test_analyze_independent.py` | 29项分析器真值及真实临时文件回执链测试 |

`validate_config(cfg, arm=None, formal=False, source=None)` 返回 `{valid, errors, required_receipts, ...}`，不修改输入。兼容实际工程字段 `classification_coefficient`、`localization_coefficient`、`teacher`、`reference`、`paths`。允许 arm 为 N/C0/C1/C1_y/L1/L_GT；source 为 paired/shuffled/same_modal。joint、CL 和两个非零 KD 系数被拒绝。

配置校验不是启动许可。返回的 `formal_receipts_verified` 始终为 false；实际 recipe、兼容、校准、几何、canary、模型生命周期、source 内容和 lease 的文件与状态核验由 trainer 执行，不能因纯逻辑检查通过就补写 `READY`。

## 分析器输入与调用

正式文件入口（下文manifest接口示例由最初29测版本保留；**现行合同绑定规则见本节后附“独立审阅修正”**，manifest不能直接提供有效评价合同）：

```text
python analyze_independent.py --manifest frozen_analysis_manifest.json --output NEW_analysis.json
```

输出采用排他创建，不覆盖历史分析。manifest 的每个 run 至少包含：

```json
{
  "arm": "C1",
  "source": "paired",
  "seed": 42,
  "source_arm": "C1",
  "path": "/server/project/runs/full_C1_s42_attempt1",
  "protocol_id": "actual_frozen_common_recipe_identity",
  "expected_val_images": 1469,
  "evaluator_contract": {
    "version": "actual_pinned_and_reviewed_evaluation_contract",
    "imgsz": 640,
    "batch": 32,
    "precision_and_nms": "explicit_effective_settings_to_be_bound_by_execution"
  }
}
```

例中的评价合同值不是已冻结执行值。独立审阅指出直接信任manifest仍然不够；现行loader**忽略manifest中的evaluator_contract/expected_val_images**，采用下面真实回执绑定方式。Drone/LLVIP分别绑定1469/2406；旧、新评价源码语义等价必须由独立兼容回执证明。

### 独立审阅修正：实际评价和干预身份

未编写本分析器的L实现agent独立审阅后发现并已修复：

1. 评价合同只能取`eval_evidence`的receipt config snapshots中的`evaluation_contract.json`。schema为`rgbir-evaluation-contract-v1`；包含完整`roster`、`expected_val_images`、`endpoint`、`official_test_accessed:false`、`evaluator_identity`与`effective_kwargs`（imgsz/batch/workers/half/conf/iou/max_det/agnostic_nms）。检查resolved字段与bound eval cfg一致，实际evaluator源码文本也纳入比较，不能用同一字符串ID遮住不同评价实现。
2. 旧N/C0无新JSON时仍保留原始描述性AP，但不得直接升级。可提交`evaluation_compatibility_receipt`：status ACCEPTED、reviewer、每个checkpoint/seed的精确`bound_training_config`与`bound_evaluation_config`、所有实际eval trainer副本的`evaluation_source_copies`直接bytes一致，以及指向已复核标准评价源码副本的`canonical_evaluator_source_copies`。这表示独立接受的语义桥接，不能自动伪造。
3. 从实际train receipt config提取T/R、辅助数据、选择配置、λ、载体共同参数；L几何JSON也必须进入train receipt config snapshots。C1−C0允许预定的载体与λ差异但T/R/选样相同；C1−C1_y核同λ/温度/选样，只允许η内容改变；L1−L_GT核同λ/门/温度和实际几何内容。没有身份不输出内容支持。
4. 对每个arm/source分别检查跨seed干预同质性；两臂在seed123一起换教师或λ也会撤下聚合，不能只凭逐seed匹配宣称固定协议三seed。arm汇总采用同一规则。
5. harm分类指标清单来自train cfg的`expected_nc`，双方同时缺同一类别也不能CLEAR。
6. 继承的receipt verifier优先取独立release的`task_conditional_reference/analyze_results.py`；接受回执绑定的是实际加载的该文件副本，不是可变的邻接开发目录。

以上分别增加了manifest覆盖无效、缺bound合同、effective_kwargs不一致、实际评价源码不同、内容T/R/λ/几何不一致、跨seed共同换teacher、双方缺同一类别的拒绝测试。

文件 loader 复用已有 `rgbir_task_conditional_v1/analyze_results.py::load_endpoint`，逐一读取完整训练、评价与 source/metric/split 副本：

- 训练 E200 完成、train/eval completed receipt、seed/arm/method/last.pt 身份一致。
- last/EMA dev endpoint、测试未访问、完整无重复 roster 与 receipt 副本相同。
- 实际 receipt 绑定的训练配置与 launch copy 一致，不能用目录名或手填 AP 代替。
- 新层额外检查 arm/source 不被 manifest 改名，跨 seed 与跨臂 recipe/roster/evaluator 一致。

旧 N 的 raw `weight0` 和 C0 的 `paired/c/c_shuffled/c_same_modal` 明确映射。新评价 JSON 必须显式写 `arm` 和 `source`；不能把 paired 端点仅通过 manifest 改成 shuffled/same_modal。C1_y/L_GT 不是四臂来源。

`analyze_records(...)` 是纯逻辑内核，只用于 loader 已验证的标准化记录或明确标记的合成真值。它无法证明调用方自行创建的字典来自真实实验；正式 CLI 不从 manifest 接受这种内联 AP 替代文件验证。

## 预先冻结的判读

- 主差值：C0−N、C1−N、C1−C0、C1−C1_y、C1_y−N/C0、L1−N、L_GT−N、L1−L_GT。
- 三 seed 固定 0/42/123，百分数/pp 明确，ddof=1；缺 seed、失效 receipt、重复 attempt 或混 recipe 不升级。
- C1 升级需对 N/C0 三 seed 全正且均值≥0.10 pp；C1_y 若同样超过 N/C0，且 C1 不三 seed 稳定胜 C1_y，则建议 `PROMOTE_C1_Y`。
- C1−C1_y、L1−L_GT 三 seed 全正且均值>0才给出对应内容支持标记，仍不是单位梯度信息优势。
- L1 对 N 满足效用门后还需固定定位诊断相容；未完成该诊断时不自动扩展。
- 损伤触发：AP50均值下降>0.20 pp、任一类别AP三 seed同降、背景FP/image三 seed同升。`REVIEW_REQUIRED` 暂停扩展，**从不下发训练早停**。缺少固定操作点诊断时为待诊断。
- 四臂就绪按每个方法独立检查 paired/shuffled/same_modal/N 三 seed，不将 C0 控制借给 C1，不将 L_GT 借作 same-modal；表中齐备与实际胜出是不同事实。
- LLVIP 与 Drone 分开聚合，不能跨数据集借 N。

固定错误分析为 confidence=.25、match IoU=.50、coarse IoU=.10，背景定义为对所有 GT IoU<.10，单位 FP/image。文件需绑定 seed、checkpoint 与完整 roster；模型报告的 best-F1 recall 不用于代填背景错误。

`proposed_decision` 是数值判据建议，`expansion_status` 受损伤/缺诊断门限制，`auto_expansion_eligible` 还要求实现检查和独立分析器接受。未接受时即使合成真值全正也保持 false。

## 独立接受方式

外部审阅通过后，CLI 可传 `--accepted-receipt path`。回执需为 `status=ACCEPTED`、有 reviewer，并在 `source_snapshots` 绑定三个真实源码副本：

```text
analyze_independent.py
protocol.py
legacy_analyze_results.py
```

最后一项绑定实际选中的冻结文件/回执verifier。运行时对副本与实际源码直接比较bytes，不计算hash。源码改变使旧接受失效；测试通过本身不签发接受回执。本次没有编造或自行签发接受。

## 本地验证及修复记录

环境：`D:/Anaconda/envs/KGJ_proj/python.exe`，纯 CPU。逻辑无第三方依赖；实际旧文件 loader 依赖项目已有 PyYAML，不导入 Torch。测试仅使用 TemporaryDirectory 创建明确的合成回执，没有访问服务器权重/图像。

实际执行：

```text
python -m unittest -v test_protocol test_analyze_independent
Ran 29 tests in 0.168s
OK
```

覆盖已知值 `[1,2,3]` 样本SD=1、配对差 `[.1,.2,.3]` 的均值/SD、阈值边界、缺seed、失效train receipt、原始AP与绑定副本不一致、混入best/test、改变评价合同/roster、跨seed变workers、重复attempt、C1_y回退、harm、未接受分析器、LLVIP与Drone隔离及四臂。

首次加入实际文件失效回执测试时发现 `metrics_percent=None` 导致验证器 AttributeError；已修为把缺失指标当不可用并返回验证错误，未补零。修复后同29项全部通过。该错误只影响新分析器临时合成测试，没有正式结果被升级。

独立审阅修正后曾执行`Ran 36 tests in 0.364s / OK`，外部复核者以该版本接受三个源码。随后仅增加旧评价桥接绑定/副本篡改回归测试，未修改被接受的源码；最新执行`Ran 37 tests in 0.458s / OK`。早先29项记录保留为阶段证据，不用它覆盖新的检查范围。

## 既有外部方法复用与准备清单

只读复用 `08_实验日志/2026-09-07_audit_RGBIR实施起点/COMPARATORS.md`，未重新检索文献或评估 AP：

| 方法 | seed0 / 42 / 123 mAP (%) | mean±样本SD | 当前可用身份 |
|---|---|---|---|
| CMDistill corrected | 53.900286 / 53.244840 / 53.669168 | 53.604765±0.332435 | PROTOCOL-ADAPTED 历史参照 |
| CCLKD partial | 54.066006 / 54.493295 / 54.332596 | 54.297299±0.215820 | PROTOCOL-ADAPTED，仅 LLD+CCL，缺 FLD/RLD |

两者历史workers8，现N/C0 workers4；辅助IR GT权限不同。CCLKD三seed的新评价回执和1469张有序roster已经核验，但不能补齐历史训练源码绑定。CMD旧metric record缺新合同的明确units/roster绑定，不伪装成新端点。两者不能直接进入当前严格配对分析器，也不能用其分数宣称打赢完整作者复现。

最小后续准备顺序：

1. 保留这些历史端点/源码与身份，列清训练/评价合同缺口；补评估只改善端点证据，不能消除训练差异。
2. BCKD式分类响应、FGD式特征、LD式定位只准备已授权的三类协议适配清单：实际调用loss完整性、student/T/R/init、增强/更新预算、layer与reduction、辅助IR标签、训练附加模块、温度/剂量及canary。
3. 未获得实际实现/回执者均为 PREPARE_ONLY，不产生占位AP；完整CCLKD未跑，不把partial结果用来否定整篇方法。
4. 最终保留分支明确后才把外部训练资源单列，当前不新增其GPU作业。

原证据入口：`08_实验日志/2026-09-06_ops_OEv1优先级与对比实验/comparison_shortlist.md` 与上述 `COMPARATORS.md`。本文不更新老文件，也不改其历史结论。
