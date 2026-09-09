# RGB–SAR 数据、复现与历史方法独立复核

**结论：OGSOD、SiXiang 和 SpaceNet6-OTD 是不同实验数据身份。现存证据支持 SiXiang full-CMD 的受限开发正信号、SpaceNet6 的 OS-SSL 预训练正信号；尚不支持历史自研监督 KD 已稳定成立。配对信号、教师优势、oracle 空间和学生 AP 收益必须分别报告。**

- date：2026-09-09。
- auditor：协作 agent `/root/audit_rgbsar_history`；独立检查者。可观察身份为本会话 agent，工具未提供可核实的具体模型版本，**不声称跨模型审阅**。
- overall_verdict / integrity_status：`warn`。本地指定结论与部分 primary 小产物可核；远端完整训练、数据和权重链未重验，不构成整项目 integrity pass。
- reason_code：`LOCAL_PRIMARY_AND_ARCHIVED_EVIDENCE_ONLY`。
- 操作范围：只读指定文档、YAML、已有 JSON/CSV 与少量源码；只新增本文件。没有 SSH、GPU、训练、评估、统计重算、SHA 计算、凭据读取、原件改写或 Git 操作。
- 使用技能：`C:/Users/MSI-PC/.codex/skills/experiment-audit/SKILL.md`。遵循本轮明确访问限制，不执行技能默认要求的重算哈希，不新增其他审计文件。
- 数字口径：AP 表用百分数、差值用百分点 pp，SD 沿用已有产物的 sample SD；NMI、恢复率等保持自己的量纲，不能当 AP。下面“已核”表示读取现存内容，不代表重新运行了对应 analyzer。

## 1. 数据身份与真正完成的数据分析

| 数据 | 已核实身份与划分 | 可用观察 | 不能推出的结论 |
|---|---|---|---|
| OGSOD-1.0 | YAML 为三类 bridge/harbor/storage_tank；SAR 与 RGB 分目录；`val=test=images/test`。canonical YOLO11n E400/B64/256；历史 split 文档存在 14664/3667 与 14665/3666 两种计数，本轮没有原始目录清点 | 正式历史 n=2 的 RGB/SAR mAP 58.7555/48.8250，约 9.93 pp 教师优势；这组 raw CSV 在 L20，本轮只能核本地审计表 | 不能称 untouched test；不能把教师差距当可蒸馏上限；不要合并 90/94 重训或不同 workers/runtime 的结果 |
| SiXiang coarse10 | YAML 为十类舰船，独立 train/val/test；源为 3182 组三模态 512 chips，138 scenes，scene-clean 110/14/14 scenes、2305/304/573 图；原 shipped split 有 scene overlap，已废弃 | 历史三 seed RGB/SAR mAP 64.441/38.979，gap 25.462 pp；本地原审计同时注明 baseline lineage 未完全闭合 | 配置足以认定其与 OGSOD 是不同实验身份；本轮未读图像字节，不能声称两者绝无重叠图像。scene-clean 也未完成 acquisition/site/near-duplicate 与历史 test-access 全审计 |
| SpaceNet6-OTD | 本项目是油罐检测派生任务。公开 annotation JSON 对应 622 train+200 test；论文文本写 620+200。SAR-Intensity 与 PS-RGB 各 3401 图；OS-SSL 排除 200 test 后 3201 对 | D0 图像为 900×900；每图约 20 个对象。optical crop 继承 SAR-derived tank labels，已有 JSON 明示 annotation-source bias | 不可直接写成所有 SpaceNet6 任务或建筑分割的结论；光学标签不是独立标注的几何真值 |

身份依据：

- [OGSOD YAML](../../03_现行工程/ogsod400_clean_protocol/configs/datasets/ogsod_hbb_sar.yaml)、[SiXiang YAML](../../03_现行工程/ogsod400_clean_protocol/configs/datasets/sixiang_hbb_sar.yaml)、[SiXiang protocol](../../03_现行工程/ogsod400_clean_protocol/configs/protocols/sixiang_yolo11n_nomosaic_direct300_sgd_b64_v1.yaml)。
- [OGSOD 原审计](../../03_现行工程/ogsod400_clean_protocol/refine-logs/research_reset_20260722_v2/OGSOD_EXPERIMENT_AUDIT.md)、[SiXiang 原审计](../../03_现行工程/ogsod400_clean_protocol/refine-logs/research_reset_20260722_v2/SIXIANG_EXPERIMENT_AUDIT.md)。后者旧 SX-APR pending 状态由下文 union 终态取代。
- [SpaceNet6 复现协议及数据来源](../../03_现行工程/SpaceNet6_OTD_official_reproduction/docs/REPRODUCTION_REPORT_DRAFT.md)。原始全量图像、split 清单及 annotation 内容本轮 `unavailable`，没有重新确认地理配准精度。

### D0：已量化配对结构，结果主要落在 context

本轮实际读取 [data_profile.json](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/optical_sar_property_d0_20260829/data_profile.json)、[confirm_summary.json](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/optical_sar_property_d0_20260829/confirm_summary.json) 的相关字段及 [D0_REPORT.md](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/optical_sar_property_d0_20260829/D0_REPORT.md)。

| 数据 / prior-train 内部性质分析 | property-fit | property-confirm | confirm 背景占比均值 | 冻结 P4：paired context NMI effect / 95% CI | object−matched-context residual / 95% CI |
|---|---:|---:|---:|---|---|
| OGSOD | 7168 对、19046 对象 | 3072 对、7826 对象 | 96.392% | +0.051169 [0.048774, 0.053504] | −0.006955 [−0.009283, −0.004652] |
| SpaceNet6 | 350 对、7219 对象 | 148 对、2990 对象 | 94.667% | +0.105569 [0.076678, 0.142570] | −0.080620 [−0.116435, −0.051679] |

终态 `DEFER_CONTEXT_ONLY`：正确配对的局部 context 信号可重现，当前聚合对象区域没有额外胜过匹配 context。它是图像描述子性质审计，`eval_type=self_supervised_proxy`，**不是 detector-feature CKA、模型可学习性或 AP 结果**。对象聚合负值也不能消除全部类别/尺度条件下的局部正信号。这里未访问 method-dev 或 official evaluation。

### D0+ 与 CAP-D：有条件的可修复空间，未过预设门

- [D0+ confirm](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/error_complement_d0p_20260829/confirm_summary.json) 的冻结 P5：OGSOD exact EO 恢复率 0.113744、relaxed-SAR null 0.089751；SpaceNet6 分别 0.147560/0.066432。对应 bootstrap effect 为 +0.023993 [0.019786, 0.028442] 与 +0.081129 [0.049785, 0.118844]。存在正差，仍因 recovery/best-null 门不足而 `KILL_NO_SHARED_ERROR_COMPLEMENT`。本轮只读其指标段、decision 及 [governance](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/error_complement_d0p_20260829/GOVERNANCE_STATE.json)，未重审全部逐对象预测；不把正差冒充模型改进。
- [CAP-D primary JSON](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/ogsod_capability_v2_clean_20260831/gates/cap_d.json)：2120 图的固定 SAR decoded candidate bank，joint oracle mAP headroom +1.3452 pp [0.6987, 2.1794]，class +0.8664、veto +0.9237 pp；终态 `KILL_NO_PRACTICAL_FIXED_BANK_HEADROOM`。这个结论限定候选库、冻结分组与门槛，既不是端到端 KD 理论上界，也不是“RGB 没有信息”。

旧《配对质量与蒸馏潜力评估》把“配对好、教师强”直接排成潜力星级，并宣称特征模仿已判死、定位已近天花板，超过以上证据。当前应分别写**数据/标签对应、配对描述子、检测器错误互补、训练期载荷与最终学生效果**。

## 2. 外部方法复现：确切做到哪一级

### SpaceNet6 原公开实现 R1

实际读取 [R1 三 seed JSON](../../03_现行工程/SpaceNet6_OTD_official_reproduction/archive/governance/R1_THREE_SEED_SUMMARY.json)，并抽查 [baseline 终态 receipt](../../03_现行工程/SpaceNet6_OTD_official_reproduction/archive/governance/R1_BASELINE_SEED168120232_TERMINAL_RECEIPT.json)：Faster R-CNN 系列，固定 200 图 SAR test，COCO bbox AP50，E60 终态，seeds 168120232/3407/20260820。完整 evaluator 源码、全部日志及权重本轮 `unavailable`。

| R1 | 逐 seed AP50 | mean±SD | 论文目标 AP50 |
|---|---|---|---:|
| Baseline | 46.2 / 45.4 / 47.0 | 46.20±0.80 | 45.7 |
| AKD | 44.4 / 44.9 / 46.2 | 45.17±0.93 | 47.6 |
| released OS-SSL | 53.7 / 52.4 / 52.7 | 52.93±0.68 | 54.6 |
| Full | 52.9 / 53.2 / 51.7 | 52.60±0.79 | 56.2 |

可报告：baseline 接近论文；OS-SSL−baseline +6.733±0.929 pp、3/3 正；released AKD−baseline −1.033±0.681 pp、3/3 负；Full 论文数值没有复现。**OS-SSL 的正效应复现，不等于 54.6 精确数值已经复现。**

严格边界：公开材料缺 TIFF→PNG 精确规则、完整 SSL 数据/切块/seed；released SSL metadata 是 500ep+增强，论文写 300ep+无增强。原公开最终 checkpoint 的 R0 输入身份门未解，不能叫 bit-exact 作者复现。已存在 [AKD 几何诊断](../../03_现行工程/SpaceNet6_OTD_official_reproduction/docs/AKD_GEOMETRY_ALIGNMENT_DIAGNOSTIC.md)：SAR/框水平翻转，离线 optical feature 保持原方向；本轮读取诊断原件，没有重新执行该算子。这限定 released implementation 的结论，不证明作者论文或所有 AKD 实现无效。

### YOLO OS-SSL v2：适配复现中的强正例

冻结 [method contract](../../03_现行工程/SpaceNet6_OTD_official_reproduction/archive/governance/YOLO_OSSSL_3SEED_V2_METHOD_CONTRACT.md) 为 YOLO11n、10,000 SSL optimizer updates、physical B64、同 seed SSL→detector；四臂 native/SAR-only SSL/shuffled/paired、seeds0/42/123、final epoch；不是论文完整 300/500ep SSL。

| 数据、模型与评价 | paired−native mean±SD | 逐 seed 0/42/123 | paired−SAR-only | 结论 |
|---|---|---|---|---|
| SpaceNet6，YOLO11n E300/640，test200 AP50 | +3.328±0.909 pp | +2.401/+3.367/+4.217 | +3.771±1.231 | 三 seed 胜全部控制 |
| OGSOD，YOLO11n E400/256，development test3667 mAP | +0.195±0.423 pp | +0.539/+0.322/−0.277 | +0.103±0.427 | 不稳，预设门 FAIL |
| SiXiang，YOLO11n E300/512，scene-clean val304 mAP | −0.449±0.961 pp | −0.152/−1.523/+0.328 | +1.040±1.685 | 不胜 native，FAIL |

以上直接核自 [SpaceNet6 analyzer JSON](../../03_现行工程/SpaceNet6_OTD_official_reproduction/archive/governance/terminal_analyses/yolo_osssl_v2_spacenet6_full_3seed.json)、[OGSOD JSON](../../03_现行工程/SpaceNet6_OTD_official_reproduction/archive/governance/terminal_analyses/yolo_osssl_v2_ogsod_full_3seed.json)、[SiXiang JSON](../../03_现行工程/SpaceNet6_OTD_official_reproduction/archive/governance/terminal_analyses/yolo_osssl_v2_sixiang_full_3seed.json)。原训练 CSV/权重未重读。配对 donor map 固定为 seed42 的一张 derangement，三 optimizer seeds 不等于三张独立 null map。正例属于**跨模态预训练**，不能充当新增监督 KD 收益。

历史 OGSOD 的 FGD/LD/CMD 不能统称原论文精确复现：FGD 删去 global relation，应称 focal-only；LD/CMD 为 YOLO11 适配；旧 FGD/LD/CMD checkpoint CRC 失败由原审计记录。CoLD、FED-CHDistill、YOLO-CMFM 的论文数值/当前完整复现原件未在本轮读取，标记 `unavailable`，不因代码或项目目录存在就宣布复现成功或失败。

## 3. 监督 KD 与原创方法的现有终局

| 版本 | 本轮核到的证据 | 科学范围 |
|---|---|---|
| OGSOD full RGB-CMD matched exact400 | 原审计表：严格 seeds42/123，paired−H_S −0.262±0.204 mAP pp；paired−shuffle +3.053 仅 s42 | 配对重要仍可输强 same-modal；L20 原始报告/CSV 本轮 unavailable。不能用含 s0 的描述均值差 −0.389 pp 替换严格 n=2 delta |
| OGSOD H_F source-swap direct50 | 原 JSON 的 s42/s123 为 −0.885/−0.978 pp；严格均值 −0.9315±0.0658 pp | 仅 P3/P5 feature 换 RGB source、其余 SAR relation/output 固定的短协议有害；不是全部 feature KD 的否定 |
| SiXiang full RGB-CMD A8+A9 | 已读 accepted 12-cell 表：paired−H_S +1.0507±0.9002 pp；逐 seed +1.367/+0.035/+1.750；paired−shuffle +5.6637±2.6640，paired−weight0 +1.6510±1.3565 | scene-clean val 的 `EXPLORATORY_PILOT`；3/3 正但 s42 极小。A9 替换两格工程失败，不增加独立重复。已有 CMD 适配的 premise，不是新方法 novelty |
| SX-APR B2+B2E | 21 logical cells；已读 union 表，六个预设门全部 FAIL；P−H 均值 +0.296 pp，P−U1 −0.007 pp，P−前景 GT gate −0.804 pp、3/3 负 | `REDESIGN`，不能继续写“盲态运行中”；六门失败不是六个均值皆负 |
| MM-ARCS SiXiang M1 | 已读 committed joint JSON：前瞻42/123为 −2.663/+0.905 pp，1/2 PASS，mean −0.879 pp；seed0明确不计入 | `DEFER_PORTABILITY_HETEROGENEITY`，不能事后凑成2/3通过；只P−H也不能完成RGB内容归因 |
| MM-ARCS OGSOD M0 | 已读 TERMINAL_RECEIPT：`DEFER_ANALYZER_INTEGRITY`，no accepted commit-last bundle，正式 analysis attempts=0 | 证据工程阻塞，不能报告方法排名、正负效应，不能手算 embargoed CSV 替代 |
| SpaceNet6 P3 optical DFL | 已读 full622 locked FP 九行 CSV；YOLO11n，622 train/E300/B32-A2，三seed final，rect test200。native/P3 mAP29.747±.212/29.864±.904；Δ +.117 pp，逐seed −.748/+.664/+.435，AP50 Δ−.003 pp | development +.69 AP50 没有稳定转成最终收益；不能宣称定位KD普遍稳定。后续监督臂共用OS-SSL初始化，微小边际不推翻SSL正例 |

直接证据：

- [H_F/H_S JSON](../../03_现行工程/ogsod400_clean_protocol/refine-logs/HF_HS_THREE_SEED_ANALYSIS_20260717.json)。该旧 JSON 聚合含 s0；正式 n=2 范围依据 OGSOD 原审计，本轮没有重算聚合。
- [A8+A9 accepted 表](../../03_现行工程/ogsod400_clean_protocol/results/sixiang_rgb_attr_v1_20260722_a8_a9_composite/sixiang_rgb_attr_a8_a9_final.md)、[SX-APR union 表](../../03_现行工程/ogsod400_clean_protocol/results/sixiang_anchor_residual_v1_20260722_union_b2_b2e_analysis/sixiang_anchor_residual_union_final.md)。这些是 analyzer 逐 run 输出，不是完整训练曲线。
- [MM SiXiang joint JSON](../../03_现行工程/ogsod400_clean_protocol/refine-logs/mm_arcs_sixiang_p3far_m1_joint_analyzer_recovery_v21_20260727T235200/committed_evidence_mirror/JOINT_ANALYSIS_BUNDLE.json)、[MM OGSOD terminal](../../03_现行工程/ogsod400_clean_protocol/refine-logs/mm_arcs_m0_exact400_20260727_050437/TERMINAL_RECEIPT.json)。
- [P3 full622 CSV](../../03_现行工程/SpaceNet6_OTD_official_reproduction/runs/p3_full622/locked_test_fp/results.csv)、[锁定评价说明](../../03_现行工程/SpaceNet6_OTD_official_reproduction/docs/P3_FULL622_LOCKED_RESULTS.md)。

**SpaceNet6 test 的额外边界：**R1 与 YOLO OS-SSL 文档已经使用同名 fixed/frozen test200，日期早于 P3 full622 的 locked 评价。因此本轮能确认的是“P3 方法冻结、九端点齐后统一评价”的阶段性锁定，不能把这 200 图称为整个项目历史从未读取的独立测试。本轮未逐图比对两份 annotation，严格字节同一性仍 unavailable。

## 4. LADD 历史：为何早期漂亮数字不能继承

已读 [reload 混杂原记录](../../06_历史工程_只读/LADD_public/docs/experiments/LADD_RELOAD_CONFOUND_20260623_CN.md)、[7/12 结果报告](../../06_历史工程_只读/LADD_public/docs/reports/LADD_PROGRESS_REPORT_POST_RELOAD_20260712_CN.md)、[rescue 最终审计](../../06_历史工程_只读/LADD_public/docs/experiments/DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706_CN.md)，并抽查 [rescue 逐行 CSV](../../06_历史工程_只读/LADD_public/figures/jstars_v1/tables/T-DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706.csv)。

- 旧 no-mosaic baseline/LADD/reload mAP 为 55.654/57.662/57.982；原 baseline 未排除额外 continuation 预算，不能把约+2 pp全部归给分解或RGB。
- 旧 YOLO-init exact400 seeds0/1/2：plain +0.305±0.605 pp、singleproj +0.682±0.738 pp，均2/3正；这些是历史协议分类，原报告自己注明不少行 provenance 尚 provisional。
- registered rescue YOLO11n E400/B64/256、MuSGD lr.001、no-warmup、cosine、BN冻结：plain−det −8.530±.442 pp，singleproj−det −6.874±.489 pp。seed42/123复用 seed0 teacher/A1，多条 source/BN/phase/schedule 轴同时变化，否定该 rescue 配方，不能隔离为“LADD机制的纯因果效应”。
- 历史 KD-to-z/KD-to-u/no-probe 56.416/56.717/57.159 弱化预期 shared/private 语义，但这些原 run 未在本轮打开，限作历史诊断转述。
- 本轮直接读 [identity modules.py](../../03_现行工程/ogsod400_clean_protocol/methods/ladd_identity/src/ladd_identity/modules.py) 与 [loss.py](../../03_现行工程/ogsod400_clean_protocol/methods/ladd_identity/src/ladd_identity/loss.py)：`private=x-shared; corrected=private+shared` 代数等于原特征，KD 为全空间 smooth-L1，无前景 mask。只能讨论这个具体 identity 实现，不能把所有 LADD/LCSR/CoRe 版本都称恒等。
- CoRe-LADD 的 accepted detector 效果原件本轮 unavailable；不应把“有实现/静态测试、尚未找到终态”写成已科学证伪。

## 5. 明确撤回或降级的旧说法

1. **FreqMix“SiXiang +9 AP”撤回。** 已读 [20260905 小文件快照](../2026-09-05_audit_跨模态蒸馏全项目复盘/replication_sources.json) 的原路径、E400最后行和run_status：实际 OGSOD/256/E400/B64，gray donor s42/123 mAP47.893/47.752，SAR donor47.939/47.692；既有native50.226/50.661，H_S51.279/51.042。已有统计为gray/SAR−native −2.621/−2.628 pp。连“+9来自自模态增强”也不成立。快照不是本轮远端重采；完整字节/权重链 unavailable。
2. **FreqMix SAR-self 并非纯 SAR null。** 9/5源码摘录记载 SAR donor 的幅度 RMS 仍来自 shuffled gray，还带 H_S KD；不能给它纯SAR零光学身份。两个seed的负结果不能推广所有频域方法。
3. **“定位/分布级KD是唯一多数据集稳定正的家族”降级。** P3 locked结果、OGSOD/SiXiang router现存结论均不足以支持。旧稿的P3 paired−shuffle +.112 pp还可能误取PEMT数字，缺P3原件不引用。
4. **OGSOD D1/D2“类别级才是主通道”降级。** [findings.md](../../03_现行工程/SpaceNet6_OTD_official_reproduction/findings.md)记录same-class +.657、exact-pair +.391仅seed42，D2降低NLL却输confidence-only D1；same-class signal可能涉及尺度/校准。所引`runs/cross_modal_component_router/seed42_decision.json`本地缺失，远端禁止访问，本轮标`unavailable`，不能作为独立核验的通道裁决。
5. **“配对好/教师强/特征相近 ⇒ 学生能收益”撤回推导。** 有三种独立反例：D0 context signal但无对象超额，CAP-D有小oracle空间但未过门，full-CMD paired>shuffle却输H_S。
6. **“SpaceNet6已完成真实板端部署效果”不成立。** full622部署说明仍记录`not_run_on_board`；QCS6490 AOT compile不是硬件AP/延迟实测。本轮未打开板端包原件，不能给板端通过结论。

## 6. 完整性检查与可引用范围

| check | status | 本轮具体覆盖 |
|---|---|---|
| gt_provenance | warn | 读两数据YAML、SiXiang split审计、SpaceNet6协议及D0标签来源；未读全图像/annotation；继承SAR标签风险已显式列出 |
| score_normalization | warn | 核JSON metric_scale、COCO AP50 receipt、P3逐行CSV header，分清AP/precision/NMI/恢复率；未重跑指标实现 |
| result_existence | pass（局部） | 上述A8/SX/MM terminal、OS-SSL/R1 JSON、P3 CSV实际存在并已读；router原件缺失单列unavailable，不由摘要补齐 |
| dead_code | warn | 只静态读identity两个源码文件；未重建历史执行bundle；文档中的AKD翻转缺陷按其历史诊断身份陈述 |
| scope | pass（本报告） | 逐项约束数据/模型/split/种子/端点；A9修复不增n，n=2不升级三seed，方法/预训练/代理/工程状态不合并 |
| eval_type | mixed | detector AP为real_gt；D0 NMI为self_supervised_proxy；D0+/CAP-D为依赖GT的诊断/oracle，非新detector效用；源码/receipt是工程证据 |

主要 claim 裁决：

- `C-DATA-IDENTITY`：supported，OGSOD与SiXiang不是同一实验数据定义；绝对图像不重叠未核。
- `C-SN6-OSSSL`：supported（限定协议），四臂三seed短剂量YOLO预训练正信号。
- `C-SX-CMD`：needs_qualifier，A8/A9受限开发premise，有时间块修复与一张donor-map边界。
- `C-SUPERVISED-NEW-METHOD`：unsupported，当前所核历史证据没有稳定成立的自研监督KD主方法。
- `C-FREQMIX-PLUS9`：unsupported，已知数据身份与比较对象错误。
- `C-ALL-FEATURE-KD-DEAD`：unsupported，局部版本负结果不能升级普遍否定。

本轮没有为未打开的远端/完整raw链给pass；没有核验或重算输入哈希，`audited_input_hashes=not_computed_by_explicit_scope`。上述链接为declared input set的主要可引用证据，另读工作区README、实验索引、AGENTS.md、9/5两篇研究综述及证据总表作入口校准。**审阅完成指此文件落盘，不表示未完成实验、原始复现或历史证据缺口已被完成。**

## 7. 对主报告的交叉复核追加（2026-09-09）

复核者仍为 `/root/audit_rgbsar_history`。只读 [主报告](../../07_研究分析/项目三线进度与证据总整理_20260909.md)，范围限RGB–SAR数据、外部复现及历史方法段；复用上文已经检查的证据，不扩大检索、不改主报告、不计算哈希。以下行号对应本次读到的版本。

**总体：主要数字、RGB–SAR并非无空间的边界、LADD/SX-APR/MM终态和SN6阶段性test锁定的处理正确。有两项语义必改、一项评价范围应补齐。**

| 项 | 主报告位置与原文 | 问题和具体修订 |
|---|---|---|
| 必改 F1 | 第37行“相对强 SAR 教师为 +1.051±0.900 pp” | 比较对象是接受same-modal KD的学生锚点 `H_S`，不是冻结SAR教师detector。改为“相对强同模态SAR蒸馏学生锚点H_S，mAP提高+1.051±0.900 pp”。A8表H_S均值40.3453与历史SAR teacher/baseline约38.979不是同一行，不得混称。 |
| 必改 F2 | 第26行“比 OGSOD 有更强的特定配对监督信号” | OGSOD/SX数据、类别、分辨率、训练时长与配对control协议不同，没有跨数据集匹配的“信号强弱”estimand。改为“在自身scene-clean协议上，full RGB-CMD相对H_S有受限开发正信号，但自研选择方法尚未稳定优于强控制”。保留不同结果，去掉跨数据集强弱排序。 |
| 补齐 F3 | 第136–137行R1和YOLO OS-SSL表 | 两行均应显式标“固定test200 AP50”。R1最好补“E60、seeds168120232/3407/20260820”；YOLO补“E300/B64/640、seeds0/42/123”。这能使读者直接看清较早复现已经评价test200，避免把其65.1 AP50与P3的29.9 mAP误拼。第166行同名test200曾使用的限制本身正确，无需删弱。 |

非阻断措辞建议：第29行“光学教师本身没有更强的信息”可改成“光学教师没有更强的检测表现”。现存baseline gap直接测得后者，并没有识别模态的信息论优势。第39行“旧CKA”最好注明相关错误主要来自早期RGB–IR探针，避免读者误以为D0的RGB–SAR描述子也被判无效。

已接受、不要求重写的范围：OGSOD三类/SiXiang舰船coarse10/SN6-OTD油罐任务身份正确；D0的NMI及object-context值没有写成AP；D0+/CAP-D阴性没有推广至端到端全部RGB–SAR；SN6 OS-SSL被归为外部预训练适配正例；P3 +.117 pp、2/3正与AP50近零正确；LADD的预算混杂和rescue多轴限制、SX六门失败、MM数据集分开的裁决均正确。主报告未被本agent改动。
