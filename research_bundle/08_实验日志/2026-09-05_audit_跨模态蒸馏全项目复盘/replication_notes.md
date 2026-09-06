# 复现线独立审计：FreqMix、94 重训、SpaceNet6

> 一句话结论：SpaceNet6 的 paired OS-SSL 有可信的三 seed 正增益，但后续监督蒸馏增量很弱；最显眼的 FreqMix“SiXiang +9 AP”是错误比较，远端原始 args/CSV 确认它实际是 OGSOD E400，且低于同主要 recipe 的 native 与 H_S。不能把所有尝试统称无效，也不能把历史总结当作已验证结果。
>
> 审计日期：2026-09-05；远端最后补证 18:55 CST。审计方式：本地定向只读文档/JSON，随后经主 agent 授权，用 SSH 别名 94 只读小型脚本、args、CSV、run_status 与结果 JSON。未加载 checkpoint、未执行 GPU 任务、未改服务器。只新增本文件与 replication_sources.json。

## 1. 范围、设置与证据等级

本审计不新跑方法、不更改已终结科学 gate。覆盖：
- 04_方法演化档案/2026-08_服务器环境总结 的实验总结。
- 03_现行工程/SpaceNet6_OTD_official_reproduction 的复现、OS-SSL、P3、方法库、能力审计。
- 94:/mnt/dataset/yudongfang/projects/ydf/freqmix_hs_gp94_20260819，以及相邻 tri_rank_94/results 的 OGSOD E400 native/H_S 与 SiXiang native/FreqMix。

下文 SN6/ 表示 E:/SHARE/光sar/03_现行工程/SpaceNet6_OTD_official_reproduction/。
远端 ROOT 表示 /mnt/dataset/yudongfang/projects/ydf/。

证据分级：
1. 远端直接读取的 CSV 末行、行数、run_status、args 和当前源码：本次新增核实事实。完整原始文件 SHA-256 与小产物内容/必要代码片段记录在 replication_sources.json。
2. 本地终态 analyzer JSON：有历史 CSV/audit 路径及 SHA，但本轮未重新读取其全部远端源。
3. 叙事总结 md：只能作为线索，发现冲突时由 1/2 级证据纠正。
4. 当前源文件不等于训练当时的完整执行 bundle；本轮没有重建所有依赖、初始权重与数据字节的训练时身份，不将 args 主要字段匹配升级为 bit-exact 控制。

## 2. FreqMix：本次远端补证得到的重大勘误

### 2.1 数据集身份：是 OGSOD，不是 SiXiang

历史 REAL_COMPLETE_ANALYSIS.md:10–23 将 e400_gray/e400_sar 的 0.478 当作 SiXiang，与“SAR300 0.388”相减，得到 +9 AP50-95。
远端事实与之冲突：
- ROOT/freqmix_hs_gp94_20260819/run_freqmix_hs_gp94.sh 明确 DATA=.../datasets/OGSOD-1.0，imgsz=256，lr0=0.005。
- 六个 e400_* 单元的 _trainer/train/args.yaml 均为 epochs400、batch64、imgsz256、SGD、lr0=.005、lrf=.01、mosaic0、nbs64。
- 使用的 ogsod_94_sar.yaml 为三类 bridge/harbor/storage_tank，train=images/train，val=test=images/test。
- 该 YAML 与 tri_rank_94/code/configs/datasets/ogsod_94_sar.yaml 的 SHA-256 完全相同：
  7b20fa2e2bae4f9825856c3cff6fc6783d4e08b753bce4520aedcc3e9db35173。
- 六个单元 CSV 均恰有400行，run_status 均 COMPLETED。

因此，“+9”首先是数据集身份混淆与训练协议混比，不能表述为已经真实获得的增强收益。

### 2.2 同数据集同主要 recipe 的端点结果

指标为 CSV 第400行 metrics/mAP50-95(B)，下表乘100，单位百分点；两个 seeds 的 mean±sample SD。

| OGSOD E400 方法 | seed42 | seed123 | mean±SD | 相对 native mean | 相对 H_S mean |
|---|---:|---:|---:|---:|---:|
| native_only | 50.226 | 50.661 | 50.4435±0.3076 | — | -0.7170 |
| H_S | 51.279 | 51.042 | 51.1605±0.1676 | +0.7170 | — |
| H_S + gray-shuffled FreqMix | 47.893 | 47.752 | 47.8225±0.0997 | -2.6210 | -3.3380 |
| H_S + SAR-shuffled FreqMix | 47.939 | 47.692 | 47.8155±0.1747 | -2.6280 | -3.3450 |
| H_S + gray-sign-randomized | 36.951 | 36.915 | 36.9330±0.0255 | -13.5105 | -14.2275 |

对应路径：
- ROOT/tri_rank_94/results/native_only_s42_e400/results.csv
- ROOT/tri_rank_94/results/native_only_s123_v2_e400/results.csv
- ROOT/tri_rank_94/results/hs_ogsod_gp94_20260818/e400_s{42,123}/results.csv
- ROOT/freqmix_hs_gp94_20260819/results/e400_{gray,sar,sign}_s{42,123}/results.csv

逐 seed 同向：
- gray−native = -2.333/-2.909 pp；gray−H_S = -3.386/-3.290 pp。
- SAR−native = -2.287/-2.969 pp；SAR−H_S = -3.340/-3.350 pp。
- gray−SAR mean 仅 +0.007 pp，且逐 seed 方向反转。

与 H_S 的 args 全字段比较，仅 data路径、project、save_dir、teacher_data 路径不同；data YAML 内容 SHA 相同，teacher_data 从 RGB 参考换灰度参考是方法操作的一部分。model、seed、训练超参、H_S route 与 SAR teacher路径均一致。与 native 的普通训练字段相同，但 FreqMix 多出真实 H_S KD 配置。代码目录不同且未完整追溯执行 bundle，所以这里只称“同主要 recipe 的现存对照”，不冒称全栈精确匹配。

结论：**不支持 FreqMix 的 +9 AP，也不支持“有大增益但其实是增强”的较弱叙事。现存 OGSOD E400 对照显示该配置低于 native/H_S。** 只有两个 seeds，这不是对所有频域方法的普遍裁决。

### 2.3 方法定义：SAR donor 臂也不是纯 SAR-only null

ROOT/freqmix_hs_gp94_20260819/code_run/src/ogsod400_core/pemt_v5_direct.py 当前 SHA：
082a738f04fff6a0d5d464a3d64d6b4f0ad94a7c884f1b90065d16cbe616a210。

源码 :595–604：
- gray_donor = gray.roll(1, dims=0)，gray_high=其高频。
- target_rms 来自 shuffled gray 高频。
- gray arm 直接取 gray_high。
- SAR arm 用 image.roll(1) 的 SAR 高频，但乘 target_rms / RMS(SAR_high)。
- 故 SAR donor 的空间高频来自 SAR，幅度仍依赖 optical-gray，不能称完全不使用光学信息。

源码 :1752–1763：
- student 前向使用 mixed image。
- H_S criterion 仍接收原始 clean SAR batch 与 mixed student predictions。
- args 显示 feature/relation/output teacher_source 均 SAR，训练 CSV 有非零 train/kd_loss。
- 这些 e400 臂不是无 KD 臂，也不包含 paired-gray counterpart；无法由这组实验直接估计 paired-specific 增量。

严谨表述是“灰度与 SAR donor、保持灰度 RMS 的两种扰动效果接近；两者均未超过现存 H_S/native”，而不是“纯自模态增强完全解释 +9”。

preflight_evidence.json 内 sar_only_metrics 是 launcher 对 best.pt 独立重评的值；本报告统一使用 CSV final epoch endpoint，不把 best 重评与 last 端点混用。

### 2.4 真正 SiXiang FreqMix 是另一条线

ROOT/tri_rank_94/mech_queue_fm_tw_94.sh 明确 SiXiang、E300、512、lr0=.01，arm=freqmix_paired/freqmix_shuffled，属于 native criterion 的输入修改，不是上述 H_S E400。
本轮已核查的结果：
- sx_native_only_s42_e300 mAP=.39509，s123=.40396。
- mech_sx/freqmix_paired_s42_e300 mAP=.10570，run_status COMPLETED。
- mech_sx/freqmix_shuffled_s42_e300 mAP=.39873，run_status COMPLETED。
- paired_s123 的 run_status 文本仍写 RUNNING，但根目录无终态 results.csv；本轮未查询进程，不能把这个历史状态称为当前活跃任务。
- 没有足够证据把该线声明为三 seed终结方法结论；s42 paired 大幅退化只作该实现下的诊断，不能补成“shuffled≈native≈sar-self 的多 seed事实”。

## 3. SpaceNet6 官方复现：有正也有负，失败原因须拆开

权威本地小产物：SN6/archive/governance/R1_THREE_SEED_SUMMARY.json:3–23。
12/12 单元终态接受；seed=168120232/3407/20260820；指标为固定200张SAR test的AP50。

| 方法 | mean±SD AP50 | 论文目标 |
|---|---:|---:|
| Baseline | 46.20±0.80 | 45.7 |
| AKD | 45.17±0.93 | 47.6 |
| released OS-SSL | 52.93±0.68 | 54.6 |
| Full | 52.60±0.79 | 56.2 |

- AKD−baseline = -1.033±0.681 pp，3/3负。
- OS-SSL−baseline = +6.733±0.929 pp，3/3正。
- Full−OS-SSL = -0.333±0.987 pp。
- 合理结论：baseline复现、OS-SSL正效应支持、AKD正效应不支持、Full论文数值未复现。

这不是 bit-exact 作者复现：SN6/docs/REPRODUCTION_REPORT_DRAFT.md:9–15 指出公开资料缺 TIFF→PNG、OS-SSL dataset/切块/seed。公开最终checkpoint在主输入AP50=37.6；诊断最佳52.0，未到56.2，R0判定输入身份未解（:60–84）。
发布SSL metadata显示500ep+增强+LARS，论文描述300ep+无增强（:55–56），不能混写同一协议。

实现风险也有具体证据：SN6/docs/AKD_GEOMETRY_ALIGNMENT_DIAGNOSTIC.md:5–15，SAR和框随机水平翻转，但离线teacher feature不随之翻转。AKD结论只限定 released implementation，不能由负结果推断“跨模态知识普遍不可转移”。

## 4. 被后续 P3 叙事掩盖的 OS-SSL 正例

本地终态分析：
- SN6/archive/governance/terminal_analyses/yolo_osssl_v2_spacenet6_full_3seed.json
- 同目录 yolo_osssl_v2_ogsod_full_3seed.json
- 同目录 yolo_osssl_v2_sixiang_full_3seed.json

各文件12个audited detector cells、3 paired SSL/detector seeds=0/42/123；:45为指标，:48为paired-native，:57为paired-SAR-only，:78为全seed全对照gate。

| 数据集 | 指标 | paired-native，mean±SD pp | paired-SAR-only SSL pp | 状态 |
|---|---|---:|---:|---|
| SpaceNet6 | AP50 | +3.328±0.909 | +3.771±1.231 | 3/3胜所有对照，PASS |
| OGSOD | mAP50-95 | +0.195±0.423 | +0.103±0.427 | native对比2/3，gate FAIL |
| SiXiang | mAP50-95 | -0.449±0.961 | +1.040±1.685 | native对比1/3，FAIL |

这是本地 analyzer，并有源CSV/审计SHA；本轮没有重新读取这些远端源CSV。
SN6/archive/governance/YOLO_OSSSL_3SEED_V2_GOVERNANCE_STATE.json:4 已 COMPLETE，18 SSL和36 detector完成；旧 docs/YOLO_OS_SSL_EXPERIMENT_TRACKER.md 是v1封存，不应读取其RUNNING/TODO为现状。

结论：SpaceNet6跨模态预训练确有可归因正信号，泛化到OGSOD/SiXiang不成立。后续监督P3的三臂都复用该初始化，因此“监督P3很弱”不意味着“OS-SSL很弱”。

## 5. P3监督蒸馏：开发集正例没有稳定转成锁定测试增益

本地原始汇总 SN6/runs/p3_full622/locked_test_fp/summary.json 已与
SN6/docs/P3_FULL622_LOCKED_RESULTS.md:25–38 核对：

| 方法 | locked mAP50-95 mean±SD |
|---|---:|
| native | 29.747±0.212 |
| paired P3 | 29.864±0.904 |
| same-modal | 29.629±0.505 |

P3-native AP50/AP75/mAP = -0.003/+0.325/+0.117 pp；逐seed mAP=-0.748/+0.664/+0.435。
开发集 +0.69/+0.82/+0.58 pp 不应替代这个final locked结果。
collect等权辅助诊断mAP=-0.439 pp[-1.039,+0.098]（文档:46–57）；estimand与全体AP不同，只作异质性诊断。

teacher质量仍强：locked optical为91.646/59.010/54.600，SAR teacher为63.886/21.490/28.682 AP50/AP75/mAP；“教师强”成立，但没有推出student可达收益。

方法库：SN6/refine-logs/FP_METHOD_ZOO_FINAL_CONCLUSION.md:27–43：
- PEMT seed42相对P3 mAP+0.440；
- 三seed均值变成AP50 -0.271/AP75 +0.567/mAP -0.055；
- paired PEMT相对shuffled mAP+0.437，但仍没超过简单DFL。
“能够保留配对信息”和“能够训练更强detector”是两个不同问题。
旧LADD/APR/PRW/RankKD/HSCARD/POPA/GBRD、QAT selector、RCFMD/CMV见
SN6/docs/METHOD_INVENTORY_AND_EVIDENCE.md:80–193。
没有证据支持继续将失败组合换名或扫更多权重来恢复原claim。

部署链值得保留但要缩小表述：
- 同fixed640链 P3-native FP mAP=-0.019，fake-W8=+0.337，ORT-QDQ=+0.226 pp。
- rect FP与fixed640的AP75方向不同，微小效应对评估布局敏感。
- 九个QCS6490 AOT context只证明可编译；真实板端仍not_run_on_board，不能称测得延迟/能效/AP保持。
- 直接依据 SN6/docs/P3_FULL622_LOCKED_RESULTS.md:59–118。

## 6. OGSOD能力审计：小正效应未达实用门，不是模态方向无价值

本地原始JSON：
SN6/runs/ogsod_capability_v2_clean_20260831/gates/cap_d.json:43：
joint mAP headroom +1.345 pp[0.699,2.179]；class +0.866；veto +0.924。
CI下界均正，但joint未达2pp、class/veto未达1pp，终态KILL_NO_PRACTICAL_FIXED_BANK_HEADROOM。

该结论只针对固定SAR proposals和class-specific candidate bank内的决策修正空间，
不等于“RGB无信息”或“end-to-end OGSOD跨模态KD不可能”。
SN6/refine-logs/ogsod_capability_v2_formal/EXPERIMENT_TRACKER.md:12–19：
teacher/donor中止弃用；CAP-A/B/C未运行。
v1 DEFER_GROUPING（SN6/findings.md:157–162）发生在前检，未开正式训练；这是协议/数据分组不足，不是机制科学失败。

## 7. 叙事总结中的可定位错误

1. 04_方法演化档案/2026-08_服务器环境总结 的 COMPLETE/FINAL/REAL 多篇自称2025-01-28，却引用2026-08实验；含未验证的300ep性能估算和“包装叙事”建议，不能作终态科学证据。
2. REAL_COMPLETE_ANALYSIS.md 的“SiXiang +9 AP”与本次远端数据身份不符；05_实验证据_按服务器/实验证据对照总表.md:152/159/199 又传播该结论。
3. 07_研究分析/配对质量与蒸馏潜力评估_20260905.md:27 的“FreqMix shuffled≈native≈sar-self，+9来自增强”比现有证据更强，且与本次OGSOD native/H_S事实冲突。
4. 同文件:26给 P3-DFL 的 paired-shuffled +0.112 pp，疑似来自 FP_METHOD_ZOO_FINAL_CONCLUSION.md:40 的 PEMT AP50 +0.112；不能混用方法身份。
5. 同文件把P3开发集+0.69标“量小但稳”，没有承载locked-test反转，应注明development。
6. SN6/docs/PROJECT_WIDE_AUDIT_CORRECTIONS_20260901.md:21–28 已记录旧PRW winner、已终结线写成运行中、方向读反等错误；说明必须由终态产物反向校准叙事。
7. CURRENT_STATUS_AND_NEXT_STEPS.md 也含历史凭据；不在本审计复制其值。此前仅避开 FINAL_SERVER_SUMMARY.md 不足以保证其他环境总结可整体复制。

## 8. 价值判断、局限与下一步

结论不能简化为“全项目方法都失败”。SpaceNet6 paired OS-SSL确有三seed正例，现阶段失败主要集中在强初始化/强native之上的监督额外增益，以及将教师/表征优势变成SAR可达的任务收益。
同时不能维持“FreqMix已获+9且只是归因不足”的乐观叙事；本次补证说明收益本身源于错误比较。

建议：
- 总览将每个结果绑定数据集、split、指标、训练阶段、recipe、seed和端点；先撤销+9，再考虑任何新跑法。
- 对SpaceNet6保留OS-SSL预训练正例与监督P3边际弱的分层叙事；锁定测试已访问，不能继续调参救援。
- 对新RGB-T线先补协议匹配no-KD/same-modal/shuffled，用最小效应确定任务收益；不要直接从“教师强/CKA提高/可解码”推出迁移成功。
- 如将FreqMix作基线，必须重新定义完全不使用光学幅度的SAR-only null、区分H_S与native criterion；但当前没有实证理由优先继续投入该配置。
- 远端旧RUNNING文本不代表活进程；本次只查文件，没有做训练进程盘点。
- 不把两seed的FreqMix负差升级成“所有频域方法失败”；不把固定候选库headroom外推为端到端理论上界。

## 9. 产物路径

- 本文件：08_实验日志/2026-09-05_audit_跨模态蒸馏全项目复盘/replication_notes.md。
- 小型远端证据：同目录 replication_sources.json（56个去重文件记录，含路径、SHA、CSV末行/行数、args和必要脚本；另含带行号的运行库摘录与12组args差异）。
- 权重/数据仍在原服务器位置；本轮未下载大文件、未移动覆盖原始结果。

