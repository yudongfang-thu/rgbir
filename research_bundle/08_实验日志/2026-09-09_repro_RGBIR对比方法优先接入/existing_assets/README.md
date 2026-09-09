# 既有外部方法资产：本地有界核查

**若只选一个最小实际接入，建议复用 CMDistill-corrected 的既有 YOLO11n 训练链，明确标 `PROTOCOL-ADAPTED`。它已具备两个数据集配置、PCCFD/SLRD/IBCLD 三部分、22 个测试函数及双数据集各三 seed 的历史 E200 端点；这不等于作者代码复现，也不表示旧数字已能与当前 N/C0/C1 正式相减。**

2026-09-09，执行者 `/root/baseline_feature_analysis`。只读本地当前工程/历史归档/已保存小回执，补读三个作者仓库主页；未 SSH、GPU、加载权重、重新计算 AP、改旧源或全局索引，未计算新 hash。新产物为本文、[assets.json](assets.json) 和只读取旧记录的 [build_asset_summary.py](build_asset_summary.py)。20 个列出资产均存在；六份 CMD 原指标及三份 CCLKD 独立评价已提取到小 JSON，保留来源容器与原远端路径。没有把本轮文件存在性检查称为独立科学验收。

## 五个方法究竟已经有什么

| 方法 | 可复用的本地实物 | 已执行与缺口 | 当前接入判断 |
|---|---|---|---|
| CMDistill | 当前 `tools/train_rgbt_cmdistill.py`、`yolo_osssl/rgbt_cmdistill_kd.py`、双数据集 YAML、22 个 CPU 测试函数；历史 LADD 另有一条 CMDistill-style 线 | corrected/adapted 在 Drone、LLVIP 各三 seed E200 完成；未绑定作者可运行仓库，未追认当前文件与历史训练源逐字节一致 | **现成工程最多、优先最小接入**；只保留 corrected 身份 |
| CCLKD | 当前 `train_rgbt_cclkd.py` / `rgbt_cclkd_kd.py`、双 YAML；历史 LADD 有 online HBB 及 YOLOv5 joint trainer | 当前 RGBIR 已执行三 seed 仅 LLD+CCL；FLD/RLD 函数存在不等于正式 trainer 调用了它们。旧完整训练源快照缺口仍在 | 可复用组件和历史 partial 端点；不是最小的完整 CCLKD 复现 |
| FGD | 历史 LADD 的 `_fgd_style_loss`、launcher、smoke；当前外部准备有作者源路径/YOLOX 配方 | 历史 FGD-style **没有可训练 Global context**；旧局部/不同数据集结果不代表完整 FGD。当前 RGBIR 完整 FGD 尚未实现/训练 | 能复用 mask/attention 思路，需补原 Global 与 optimizer 接线，不能只换名称 |
| LD | 历史 LADD 的 DFL Main + teacher-quality VLR-style loss；当前准备有作者 Main/VLR/head/ATSS 方案 | 历史 VLR 不是作者 ATSS/DIoU 区域；更早版本甚至误蒸馏分类 logits，已被原说明作废。无当前两数据集完整 LD 端点 | 需要明确 Main/VLR、16-bin 与 TAL 旁路适配，非立即可用完整原方法 |
| BCKD | 作者仓库/配置与损失路径、20260907 本地可用性 JSON及适配 proposal | 在本次限定目录内未发现当前 RGBIR 训练实现或端点。仅 BCDL 分类项是 partial，完整 BCKD 还含定位 | 作者代码来源清楚，但当前移植工程尚未完成；不是现成训练入口 |

“当前 RGBIR 未实现”与“历史工程没有任何代码”不同。历史 LADD 路径有大量 FGD/LD/CCLKD 适配和旧结果，不能为了省工程把它们的 SAR→RGB 方向、OGSOD/SiXiang、YOLO11n/s/x、E400/E800、best/current 指标混入现在的 Drone/LLVIP RGB-only HBB 对照。

### 精确源码入口

当前工程根为 `03_现行工程/SpaceNet6_OTD_official_reproduction/`：

- CMD：[trainer](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/tools/train_rgbt_cmdistill.py)、[loss](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/rgbt_cmdistill_kd.py)、[同步配对 adapter](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/rgbt_hnewa_pairing.py)、[Drone 配置](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/configs/research/rgbt_cmdistill_protocol_drone.yaml)、[LLVIP 配置](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/configs/research/rgbt_cmdistill_protocol_llvip.yaml)、[loss 测试](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/tests/test_rgbt_cmdistill_kd.py)、[trainer 测试](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/tests/test_train_rgbt_cmdistill.py)。本轮只核测试存在/内容，未再次运行它们。
- CCLKD：[当前 trainer](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/tools/train_rgbt_cclkd.py)、[当前 loss](../../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/rgbt_cclkd_kd.py)。历史完整结构另见 [online HBB trainer](../../../06_历史工程_只读/LADD_public/cclkd_reproduction/code/train_cclkd_online_hbb.py) 与 [YOLOv5 trainer](../../../06_历史工程_只读/LADD_public/cclkd_reproduction/yolov5_sanity/code/train_yolov5_cclkd_full.py)。后两者不是当前 frozen-IR partial 的同一实现。
- 历史 FGD/LD：[共用 loss.py](../../../06_历史工程_只读/LADD_public/ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py)、[FGD 限定说明](../../../06_历史工程_只读/LADD_public/comparison/fgd/README.md)、[LD 修正历史](../../../06_历史工程_只读/LADD_public/comparison/ld/README.md)。历史目录本地可读，不表示本轮已确认这些文件仍在 90 的同路径；本轮没有 SSH。
- BCKD/完整 FGD/完整 LD 的当前准备：[EXTERNAL_BASELINE_PREPARATION.md](../../2026-09-07_train_IndependentKD实施/EXTERNAL_BASELINE_PREPARATION.md)、[proposals](../../2026-09-07_train_IndependentKD实施/external_baseline_proposals.json)、[作者仓库元数据](../../2026-09-07_train_IndependentKD实施/external_baseline_availability.json)。后者是 metadata/tree 路径记录，不是已安装且验收通过的完整代码包。

## 作者代码与适配身份

本次重新确认 [BCKD 作者仓库](https://github.com/TinyTigerPan/BCKD)、[FGD 作者仓库](https://github.com/yzd-v/FGD)、[LD 作者仓库](https://github.com/HikariTJU/LD) 的主页可读。BCKD 原入口基于 MMDetection2.28.2/GFL，并非 YOLO11/RGBIR即插即用；有作者 checkpoint/log 链接，不代表它们已下载、可直接当本数据集 IR 教师。

CMD 当前 loss 注释称“没有官方代码”；本轮应收窄为：**在所核本地资产中没有已确认作者可运行仓库**，一次精确 DOI/GitHub 搜索也未返回结果，不构成全网不存在的证明。CCLKD 同样保留“未绑定作者实现”的限制，不把本项目 paper-structured 重写当作者原码。

CMD 当前 corrected 的不可省略适配：PCCFD 用 `1-r`、逻辑框项用 `1-IoU`；三层 PCCFD 等权、最深层逐图关系矩阵、未实现论文不明确的 adaptive layer；共享 dense anchor 解码，IoU 仅教师 max class≥.5，BCE 覆盖所有 anchors×classes；三个大项系数均1，平均 KD 以实际 B 乘回 native loss。不能把这些约定换成 C0/L2 的对象质量门后仍称同一 CMD。CLI 虽还接受 literal，正式最小接入必须明确只选 corrected。

三类作者方法迁移的关键差异已有 [原准备](../../2026-09-07_train_IndependentKD实施/EXTERNAL_BASELINE_PREPARATION.md) 支撑：BCKD 的学生差值权重不能 detach 或把 BCE 换 KL；FGD 要保留前景/背景/注意力/Global 四项及可学习辅助参数；LD 的 Main/VLR 选择、温度/reduction、GFL17 bins→YOLO16 bins必须声明。LLVIP 单类不适合验证跨类别 softmax 暗知识，不能为凑非零悄悄改 loss。

## 过去“三 seed”究竟是什么比较

以下 mAP50–95 为百分数，SD 为三 seed 样本 SD；顺序0/42/123。数字来自已有独立 JSON / 已接受汇总，不使用训练 CSV AP，也不是本轮重新评估。

| 历史结果 | seed0 | seed42 | seed123 | mean±SD |
|---|---:|---:|---:|---:|
| Drone 历史 native（cgkd_w1） | 54.357648 | 53.815556 | 53.687433 | 53.953546±0.355778 |
| Drone CMDistill-corrected | 53.900286 | 53.244840 | 53.669168 | 53.604765±0.332435 |
| Drone CCLKD partial | 54.066006 | 54.493295 | 54.332596 | 54.297299±0.215820 |
| LLVIP CMDistill-corrected | 34.060274 | 33.443988 | 33.316849 | 33.607037±0.397629 |

CMD 双数据集六端点远端根为 `/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_cmdistill_adapted_v2/{dataset}_seed{seed}_b32_e200/`，本地原值在 [历史小回执容器](../../2026-09-08_audit_项目与94最新全景/historical_inventory/remote_snapshot_20260908.json)。每臂 CSV 记录200行/末轮200；独立 `metrics_record.json` 指向 last、Drone val / LLVIP `evaluation_role=dev`。LLVIP 的 JSON `split=val` 是 YAML槽名，不代表官方 test。六份原 JSON已提取到 assets.json，并保留容器路径，未复制完整主机状态或权重。

Drone CMD−它的历史 N 为 −0.348781±0.291793 pp（3/3负）；CCLKD partial−历史 N 为 +0.343753±0.550510 pp（−/+/+）。[已接受比较汇总](../../2026-09-07_audit_RGBIR实施起点/comparator_analysis.json) 明确：历史 CMD/N 的 ordered roster 与现 OEv1 是否相等仍是 null，旧 native receipt 没有新 schema 要求的 terminal status，不能自动由新 analyzer 准入。LLVIP 此处没有同三 seed、同源完整控制核对，**不拿一个 mature N42 减三 seed CMD 均值写增益**。

CCLKD 三份新独立评价明确 `CCLKD-ADAPTED-PARTIAL-LLD-CCL-HISTORICAL`、`fixed_budget_last_ema`、full1469、fraction原值，来源 [RESULTS](../../2026-09-06_ops_OEv1优先级与对比实验/cclkd/RESULTS.md)。统一重评补了评价侧证据，没有修复旧训练源码绑定，也没补出 FLD/RLD。

这些历史训练 workers8；当前 N/C0 主线 workers4，两种设置已知会改变增强流。教师历史路径相同不是内容/版本完全相同的证明，学生新检测头也不能仅凭同一预训练文件路径追认逐张量一致。另 C0 直接利用 IR GT 与 RGB参考，CMD/CCLKD 核心 KD 没有同等额外监督，方案成本和信息权限要单列。因此旧 CMD53.60/CCLKD54.30 不能直接与当前 C054.69 相减，宣称打赢作者方法。

## 一个最小可执行接入建议

只准备 **CMDistill-corrected 新90 seed42**，双数据集共用现成算子/loader/evaluator，不新增大方法矩阵。以根代理选定的主数据集先完成技术接入，第二数据集只切对应已治理配置；本条不代替根代理的实际调度决定。执行前所需差异很有限：

1. 复制当前 `train_rgbt_cmdistill.py`、`rgbt_cmdistill_kd.py`、`rgbt_hnewa_pairing.py` 和原 `paired_detection.py` 到独立新 source 目录，复用已准备 native payload 的其他导入/guard/receipt依赖。输入必须为真正新90 IR教师；若94旧权重不可得，等待新90 IR baseline完成，不能填通用预训练或 fake R/T冒充教师。CMD 不需要 RGB参考模型。
2. 两配置只绑定新90普通预训练、完整 prepared YAML/mapping、实际 IR last身份；冻结原 E200/640/B32/nbs64/SGD/增强和明确 workers，与新90 N使用同一版本。仍标 `PROTOCOL-ADAPTED`。原 trainer 参数 `--workers` 默认8，CLI覆盖应显式记录；别只改 YAML误以为生效。
3. 先用现成22个测试函数做实际 pinned CPU 接线核对，再以原函数的真实 batch 技术 canary验 `boxes/scores/feats`、各项 finite/KD梯度、教师冻结、source/data/RNG 和性能。原 `--validate-only` 仅查部分文件，不证明这些；原 `--max-steps` 是 optimizer调用数，不保证 AMP成功更新。沿现有统一lease做有界测量，不复用旧 shell 的无限资源重试和固定10000MiB预估为实测。
4. 若复用新90 native baseline作为正式分母，先核共同学生初始化、RGB实际流、native loss/优化以及保存端点一致；有 wrapper差异就补最小同路径 weight0入口并留新身份，不能继续借旧94分母。历史 trainer 内部关闭验证，last独立 full-dev评价沿已接受接口绑定完整名单/GT数量/类别/度量单位；不要用旧 `eval_rgbt_detector.py` 的薄回执直接假定满足新合同。
5. 性能短测只用于单完整任务≤10h准入，不用 AP选择λ/种子/子集。历史 CMD 六次 CSV训练约 Drone5.83–7.86h、LLVIP3.09–4.21h，仅是旧94不同负载下的时间，不能承诺90速度。新的last端点出现后按实际共同协议报告；seed42不是三seed结论，且paired−N不等于IR独特贡献。

如果根代理要求“第一项必须作者代码原样复现”，该选择不满足：应改选作者仓库的独立方法，并承担其原框架/模型与本任务不同的适配成本。不能为了叫作复现，隐去 CMD 的 corrected / protocol-adapted 身份。

本轮建议只基于现成资产和接入成本，不基于选择一个历史表现较弱的对手。公开原论文的LLVIP/Drone适用性由另外两条独立文献任务核对；本文不重复给论文排行榜或新方法胜负判断。

## 本轮追加：BCKD 分类算子已实做

上文表格记录资产初查状态。根代理随后授权把 BCKD 准备推进到一个真实 CPU 算子小包：[bckd_operator/README.md](bckd_operator/README.md)。现已固定作者提交下载 kd_loss/config/head三原件，仅去decorator提取 `novel_kd_loss`，与独立 softplus实现对照loss/学生梯度，6/6小检查通过。明确确认默认beta1、实际分类温度1、教师detach而学生差值权重不detach，以及加权BCE不能静默替为KL；小回执保留首次Python3.8 metadata兼容错误和修正后通过。

因此“BCKD只有资料准备”现更新为“**BCDL逐anchor分类算子CPU复验完成，完整trainer/head/reduction和双数据集训练仍未接入**”。这不改变最小现成完整训练链为CMD-adapted的判断；若后续优先作者算子，BCKD分类partial已有可直接调用的小函数。assets.json保留初查快照，不把追加执行伪装成初查已有资产。

根代理随后在90现有Torch2.10环境复跑同一BCDL小检查，6/6通过，原回执已附算子目录。同时已准备 [cmd90_port](cmd90_port/README.md)：仅三份原源码、两份原测试与双数据集90模板，保留真实IR教师 `TO_BIND` 和 `PROTOCOL-ADAPTED` 身份；不创建新队列、不启动训练，不把CMD待教师绑定变成其他复现路径的总阻塞。
