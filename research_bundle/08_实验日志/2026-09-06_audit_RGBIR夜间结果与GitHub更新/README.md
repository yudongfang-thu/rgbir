# RGBIR夜间结果与GitHub更新（2026-09-06）

> 21:46新快照：OEv1新增N0=54.346、P123=54.637，现3/6有效端点但0/3同seed完整配对；OS-SSL首次出现同seed paired−shuffled CSV差值：+0.669 mAP/+0.495 AP50。新证据与17:30审计一并追加到原GitHub分支，不据此升级三seed收益或归因结论。

## 目的
检查最新训练/评估结果，明确相对上一快照的变化，并使外部模型能通过GitHub独立复核。

## 设置
沿用monitor-experiment与analyze-results工作流；不改变冻结方法、阈值或训练队列。OEv1只认固定E200 last/EMA独立端点，OS-SSL区分CSV末轮与独立评估，并保留初始化混杂边界。发布分支research/full-evidence-20260906。

## 结果
### OEv1：新增两个可靠独立端点

单位：AP为百分数，差值为百分点。全部端点是E200 last/EMA、1469唯一图像的开发val；不使用训练CSV中错位或占位的AP。

| 臂/seed | 本轮mAP50–95 | 本轮AP50 | 历史同seed native mAP | 相对历史N的描述性差值 |
|---|---:|---:|---:|---:|
| weight0 / 0（新增） | 54.346185 | 76.992515 | 54.357648 | −0.011463 |
| paired / 42（保持） | 54.658162 | 77.069620 | 53.815556 | +0.842606 |
| paired / 123（新增） | 54.636847 | 77.304557 | 53.687433 | +0.949415 |

历史对照只提供背景，不能代替本轮同代码P−N。P42、P123接近，且分别高于各自历史N，是继续完成试验的正面线索；仅两个paired终点不能称作稳定三seed收益。N0与历史N0接近，说明该seed未见明显基线路径性能偏移，不能证明所有seed的loader/trainer完全等价。

| student seed | paired完成轮数 | weight0完成轮数 |
|---|---:|---:|
| 0 | 28/200，运行 | 200/200，已独立评估 |
| 42 | 200/200，已独立评估 | 135/200，运行 |
| 123 | 200/200，已独立评估 | 20/200，运行 |

截至21:46:12，3/6完整端点、0/3完整配对；三seed汇总保持null。N0/P123完成后队列已自动接续P0/N123，无需重启。当前N42按最近5轮速度还约4小时，P0/N123分别约11.7/12小时，受负载影响。

三个端点的train/eval COMPLETED receipt、E200 completion、指标副本、source引用、1469唯一val清单及跨端点清单相等性全部通过独立CPU核验。完整检查见[review_oev1_endpoints_complete.json](review_oev1_endpoints_complete.json)。首次本地检查因三个4.3MB配对manifest未被小文件采集器复制而失败；现已补齐，[receipt_supplement_sources.json](receipt_supplement_sources.json)记录来源。旧失败检查保留，不能误读为训练或指标失败。

### OS-SSL：新增paired123完成，以及第一个同seed内部差值

| SSL臂/seed | E200 CSV mAP50–95 | E200 CSV AP50 |
|---|---:|---:|
| paired / 123（新增） | 53.942 | 75.849 |
| shuffled / 123 | 53.273 | 75.354 |
| IR-only / 42 | 54.623 | 76.822 |

同seed paired123−shuffled123为 **+0.669 mAP / +0.495 AP50 pp**。这次差值可作为有方向的单seed内部观察；+0.495不能舍入成通过冻结的+0.5 AP50门，更不能替代三seed判定。paired123相对旧native123的CSV差为+0.244 mAP/+0.091 AP50，但旧native检测头初始化不同，不能把此差归给SSL。

21:46:07时3/9微调完成，shuffled0完成131/200并运行，余5个排队。仍无OS-SSL独立last评估，stdout的best不能补位；不把这些CSV值与OEv1独立last数值直接排名。详见[OS-SSL更新](osssl/README.md)及[原始复算](osssl/summary.json)。

## 结论
1. OEv1由1/6推进至3/6有效端点，新增第二个paired结果仍高于历史同seedN，且新weight0 seed0接近历史N0。信号比17:30更完整，但预注册同代码净收益仍未可计算，不能用两个P平均减单个N。
2. OS-SSL出现第一个同seed paired−shuffled正差，是配对作用的初步线索；证据仍限于一次SSL预训练、单个微调seed和CSV口径。尚不能宣布配对归因成立、迁移门通过或优于IR-only。
3. 17:31确认的W1与SSL检测头初始化混杂仍然存在，三SSL臂259个非骨干张量彼此一致的证据也保留。新增正差没有消除“同模板零SSL native”和“RGB-only SSL”两项控制缺口。
4. OEv1终态CSV列数缩短与progress旧running字段已定位为日志表现，独立评估不受影响；新N0/P123也出现相同CSV终行格式。本次采集只解释epoch/time/train-loss字段，原始CSV不改写。

## 产物路径
本目录保存采集/分析/发布脚本、原始小证据、独立检查和上传回执；大权重与数据集留服务器原位。

- OEv1源：94 `RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/`及`rgbir_object_evidence_expand_20260906/`。
- OEv1新快照：94 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/audit_rgbir_night_20260906_2144/`；本地oev1_snapshot/保存对应小产物和补采完整终态证据链。
- OS-SSL源：94 `RGBT_campaign/runs/osssl_ir_20260906/`、`artifacts/osssl_ir_20260906/`；本地osssl/有73个原始小文件及17:31初始化检查副本，日志片段明确标partial。
- 同次发布包含17:30晚间audit及本21:46快照；原08:53快照保留。GitHub分支：`research/full-evidence-20260906`，根LATEST_RESULTS.md作为最新阅读入口。

## 局限与下一步
保持冻结的训练/阈值/队列不变，先等N42形成第一个完整P−N，再等三seed。OS-SSL按原协议补独立last评估、另行冻结初始化一致的native与RGB-only SSL控制；本轮只读采集和发布，不增加GPU任务。

公开包不收录原始数据集、checkpoint、凭据、第三方论文全文或环境缓存。代码和数值按原字节保留，Markdown仅适配GitHub导航。快照非实时仪表盘；不以跨seed未配对数字、旧native或训练CSV替代同代码完整对照。具体发布校验、文件范围和Git提交以本目录PUBLISH_RECEIPT.json为准。
