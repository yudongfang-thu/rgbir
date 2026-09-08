# 首批选择覆盖的有限解释

**本批的 9/10 个相反状态对象，都是 GT 辅助一对一分配框的置信度差异；不能称 9 个实际可修复漏检、10 个教师检测错误，或其中 3 个是负迁移。** 一次 forward 没有训练或梯度证据。CPU 复核及逐对象 GT/框/门原值见 [interpretation_cpu.json](interpretation_cpu.json)，执行源见 [interpretation_cpu.py](interpretation_cpu.py)。原 producer、accepted analyzer 和阈值均未修改。

## 原值与门链

范围是 LLVIP 固定首个真实 train batch：32 图、80 个增强后 RGB GT、80 个配对 IR GT、1 个空 GT 图。80 对增强 GT 的 RGB/IR xyxy 在本批全部 exact 相同；S/R 的分配状态及 scalar evidence 也都是 80/80 exact。此批 19 个反向状态差异不是 RGB/IR GT 坐标不同造成。独立用导出 GT/box 重算 IoU，最大绝对差 2.16e-7，状态判定一致。

| 一对一分配框状态 | 对象 | 原 eligible | 原 selected |
|---|---:|---:|---:|
| T 正确、S 低置信 | 9 | 8 | 7 |
| S 正确、T 低置信 | 10 | 9 | 3 |
| S/T 都正确 | 54 | 43 | 19 |
| S/T 都未达正确标准 | 7 | 4 | 3 |

完整原计数为 matched=80、valid-region=80、reference-candidate=79、base=79、teacher-correct-base=78、eligible=64、selected=32、normalizer=79。`teacher_correct_own` 实际是 **78/80 true**，不是 80 全过；false 是 row17（S/T 均无 assigned candidate，非 base）和 row64（S/T assigned 均低置信，base）。

9 个 T-assigned 正确/S-assigned 低置信对象均有有效区域、R candidate 和 teacher any-correct。row13 的 q=0 被排除；row31 q=.005904、eligible rank51，未进整批前32；其余7个被选。10 个相反对象的 teacher any-correct 全部 true，row58 q=0，另6个 q 排名未入选，row2/11/44 入选。没有改按图配额或重选。

## 为什么 assigned 状态与实际 teacher 门不一致

旧 baseline 的空间分配先保留置信度≥.05 的 raw anchor，再作最大配对数量、其次最大总 IoU 的一对一匹配；置信度不会在匹配中优先排序。匹配完成后才检查 .25 置信度/.5 IoU/正确类别。它可能分配到定位很好的低置信框。其目标是整图的一对一总匹配，**不能没有全候选就说每个对象拿到最高 IoU 框**。

原 teacher 门则问“是否存在一个类别正确、置信度≥.25、对 own GT IoU≥.5 的 dense candidate”，不要求那个框正是一对一 assigned 框。因此 10 个 T-assigned 低置信对象的 true 门证明另有满足原判定的候选存在；只缺具体 anchor/box 身份，不是两套判断矛盾。全部 17 个 T-assigned 未达正确标准对象中，15 个通过原 teacher any-correct 门。这仍不等于 15 个 native NMS 后正确检出，因为 dense any-candidate 不执行 NMS，也不要求候选与 GT 一对一。

在已保存的同帧 T-assigned 框子集中，找到了 **1 个直接证人**：

| 本对象 | T assigned anchor / conf / IoU | 另一个已保存 T anchor / conf / 对本对象 IoU |
|---|---|---|
| row58，image24，稳定 ID 见 JSON | 7530 / .091382 / .794697 | 7571 / .806243 / .660525，原分配给同帧 row60 |

另一个框足以通过 row58 的原正确性门。其余9个反向对象在**已保存 assigned 框子集**中未找到可列出的正确替代框；完整 dense 候选及其 ID 没有保存，不能据此断言不存在替代框，也不能确定是单对象 IoU 优先还是一对一竞争造成了具体分配。此证人只解释原 gate 的存在性，不证明 NMS 后该对象检出。

q 也不是两个 assigned 框置信度的差。它来自 P3/P4 GT 前景/背景区域的原 evidence 汇总。例如 row13 的 S/T assigned conf 是 .085404/.815233，但 R/T region evidence 为 8.188158/8.077787，q=0；row44 的 S/T assigned conf 是 .845433/.068787，R/T evidence 为 .432264/4.431097，q=.488361，整批 rank1。两种观测针对不同 anchor/区域统计，不能把它解释为 selector 明知教师错误仍给错误框高权重。

## 19 个对象的原始分配原值

row 是冻结导出里的 0 起 `rgb_global_row`；完整 stable IDs、两模态 GT、assigned xyxy、逐层区域和所有门均保留在 JSON。未选以“—”表示。T正确/S错9个均为 S low_confidence，S正确/T错10个均为 T low_confidence；这些低置信框的类别及 IoU 都已过正确性门。

| row | 状态桶 | S anchor | S conf | S IoU | T anchor | T conf | T IoU | q | selected rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | S正确/T低置信 | 1345 | .284576 | .879735 | 6832 | .129843 | .965585 | .103033 | 10 |
| 11 | S正确/T低置信 | 6525 | .371137 | .742148 | 6566 | .157137 | .820933 | .024155 | 31 |
| 13 | T正确/S低置信 | 4368 | .085404 | .896810 | 4450 | .815233 | .803334 | 0 | — |
| 20 | T正确/S低置信 | 4146 | .178956 | .802556 | 7513 | .395134 | .882247 | .064985 | 16 |
| 25 | T正确/S低置信 | 7233 | .059647 | .741111 | 7194 | .777975 | .874476 | .051299 | 19 |
| 26 | T正确/S低置信 | 7276 | .128088 | .500482 | 7316 | .498047 | .925474 | .349574 | 2 |
| 28 | S正确/T低置信 | 3238 | .558327 | .752957 | 2997 | .140336 | .973655 | .021823 | — |
| 31 | T正确/S低置信 | 7525 | .157137 | .844692 | 7685 | .662796 | .959052 | .005904 | — |
| 33 | T正确/S低置信 | 5984 | .168309 | .776598 | 7872 | .693473 | .788137 | .127483 | 9 |
| 35 | S正确/T低置信 | 5191 | .731059 | .913582 | 5272 | .203075 | .956678 | .002582 | — |
| 44 | S正确/T低置信 | 6682 | .845433 | .791670 | 6762 | .068787 | .840094 | .488361 | 1 |
| 51 | T正确/S低置信 | 2268 | .094343 | .720705 | 7053 | .631594 | .846562 | .051461 | 17 |
| 58 | S正确/T低置信 | 7490 | .324235 | .935874 | 7530 | .091382 | .794697 | 0 | — |
| 65 | S正确/T低置信 | 7377 | .800068 | .871544 | 7577 | .086323 | .821187 | .021723 | — |
| 69 | T正确/S低置信 | 3361 | .130285 | .848528 | 7281 | .762778 | .947601 | .087069 | 12 |
| 70 | S正确/T低置信 | 3714 | .684264 | .858599 | 3633 | .074231 | .923161 | .001787 | — |
| 71 | S正确/T低置信 | 7247 | .603932 | .843504 | 3694 | .149531 | .870450 | .001181 | — |
| 72 | S正确/T低置信 | 7088 | .472683 | .859189 | 7128 | .120853 | .788391 | .007311 | — |
| 79 | T正确/S低置信 | 2931 | .172172 | .864045 | 2691 | .403567 | .919095 | .127769 | 8 |

## 对旧分析的限定与最小缺口

旧 LLVIP dev200 表中 146 个“IR 可修复”代理对象含90个 low_confidence，这个 **90/146 只能读成 GT 辅助一对一分配框低置信的桶构成**，不能直接指导实际 miss 数量或可达 KD 收益。不能把本批 train 的 proxy 差异率反向校正旧 dev200，更不能改写完整 dev AP/TIDE。9/10 同样只是本批 assigned-state 桶；被选双方都正确的19个也不能被称作“浪费”，没有梯度/训练结果支持该判断。

现有数据已经闭合实际配对、GT 坐标、原门/q/排序、全部分母、19个框判定、1个具体替代证人。**无法闭合 S/R/T 完整 any-correct 候选清单、另外9个教师替代 anchor、native NMS 后的一对一正确性**：这次产物没有完整 raw tensor/候选框，也没有 NMS 输出。只对80个 assigned 框做 NMS会漏掉其他真实候选，不能冒充原生推理。

下一最小有信息价值的跟进应保持同首批/frame/稳定 GT IDs/模型初始状态/AMP/mode/增强与原阈值，另立 attempt 记录 S/R/T 各自 any-correct 候选数量及至少一个 witness（anchor、class、conf、box、own-GT IoU），同时用 pinned native postprocess 的原 NMS 参数输出该批 post-NMS 框与一对一 GT 正确性。用 raw、assigned、post-NMS 三种口径的交叉表检验代理是否把其他候选仍能检出的对象计为低置信。原 selector、λ与训练矩阵均无需改变，不能用它计算完整 dev AP。若没有当次 raw 缓存，当前 JSON 无法纯 CPU 恢复这些候选；须由 root 另行决定能否按冻结 first-batch stream exact 重放一次推理，**本次复核没有启动该重放或任何 GPU/训练**。

证据入口：[原对象 JSONL](evidence_1252/probe/objects.jsonl)、[完成合同](evidence_1252/probe/completion_receipt.json)、[原导出合同](evidence_1252/probe/export_contract.json)、[accepted 汇总](analysis_1252/summary.json)。
