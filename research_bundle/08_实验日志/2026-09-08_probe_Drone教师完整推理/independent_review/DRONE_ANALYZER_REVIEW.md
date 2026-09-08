# Drone 配对 CPU 分析独立验收

**ACCEPTED：`cpu_analysis/output_attempt2` 支持完整 dev 上固定 GT 关联的检测互补描述；不支持几何准入、可蒸馏收益或训练选择覆盖。**

源码及 8 项合成真值独立复核通过（`DRONE_ANALYZER_CPU_REPLAY.json`）。原 PairDataset、`_iou` / `_match_objects` 源与 pinned legacy 文件字节一致；三份原生匹配源及 helper 与已接受版本字节一致。attempt1 在任何检测统计前，因源码快照清单包含 `__pycache__` 目录而失败；修订仅为文件清单增加 `is_file()`，失败产物与源码快照保留，分析规则未变。

1469 帧绑定执行原 PairDataset 私有构造器的 `strong_by_weak=None` 唯一 stem 分支，远端 canonical 由已接受 capture 合同逐项查表。独立验证实际新 mapping 与 1469 个唯一 stem 对应及两份完整 roster 双射 exact，顺序保持 RGB 实际评价顺序。预期旧 val JSON 实查不存在，已保留勘误；新 mapping 明确是本次执行原 fallback 的产物，不冒称存在的旧输入。

逐帧执行原最大基数、再最大总 IoU 的同类 GT assignment，独立重建全部 **21568 对 GT**；它与 native GT/pred 匹配是两个不同规则。双方 own-GT 总量分别闭合 **22462 / 24490**，未配对 **894 / 2922**。全部 **46952 个 own-GT** 的框、类、稳定缓存 ID 和双向伙伴 ID 一致；未配对的对侧状态保持 null。

另直接执行固定原生 `box_iou` / `match_predictions` 函数，共 **11752 次**独立条件调用，核对 **148812 条原预测身份**、完整 TP 位、GT/pred 对及过滤后 ID。四组状态及 IoU 与保存输出一致，IoU 最大差 **0**。score > .25 从原缓存保序过滤；.5/.75 各自调用匹配，不从较低阈值的配对过滤生成较高阈值结果，也不再次运行 NMS。

| 固定条件 | 双方正确 | 仅 T 正确 | 仅 N 正确 | 双方未匹配 |
|---|---:|---:|---:|---:|
| 原低阈值缓存，IoU .5 | 20756 | 554 | 173 | 85 |
| 原低阈值缓存，IoU .75 | 16249 | 2817 | 1260 | 1242 |
| score > .25，IoU .5 | 18309 | 1873 | 587 | 799 |
| score > .25，IoU .75 | 14767 | 3399 | 1414 | 1988 |

表中每行分母均为 **21568 对 GT**。五类配对/未配对表、对象比例与图数、各模型完整 TP/FP/FN、两个置信过滤转移表及 16 格 IoU 联合表均独立闭合。图数有重叠不能直接相加；完整模态 FP 是该匹配规则下未获 TP 的预测，不能全部称为背景。

T 始终相对独立 IR own-GT 判断，再把已配对对象的状态联系到 RGB 伙伴。表中的“仅 T 正确”是固定标签关联下的差异模式；原生尺寸一致、同名帧及 GT IoU 门均不认证物理配准。N 为 weight0 协议 N42，T 为历史 IR42，不能从表格推出同配方训练增益。

本次执行环境为 local Torch 1.8.0 CPU float32，调用固定 Ultralytics 8.4.115 原生函数；不宣称跨运行时复现原 GPU TP bitwise。没有 GPU、forward、NMS、权重加载、新 hash 或 AP 重估。原 attempt2 的 PENDING 状态保留，此文件提供外部限定验收。

独立实际回执：`DRONE_TABLES_ACTUAL_RECEIPT.json`；完整复算脚本：`review_drone_tables.py`。当前源码、原始缓存及作者输出均未改。
