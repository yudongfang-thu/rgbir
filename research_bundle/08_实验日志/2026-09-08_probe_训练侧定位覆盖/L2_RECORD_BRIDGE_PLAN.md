# 首批原 L2 记录与原生定位对象的关联审计

本轮首32缓存CPU定位读出已完成，11个双方粗检出而仅T达到IoU.75。只读检查发现，旧 `2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip/calibration_batches.jsonl` 第1行的32图顺序与当前原stream exact；其79个L2 base record的batch图索引、RGB/IR全局GT行、类别及双GT框与当前80对象记录逐项一致。原首批记录为base79、reference_gap7、selected7。

下一项只读关联固定如下：核对两处cfg的模型/参考/教师及该批身份；保留全部79条历史base record，通过已经核实的图与双GT行连接到稳定ID，未落入base的对象明确标记未进base。原selected不重算或改阈值，关联IoU.5/.75的实际匹配状态及原C选择，统计11个定位机会在历史L2 base/reference_reliable/reference_gap/teacher_quality/selected中的覆盖与排除原因。

这是旧L2校准同批记录的审计，不冒称在此次见证forward中重新运行L2，也不需要GPU。模型参数校准逐批恢复、R/T冻结、BN冻结、原流和逐记录身份事实分别记录；旧L2 raw FP32 boxes与native AMP解码的框差异不偷偷当同一数组。可用dense_witness锚与框做实际重叠字段核验，覆盖不足就如实说明，不能补造整张raw输出exact。

同时区分native-box .70/.05代理与实际L2锚选择。既有代理读数4/11仅是native框门，历史L2有自己的confidence-first/unique-owner/P3P4/support条件。最终归因必须来自旧base_records实际字段，不能把代理数字当实际掩码。若缺关键身份/字段则阻塞该关联，不追加前向或放宽门。

保留脚本、输入stat、小结果、逐ID记录及独立复核；不修改旧校准回执、系数或训练结论，不把这项审计作为扩大训练矩阵的许可。
