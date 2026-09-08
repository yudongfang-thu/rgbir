# 稳定GT身份与实际C选择映射：限定独立复核

只读源码范围 `release/mapping_trace.py`、`release/raw_object_state.py`、`release/export_selection.py`，对照本地 pinned IndependentKD 的 `selection_adapter.py` 与冻结OEv1 `_choose`。没有启动GPU、载入权重、读真实新覆盖结果或计算hash。

稳定ID链可接受进入真实首批核验：源身份明确是native verified `dataset.labels` 的行号，含canonical源图路径，不冒称原XML或文本行号。tap仅读取唯一原RandomPerspective最终filter bool，在同一原变换返回值上记录保留原行，再与实际collated cls/bboxes/global batch row exact对齐。没有最近邻重配身份、第二次transform或新的随机抽样。实际pinned Pair/Tracked类及native过滤函数体的8项CPU检查独立复跑全部通过，见 [INDEPENDENT_MAPPING_CPU.json](INDEPENDENT_MAPPING_CPU.json)，包含同类中间行删除后保留 `[0,2]`、空集、RNG/loader generator/payload不变及身份异常拒绝。CPU结果不能替代真实首批对原stream的检查，producer仍硬要求该项。

实际C门链源码核对无确定错位：完整当前batch调用既有selected-only API的full诊断；thin/full matched/base IDs、映射、valid、q、eligible与selected/rank exact比较；为暴露nonbase丢弃项，用相同原helper重建，按全局RGB/IR行和matched顺序再次与full selector统计和tensor比较。base为region与R粗候选共同成立；teacher own正确与q>0在base之后约束eligible；原 `_choose` 对整批eligible取ceil(rho×N)，q降序、exact tie保留matched顺序。没有逐图重选。返回的matched/base为0起、selected/eligible rank为1起，聚合器按此接受。

导出器6组CPU真值也已独立复跑通过，见 [INDEPENDENT_EXPORT_CPU.json](INDEPENDENT_EXPORT_CPU.json)：实际pinned selector与selected-only API的FP16 thin/FP32 full exact；交错global GT顺序、未配对与空图；nonbase/q=0仍保留；own/cross状态优先级；any-candidate与一对一分配反例；错位身份拒绝及输入/梯度未改。没有新模型前向。

S/R/T状态使用GT辅助的一对一粗空间分配，与真实C的any-candidate门不同；特别保留T相对IR own GT主状态和同框对RGB GT次级状态。未配对T未知/null，而不是错误。源码将这两层分开，聚合器也分别输出，不从状态proxy反推actual gate。

接受范围仅稳定映射、原选择门导出与一次真实前向探针准备。没有AP、负迁移消除、总体修复率或物理配准结论；增强丢弃的原GT不能误计为S错误。
