# Drone AP交叉复核

> 六端点×AP50/AP75经独立COCO评价器重算，在使用与TIDE相同的101个recall点表示时最大绝对差2.842e−14pp。

原始GT/预测缓存不变，执行入口为`root_coco_crosscheck.py`，成功产物在`root_coco_review_v3`。默认COCO的`np.linspace`与TIDE的整数除以100在浮点边界有微小差异；v3同时保留默认COCO与统一网格结果，没有修改TIDE数字。此项核对TIDE AP实现，不把它等同于工程pinned指标。

v1 Windows默认文本编码错误、v2默认网格严格零差断言失败的产物均保留。v3的网格差异解释只修正交叉复核条件，不构成训练结果变更；数值单位均明确为百分制AP/pp。
