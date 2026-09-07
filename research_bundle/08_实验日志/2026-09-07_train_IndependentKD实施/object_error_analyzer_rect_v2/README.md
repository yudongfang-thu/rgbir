# 对象分析器 rect v2：名义尺寸与实际画布技术修复

**DRAFT，尚未独立接受，尚未运行真实分析。** 原v1源码/测试/规则及其接受回执不改；真实第一次CPU失败保存在相邻`old_c0_object_diagnostics_v1/failed_cpu_attempt1`及94原输出。

本版复制三个文件后最小修复：删除实际canvas最大边必须等于640的错误条件，保留shape有限正整数、成对原图/画布/GT完全相同；名义640由bound配置和实际合同共同检查。实际shape只来自receipt-bound objects，contract提供实际loader顺序、batch与名义输入设置，据此核每个实际batch共享同一canvas并输出观察摘要。未声称contract自身有shape列表，未猜测原图resize比或改框。

`canvas640_*`标签一一改成`input640_*`，指名义imgsz640；原始32²/96²面积阈值、置信度.25、匹配IoU.50、背景IoU<.10、修复/损伤分母均不变。输出schema改v2用于识别技术修复，供下游消费的ERROR_CONTRACT保持原定义。

保留旧15项测试的全部断言语义；仅补齐文件fixture的实际loader顺序及名义imgsz/batch配置。新加544×672正确匹配、padding平移面积不变、成对canvas/批内shape不一致拒绝（含混合原图纵横比）以及receipt-bound矩形画布载入测试。本版作者只执行这些合成CPU验证，真实三seed待根独立review和新部署后再跑。
