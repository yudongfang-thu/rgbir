# 初始参照小包独立复核

**通过来源限定复核，可作为新三臂的“成熟初始化”第二参照；不能代替本轮匹配N。** 原值：mAP50–95=`0.3287840510939928`，AP50=`0.7161396006544973`，AP75=`0.24282631410771383`，均为fraction。未重新评价或计算AP。

独立回读确认：本包与旧条目的receipt/contract/roster逐字节一致，remote receipt副本一致；完整dev为2406图/7879 GT、person。receipt与contract的visible42 last.pt路径、5482010 bytes、mtime_ns=1788127857140185407，与已执行DFL读出的S/R初始化stat一致。新N/L3-DFL/L3-GT配置使用该模型路径及相同full-dev YAML/版本；native配置与旧实际保存配置逐字节一致，新evaluator参数常量与旧执行参数一致（FP32、640/B32/workers4、conf=.001、NMS IoU=.7、max_det300）。只解析源码常量，不导入或执行评价器。

旧评价scope仍是`INITIAL_BASELINE_NATIVE_RECHECK`，其`accepted_endpoint_claim=false`保留；本复核不将其改造为新的原始预测AP验收或FT完成端点。新三臂尚须各自实际初始化stat、完整dev roster与actual effective profile闭合后，才能报告相对该初始化的差。当前配置匹配不等于未完成运行已匹配，且初始化差不能替代DFL−N/DFL−GT。

复核者`/root/ap_error`；[机器回执](INDEPENDENT_REFERENCE_REVIEW.json)与[可重复只读入口](independent_verify_small_package.py)。无新SSH/GPU/forward/权重读取/hash；未修改已有包或重复收集。
