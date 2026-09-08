# 实际 v2 八批校准独立审计

**PASS（限校准计算、剂量与资源记录）**。从8条实际梯度记录独立重算，7/8批有限非零，固定共享 λ=0.6900524651944485；DFL与GT全部649个基础对象和25个选中对象同掩码、同anchor、同分母。第3批无选中，保留为0梯度，没有替换批次。

八批资源门全部通过。NVML峰5454 MiB、framework allocated峰4390.573 MiB、reserved峰4954 MiB、进程树RSS峰17280 MiB；原规则计算的显存预约5888 MiB，低于不变的8192 MiB。batch2–8开始allocated均为82.584 MiB，清理后为382.5889–382.5898 MiB，reserved全程4954 MiB，没有empty_cache；固定八批内未再出现原跨批累积现象。

v2首批完整JSON记录与失败attempt1首批逐值一致，包括32文件、梯度norm/cosine和完整对象选择统计。执行config与v2独立快照逐字节一致；source manifest指向reviewed release_v2。全模型初始状态/逐批恢复/BN冻结/零optimizer与EMA更新由执行路径断言和终态记录支持，本审计没有另跑模型重建梯度。

共享λ不等于相同梯度剂量：非零批实际 `λ·norm(B·KD)/norm(native)` 的中位数，DFL=0.100000000，GT=0.122837405；每批余弦和剂量都保留于RECOMPUTATION.json。GT没有单独重校准。

校准计算耗时37.566秒。收集快照的外层dispatcher状态仍为RUNNING，因此本审计不宣称外层进程或后续canary/FT3已完成；它不阻断也不绕过原encoded queue的后续实际门槛。没有读取新AP，不作KD效用或物理配准结论。原失败attempt1保留，v2仅在本固定八批范围证实资源恢复稳定。
