# 预算算术独立复核

**PASS_ARITHMETIC_SCOPE：最终 budget.py 的 7 项合成 CPU 真值通过。** 本复核未读取新 AP、实际 canary 测量或 checkpoint，未执行 SSH、GPU、scheduler，也未计算 hash。

固定两臂分别使用全部实测 batch 时长均值，512 批外推、非 batch 开销和前 30 批像素审计 I/O 均进入 1.2 倍训练估计；已花执行时间与两次各 60 秒评价预留计入 2700 秒总额。真值覆盖恰好 2700 秒接受、略超拒绝、两臂不同速度及慢批不可删除、缺测/非有限/错臂拒绝、I/O 不遗漏，以及实测峰值加余量后的向上取整。最新代码另将 batch 时长与 I/O 合计超 stage 时间的输入拒绝，此负例也通过。

有效回执为 `BUDGET_INDEPENDENT_CPU_attempt2.json`，当时源码为同名前缀的 `_budget_source.py`；执行前后逐字节一致。初版回执、源码和测试保留为 attempt1。复现入口为 `review_budget_cpu.py --output <新的 JSON 路径>`。

范围仅为公式和输入校验：本回执不代替真实 RUNNING 到终态的计时、ADMISSION 等待分离、进程终止、动态资源准入或实际 45 分钟截止执行验证。`measured_framework_peak_mib` 当前保存的是 NVML/框架峰值三者的最大值，预约使用此最大值正确；该字段应按“预约基准峰值”解释，不能单独当作框架仪表原值。
