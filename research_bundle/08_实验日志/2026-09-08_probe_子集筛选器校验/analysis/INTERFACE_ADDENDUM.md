# 终态接口补充

在读取实际新 AP 前，核对已完成的 `release/evaluate_short_screen.py` 源码：完成回执的 scope、endpoint、系数、BN、初始化 stat、三条子集路径、完整 dev 人口及 fraction 指标字段均与分析器一致。

该 evaluator 在预算截止时以 `short_evaluation_failure.json` 保存 `SUBSET_SCREEN_INCOMPLETE`。分析器因此显式接收同 scope 的这个状态，输出 `INCOMPLETE`、指标和差值均为 null，并保留原因；普通异常仍为 `FAILED`。错误或缺失 scope 拒绝，完成和失败文件同时存在仍为 `CONFLICT`，不选择任一结果。

原已审阅源码和测试分别保留为 `analyze_subset_e8_before_incomplete.py`、`test_subset_analyzer_cpu_before_incomplete.py`。仅在原缺失/失败真值中补接口情况，`CPU_attempt2.json` 仍为 6 项全部通过；未改变 AP、单位、差值、固定臂身份或解释规则，未读取新 AP。
