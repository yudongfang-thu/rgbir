# 对象诊断工具实施记录

**结论：首版完成，15/15 已知真值 CPU 测试通过，状态 NOT_ACCEPTED，待独立审阅。未读取新 C1 AP 或运行 GPU。**

2026-09-07。文件：`object_error_analysis.py`、`test_object_error_analysis.py`、`OBJECT_ERROR_ANALYSIS_RULES.md`，均位于实施日志而非冻结训练模块。

验证命令：`D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_object_error_analysis -v`；15项通过，0.716s。覆盖修复/损伤不同分母、重复预测与空GT图、类别错误、.25/.50/.10阈值边界、确定性匹配、GT数组顺序、640 canvas尺度、UNKNOWN及分组图像分母、输入不变、原始objects/contract字节回执、默认draft不能误入accepted analyzer。

输出默认仅有 draft_contract，接入正式 analyzer 需独立review绑定本脚本/fixture/规则三份源码。`localization_diagnostics_consistent`始终null，不能用全检测层面的修复率替代L的同anchor/几何/梯度证据。未改已接受 analyzer 或训练代码。
