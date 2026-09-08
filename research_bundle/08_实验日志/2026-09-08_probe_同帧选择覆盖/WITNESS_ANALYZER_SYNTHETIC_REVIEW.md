# Witness analyzer 独立轻量复核

范围内 READY。只读回读 `witness_analyzer.py` / `test_witness_analyzer_cpu.py` 并独立执行4项合成真值，4/4 PASS，见 `WITNESS_ANALYZER_INDEPENDENT_CPU.json`。未重新计算真实 witness 结果。

确认 producer 唯一 scope、实际 multi_label=true、>=/.25 与 >/.25 区别、原 records 除 detector 外 exact、own-GT/stable ID、T 原 dense门 exact、native prediction frame/模型内不复用、四种定义和全部分母闭合。GT匹配不是另写近似算法，依赖 producer 的实际 native return-frame 配对及逐TP验收。结果仅单批阈值诊断，不升级为 AP、学习效果或负迁移。

真实产物验收与 pinned installed-native CPU 由 root/loc 的另行回执负责；本回执不替代该证据。
