# Anchor identity join

有效结果在 `output_attempt3/`，待独立验收。11个定位机会的历史R候选与native S/R匹配anchor全部相同；其中实际L2选中的4个学习anchor也全相同。未selected的另外7个只能称候选位置相同，没有实际L2学习。全selected7为same5/different1/native-GT-unmatched1。7个实际T候选均不是native T匹配anchor；这不意味着目标错误，原候选规则不同，框与目标问题由另项审阅。

程序只核真实frame/GT/prediction/anchor身份，逐对象点位、原native/FP32框、原门与null均保留。历史S学习位置的DFL与decode未记录，不能把R或当前native值填成历史S。C区域池化没有单一学习anchor，C selected与L2 selected分开。范围见 `PLAN.md`，整数表、缺字段和输入stat见有效summary。

5项CPU真值在CPU_attempt1/2均PASS。最初两次直接Python脚本启动在进入程序前报源UTF-8解析错误，虽然相同文件的read_bytes/decode/compile和import测试都通过；原因未确定，不推测服务器/数据问题。output_attempt1/2保留该失败，原UTF8源在source_utf8_attempt1.py。仅将非ASCII显示字符串换成等价Unicode转义，AST前后exact（SOURCE_DISPLAY_ENCODING_FIX.json），分析逻辑、输入和门完全不变，随后output_attempt3成功。

复跑使用新输出目录：

```text
python test_anchor_join_cpu.py --output NEW_CPU.json
python analyze_anchor_join.py --core CORE_OUTPUT --bridge BRIDGE_OUTPUT --witness WITNESS_PROBE --l2-source EXECUTED_LOCALIZATION_BOX_V2.py --output NEW_OUTPUT
```

不重跑匹配、selector、模型或GPU；没有新训练、梯度或增益主张。
