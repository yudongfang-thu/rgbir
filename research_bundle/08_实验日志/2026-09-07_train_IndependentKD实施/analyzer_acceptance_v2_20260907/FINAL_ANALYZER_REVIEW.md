# 独立分析器修订版接受审阅

> **接受本目录直接源码副本所绑定的分析器，范围限receipt约束的开发集描述统计和固定单分支决策逻辑。独立重跑28项analyzer＋8项protocol，共36项全部通过。此接受不批准训练、不认定任何新AP、几何或方法有效。**

审阅者为 `/root/review_l1_spec`，不是analyzer/protocol作者。首轮源码、29项测试及拒绝原因保存在父目录 `ANALYZER_REVIEW.md` / `analyzer_review_sources_v1/`。作者 `review_matrix_spec` 按发现逐项修复后，审阅者重新阅读修订函数及测试，并在本地CPU实际执行36项测试。

## 已修复并核验

1. **实际评估合同绑定。** file-loader不再用manifest自报evaluator/count作为事实，转而读取eval receipt绑定的合同、有效配置及实际evaluator源码内容。manifest同字符串不能掩盖源码/half等实际差异。旧端点若缺合同，仍可保留原始AP但不通过正式endpoint验证；兼容桥接需独立accepted回执、精确train/eval配置、实际源码独立副本和canonical等价源码。
2. **内容对照身份。** receipt绑定干预记录进入比较：C1/C1_y的T/R、selection、λ、共同payload；L1/L_GT额外geometry内容；same-modal允许预登记的教师/数据差异。缺失或不符时不能产出完整可支持的内容比较。
3. **跨seed同质性。** 两臂在seed123一起换teacher/λ，不再被当作固定教师与剂量的三seed实验：每seed原值可以保留，但聚合stats被withhold，arm summary也不报告同质三seed。首轮确切合成反例另存父目录 `analyzer_cross_seed_counterexample_v1.json`，其中数值不是实验AP。
4. **类别完整性。** harm review要求实际bound cfg的expected_nc所定义的完整类别ID，防止两臂同时漏一类后错误CLEAR。
5. **冻结依赖。** `_legacy_path` 优先本v2冻结的 `task_conditional_reference/analyze_results.py`；本接受回执绑定这个实际依赖文件的直接副本。

基础逻辑仍保持显式metric单位、ddof=1、严格seed配对、完整E200 last/EMA与val总体、重复attempt拒绝、receipt失效拒绝、背景FP固定操作点、缺损伤诊断不升级、无自动早停、四臂身份分离。

## 接受与使用

`analyzer_acceptance.json` 的source_snapshots指向本目录source_copies。创建后实际运行 `verify_analyzer_acceptance(...)` 返回True；没有计算hash。修改任何被绑定源码将使该检查失败，须再审阅。

本接受验证的是分析器实现，不替代真实实验的source/config/model/几何/数据/梯度/资源验收。`implementation_checks_passed`在接受回执中保持false，不能拿该回执自动填写为true。方法内容与归因仍由真实三seed、GT/目标类控制、四臂和错误诊断证明。

对于reviewer创建的合成字典，`analyze_records`只是数学内核；真实使用必须走file-loader/manifest且绑定原始receipt。ACCEPTED bridge本身还需真正独立审核，不能仅复制字段制造兼容。没有新方法结果时，本次接受不会生成预期AP或升级结论。

## 实际测试

- Python3.8.0，本地CPU，无GPU、无SSH。
- `python -m unittest discover ... -p test_analyze_independent.py -v`：28项通过。
- `python -m unittest discover ... -p test_protocol.py -v`：8项通过。
- 本次没有修改analyzer/protocol作者文件，source_copies为接受时直接字节副本。
