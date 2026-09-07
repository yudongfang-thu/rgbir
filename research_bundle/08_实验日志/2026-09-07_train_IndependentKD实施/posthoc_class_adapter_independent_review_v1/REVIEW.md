# 补评逐类指标适配器的独立审阅

**结论：ACCEPT。此额外适配接口的代码范围接受，未发现阻塞问题。17/17 CPU 测试与真实六端点独立执行通过；不接受方法有效性，不改变 CLI 的 DRAFT 身份或自动扩展禁用状态。**

日期：2026-09-07。审阅者 `/root/review_c1_spec` 未编写或修改根实现的 `posthoc_class_adapter.py`、测试或实际 manifest 准备脚本。已阅读这三份源码、README、DEVELOPMENT_FAILURES、actual_manifest 和 actual_analysis_attempt2，并核对冻结分析器的 load_endpoint、单位转换、配对、harm_review 接口。

## 适配身份与来源

只有 manifest 显式 `posthoc_class_metrics=true` 时启用。原端点先由冻结分析器验证其训练/评价、协议、完整名单和已接受 observed bridge；原有逐类字段非空时拒绝覆盖。新 metric 的 seed/checkpoint/实际原arm、规范方法arm、source、dataset、endpoint、method_id、val/test标记与旧端点及实际补评 receipt/completion 逐项核对。

实际 receipt 的 seed 位于顶层，其余方法身份位于 inputs；当前实现已按这个真实 schema 读取，两侧 seed 均有验证。没有保留开发首轮把 seed 错从 inputs 读取的错误。

补评 metric/receipt/completion/contract/roster 和九份实际评价源码均从根已接受 observed bridge 的独立副本取得，并再次与其真实输入原字节比较。指标还需通过原 receipt 的 metric_snapshot 别名绑定，七份共同源码与 canonical 对齐，合同与原端点完整名单一致。此范围使用已接受 bridge 的实际来源映射，不声称旧训练当时保存了补评字段。

适配只向深复制的内存记录添加 `per_class_percent`、额外三种逐类AP值及 `per_class_provenance`。provenance 明确标记为另行执行的 post-hoc 评价，指向metric/receipt/observed bridge并保留checkpoint和seed。原五项汇总指标不变，旧JSON不写入。

## 单位、完整性和配对

类别ID必须是整数且唯一，集合必须覆盖原端点的全部预期类别；bool ID、缺类、重复类、非有限值和越界AP被拒绝。三个逐类指标通过冻结 to_percent 转成百分数，再与各自总指标的宏平均核对。真实六模型的五类AP/AP50/AP75都独立检查为补评原fraction乘100，未混用pp和fraction。

三seed配对、样本SD和harm触发继续调用冻结分析器。新增字段不改变原协议、roster、教师/参考或训练recipe，也不让未完成C1出现AP端点。实际六记录仍只包含 N/C0 × seed0/42/123。

## 独立执行结果

- `python -m unittest test_posthoc_class_adapter -v`：17/17通过，输出在 `cpu_tests.log`。
- `run_independent_checks.py` 调用真实CLI重新加载六个端点，生成本目录新文件 `actual_analysis_independent.json`；该文件解析后与根提交的 `actual_analysis_attempt2.json` 完全一致。六记录均complete且各有五类。
- 显式调用冻结 `paired_comparison(records,'C0','N')`：三seed完整且无身份问题，原均值和样本SD仍为 +0.2666552104 ±0.1443728999 pp。
- 显式调用冻结 `harm_review(records,'C0')`：**REVIEW_REQUIRED，missing=[]**。seed0/42/123 的背景误检仍分别增加 **3/5/46**，触发“三seed背景FP/image均增加”。逐类AP补齐没有将此结论清为CLEAR；该状态仍是审阅触发，不证明具有实质负迁移。
- opt-out 保持只有原记录；用调用方持有的真实记录测试 opted-in 路径，原字典没有被修改。重新调用CLI写同一输出触发 FileExistsError，既有输出原字节不变。
- 运行前后逐字节核对30份旧 completion/metric/train receipt/eval receipt/roster，全部不变；根源码也未被审阅改动。

主要完整检查结果为 `independent_execution_checks.json`。根另提供的 pinned CPU 回执记录17项通过；本审阅只读取该文件，不把它说成审阅者自己的远端执行。审阅者没有GPU、SSH或联网操作。

## 接受边界

独立 `review_receipt.json` 绑定本次确切五份适配/测试/准备脚本/文档副本，并另存三个冻结分析器依赖源码，以便定位此接受范围。CLI仍写 `DRAFT_AWAITING_INDEPENDENT_REVIEW`、analyzer_accepted=False、implementation_checks_passed=False，两类自动扩展资格均False；本次审阅没有修改这些值。

已接受的rect-v2对象诊断是本次显式manifest的既有输入。适配器不重算对象匹配，不改阈值，不推断昼夜标签，不更新对象诊断的科学接受范围。C1仍待真实端点，四臂归因、正式方法贡献、自动排程资格和论文主张均不由这份额外接口审阅签出。
