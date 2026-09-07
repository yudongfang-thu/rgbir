# 独立分析器首轮审阅

> **首轮21项analyzer＋8项protocol＝29项CPU测试实际通过；不是26项。基础数值逻辑可以复用，但正式接受仍需补充评估身份和内容对照身份绑定，当前不生成ACCEPTED回执。**

审阅者：`review_l1_spec`，不是被审代码作者。读取 `protocol.py`、`analyze_independent.py`、两份测试和旧receipt loader，并在本地Python3.8实际运行两组unittest。只读源码，没有修改作者文件；修订已发给作者 `review_matrix_spec`。

## 已核验正确的部分

- metric单位显式转换；拒绝bool/NaN/越界；ddof=1样本SD；逐seed差而非臂SD相减。
- 缺seed/失效receipt/endpoint或roster不一致不能完成三seed比较；跨seed recipe差异不聚合。
- 不按一个好seed提前结束，`stop_running_experiments=false`；三seed方向门属于投入判据。
- N/类别/定位单分支系数防串线；N/C0逻辑明确；protocol.valid不等于receipt已核验。
- 固定阈值背景FP审计不由YOLO best-F1 recall填充；缺固定操作点诊断会阻止自动升级。
- L_GT/C1_y不冒充shuffled/same_modal；数据集分组防止LLVIP借用Drone N。
- 训练/评估完整receipt及bound metric副本可以拒绝失效或单独篡改AP；manifest不能把已有paired arm简单重标为same_modal。
- `analyzer_accepted`默认为false；CLI需独立review receipt和直接byte相等的reviewed source copies，测试通过本身不会写accepted状态。

## 正式接受前需要修复

### P1：evaluator_contract现在来自manifest声明

`load_endpoint`直接将spec.evaluator_contract填入记录，而旧loader只确认eval source/config snapshot文件存在，没有读取实际precision/NMS/imgsz/batch等有效设置。这能检查两个manifest声明是否相同，不能证明实际评估相同。

要求：从receipt绑定的评估配置/有效kwargs取比较合同；已有旧端点缺该合同则保持descriptive-only，或以明确的独立兼容检查回执桥接。不能靠给两个manifest都写相同字符串放行正式比较。

### P1：内容对照缺少干预身份比较

当前PAIR_FIELDS只有native recipe与数据/评估。被复用的旧 `_recipe` 明确排除teacher/reference/λ/geometry/方法ID；因此C1与C1_y、L1与L_GT即使这些身份不同，仍可能在native recipe相同的情况下产生 `non_target_content_supported` / `teacher_content_supported`。

要求：从训练receipt绑定的完整cfg提取干预合同；按比较类型建立允许差异清单。C1/C1_y必须同T/R、选择、τ、clip、λ，仅payload的eta定义允许不同；L1/L_GT必须同T/R、geometry/support版本、selection、τ、λ，仅target定义不同。N对照与same-modal对照允许哪些差异另按预登记规则核对，不能要求所有字段全相等，也不能全部忽略。

### P2：旧loader依赖应指向冻结副本

`_legacy_loader` 与 `verify_analyzer_acceptance` 当前都指邻接旧v1/analyze_results.py。v2已冻结 `task_conditional_reference/`，实际运行与接受回执应绑定同一个选中的冻结依赖。避免部署后与review源不同或邻接旧目录变化。

## 其余解释边界

`analyze_records`是已验证记录的纯逻辑内核，测试通过手工真值dictionary调用是正常单元测试，不应被报告成真实端点验证。`proposed_decision=L1_EFFECTIVE`与`expansion_status=AWAIT_LOCALIZATION_DIAGNOSTICS`可以同时出现，报告者必须用完整状态，不能摘取前者当作定位机制通过。

当前29测试缺少上述bound evaluator和内容合同反例，因此全绿不足以接受。修复后应增加“改变实际bound evaluator/λ/teacher/geometry，但manifest表面相同”的拒绝测试，再由独立审阅者复查源码与直接byte副本。

首轮源码副本：本日志 `analyzer_review_sources_v1/`，含analyze_independent.py、protocol.py、两份测试及legacy_analyze_results.py。此目录只是审阅证据，**没有ACCEPTED JSON**。本轮没有产生训练结论、真实AP或自动扩展授权。

## 后续修订状态（保留首轮发现）

作者修复上述问题及后续发现的跨seed干预同质性、完整类别inventory后，独立审阅者重新阅读并实际运行36项测试通过。接受时源码副本和严格范围见 `analyzer_acceptance_v2_20260907/FINAL_ANALYZER_REVIEW.md` 与同目录 `analyzer_acceptance.json`；直接字节检查实际返回True。首轮源码/反例不覆盖，接受不代表真实新方法/几何/训练资格通过。
