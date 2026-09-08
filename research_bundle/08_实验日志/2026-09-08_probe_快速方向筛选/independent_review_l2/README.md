# L2 算子独立快审：CPU 限定通过

**未发现阻碍有界新路径 canary 的确定算子问题。独立重跑作者 13 项真值全部 PASS，另补 4 项 CPU 检查全部 PASS；不涉及 GPU、AP 或训练准入，不解除原 L1 几何阻塞。** 审阅源为 `newentry/release/localization_box_v2.py`（13,285 字节）及 pinned `selection_adapter` / `task_conditional_reference/localization_loss.py`，未修改作者源。

[作者真值独立执行回执](author_tests_independent_run.json)、[独立补测源码](independent_checks.py)、[补测回执](independent_checks_receipt.json)。CUDA 未初始化。

已核对 R 的 confidence→IoU→anchor ID 排序、未 clamp 的原 DFL 支持、两侧全部 GT 的 native unique owner、coarse R base 先于后续质量门且保留全 base 分母。teacher 独立 anchor 采用 own-IoU→confidence→ID；选择不依赖学生值。学生只在 R anchor 通过相同 float32 DFL 期望解码回传梯度，teacher/reference 和 score 门均 detached。L2-box 与 L2-GT 使用同一选择与分母，后者目标为相同单位 GT 框；均保留 native loss，lambda 与 batch multiplier 由外层统一应用。

补测确认真实 LLVIP 单类 nc=1 可以产生非零有限学生 DFL 梯度；R 最高 confidence 即使 IoU 较低仍优先选取，提高另一个完美框的 confidence 会使其进入 R gap 排除；这些行为符合代码固定合同。

必须保留的解释边界：IR GT→RGB GT 的独立正 x/y 仿射对 box-vs-GT IoU 不变，因此 mapped RGB IoU 门不是第二份独立物理几何证据；它在这一坐标定义下等价于 teacher own-IoU 的更强阈值。RGB/IR 共享 GT 时映射为恒等，仍不代表真实配准已认证。L2 是 GT 条件下对象相对框回归候选，不是原 L1 同 anchor/bin 蒸馏获得准入。已用已知真值验证这一边界，不新增数据分析或方法调参。

本次未审阅真实训练外层的 lambda 缩放、BN 冻结、EMA、数据流及资源日程；这些由本轮新入口已有 canary/整合检查承接，不据 CPU 算子 PASS 声称已经训练有效。
